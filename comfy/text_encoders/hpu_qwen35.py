"""Modular HPU compilation, channel-last GDN state flow, and shared FP8 quantization for Qwen3.5 MTP."""
from functools import lru_cache
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops

from . import hpu_kv_attention
from .llama import FixedKVCache, apply_penalty, apply_rope, penalty_active

if comfy.model_management.hpu_available:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA


@lru_cache(maxsize=4)
def _divisor(device):
    return torch.tensor(240.0, dtype=torch.float32, device=device)


def is_dynamic_fp8(linear):
    return (getattr(linear, "layout_type", None) == "HPUFP8E4M3Layout"
            and not getattr(linear, "_full_precision_mm", False)
            and getattr(linear, "bias", None) is None
            and not getattr(linear, "weight_function", None)
            and not getattr(linear, "bias_function", None)
            and getattr(linear, "input_scale", None) is None
            and getattr(linear, "pre_quant_scale", None) is None
            and not hasattr(linear, "_v")
            and getattr(linear, "weight_lowvram_function", None) is None
            and getattr(linear, "bias_lowvram_function", None) is None)


def _weight_pair(linear):
    """Extract (qdata, scale) for dynamic FP8 or (weight, None) for dense linears."""
    if is_dynamic_fp8(linear):
        return linear.weight._qdata, linear.weight._params.scale
    return linear.weight, None


def _rms_norm(x, weight, eps, add_one=True):
    xf = x.float()
    normed = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return normed * (1.0 + weight) if add_one else normed * weight


def _quantize_fp8(x, divisor):
    flat = x.reshape(-1, x.shape[-1])
    scale = flat.abs().amax(-1, keepdim=True).float().clamp_min(1e-12) / divisor
    quantized = torch.ops.hpu.cast_to_fp8_v2(flat, scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    return quantized, scale


def _fp8_proj_raw(quantized, scale, w_qdata, w_scale, orig_shape, orig_dtype):
    out = torch.ops.hpu.fp8_gemm_v2(
        quantized, False, w_qdata, True, None, orig_dtype,
        scale, w_scale, None, False,
    )
    return out.reshape(*orig_shape[:-1], w_qdata.shape[0])


def _linear_raw(x, w, ws, divisor):
    if ws is not None:
        qx, sx = _quantize_fp8(x, divisor)
        return _fp8_proj_raw(qx, sx, w, ws, x.shape, x.dtype)
    return F.linear(x, w, None)


def _mlp_raw(x, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s, divisor, shared_quant):
    xm = _rms_norm(x, post_nw, eps, add_one)
    if shared_quant and gate_s is not None and up_s is not None:
        qm, sm = _quantize_fp8(xm, divisor)
        gate = _fp8_proj_raw(qm, sm, gate_w, gate_s, xm.shape, xm.dtype)
        up = _fp8_proj_raw(qm, sm, up_w, up_s, xm.shape, xm.dtype)
    else:
        gate = _linear_raw(xm, gate_w, gate_s, divisor)
        up = _linear_raw(xm, up_w, up_s, divisor)
    return _linear_raw(F.silu(gate) * up, down_w, down_s, divisor)


def _apply_partial_rope(xq, xk, rotary, rotary_dim, native_rope):
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]
    if native_rope:
        xq_rot, xk_rot = hpu_kv_attention._native_rope_forward(xq_rot, xk_rot, rotary)
    else:
        xq_rot, xk_rot = apply_rope(xq_rot, xk_rot, rotary)
    return torch.cat((xq_rot, xq_pass), dim=-1), torch.cat((xk_rot, xk_pass), dim=-1)


def _delta_vh(state, keys, values, decay, beta, parents, doublings):
    """Parallel FP32 delta recurrence on [B, vh, R, hd] tensors without inner transposes."""
    rows = decay.shape[-1]
    eye = torch.eye(rows, dtype=decay.dtype, device=decay.device)
    pos = torch.arange(rows, device=parents.device)
    transition = decay.unsqueeze(-1) * (parents.unsqueeze(-1) == pos).to(decay.dtype)
    paths, power = eye, transition
    for _ in range(doublings):
        paths = paths + power @ paths
        power = power @ power
    initial = (paths @ (decay * (parents < 0)).unsqueeze(-1)).squeeze(-1)
    k_t = keys.transpose(-1, -2)
    dependencies = (keys @ k_t) * paths * beta.unsqueeze(-1) * (1 - eye)
    updates = beta.unsqueeze(-1) * (values - initial.unsqueeze(-1) * (keys @ state))
    power = -dependencies
    for _ in range(doublings):
        updates = updates + power @ updates
        power = power @ power
    return updates, paths, initial, k_t


def _gdn_layer_core(x, w_tup, s_tup, ctl_tup, divisor, cfg_tup, shared_quant, commit_doublings, tree_doublings):
    (in_nw, qkv_w, qkv_s, z_w, z_s, wa, wb, conv_w_cl, conv_bias, g_decay, dt_bias, gdn_nw,
     out_w, out_s, post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s) = w_tup
    (conv_st, rec_st, p_proj_w, p_k_w, p_v_w, p_d_w, p_b_w,
     p_proj_r, p_k_r, p_v_r, p_d_r, p_b_r) = s_tup
    slots, live, conv_commit_idx, conv_src, tree_parents, commit_parents = ctl_tup
    kh, vh, k_hd, v_hd, taps, eps, add_one = cfg_tup
    key_dim = kh * k_hd
    groups = vh // kh
    scale = k_hd ** -0.5
    bs, seq, _ = x.shape

    xn = _rms_norm(x, in_nw, eps, add_one)
    if shared_quant and qkv_s is not None and z_s is not None:
        qx, sx = _quantize_fp8(xn, divisor)
        proj = _fp8_proj_raw(qx, sx, qkv_w, qkv_s, xn.shape, xn.dtype)
        z = _fp8_proj_raw(qx, sx, z_w, z_s, xn.shape, xn.dtype)
    else:
        proj = _linear_raw(xn, qkv_w, qkv_s, divisor)
        z = _linear_raw(xn, z_w, z_s, divisor)

    channels = proj.shape[2]
    # Channel-last [B, T, C] depthwise causal conv avoids scalar innermost-dim gathers on TPC.
    prev_proj = p_proj_r.index_select(1, slots)
    committed_conv = torch.cat((conv_st, prev_proj), dim=1).index_select(1, conv_commit_idx)
    conv_st.copy_(committed_conv)
    win = torch.cat((committed_conv, proj), dim=1)
    gathered = win.index_select(1, conv_src).view(bs, seq, taps, channels).float()
    acc = (gathered * conv_w_cl).sum(2)
    if conv_bias is not None:
        acc = acc + conv_bias
    y = acc.to(proj.dtype).float()
    current = (y / (1.0 + torch.exp(-y))).to(proj.dtype)
    p_proj_w.copy_(proj)

    beta = F.linear(xn, wb).sigmoid().float().transpose(1, 2)
    decay = torch.exp(g_decay * F.softplus(F.linear(xn, wa).float() + dt_bias)).transpose(1, 2)
    q_cl = current[:, :, :key_dim].reshape(bs, seq, kh, k_hd).float()
    k_cl = current[:, :, key_dim:2 * key_dim].reshape(bs, seq, kh, k_hd).float()
    v_cl = current[:, :, 2 * key_dim:].reshape(bs, seq, vh, v_hd).float()
    q_cl = q_cl / q_cl.square().sum(-1, keepdim=True).sqrt().clamp(min=1e-12) * scale
    k_cl = k_cl / k_cl.square().sum(-1, keepdim=True).sqrt().clamp(min=1e-12)
    if groups > 1:
        q_cl = q_cl.repeat_interleave(groups, dim=2)
        k_cl = k_cl.repeat_interleave(groups, dim=2)
    q_cl = q_cl.transpose(1, 2)
    k_cl = k_cl.transpose(1, 2)
    v_cl = v_cl.transpose(1, 2)
    p_k_w.copy_(k_cl)
    p_v_w.copy_(v_cl)
    p_d_w.copy_(decay)
    p_b_w.copy_(beta)

    pk = p_k_r.index_select(2, slots)
    pv = p_v_r.index_select(2, slots)
    pd = torch.where(live.view(1, 1, -1), p_d_r.index_select(2, slots), 1.0)
    pb = torch.where(live.view(1, 1, -1), p_b_r.index_select(2, slots), 0.0)
    u, paths, init, pk_t = _delta_vh(rec_st, pk, pv, pd, pb, commit_parents, commit_doublings)
    committed_state = rec_st * init[..., -1, None, None] + (pk_t * paths[..., -1, None, :]) @ u
    rec_st.copy_(committed_state)

    tu, tpaths, tinit, tk_t = _delta_vh(committed_state, k_cl, v_cl, decay, beta, tree_parents, tree_doublings)
    out = (tinit.unsqueeze(-1) * (q_cl @ committed_state) + ((q_cl @ tk_t) * tpaths) @ tu).transpose(1, 2)
    out = out.to(x.dtype).float()
    rstd = torch.rsqrt(out.square().mean(-1, keepdim=True) + eps)
    gated = ((out * rstd * gdn_nw).to(x.dtype) * F.silu(z.reshape(bs, seq, vh, v_hd))).reshape(bs, seq, -1)
    x = x + _linear_raw(gated, out_w, out_s, divisor)
    return x + _mlp_raw(x, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s, divisor, shared_quant)


def _gdn3_stage_fn(x, w0, s0, w1, s1, w2, s2, ctl_tup, divisor, cfg_tup, shared_quant, cd, td):
    x = _gdn_layer_core(x, w0, s0, ctl_tup, divisor, cfg_tup, shared_quant, cd, td)
    x = _gdn_layer_core(x, w1, s1, ctl_tup, divisor, cfg_tup, shared_quant, cd, td)
    return _gdn_layer_core(x, w2, s2, ctl_tup, divisor, cfg_tup, shared_quant, cd, td)


def _gdn1_stage_fn(x, w0, s0, ctl_tup, divisor, cfg_tup, shared_quant, cd, td):
    return _gdn_layer_core(x, w0, s0, ctl_tup, divisor, cfg_tup, shared_quant, cd, td)


def _stage4_fn(x, w0, s0, w1, s1, w2, s2, aw, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask,
               ctl_tup, divisor, gdn_cfg, attn_cfg, shared_quant, native_rope, use_fused_sdpa, cd, td):
    x = _gdn_layer_core(x, w0, s0, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td)
    x = _gdn_layer_core(x, w1, s1, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td)
    x = _gdn_layer_core(x, w2, s2, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td)
    return _attn_stage_fn(
        x, aw, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask,
        divisor, attn_cfg, shared_quant, native_rope, use_fused_sdpa,
    )


def _kv_commit_slot_fn(ks, vs, src, dst):
    for k, v in zip(ks, vs):
        k.index_copy_(2, dst, k.index_select(2, src))
        v.index_copy_(2, dst, v.index_select(2, src))
    return dst


def _build_tree_sdpa_mask_fn(key_cache, prefix_len, tree_mask_bits, seq):
    cap = key_cache.shape[2]
    rel = torch.arange(cap, device=key_cache.device).view(1, 1, 1, -1) - prefix_len.view(-1, 1, 1, 1)
    tree_vis = ((tree_mask_bits.view(1, 1, seq, 1) >> rel.clamp(0, seq - 1)) & 1).bool()
    visible = (rel < 0) | ((rel >= 0) & (rel < seq) & tree_vis)
    return torch.zeros(visible.shape, device=key_cache.device, dtype=key_cache.dtype).masked_fill(~visible, float("-inf"))


def _attn_stage_fn(x, w_tup, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask, divisor,
                   cfg_tup, shared_quant, native_rope, use_fused_sdpa):
    (in_nw, q_w, q_s, k_w, k_s, v_w, v_s, qn_w, kn_w, o_w, o_s,
     post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s) = w_tup
    num_heads, num_kv_heads, head_dim, rotary_dim, eps, add_one = cfg_tup
    bs, seq, _ = x.shape

    xn = _rms_norm(x, in_nw, eps, add_one)
    if shared_quant and q_s is not None and k_s is not None and v_s is not None:
        qx, sx = _quantize_fp8(xn, divisor)
        qg = _fp8_proj_raw(qx, sx, q_w, q_s, xn.shape, xn.dtype)
        xk = _fp8_proj_raw(qx, sx, k_w, k_s, xn.shape, xn.dtype)
        xv = _fp8_proj_raw(qx, sx, v_w, v_s, xn.shape, xn.dtype)
    else:
        qg = _linear_raw(xn, q_w, q_s, divisor)
        xk = _linear_raw(xn, k_w, k_s, divisor)
        xv = _linear_raw(xn, v_w, v_s, divisor)

    qg = qg.view(bs, seq, num_heads, head_dim * 2)
    xq, gate = qg[..., :head_dim], qg[..., head_dim:].reshape(bs, seq, -1)
    xq = _rms_norm(xq, qn_w, eps, add_one).transpose(1, 2)
    xk = _rms_norm(xk.view(bs, seq, num_kv_heads, head_dim), kn_w, eps, add_one).transpose(1, 2)
    xv = xv.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
    xq, xk = _apply_partial_rope(xq, xk, rot, rotary_dim, native_rope)

    kc.index_copy_(2, wpos, xk)
    vc.index_copy_(2, wpos, xv)

    if use_fused_sdpa:
        out = FusedSDPA.apply(xq, kc, vc, sdpa_mask, 0.0, False, None, "fp32", False).transpose(1, 2).reshape(bs, seq, -1)
    else:
        import comfy_kitchen
        o_pre, lse = comfy_kitchen.flash_attention_decode_gqa(xq, kc, vc, prefix_len, return_lse=True, causal=False)
        out = comfy_kitchen.flash_attention_decode_tree_merge(o_pre, lse, xq, xk, xv, tree_mask, torch.empty_like(o_pre))
    out = out * gate.sigmoid()
    x = x + _linear_raw(out, o_w, o_s, divisor)
    return x + _mlp_raw(x, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s, divisor, shared_quant)


def _sample_prep(logits_rows, verify_tokens, pen_mask, ancestor_indices,
                 temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, fp32_penalty=False):
    rows_lg = logits_rows.float() if fp32_penalty else logits_rows
    if pen_mask is not None and penalty_active(repetition_penalty, presence_penalty):
        rows = rows_lg.shape[0]
        anc_toks = verify_tokens[0].index_select(0, ancestor_indices.reshape(-1)).reshape(ancestor_indices.shape)
        row_mask = pen_mask.unsqueeze(0).expand(rows, -1).clone()
        row_mask.scatter_(1, anc_toks, torch.ones_like(anc_toks, dtype=torch.bool))
        rows_lg = torch.where(row_mask, apply_penalty(rows_lg, repetition_penalty, presence_penalty), rows_lg)

    if temperature != 1.0:
        rows_lg = rows_lg / temperature

    top_indices = None
    if top_k > 0:
        k = min(top_k, rows_lg.shape[-1])
        rows_lg, top_indices = torch.topk(rows_lg, k, dim=-1)

    rows_lg = rows_lg.float()
    if min_p > 0.0:
        probs_before = F.softmax(rows_lg, dim=-1)
        top_probs, _ = probs_before.max(dim=-1, keepdim=True)
        rows_lg = rows_lg.masked_fill(probs_before < (min_p * top_probs), torch.finfo(rows_lg.dtype).min)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(rows_lg, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs >= top_p
        sorted_remove = torch.cat((torch.zeros_like(sorted_remove[..., :1]), sorted_remove[..., :-1]), dim=-1)
        remove = torch.zeros_like(rows_lg, dtype=torch.bool)
        remove.scatter_(1, sorted_indices, sorted_remove)
        rows_lg = rows_lg.masked_fill(remove, torch.finfo(rows_lg.dtype).min)

    return F.softmax(rows_lg, dim=-1), top_indices


def _verify_head_fn(x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices, rand_u, divisor,
                    eps, add_one, do_sample, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty):
    if final_nw is not None:
        x = _rms_norm(x, final_nw, eps, add_one)
    logits = _linear_raw(x, lm_w, lm_s, divisor)
    if not do_sample:
        rows_lg = logits[0]
        if pen_mask is not None and penalty_active(repetition_penalty, presence_penalty):
            rows = rows_lg.shape[0]
            anc_toks = verify_tokens[0].index_select(0, ancestor_indices.reshape(-1)).reshape(ancestor_indices.shape)
            row_mask = pen_mask.unsqueeze(0).expand(rows, -1).clone()
            row_mask.scatter_(1, anc_toks, torch.ones_like(anc_toks, dtype=torch.bool))
            rows_lg = torch.where(row_mask, apply_penalty(rows_lg, repetition_penalty, presence_penalty), rows_lg)
        return x, rows_lg.argmax(dim=-1).unsqueeze(0)
    probs, top_indices = _sample_prep(
        logits[0], verify_tokens, pen_mask, ancestor_indices,
        temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty,
    )
    cdf = probs.cumsum(dim=-1)
    sampled = (cdf <= rand_u).sum(dim=-1, keepdim=True).clamp(max=probs.shape[-1] - 1)
    if top_indices is not None:
        toks = top_indices.int().gather(1, sampled).long().transpose(0, 1)
    else:
        toks = sampled.transpose(0, 1)
    return x, toks


def _mtp_draft_chain_fn(embed_w, mtp_w, lm_w, lm_s, root_token, root_hidden, rotary_chain,
                        mtp_kv_key, mtp_kv_val, mtp_base, gather_idx, divisor,
                        cfg_tup, depth, topk_k, shared_quant, native_rope):
    (pre_emb_nw, pre_hid_nw, fc_w, fc_s, in_nw, q_w, q_s, k_w, k_s, v_w, v_s,
     qn_w, kn_w, o_w, o_s, post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s, out_nw) = mtp_w
    num_heads, num_kv_heads, head_dim, rotary_dim, eps, add_one = cfg_tup
    cur_tok = root_token
    cur_hid = root_hidden
    mtp_positions = mtp_base + torch.arange(depth, device=mtp_base.device, dtype=torch.int64)
    mtp_seqlens = (mtp_positions + 1).to(torch.int32)
    tops = []
    for k in range(depth):
        emb = F.embedding(cur_tok, embed_w).to(cur_hid.dtype)
        fused_in = torch.cat((_rms_norm(emb, pre_emb_nw, eps, add_one), _rms_norm(cur_hid, pre_hid_nw, eps, add_one)), dim=-1)
        x = _linear_raw(fused_in, fc_w, fc_s, divisor)
        bs, seq, _ = x.shape

        xn = _rms_norm(x, in_nw, eps, add_one)
        if shared_quant and q_s is not None and k_s is not None and v_s is not None:
            qx, sx = _quantize_fp8(xn, divisor)
            qg = _fp8_proj_raw(qx, sx, q_w, q_s, xn.shape, xn.dtype)
            xk = _fp8_proj_raw(qx, sx, k_w, k_s, xn.shape, xn.dtype)
            xv = _fp8_proj_raw(qx, sx, v_w, v_s, xn.shape, xn.dtype)
        else:
            qg = _linear_raw(xn, q_w, q_s, divisor)
            xk = _linear_raw(xn, k_w, k_s, divisor)
            xv = _linear_raw(xn, v_w, v_s, divisor)

        qg = qg.view(bs, seq, num_heads, head_dim * 2)
        xq, gate = qg[..., :head_dim], qg[..., head_dim:].reshape(bs, seq, -1)
        xq = _rms_norm(xq, qn_w, eps, add_one).transpose(1, 2)
        xk = _rms_norm(xk.view(bs, seq, num_kv_heads, head_dim), kn_w, eps, add_one).transpose(1, 2)
        xv = xv.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
        xq, xk = _apply_partial_rope(xq, xk, rotary_chain[:, :, k:k + 1], rotary_dim, native_rope)

        mtp_kv_key.index_copy_(2, mtp_positions[k:k + 1], xk)
        mtp_kv_val.index_copy_(2, mtp_positions[k:k + 1], xv)
        out = hpu_kv_attention.single_decode(xq, mtp_kv_key, mtp_kv_val, mtp_seqlens[k:k + 1])
        x = x + _linear_raw(out * gate.sigmoid(), o_w, o_s, divisor)
        x = x + _mlp_raw(x, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s, divisor, shared_quant)

        n1 = _rms_norm(x, out_nw, eps, add_one)
        logits = _linear_raw(n1, lm_w, lm_s, divisor)[:, -1]
        if topk_k > 1:
            top = logits.topk(topk_k, dim=-1).indices
            cur_tok = top[:, :1]
        else:
            cur_tok = logits.argmax(dim=-1, keepdim=True)
            top = cur_tok
        cur_hid = x
        tops.append(top)
    stacked = torch.stack(tops, dim=0)
    flat_cand = torch.cat((root_token.reshape(-1), stacked.reshape(-1)))
    verify_tokens = flat_cand.index_select(0, gather_idx).unsqueeze(0)
    return verify_tokens, stacked


@lru_cache(maxsize=1)
def _compiled_gdn3():
    return torch.compile(_gdn3_stage_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_gdn1():
    return torch.compile(_gdn1_stage_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_tree_mask():
    return torch.compile(_build_tree_sdpa_mask_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_attn():
    return torch.compile(_attn_stage_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_verify_head():
    return torch.compile(_verify_head_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_stage4():
    return torch.compile(_stage4_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_kv_commit_slot():
    return torch.compile(_kv_commit_slot_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_mtp_draft_chain():
    return torch.compile(_mtp_draft_chain_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


def reset_compile_caches():
    for fn in (_compiled_gdn3, _compiled_gdn1, _compiled_stage4, _compiled_kv_commit_slot,
               _compiled_tree_mask, _compiled_attn, _compiled_verify_head, _compiled_mtp_draft_chain):
        fn.cache_clear()


def tree_ancestor_indices(tree, device):
    """Precompute [rows, max(1, rows - 1)] ancestor table for vectorized penalty masking."""
    rows = tree.rows
    width = max(1, rows - 1)
    table = [[0] * width for _ in range(rows)]
    for r in range(1, rows):
        chain = []
        a = r
        while a > 0:
            chain.append(a)
            a = tree.parent[a]
        for j, idx in enumerate(chain[:width]):
            table[r][j] = idx
    return torch.tensor(table, dtype=torch.long, device=device)


def tree_draft_gather_indices(tree, topk_k, device):
    """Precompute [rows] gather indices mapping [root, tops.reshape(-1)] -> verify_tokens."""
    gather = [0] * tree.rows
    for k in range(tree.chain):
        for j, child in enumerate(tree.children[k]):
            gather[child] = 1 + k * topk_k + j
    return torch.tensor(gather, dtype=torch.long, device=device)


def _tree_gdn_tables(tree, window, taps, device):
    """Precompute static per-tree conv gather indices and GDN parent map on host."""
    rows = tree.rows
    parent = list(tree.parent[:rows])
    conv_src = []
    for r in range(rows):
        p = r
        missing = 0
        row_taps = [window + r]
        for _ in range(1, taps):
            p = parent[p] if p >= 0 else -1
            if p < 0:
                missing += 1
                row_taps.append(window - missing)
            else:
                row_taps.append(window + p)
        conv_src.extend(row_taps)
    prog = [int(x) for x in tree.ctl[len(tree.ctl) // 2:len(tree.ctl) // 2 + rows].tolist()]
    prog_rows = [p & 15 for p in prog]
    prog_committed = [(p & 32) != 0 for p in prog]
    tree_parents = [-1] * rows
    last_committed = -1
    for pos_i in range(rows):
        r = prog_rows[pos_i]
        tree_parents[r] = last_committed
        if prog_committed[pos_i]:
            last_committed = r
    max_depth = max(tree.depth[:rows]) if rows > 0 else 1
    tree_doublings = max(1, max_depth.bit_length())
    return (
        torch.tensor(conv_src, dtype=torch.long, device=device),
        torch.tensor(tree_parents, dtype=torch.long, device=device),
        tree_doublings,
    )


def can_compile_mtp(qwen, embeds, pkv, mtp_kv, qwen_transformer_cls, linear_kv_cls):
    if (embeds.device.type != "hpu" or embeds.dtype != torch.bfloat16 or embeds.shape[0] != 1
            or qwen.mtp is None or type(qwen.model) is not qwen_transformer_cls
            or comfy.model_management.args.disable_comfy_compiler or comfy.model_management.in_training
            or torch.is_grad_enabled() or comfy.model_management.comfy_kitchen_attention_enabled()
            or "forward" in qwen.model.__dict__ or "forward" in qwen.mtp.__dict__
            or comfy.ops.has_forward_hooks(qwen.model) or comfy.ops.has_forward_hooks(qwen.mtp)):
        return False
    for root in (qwen.model, qwen.mtp):
        if any(p.device != embeds.device or p.dtype != torch.bfloat16 for p in root.parameters()):
            return False
        for module in root.modules():
            if isinstance(module, comfy.ops.CastWeightBiasOp):
                if (module.weight_function or module.bias_function or hasattr(module, "_v")
                        or getattr(module, "weight_lowvram_function", None) is not None
                        or getattr(module, "bias_lowvram_function", None) is not None):
                    return False
                if getattr(module, "_full_precision_mm", False):
                    return False
                if getattr(module, "bias", None) is not None and not isinstance(module, torch.nn.Conv1d):
                    return False
    if type(mtp_kv) is not FixedKVCache or qwen.mtp.layers[0].self_attn.comfy_attention.method is not None:
        return False
    if qwen.mtp.layers[0].mlp.merged_mlp:
        return False
    for layer, cache in zip(qwen.model.layers, pkv):
        if layer.mlp.merged_mlp:
            return False
        if layer.layer_type == "linear_attention":
            if type(cache) is not linear_kv_cls:
                return False
            if getattr(layer.linear_attn.in_proj_a, "layout_type", None) is not None:
                return False
            if getattr(layer.linear_attn.in_proj_b, "layout_type", None) is not None:
                return False
        elif type(cache) is not FixedKVCache or layer.self_attn.comfy_attention.method is not None:
            return False
    return True


def build_mtp_runner(qwen, embeds, pkv, mtp_kv, tree, ftab, sampling, qwen_transformer_cls, linear_kv_cls, standard_freqs_fn,
                     *, shared_quant=True, use_fused_sdpa=True, kv_bucket=0, fuse_stage4=False):
    """Create request-local modular compiled draft/verify/commit runner for HPU Qwen3.5 MTP."""
    if not can_compile_mtp(qwen, embeds, pkv, mtp_kv, qwen_transformer_cls, linear_kv_cls):
        return None

    device = embeds.device
    dtype = embeds.dtype
    cfg = qwen.model.config
    divisor = _divisor(device)
    standard_rotary = getattr(qwen.model.compute_freqs_cis, "__func__", None) is standard_freqs_fn
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    native_rope = standard_rotary and rotary_dim == 64

    kv_caches = tuple(c for c in pkv if type(c) is FixedKVCache)
    if not kv_caches:
        return None
    for c in kv_caches:
        torch._dynamo.mark_dynamic(c.key, 2)
        torch._dynamo.mark_dynamic(c.value, 2)
    torch._dynamo.mark_dynamic(mtp_kv.key, 2)
    torch._dynamo.mark_dynamic(mtp_kv.value, 2)

    eps = cfg.rms_norm_eps
    add_one = cfg.rms_norm_add
    kh, vh = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    k_hd, v_hd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    taps = cfg.conv_kernel_size
    window = taps - 1
    gdn_cfg = (kh, vh, k_hd, v_hd, taps, eps, add_one)
    attn_cfg = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, rotary_dim, eps, add_one)

    rows = tree.rows
    bs = embeds.shape[0]
    gdn_ws, gdn_conv_st, gdn_rec_st = [], [], []
    gdn_ping = []  # per GDN layer: [side0_tuple, side1_tuple]
    attn_ws = []
    # Build execution plan grouping contiguous 3-GDN runs (and optionally trailing Attn into stage4).
    plan = []
    gdn_run = []
    gdn_i = 0
    kv_i = 0
    for layer, cache in zip(qwen.model.layers, pkv):
        if layer.layer_type == "linear_attention":
            if cache.g_decay is None:
                return None
            g = layer.linear_attn
            channels = g.key_dim * 2 + g.value_dim
            cw_cl = g.conv1d.weight.reshape(channels, taps).float().flip(1).transpose(0, 1).view(1, 1, taps, channels).contiguous()
            cb = g.conv1d.bias.float().view(1, 1, channels).contiguous() if g.conv1d.bias is not None else None
            gdn_ws.append((
                layer.input_layernorm.weight,
                *_weight_pair(g.in_proj_qkv),
                *_weight_pair(g.in_proj_z),
                g.in_proj_a.weight, g.in_proj_b.weight, cw_cl, cb,
                cache.g_decay, cache.dt_bias, cache.norm_weight.float(),
                *_weight_pair(g.out_proj),
                layer.post_attention_layernorm.weight,
                *_weight_pair(layer.mlp.gate_proj),
                *_weight_pair(layer.mlp.up_proj),
                *_weight_pair(layer.mlp.down_proj),
            ))
            gdn_conv_st.append(cache.conv_state.transpose(1, 2).contiguous())
            gdn_rec_st.append(cache.recurrent_state)
            sides = []
            for _ in range(2):
                sides.append((
                    torch.zeros(bs, rows, channels, device=device, dtype=dtype),
                    torch.zeros(bs, vh, rows, k_hd, device=device, dtype=torch.float32),
                    torch.zeros(bs, vh, rows, v_hd, device=device, dtype=torch.float32),
                    torch.zeros(bs, vh, rows, device=device, dtype=torch.float32),
                    torch.zeros(bs, vh, rows, device=device, dtype=torch.float32),
                ))
            gdn_ping.append(sides)
            gdn_run.append(gdn_i)
            gdn_i += 1
            if len(gdn_run) == 3 and not fuse_stage4:
                plan.append(("gdn3", tuple(gdn_run)))
                gdn_run = []
        else:
            a = layer.self_attn
            attn_ws.append((
                layer.input_layernorm.weight,
                *_weight_pair(a.q_proj),
                *_weight_pair(a.k_proj),
                *_weight_pair(a.v_proj),
                a.q_norm.weight, a.k_norm.weight,
                *_weight_pair(a.o_proj),
                layer.post_attention_layernorm.weight,
                *_weight_pair(layer.mlp.gate_proj),
                *_weight_pair(layer.mlp.up_proj),
                *_weight_pair(layer.mlp.down_proj),
            ))
            if fuse_stage4 and len(gdn_run) == 3:
                plan.append(("stage4", (tuple(gdn_run), kv_i)))
                gdn_run = []
            else:
                while len(gdn_run) >= 3:
                    plan.append(("gdn3", tuple(gdn_run[:3])))
                    gdn_run = gdn_run[3:]
                for idx in gdn_run:
                    plan.append(("gdn1", (idx,)))
                gdn_run = []
                plan.append(("attn", kv_i))
            kv_i += 1
    while len(gdn_run) >= 3:
        plan.append(("gdn3", tuple(gdn_run[:3])))
        gdn_run = gdn_run[3:]
    for idx in gdn_run:
        plan.append(("gdn1", (idx,)))

    final_nw = qwen.model.norm.weight if qwen.model.norm is not None else None
    lm_head = getattr(qwen.model, "lm_head", None)
    if lm_head is not None:
        lm_w, lm_s = _weight_pair(lm_head)
    else:
        lm_w, lm_s = qwen.model.embed_tokens.weight, None
    embed_w = qwen.model.embed_tokens.weight

    mb = qwen.mtp.layers[0]
    ma = mb.self_attn
    mtp_w = (
        qwen.mtp.pre_fc_norm_embedding.weight,
        qwen.mtp.pre_fc_norm_hidden.weight,
        *_weight_pair(qwen.mtp.fc),
        mb.input_layernorm.weight,
        *_weight_pair(ma.q_proj),
        *_weight_pair(ma.k_proj),
        *_weight_pair(ma.v_proj),
        ma.q_norm.weight, ma.k_norm.weight,
        *_weight_pair(ma.o_proj),
        mb.post_attention_layernorm.weight,
        *_weight_pair(mb.mlp.gate_proj),
        *_weight_pair(mb.mlp.up_proj),
        *_weight_pair(mb.mlp.down_proj),
        qwen.mtp.norm.weight,
    )

    gdn3_compiled = _compiled_gdn3()
    gdn1_compiled = _compiled_gdn1()
    stage4_compiled = _compiled_stage4()
    kv_commit_compiled = _compiled_kv_commit_slot()
    mask_compiled = _compiled_tree_mask()
    attn_compiled = _compiled_attn()
    head_compiled = _compiled_verify_head()
    draft_compiled = _compiled_mtp_draft_chain()

    # Flatten rotary table once so dim-0 index_select stays contiguous and vectorized on TPC.
    rot_half = ftab.shape[3]
    ftab_2d = ftab.reshape(ftab.shape[2], -1).contiguous()

    mtp_cap = mtp_kv.key.shape[2]
    mtp_base_table = torch.arange(mtp_cap, device=device, dtype=torch.int64)
    mtp_base_buf = torch.empty((1,), device=device, dtype=torch.int64)

    kv_cap = kv_caches[0].key.shape[2]
    kv_pos_table = torch.arange(kv_cap, device=device, dtype=torch.int64)
    kv_src_buf = torch.empty((1,), device=device, dtype=torch.int64)
    kv_dst_buf = torch.empty((1,), device=device, dtype=torch.int64)
    ks_full = tuple(c.key for c in kv_caches)
    vs_full = tuple(c.value for c in kv_caches)

    cpu_gen = torch.Generator(device="cpu").manual_seed(sampling["seed"]) if sampling is not None else None
    u_buf = torch.empty((rows, 1), device=device, dtype=torch.float32) if sampling is not None else None

    # Precomputed static GDN commit control tables for common on-chain pending counts (zero H2D copy).
    commit_p4 = torch.arange(4, device=device, dtype=torch.long) - 1
    commit_p8 = torch.arange(8, device=device, dtype=torch.long) - 1
    onchain_ctl = {}
    for p_max, cp in ((4, commit_p4), (8, commit_p8)):
        for pend in range(p_max + 1):
            slots_t = torch.arange(p_max, device=device, dtype=torch.long).clamp(max=rows - 1)
            live_t = (torch.arange(p_max, device=device) < pend)
            cc_idx = torch.arange(pend, pend + window, device=device, dtype=torch.long)
            onchain_ctl[(p_max, pend)] = (slots_t, live_t, cc_idx, cp)
    offchain_slots = {4: torch.zeros(4, device=device, dtype=torch.long),
                      8: torch.zeros(8, device=device, dtype=torch.long)}

    state = {
        "parity": 0,
        "pending": 0,
        "path": (),
        "on_chain": True,
        "conv_src": None,
        "tree_parents": None,
        "tree_doublings": 2,
    }

    def sync_tree(cur_tree):
        cs, tp, td = _tree_gdn_tables(cur_tree, window, taps, device)
        state["conv_src"] = cs
        state["tree_parents"] = tp
        state["tree_doublings"] = td

    sync_tree(tree)

    def _slice_kv(caches, min_len):
        if kv_bucket <= 0:
            return tuple(c.key for c in caches), tuple(c.value for c in caches)
        cap = caches[0].key.shape[2]
        vcap = min(cap, ((min_len + kv_bucket - 1) // kv_bucket) * kv_bucket)
        if vcap >= cap:
            return tuple(c.key for c in caches), tuple(c.value for c in caches)
        ks, vs = [], []
        for c in caches:
            k_s = c.key[:, :, :vcap]
            v_s = c.value[:, :, :vcap]
            torch._dynamo.mark_dynamic(k_s, 2)
            torch._dynamo.mark_dynamic(v_s, 2)
            ks.append(k_s)
            vs.append(v_s)
        return tuple(ks), tuple(vs)

    def run_draft(nt_buf, h_buf, _ftab, pos, depth, topk_k, gather_idx):
        rot_chain = ftab_2d[pos:pos + depth].clone().view(1, 1, depth, rot_half, 2, 2)
        base = mtp_kv.index
        mtp_base_buf.copy_(mtp_base_table[base:base + 1])
        mk_l, mv_l = _slice_kv((mtp_kv,), base + depth)
        verify_tokens, tops = draft_compiled(
            embed_w, mtp_w, lm_w, lm_s,
            nt_buf, h_buf, rot_chain,
            mk_l[0], mv_l[0], mtp_base_buf, gather_idx, divisor,
            attn_cfg, depth, topk_k, shared_quant, native_rope,
        )
        mtp_kv.advance(depth)
        return verify_tokens, tops

    def run_verify(verify_tokens, _ftab, pos, cur_tree, ancestor_indices, pen_mask, cur_sampling, _generator,
                   repetition_penalty=1.0, presence_penalty=0.0):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur_rows = cur_tree.rows
        for cache in kv_caches:
            cache.prepare(cur_rows)
        rot_idx = cur_tree.positions[0] + pos
        rotary = ftab_2d.index_select(0, rot_idx).view(1, 1, cur_rows, rot_half, 2, 2)
        write_pos = kv_caches[0].position[:cur_rows]
        prefix_len = kv_caches[0].seqlen
        ks, vs = _slice_kv(kv_caches, kv_caches[0].index + cur_rows)

        pend = state["pending"]
        p_max = 4 if pend <= 4 else 8
        cd = 2 if p_max == 4 else 3
        slots_t, live_t, cc_idx, cp = onchain_ctl[(p_max, pend)]
        if not state["on_chain"] and pend > 0:
            path_vals = list(state["path"]) + [0] * (p_max - pend)
            offchain_slots[p_max].copy_(torch.tensor(path_vals, dtype=torch.long), non_blocking=True)
            slots_t = offchain_slots[p_max]
        ctl_tup = (slots_t, live_t, cc_idx, state["conv_src"], state["tree_parents"], cp)

        w_side = state["parity"]
        r_side = 1 - w_side

        def gdn_state_for(idx):
            return (gdn_conv_st[idx], gdn_rec_st[idx], *gdn_ping[idx][w_side], *gdn_ping[idx][r_side])

        sdpa_mask = mask_compiled(ks[0], prefix_len, cur_tree.mask, cur_rows) if use_fused_sdpa else None
        x = F.embedding(verify_tokens, embed_w).to(dtype)
        td = state["tree_doublings"]
        for kind, idx_info in plan:
            if kind == "stage4":
                (i0, i1, i2), kv_idx = idx_info
                x = stage4_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), gdn_ws[i1], gdn_state_for(i1),
                    gdn_ws[i2], gdn_state_for(i2), attn_ws[kv_idx],
                    rotary, ks[kv_idx], vs[kv_idx], write_pos, prefix_len, cur_tree.mask, sdpa_mask,
                    ctl_tup, divisor, gdn_cfg, attn_cfg, shared_quant, native_rope, use_fused_sdpa, cd, td,
                )
            elif kind == "gdn3":
                i0, i1, i2 = idx_info
                x = gdn3_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), gdn_ws[i1], gdn_state_for(i1),
                    gdn_ws[i2], gdn_state_for(i2), ctl_tup, divisor, gdn_cfg, shared_quant, cd, td,
                )
            elif kind == "gdn1":
                i0 = idx_info[0]
                x = gdn1_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), ctl_tup, divisor, gdn_cfg, shared_quant, cd, td,
                )
            else:
                x = attn_compiled(
                    x, attn_ws[idx_info], rotary, ks[idx_info], vs[idx_info],
                    write_pos, prefix_len, cur_tree.mask, sdpa_mask, divisor,
                    attn_cfg, shared_quant, native_rope, use_fused_sdpa,
                )

        rp = cur_sampling.get("repetition_penalty", repetition_penalty) if cur_sampling is not None else repetition_penalty
        pp = cur_sampling.get("presence_penalty", presence_penalty) if cur_sampling is not None else presence_penalty
        if cur_sampling is not None:
            u_buf.copy_(torch.rand((cur_rows, 1), dtype=torch.float32, generator=cpu_gen), non_blocking=True)
            hidden, toks = head_compiled(
                x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices, u_buf, divisor,
                eps, add_one, True, cur_sampling["temperature"], cur_sampling["top_k"], cur_sampling["top_p"],
                cur_sampling["min_p"], rp, pp,
            )
        else:
            hidden, toks = head_compiled(
                x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices if pen_mask is not None else None,
                None, divisor, eps, add_one, False, 1.0, 0, 1.0, 0.0, rp, pp,
            )

        for cache in kv_caches:
            cache.advance(cur_rows)
        return hidden, toks

    def record_commit(base_pos, path):
        state["parity"] ^= 1
        state["pending"] = len(path)
        state["path"] = tuple(path)
        on_chain = (path[-1] == len(path) - 1)
        state["on_chain"] = on_chain
        if not on_chain:
            for k, r in enumerate(path[1:], 1):
                if r != k:
                    src = (base_pos + r) % kv_cap
                    dst = (base_pos + k) % kv_cap
                    kv_src_buf.copy_(kv_pos_table[src:src + 1])
                    kv_dst_buf.copy_(kv_pos_table[dst:dst + 1])
                    kv_commit_compiled(ks_full, vs_full, kv_src_buf, kv_dst_buf)

    return run_draft, run_verify, record_commit, sync_tree
