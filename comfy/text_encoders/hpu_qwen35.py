"""Modular HPU compilation, Gaudi2 TPC kernels, and batched on-device accept for Qwen3.5 MTP."""
from functools import lru_cache
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops

from . import hpu_kv_attention, hpu_qwen35_tpc
from .llama import FixedKVCache, apply_penalty, apply_rope, penalty_active

if comfy.model_management.hpu_available:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA


MAX_BATCH_STEPS = 8
DEFAULT_BATCH_STEPS = 8


@lru_cache(maxsize=4)
def _divisor(device):
    return torch.tensor(240.0, dtype=torch.float32, device=device)


@lru_cache(maxsize=4)
def _dummy_bf16(device):
    return torch.zeros((1, 1), dtype=torch.bfloat16, device=device)


@lru_cache(maxsize=4)
def _dummy_u8(device):
    return torch.zeros((1,), dtype=torch.uint8, device=device)


@lru_cache(maxsize=4)
def _dummy_i32_2d(device):
    return torch.zeros((1, 1), dtype=torch.int32, device=device)


@lru_cache(maxsize=4)
def _dummy_f32_1d(device):
    return torch.zeros((1,), dtype=torch.float32, device=device)


@lru_cache(maxsize=4)
def _dummy_f32_2d(device):
    return torch.zeros((1, 1), dtype=torch.float32, device=device)


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


def _rms_quant_raw(x, res, weight, eps, add_one, divisor, use_tpc_rms, has_res=False, store_norm=False, dummy_bf=None):
    if use_tpc_rms:
        flat_x = x.reshape(-1, x.shape[-1])
        flat_r = res.reshape(-1, res.shape[-1]) if has_res else dummy_bf
        ox, oy, qx, sx = hpu_qwen35_tpc.rmsnorm_quant(
            flat_x, flat_r, weight, int(has_res), int(add_one), int(store_norm), eps,
        )
        nx = ox.reshape(x.shape) if has_res else x
        ny = oy.reshape(x.shape) if store_norm else None
        return nx, ny, qx, sx
    nx = (x + res) if has_res else x
    ny = _rms_norm(nx, weight, eps, add_one)
    qx, sx = _quantize_fp8(ny, divisor)
    return nx, ny, qx, sx


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


def _mlp_raw(x, res, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s,
             divisor, shared_quant, use_tpc_rms=False, use_tpc_silu=False, dummy_bf=None, split_out=False):
    if shared_quant and gate_s is not None and up_s is not None:
        x, _, qm, sm = _rms_quant_raw(
            x, res, post_nw, eps, add_one, divisor, use_tpc_rms,
            has_res=(res is not None), store_norm=False, dummy_bf=dummy_bf,
        )
        gate = _fp8_proj_raw(qm, sm, gate_w, gate_s, x.shape, x.dtype)
        up = _fp8_proj_raw(qm, sm, up_w, up_s, x.shape, x.dtype)
    else:
        if res is not None:
            x = x + res
        xm = _rms_norm(x, post_nw, eps, add_one)
        gate = _linear_raw(xm, gate_w, gate_s, divisor)
        up = _linear_raw(xm, up_w, up_s, divisor)
    if use_tpc_silu and down_s is not None:
        qd, sd = hpu_qwen35_tpc.silu_mul_quant(
            gate.reshape(-1, gate.shape[-1]), up.reshape(-1, up.shape[-1]),
        )
        down_out = _fp8_proj_raw(qd, sd, down_w, down_s, x.shape, x.dtype)
    else:
        down_out = _linear_raw(F.silu(gate) * up, down_w, down_s, divisor)
    if split_out:
        return x, down_out
    return x + down_out


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


def _gdn_layer_core_tpc(x, w_tup, s_tup, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                        shared_quant, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                        in_res=None, split_out=False):
    (in_nw, qkv_w, qkv_s, z_w, z_s, wab, conv_w_2d, conv_bias_1d, g_decay, dt_bias, gdn_nw,
     out_w, out_s, post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s) = w_tup
    (conv_st, rec_st_w, rec_st_r, p_proj_w, p_conv_w, p_ab_w, p_proj_r, p_conv_r, p_ab_r) = s_tup
    conv_src_i32, ctl_gdn_i32 = ctl_tup
    kh, vh, _k_hd, v_hd, _taps, eps, add_one = cfg_tup
    bs, seq, _ = x.shape

    if shared_quant and qkv_s is not None and z_s is not None:
        x, xn, qx, sx = _rms_quant_raw(
            x, in_res, in_nw, eps, add_one, divisor, use_tpc_rms,
            has_res=(in_res is not None), store_norm=True, dummy_bf=dummy_bf,
        )
        proj = _fp8_proj_raw(qx, sx, qkv_w, qkv_s, x.shape, x.dtype)
        z = _fp8_proj_raw(qx, sx, z_w, z_s, x.shape, x.dtype)
    else:
        if in_res is not None:
            x = x + in_res
        xn = _rms_norm(x, in_nw, eps, add_one)
        proj = _linear_raw(xn, qkv_w, qkv_s, divisor)
        z = _linear_raw(xn, z_w, z_s, divisor)

    ab = F.linear(xn, wab)

    has_bias = 1 if conv_bias_1d is not None else 0
    cb_arg = conv_bias_1d if has_bias else dummy_f32
    new_conv_st, cur_conv = hpu_qwen35_tpc.gdn_conv(
        conv_st, p_proj_r, proj, conv_w_2d, cb_arg, conv_src_i32, ctl_gdn_i32, has_bias,
    )
    conv_st.copy_(new_conv_st)
    p_proj_w.copy_(proj)
    p_conv_w.copy_(cur_conv)
    p_ab_w.copy_(ab)

    committed_st, out_f32 = hpu_qwen35_tpc.gdn_rec(
        rec_st_r, p_conv_r, cur_conv, p_ab_r, ab, g_decay, dt_bias, ctl_gdn_i32, kh, vh, v_hd,
    )
    rec_st_w.copy_(committed_st)

    if use_tpc_gated and out_s is not None:
        qo, so = hpu_qwen35_tpc.gated_rmsnorm_quant(
            out_f32.reshape(-1, out_f32.shape[-1]), z.reshape(-1, z.shape[-1]), gdn_nw, vh, eps,
        )
        o_out = _fp8_proj_raw(qo, so, out_w, out_s, x.shape, x.dtype)
    else:
        o = out_f32.view(bs, seq, vh, v_hd)
        rstd = torch.rsqrt(o.square().mean(-1, keepdim=True) + eps)
        gated = ((o * rstd * gdn_nw).to(x.dtype) * F.silu(z.reshape(bs, seq, vh, v_hd))).reshape(bs, seq, -1)
        o_out = _linear_raw(gated, out_w, out_s, divisor)
    return _mlp_raw(
        x, o_out, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s,
        divisor, shared_quant, use_tpc_rms=use_tpc_rms, use_tpc_silu=use_tpc_silu,
        dummy_bf=dummy_bf, split_out=split_out,
    )


def _gdn_layer_core(x, w_tup, s_tup, ctl_tup, divisor, cfg_tup, shared_quant, commit_doublings, tree_doublings,
                    use_tpc_gdn=False, use_tpc_rms=False, use_tpc_gated=False, use_tpc_silu=False,
                    dummy_bf=None, dummy_f32=None):
    if use_tpc_gdn:
        return _gdn_layer_core_tpc(
            x, w_tup, s_tup, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
            shared_quant, use_tpc_rms, use_tpc_gated, use_tpc_silu,
        )
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

    if shared_quant and qkv_s is not None and z_s is not None:
        _, xn, qx, sx = _rms_quant_raw(
            x, None, in_nw, eps, add_one, divisor, use_tpc_rms,
            has_res=False, store_norm=True, dummy_bf=dummy_bf,
        )
        proj = _fp8_proj_raw(qx, sx, qkv_w, qkv_s, x.shape, x.dtype)
        z = _fp8_proj_raw(qx, sx, z_w, z_s, x.shape, x.dtype)
    else:
        xn = _rms_norm(x, in_nw, eps, add_one)
        proj = _linear_raw(xn, qkv_w, qkv_s, divisor)
        z = _linear_raw(xn, z_w, z_s, divisor)

    channels = proj.shape[2]
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
    o_out = _linear_raw(gated, out_w, out_s, divisor)
    return _mlp_raw(
        x, o_out, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s,
        divisor, shared_quant, use_tpc_rms=use_tpc_rms, use_tpc_silu=use_tpc_silu, dummy_bf=dummy_bf,
    )


def _gdn3_stage_fn(x, w0, s0, w1, s1, w2, s2, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                   shared_quant, cd, td, use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu):
    if use_tpc_gdn and use_tpc_rms:
        x, r0 = _gdn_layer_core_tpc(x, w0, s0, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                                    shared_quant, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                                    in_res=None, split_out=True)
        x, r1 = _gdn_layer_core_tpc(x, w1, s1, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                                    shared_quant, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                                    in_res=r0, split_out=True)
        return _gdn_layer_core_tpc(x, w2, s2, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                                   shared_quant, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                                   in_res=r1, split_out=False)
    x = _gdn_layer_core(x, w0, s0, ctl_tup, divisor, cfg_tup, shared_quant, cd, td,
                        use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)
    x = _gdn_layer_core(x, w1, s1, ctl_tup, divisor, cfg_tup, shared_quant, cd, td,
                        use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)
    return _gdn_layer_core(x, w2, s2, ctl_tup, divisor, cfg_tup, shared_quant, cd, td,
                           use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)


def _gdn1_stage_fn(x, w0, s0, ctl_tup, divisor, dummy_bf, dummy_f32, cfg_tup,
                   shared_quant, cd, td, use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu):
    return _gdn_layer_core(x, w0, s0, ctl_tup, divisor, cfg_tup, shared_quant, cd, td,
                           use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)


def _stage4_fn(x, w0, s0, w1, s1, w2, s2, aw, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask,
               ctl_tup, divisor, dummy_bf, dummy_f32, gdn_cfg, attn_cfg,
               shared_quant, native_rope, use_fused_sdpa, cd, td,
               use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, use_tpc_qk, use_tpc_sig):
    x = _gdn_layer_core(x, w0, s0, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td,
                        use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)
    x = _gdn_layer_core(x, w1, s1, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td,
                        use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)
    x = _gdn_layer_core(x, w2, s2, ctl_tup, divisor, gdn_cfg, shared_quant, cd, td,
                        use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, dummy_bf, dummy_f32)
    return _attn_stage_fn(
        x, aw, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask,
        divisor, dummy_bf, attn_cfg, shared_quant, native_rope, use_fused_sdpa,
        use_tpc_rms, use_tpc_qk, use_tpc_sig, use_tpc_silu,
    )


def _kv_commit_slot_fn(ks, vs, src, dst):
    for k, v in zip(ks, vs):
        k.index_copy_(2, dst, k.index_select(2, src))
        v.index_copy_(2, dst, v.index_select(2, src))
    return dst


def _kv_compact_ondev_fn(ks, vs, old_pos, path_slots, depth):
    ar = torch.arange(1, depth + 1, device=old_pos.device, dtype=old_pos.dtype)
    src = old_pos + path_slots[1:depth + 1]
    dst = old_pos + ar
    for k, v in zip(ks, vs):
        ksel = k.index_select(2, src)
        vsel = v.index_select(2, src)
        k.index_copy_(2, dst, ksel)
        v.index_copy_(2, dst, vsel)
    return dst


def _build_tree_sdpa_mask_fn(key_cache, prefix_len, tree_mask_bits, seq):
    cap = key_cache.shape[2]
    rel = torch.arange(cap, device=key_cache.device, dtype=torch.int32).view(1, 1, 1, -1) - prefix_len.view(-1, 1, 1, 1)
    tree_vis = ((tree_mask_bits.view(1, 1, seq, 1) >> rel.clamp(0, seq - 1)) & 1).bool()
    visible = (rel < 0) | ((rel >= 0) & (rel < seq) & tree_vis)
    return torch.zeros(visible.shape, device=key_cache.device, dtype=key_cache.dtype).masked_fill(~visible, float("-inf"))


def _prepare_verify_ondev_fn(key_cache, rot_tab, embed_w, verify_tokens, state_i32, tree_pos_i32,
                             tree_mask_bits, seq, rot_half, use_fused_sdpa, use_tpc_qk):
    pos_i32 = state_i32[17:18]
    write_pos = pos_i32 + torch.arange(seq, device=pos_i32.device, dtype=torch.int32)
    rot_idx = tree_pos_i32 + pos_i32
    if use_tpc_qk:
        rotary = rot_tab.index_select(0, rot_idx)
    else:
        rotary = rot_tab.index_select(0, rot_idx).view(1, 1, seq, rot_half, 2, 2)
    x = F.embedding(verify_tokens, embed_w).to(key_cache.dtype)
    if use_fused_sdpa:
        sdpa_mask = _build_tree_sdpa_mask_fn(key_cache, pos_i32, tree_mask_bits, seq)
    else:
        sdpa_mask = pos_i32
    return x, rotary, write_pos, pos_i32, sdpa_mask


def _attn_stage_fn(x, w_tup, rot, kc, vc, wpos, prefix_len, tree_mask, sdpa_mask, divisor, dummy_bf,
                   cfg_tup, shared_quant, native_rope, use_fused_sdpa,
                   use_tpc_rms, use_tpc_qk, use_tpc_sig, use_tpc_silu):
    (in_nw, q_w, q_s, k_w, k_s, v_w, v_s, qn_w, kn_w, o_w, o_s,
     post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s) = w_tup
    num_heads, num_kv_heads, head_dim, rotary_dim, eps, add_one = cfg_tup
    bs, seq, _ = x.shape

    if shared_quant and q_s is not None and k_s is not None and v_s is not None:
        _, _, qx, sx = _rms_quant_raw(
            x, None, in_nw, eps, add_one, divisor, use_tpc_rms,
            has_res=False, store_norm=False, dummy_bf=dummy_bf,
        )
        qg = _fp8_proj_raw(qx, sx, q_w, q_s, x.shape, x.dtype)
        xk = _fp8_proj_raw(qx, sx, k_w, k_s, x.shape, x.dtype)
        xv = _fp8_proj_raw(qx, sx, v_w, v_s, x.shape, x.dtype)
    else:
        xn = _rms_norm(x, in_nw, eps, add_one)
        qg = _linear_raw(xn, q_w, q_s, divisor)
        xk = _linear_raw(xn, k_w, k_s, divisor)
        xv = _linear_raw(xn, v_w, v_s, divisor)

    if use_tpc_qk:
        xq, xk, gate = hpu_qwen35_tpc.qk_norm_rope(
            qg.reshape(seq, -1), xk.reshape(seq, -1), qn_w, kn_w, rot,
            num_heads, num_kv_heads, int(add_one), eps,
        )
        xv = xv.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
    else:
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

    if use_tpc_sig and o_s is not None:
        qo, so = hpu_qwen35_tpc.sig_mul_quant(
            out.reshape(-1, out.shape[-1]), gate.reshape(-1, gate.shape[-1]),
        )
        o_out = _fp8_proj_raw(qo, so, o_w, o_s, x.shape, x.dtype)
    else:
        o_out = _linear_raw(out * gate.reshape(bs, seq, -1).sigmoid(), o_w, o_s, divisor)
    return _mlp_raw(
        x, o_out, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s,
        divisor, shared_quant, use_tpc_rms=use_tpc_rms, use_tpc_silu=use_tpc_silu, dummy_bf=dummy_bf,
    )


def _finish_topk_probs(rows_lg, temperature, top_p, min_p):
    if temperature != 1.0:
        rows_lg = rows_lg / temperature
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
    return F.softmax(rows_lg, dim=-1)


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

    return _finish_topk_probs(rows_lg, 1.0, top_p, min_p), top_indices


def _lm_logits_core(x, final_nw, lm_w, lm_s, divisor, dummy_bf, eps, add_one, use_tpc_rms):
    if final_nw is not None:
        if use_tpc_rms and lm_s is not None:
            _, x_norm, qx, sx = _rms_quant_raw(
                x, None, final_nw, eps, add_one, divisor, True,
                has_res=False, store_norm=True, dummy_bf=dummy_bf,
            )
            return x_norm, _fp8_proj_raw(qx, sx, lm_w, lm_s, x.shape, x.dtype)
        x = _rms_norm(x, final_nw, eps, add_one)
    return x, _linear_raw(x, lm_w, lm_s, divisor)


def _head_sample_core(x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices, rand_u,
                      divisor, dummy_bf, dummy_u8, dummy_i32,
                      eps, add_one, do_sample, temperature, top_k, top_p, min_p,
                      repetition_penalty, presence_penalty, use_tpc_rms, use_tpc_topk):
    x, logits = _lm_logits_core(x, final_nw, lm_w, lm_s, divisor, dummy_bf, eps, add_one, use_tpc_rms)

    if not do_sample:
        return x, logits.argmax(dim=-1)

    if use_tpc_topk and 0 < top_k <= 24:
        has_pen = 1 if (pen_mask is not None and penalty_active(repetition_penalty, presence_penalty)) else 0
        if has_pen:
            pm_u8 = pen_mask if pen_mask.dtype == torch.uint8 else pen_mask.to(torch.uint8)
            anc_i32 = verify_tokens[0].index_select(0, ancestor_indices.reshape(-1)).reshape(ancestor_indices.shape).to(torch.int32)
        else:
            pm_u8 = dummy_u8
            anc_i32 = dummy_i32
        k = min(top_k, logits.shape[-1])
        top_vals, top_idx_i32 = hpu_qwen35_tpc.topk(
            logits[0], pm_u8, anc_i32, k, has_pen, float(presence_penalty), float(repetition_penalty),
        )
        probs = _finish_topk_probs(top_vals, temperature, top_p, min_p)
        cdf = probs.cumsum(dim=-1)
        sampled = (cdf <= rand_u).sum(dim=-1, keepdim=True).clamp(max=k - 1)
        toks = top_idx_i32.gather(1, sampled).long().transpose(0, 1)
        return x, toks

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


def _verify_head_fn(x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices, rand_u,
                    divisor, dummy_bf, dummy_u8, dummy_i32,
                    eps, add_one, do_sample, temperature, top_k, top_p, min_p,
                    repetition_penalty, presence_penalty, use_tpc_rms, use_tpc_topk):
    return _head_sample_core(
        x, final_nw, lm_w, lm_s, verify_tokens, pen_mask, ancestor_indices, rand_u,
        divisor, dummy_bf, dummy_u8, dummy_i32,
        eps, add_one, do_sample, temperature, top_k, top_p, min_p,
        repetition_penalty, presence_penalty, use_tpc_rms, use_tpc_topk,
    )


def _verify_head_accept_fn(x, final_nw, lm_w, lm_s, verify_tokens_i32, tops_i32, pen_mask_u8, ancestor_i32, rand_u,
                           state_i32, tree_parent_i32, nt_buf, h_buf,
                           ring_toks_s, ring_acc_s, ring_path_s, ring_ranks_s,
                           divisor, dummy_bf, dummy_u8, dummy_i32, dummy_f32_2d,
                           eps, add_one, do_sample, temperature, top_k, top_p, min_p,
                           repetition_penalty, presence_penalty, depth, adaptive, use_pen_update,
                           use_tpc_rms, use_tpc_topk, use_tpc_accept):
    if use_tpc_accept and (not do_sample or (use_tpc_topk and 0 < top_k <= 24)):
        x, logits = _lm_logits_core(x, final_nw, lm_w, lm_s, divisor, dummy_bf, eps, add_one, use_tpc_rms)
        if do_sample:
            has_pen = 1 if (use_pen_update and penalty_active(repetition_penalty, presence_penalty)) else 0
            if has_pen:
                anc_i32 = verify_tokens_i32[0].index_select(0, ancestor_i32.reshape(-1)).reshape(ancestor_i32.shape)
                pm_u8 = pen_mask_u8
            else:
                anc_i32 = dummy_i32
                pm_u8 = dummy_u8
            k = min(top_k, logits.shape[-1])
            top_vals, top_idx_i32 = hpu_qwen35_tpc.topk(
                logits[0], pm_u8, anc_i32, k, has_pen, float(presence_penalty), float(repetition_penalty),
            )
            ru_arg = rand_u
        else:
            top_vals = dummy_f32_2d
            top_idx_i32 = logits[0].argmax(dim=-1, keepdim=True).to(torch.int32)
            ru_arg = dummy_f32_2d

        st_out, commit_toks, safe_commit, meta, ranks = hpu_qwen35_tpc.sample_tree_accept(
            top_vals, top_idx_i32, verify_tokens_i32, tops_i32, ru_arg, state_i32, tree_parent_i32,
            depth, int(do_sample), int(adaptive), float(temperature), float(top_p), float(min_p),
        )
        state_i32.copy_(st_out)
        if use_pen_update:
            pen_mask_u8.index_fill_(0, safe_commit.long(), 1)
        nt_buf.copy_(meta[1:2].view(1, 1).to(nt_buf.dtype))
        h_buf.copy_(x.index_select(1, meta[0:1]))
        ring_toks_s.copy_(commit_toks)
        ring_acc_s.copy_(meta[3:4])
        ring_path_s.copy_(st_out[9:17])
        if adaptive:
            ring_ranks_s.copy_(ranks[:depth])
        return meta[2:3], st_out[9:17]

    # Fallback path when top_k > 24 or top_k == 0 with do_sample=True.
    verify_tokens = verify_tokens_i32.long()
    tops = tops_i32.long()
    x, toks = _head_sample_core(
        x, final_nw, lm_w, lm_s, verify_tokens, pen_mask_u8 if use_pen_update else None, ancestor_i32.long(), rand_u,
        divisor, dummy_bf, dummy_u8, dummy_i32,
        eps, add_one, do_sample, temperature, top_k, top_p, min_p,
        repetition_penalty, presence_penalty, use_tpc_rms, use_tpc_topk,
    )
    dev = toks.device
    rows = verify_tokens.shape[1]
    toks_row = toks[0]
    draft_row = verify_tokens[0]
    tree_parent_t = tree_parent_i32.long()
    ar_rows = torch.arange(rows, device=dev, dtype=torch.long)
    ar_8 = torch.arange(8, device=dev, dtype=torch.long)
    zero_l = torch.zeros((), device=dev, dtype=torch.long)
    cur = zero_l
    active = torch.ones((), device=dev, dtype=torch.bool)
    acc_len = torch.ones((), device=dev, dtype=torch.long)
    path_slots = ar_8.clone()
    commit_toks = torch.zeros((8,), device=dev, dtype=torch.long)

    for h in range(depth):
        cur_tok = toks_row.index_select(0, cur.unsqueeze(0)).squeeze(0)
        match_c = (tree_parent_t == cur) & (draft_row == cur_tok) & (ar_rows > 0)
        has_match = match_c.any() & active
        next_node = torch.where(match_c, ar_rows, zero_l).sum()
        cur = torch.where(has_match, next_node, cur)
        acc_len = acc_len + has_match.to(torch.long)
        path_slots = torch.where((ar_8 == (h + 1)) & has_match, next_node, path_slots)
        commit_toks = torch.where(ar_8 == h, torch.where(has_match, cur_tok, commit_toks), commit_toks)
        active = has_match

    last_tok = toks_row.index_select(0, cur.unsqueeze(0)).squeeze(0)
    commit_toks = torch.where(ar_8 == (acc_len - 1), last_tok, commit_toks)

    if use_pen_update:
        safe_commit = torch.where(ar_8 < acc_len, commit_toks, last_tok)
        pen_mask_u8.index_fill_(0, safe_commit, 1)

    nt_buf.copy_(last_tok.view(1, 1))
    h_buf.copy_(x.index_select(1, cur.unsqueeze(0)))

    old_pos = state_i32[17:18].clone()
    acc_i32 = acc_len.to(torch.int32).unsqueeze(0)
    on_chain = (cur == (acc_len - 1))
    kept = torch.where(on_chain, acc_len, acc_len - 1).clamp(max=depth).to(torch.int32).unsqueeze(0)
    ps_i32 = path_slots.to(torch.int32)
    st_new = torch.cat((state_i32[:8], acc_i32, ps_i32, old_pos + acc_i32, state_i32[18:19] + kept), dim=0)
    state_i32.copy_(st_new)

    ring_toks_s.copy_(commit_toks.to(torch.int32))
    ring_acc_s.copy_(acc_i32)
    ring_path_s.copy_(ps_i32)
    if adaptive:
        ar_7 = torch.arange(7, device=dev, dtype=torch.long)
        eq = (tops[:, 0] == toks_row[:depth].unsqueeze(1))
        r_hit = torch.where(eq, ar_7.unsqueeze(0), zero_l).sum(dim=-1)
        r_val = torch.where(eq.any(dim=-1), r_hit, 7).to(torch.int32)
        ring_ranks_s.copy_(r_val)
    return old_pos, ps_i32


def _mtp_draft_chain_fn(embed_w, mtp_w, lm_w, lm_s, root_token, root_hidden, rotary_chain,
                        mtp_kv_key, mtp_kv_val, mtp_base, gather_idx, divisor, dummy_bf, dummy_u8, dummy_i32,
                        cfg_tup, depth, topk_k, shared_quant, native_rope,
                        use_tpc_rms, use_tpc_topk, use_tpc_qk, use_tpc_sig, use_tpc_silu,
                        rot_tab=None, state_i32=None, rot_half=0, out_i32=False):
    (pre_emb_nw, pre_hid_nw, fc_w, fc_s, in_nw, q_w, q_s, k_w, k_s, v_w, v_s,
     qn_w, kn_w, o_w, o_s, post_nw, gate_w, gate_s, up_w, up_s, down_w, down_s, out_nw) = mtp_w
    num_heads, num_kv_heads, head_dim, rotary_dim, eps, add_one = cfg_tup
    if rot_tab is not None and state_i32 is not None:
        pos_i32 = state_i32[17:18]
        mtp_base = state_i32[18:19]
        rot_pos = pos_i32 + torch.arange(depth, device=pos_i32.device, dtype=torch.int32)
        if use_tpc_qk:
            rotary_chain = rot_tab.index_select(0, rot_pos)
        else:
            rotary_chain = rot_tab.index_select(0, rot_pos).view(1, 1, depth, rot_half, 2, 2)
    cur_tok = root_token.to(torch.int32) if out_i32 else root_token
    cur_hid = root_hidden
    mtp_positions = mtp_base + torch.arange(depth, device=mtp_base.device, dtype=mtp_base.dtype)
    mtp_seqlens = (mtp_positions + 1).to(torch.int32)
    tops = []
    for k in range(depth):
        emb = F.embedding(cur_tok, embed_w).to(cur_hid.dtype)
        if use_tpc_rms and fc_s is not None:
            qfc, sfc = hpu_qwen35_tpc.dual_rmsnorm_cat_quant(
                emb.reshape(-1, emb.shape[-1]), cur_hid.reshape(-1, cur_hid.shape[-1]),
                pre_emb_nw, pre_hid_nw, int(add_one), eps,
            )
            x = _fp8_proj_raw(qfc, sfc, fc_w, fc_s, emb.shape, emb.dtype)
        else:
            fused_in = torch.cat((_rms_norm(emb, pre_emb_nw, eps, add_one), _rms_norm(cur_hid, pre_hid_nw, eps, add_one)), dim=-1)
            x = _linear_raw(fused_in, fc_w, fc_s, divisor)
        bs, seq, _ = x.shape

        if shared_quant and q_s is not None and k_s is not None and v_s is not None:
            _, _, qx, sx = _rms_quant_raw(
                x, None, in_nw, eps, add_one, divisor, use_tpc_rms,
                has_res=False, store_norm=False, dummy_bf=dummy_bf,
            )
            qg = _fp8_proj_raw(qx, sx, q_w, q_s, x.shape, x.dtype)
            xk = _fp8_proj_raw(qx, sx, k_w, k_s, x.shape, x.dtype)
            xv = _fp8_proj_raw(qx, sx, v_w, v_s, x.shape, x.dtype)
        else:
            xn = _rms_norm(x, in_nw, eps, add_one)
            qg = _linear_raw(xn, q_w, q_s, divisor)
            xk = _linear_raw(xn, k_w, k_s, divisor)
            xv = _linear_raw(xn, v_w, v_s, divisor)

        if use_tpc_qk:
            xq, xk, gate = hpu_qwen35_tpc.qk_norm_rope(
                qg.reshape(seq, -1), xk.reshape(seq, -1), qn_w, kn_w, rotary_chain[k:k + 1],
                num_heads, num_kv_heads, int(add_one), eps,
            )
            xv = xv.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
        else:
            qg = qg.view(bs, seq, num_heads, head_dim * 2)
            xq, gate = qg[..., :head_dim], qg[..., head_dim:].reshape(bs, seq, -1)
            xq = _rms_norm(xq, qn_w, eps, add_one).transpose(1, 2)
            xk = _rms_norm(xk.view(bs, seq, num_kv_heads, head_dim), kn_w, eps, add_one).transpose(1, 2)
            xv = xv.view(bs, seq, num_kv_heads, head_dim).transpose(1, 2)
            xq, xk = _apply_partial_rope(xq, xk, rotary_chain[:, :, k:k + 1], rotary_dim, native_rope)

        mtp_kv_key.index_copy_(2, mtp_positions[k:k + 1], xk)
        mtp_kv_val.index_copy_(2, mtp_positions[k:k + 1], xv)
        out = hpu_kv_attention.single_decode(xq, mtp_kv_key, mtp_kv_val, mtp_seqlens[k:k + 1])
        if use_tpc_sig and o_s is not None:
            qo, so = hpu_qwen35_tpc.sig_mul_quant(
                out.reshape(-1, out.shape[-1]), gate.reshape(-1, gate.shape[-1]),
            )
            o_out = _fp8_proj_raw(qo, so, o_w, o_s, x.shape, x.dtype)
        else:
            o_out = _linear_raw(out * gate.reshape(bs, seq, -1).sigmoid(), o_w, o_s, divisor)
        x = _mlp_raw(
            x, o_out, post_nw, eps, add_one, gate_w, gate_s, up_w, up_s, down_w, down_s,
            divisor, shared_quant, use_tpc_rms=use_tpc_rms, use_tpc_silu=use_tpc_silu, dummy_bf=dummy_bf,
        )

        if use_tpc_rms and lm_s is not None:
            _, _, qlm, slm = _rms_quant_raw(
                x, None, out_nw, eps, add_one, divisor, True,
                has_res=False, store_norm=False, dummy_bf=dummy_bf,
            )
            logits = _fp8_proj_raw(qlm, slm, lm_w, lm_s, x.shape, x.dtype)[:, -1]
        else:
            n1 = _rms_norm(x, out_nw, eps, add_one)
            logits = _linear_raw(n1, lm_w, lm_s, divisor)[:, -1]

        if topk_k > 1:
            if use_tpc_topk and topk_k <= 24:
                _, top_i32 = hpu_qwen35_tpc.topk(logits, dummy_u8, dummy_i32, topk_k, 0, 0.0, 1.0)
                top = top_i32 if out_i32 else top_i32.long()
            else:
                top = logits.topk(topk_k, dim=-1).indices
                if out_i32:
                    top = top.to(torch.int32)
            cur_tok = top[:, :1]
        else:
            cur_tok = logits.argmax(dim=-1, keepdim=True)
            if out_i32:
                cur_tok = cur_tok.to(torch.int32)
            top = cur_tok
        cur_hid = x
        tops.append(top)
    stacked = torch.stack(tops, dim=0)
    root_flat = root_token.to(torch.int32).reshape(-1) if out_i32 else root_token.reshape(-1)
    flat_cand = torch.cat((root_flat, stacked.reshape(-1)))
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
def _compiled_prepare_verify_ondev():
    return torch.compile(_prepare_verify_ondev_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
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
def _compiled_verify_head_accept():
    return torch.compile(_verify_head_accept_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
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
def _compiled_kv_compact_ondev():
    return torch.compile(_kv_compact_ondev_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


@lru_cache(maxsize=1)
def _compiled_mtp_draft_chain():
    return torch.compile(_mtp_draft_chain_fn, backend="hpu_backend", fullgraph=True, dynamic=None,
                         options={"force_static_compile": True})


def reset_compile_caches():
    for fn in (_compiled_gdn3, _compiled_gdn1, _compiled_stage4, _compiled_kv_commit_slot,
               _compiled_kv_compact_ondev, _compiled_tree_mask, _compiled_prepare_verify_ondev,
               _compiled_attn, _compiled_verify_head, _compiled_verify_head_accept, _compiled_mtp_draft_chain):
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
    prog_pad = prog + [0] * (8 - len(prog))
    return (
        torch.tensor(conv_src, dtype=torch.long, device=device),
        torch.tensor(conv_src, dtype=torch.int32, device=device),
        torch.tensor(tree_parents, dtype=torch.long, device=device),
        torch.tensor(parent, dtype=torch.int32, device=device),
        torch.tensor(prog_pad[:8], dtype=torch.int32, device=device),
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
                     *, shared_quant=True, use_fused_sdpa=True, kv_bucket=0, fuse_stage4=False,
                     use_tpc=None, batch_steps=DEFAULT_BATCH_STEPS, pen_mask=None, adaptive=False):
    """Create request-local modular compiled draft/verify/commit runner for HPU Qwen3.5 MTP."""
    if not can_compile_mtp(qwen, embeds, pkv, mtp_kv, qwen_transformer_cls, linear_kv_cls):
        return None

    device = embeds.device
    dtype = embeds.dtype
    cfg = qwen.model.config
    divisor = _divisor(device)
    dummy_bf = _dummy_bf16(device)
    dummy_u8 = _dummy_u8(device)
    dummy_i32 = _dummy_i32_2d(device)
    dummy_f32 = _dummy_f32_1d(device)
    dummy_f32_2d = _dummy_f32_2d(device)
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
    gdn_channels = kh * k_hd * 2 + vh * v_hd
    gdn_cfg = (kh, vh, k_hd, v_hd, taps, eps, add_one)
    attn_cfg = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, rotary_dim, eps, add_one)

    rows = tree.rows
    bs = embeds.shape[0]
    tpc_enabled = hpu_qwen35_tpc.available if use_tpc is None else (bool(use_tpc) and hpu_qwen35_tpc.available)
    use_tpc_gdn = (tpc_enabled and k_hd == 128 and v_hd % 64 == 0 and vh % kh == 0
                   and gdn_channels % 128 == 0 and taps == 4 and rows <= 8)
    use_tpc_rms = tpc_enabled and cfg.hidden_size % 256 == 0 and cfg.hidden_size <= 4096
    use_tpc_gated = tpc_enabled and v_hd == 128 and (vh * v_hd) % 256 == 0 and (vh * v_hd) <= 4096
    use_tpc_silu = tpc_enabled and cfg.intermediate_size % 256 == 0
    use_tpc_sig = tpc_enabled and (cfg.num_attention_heads * cfg.head_dim) % 256 == 0 and (cfg.num_attention_heads * cfg.head_dim) <= 4096
    use_tpc_qk = tpc_enabled and native_rope and cfg.head_dim == 256 and rotary_dim == 64
    lm_head = getattr(qwen.model, "lm_head", None)
    vocab_size = lm_head.weight.shape[0] if lm_head is not None else qwen.model.embed_tokens.weight.shape[0]
    use_tpc_topk = tpc_enabled and vocab_size <= 327680
    use_tpc_accept = tpc_enabled and rows <= 8

    gdn_ws, gdn_conv_st, gdn_rec_st = [], [], []
    gdn_ping = []
    attn_ws = []
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
            if use_tpc_gdn:
                cw = g.conv1d.weight.reshape(channels, taps).float().flip(1).transpose(0, 1).contiguous()
                cb = g.conv1d.bias.float().reshape(channels).contiguous() if g.conv1d.bias is not None else None
                wab = torch.cat((g.in_proj_a.weight, g.in_proj_b.weight), dim=0).contiguous()
                gdn_ws.append((
                    layer.input_layernorm.weight,
                    *_weight_pair(g.in_proj_qkv),
                    *_weight_pair(g.in_proj_z),
                    wab, cw, cb,
                    cache.g_decay, cache.dt_bias, cache.norm_weight.float(),
                    *_weight_pair(g.out_proj),
                    layer.post_attention_layernorm.weight,
                    *_weight_pair(layer.mlp.gate_proj),
                    *_weight_pair(layer.mlp.up_proj),
                    *_weight_pair(layer.mlp.down_proj),
                ))
                gdn_conv_st.append(cache.conv_state.transpose(1, 2).contiguous())
                # Packed 4x16 FP32 state [B, vh*(v_hd//16), 32, 64] ping-ponged across sides (zero EDMA copy_).
                packed = hpu_qwen35_tpc.pack_gdn_state(cache.recurrent_state)
                # Initial parity=0 -> r_side=1 holds initial state, w_side=0 receives committed state.
                gdn_rec_st.append((torch.empty_like(packed), packed))
                sides = []
                for _ in range(2):
                    sides.append((
                        torch.zeros(bs, rows, channels, device=device, dtype=dtype),
                        torch.zeros(bs, rows, channels, device=device, dtype=dtype),
                        torch.zeros(bs, rows, vh * 2, device=device, dtype=dtype),
                    ))
            else:
                cw = g.conv1d.weight.reshape(channels, taps).float().flip(1).transpose(0, 1).view(1, 1, taps, channels).contiguous()
                cb = g.conv1d.bias.float().view(1, 1, channels).contiguous() if g.conv1d.bias is not None else None
                gdn_ws.append((
                    layer.input_layernorm.weight,
                    *_weight_pair(g.in_proj_qkv),
                    *_weight_pair(g.in_proj_z),
                    g.in_proj_a.weight, g.in_proj_b.weight, cw, cb,
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
    kv_compact_compiled = _compiled_kv_compact_ondev()
    mask_compiled = _compiled_tree_mask()
    prep_ondev_compiled = _compiled_prepare_verify_ondev()
    attn_compiled = _compiled_attn()
    head_compiled = _compiled_verify_head()
    head_accept_compiled = _compiled_verify_head_accept()
    draft_compiled = _compiled_mtp_draft_chain()

    rot_half = ftab.shape[3]
    if use_tpc_qk:
        # [max_pos, 128] FP32: [cos0..31, cos0..31, -sin0..31, +sin0..31]
        rot_tab = torch.cat((
            ftab[0, 0, :, :, 0, 0], ftab[0, 0, :, :, 1, 1],
            ftab[0, 0, :, :, 0, 1], ftab[0, 0, :, :, 1, 0],
        ), dim=-1).float().contiguous()
    else:
        rot_tab = ftab.reshape(ftab.shape[2], -1).contiguous()
    torch._dynamo.mark_dynamic(rot_tab, 0)

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
    u_batch_buf = (
        torch.empty((MAX_BATCH_STEPS, rows, 1), device=device, dtype=torch.float32)
        if sampling is not None else None
    )

    # On-device state [19] int32 = [prog[0..7], pending, slots[0..7], pos, mtp_base].
    can_batch = use_tpc_gdn and batch_steps > 0
    state_i32 = torch.zeros((19,), device=device, dtype=torch.int32)
    state_i32[17] = int(embeds.shape[1])
    state_i32[18] = 0
    pen_mask_u8 = pen_mask.to(torch.uint8) if pen_mask is not None else None
    use_pen_update = pen_mask_u8 is not None and sampling is not None and penalty_active(
        sampling["repetition_penalty"], sampling["presence_penalty"],
    )
    ring_toks = torch.zeros((MAX_BATCH_STEPS, 8), device=device, dtype=torch.int32)
    ring_acc = torch.zeros((MAX_BATCH_STEPS,), device=device, dtype=torch.int32)
    ring_path = torch.zeros((MAX_BATCH_STEPS, 8), device=device, dtype=torch.int32)
    ring_ranks = torch.zeros((MAX_BATCH_STEPS, 7), device=device, dtype=torch.int32)

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
        "mtp_base": 0,
        "conv_src": None,
        "conv_src_i32": None,
        "tree_parents": None,
        "tree_parent_i32": None,
        "tree_pos_i32": None,
        "tree_doublings": 2,
        "last_gather": (None, None),
        "last_anc": (None, None),
    }

    def sync_tree(cur_tree):
        cs, cs32, tp, tpt32, prog32, td = _tree_gdn_tables(cur_tree, window, taps, device)
        state["conv_src"] = cs
        state["conv_src_i32"] = cs32
        state["tree_parents"] = tp
        state["tree_parent_i32"] = tpt32
        state["tree_pos_i32"] = cur_tree.positions[0].to(torch.int32)
        state["tree_doublings"] = td
        state_i32[:8].copy_(prog32)

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

    def _run_verify_layers(x, rotary, ks, vs, write_pos, prefix_len, cur_tree, sdpa_mask, ctl_tup, cd, td):
        w_side = state["parity"]
        r_side = 1 - w_side

        def gdn_state_for(idx):
            if use_tpc_gdn:
                return (
                    gdn_conv_st[idx], gdn_rec_st[idx][w_side], gdn_rec_st[idx][r_side],
                    *gdn_ping[idx][w_side], *gdn_ping[idx][r_side],
                )
            return (gdn_conv_st[idx], gdn_rec_st[idx], *gdn_ping[idx][w_side], *gdn_ping[idx][r_side])

        for kind, idx_info in plan:
            if kind == "stage4":
                (i0, i1, i2), kv_idx = idx_info
                x = stage4_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), gdn_ws[i1], gdn_state_for(i1),
                    gdn_ws[i2], gdn_state_for(i2), attn_ws[kv_idx],
                    rotary, ks[kv_idx], vs[kv_idx], write_pos, prefix_len, cur_tree.mask, sdpa_mask,
                    ctl_tup, divisor, dummy_bf, dummy_f32, gdn_cfg, attn_cfg,
                    shared_quant, native_rope, use_fused_sdpa, cd, td,
                    use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu, use_tpc_qk, use_tpc_sig,
                )
            elif kind == "gdn3":
                i0, i1, i2 = idx_info
                x = gdn3_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), gdn_ws[i1], gdn_state_for(i1),
                    gdn_ws[i2], gdn_state_for(i2), ctl_tup, divisor, dummy_bf, dummy_f32, gdn_cfg,
                    shared_quant, cd, td, use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                )
            elif kind == "gdn1":
                i0 = idx_info[0]
                x = gdn1_compiled(
                    x, gdn_ws[i0], gdn_state_for(i0), ctl_tup, divisor, dummy_bf, dummy_f32, gdn_cfg,
                    shared_quant, cd, td, use_tpc_gdn, use_tpc_rms, use_tpc_gated, use_tpc_silu,
                )
            else:
                x = attn_compiled(
                    x, attn_ws[idx_info], rotary, ks[idx_info], vs[idx_info],
                    write_pos, prefix_len, cur_tree.mask, sdpa_mask, divisor, dummy_bf,
                    attn_cfg, shared_quant, native_rope, use_fused_sdpa,
                    use_tpc_rms, use_tpc_qk, use_tpc_sig, use_tpc_silu,
                )
        return x

    def run_draft(nt_buf, h_buf, _ftab, pos, depth, topk_k, gather_idx):
        if use_tpc_qk:
            rot_chain = rot_tab[pos:pos + depth].clone()
        else:
            rot_chain = rot_tab[pos:pos + depth].clone().view(1, 1, depth, rot_half, 2, 2)
        base = mtp_kv.index
        mtp_base_buf.copy_(mtp_base_table[base:base + 1])
        mk_l, mv_l = _slice_kv((mtp_kv,), base + depth)
        verify_tokens, tops = draft_compiled(
            embed_w, mtp_w, lm_w, lm_s,
            nt_buf, h_buf, rot_chain,
            mk_l[0], mv_l[0], mtp_base_buf, gather_idx, divisor, dummy_bf, dummy_u8, dummy_i32,
            attn_cfg, depth, topk_k, shared_quant, native_rope,
            use_tpc_rms, use_tpc_topk, use_tpc_qk, use_tpc_sig, use_tpc_silu,
        )
        mtp_kv.advance(depth)
        return verify_tokens, tops

    def run_verify(verify_tokens, _ftab, pos, cur_tree, ancestor_indices, cur_pen_mask, cur_sampling, _generator):
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur_rows = cur_tree.rows
        for cache in kv_caches:
            cache.prepare(cur_rows)
        rot_idx = cur_tree.positions[0] + pos
        if use_tpc_qk:
            rotary = rot_tab.index_select(0, rot_idx)
        else:
            rotary = rot_tab.index_select(0, rot_idx).view(1, 1, cur_rows, rot_half, 2, 2)
        write_pos = kv_caches[0].position[:cur_rows]
        prefix_len = kv_caches[0].seqlen
        ks, vs = _slice_kv(kv_caches, kv_caches[0].index + cur_rows)

        if use_tpc_gdn:
            ctl_tup = (state["conv_src_i32"], state_i32)
            cd = 2
        else:
            pend = state["pending"]
            p_max = 4 if pend <= 4 else 8
            cd = 2 if p_max == 4 else 3
            slots_t, live_t, cc_idx, cp = onchain_ctl[(p_max, pend)]
            if not state["on_chain"] and pend > 0:
                path_vals = list(state["path"]) + [0] * (p_max - pend)
                offchain_slots[p_max].copy_(torch.tensor(path_vals, dtype=torch.long), non_blocking=True)
                slots_t = offchain_slots[p_max]
            ctl_tup = (slots_t, live_t, cc_idx, state["conv_src"], state["tree_parents"], cp)

        sdpa_mask = mask_compiled(ks[0], prefix_len, cur_tree.mask, cur_rows) if use_fused_sdpa else None
        x = F.embedding(verify_tokens, embed_w).to(dtype)
        x = _run_verify_layers(x, rotary, ks, vs, write_pos, prefix_len, cur_tree, sdpa_mask, ctl_tup, cd, state["tree_doublings"])

        if cur_sampling is not None:
            u_buf.copy_(torch.rand((cur_rows, 1), dtype=torch.float32, generator=cpu_gen), non_blocking=True)
            hidden, toks = head_compiled(
                x, final_nw, lm_w, lm_s, verify_tokens, cur_pen_mask, ancestor_indices, u_buf,
                divisor, dummy_bf, dummy_u8, dummy_i32,
                eps, add_one, True, cur_sampling["temperature"], cur_sampling["top_k"], cur_sampling["top_p"],
                cur_sampling["min_p"], cur_sampling["repetition_penalty"], cur_sampling["presence_penalty"],
                use_tpc_rms, use_tpc_topk,
            )
        else:
            hidden, toks = head_compiled(
                x, final_nw, lm_w, lm_s, verify_tokens, None, None, None,
                divisor, dummy_bf, dummy_u8, dummy_i32,
                eps, add_one, False, 1.0, 0, 1.0, 0.0, 1.0, 0.0,
                use_tpc_rms, use_tpc_topk,
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
        if use_tpc_gdn:
            p_list = list(path) + [0] * (8 - len(path))
            state_i32[8:17].copy_(torch.tensor([len(path)] + p_list[:8], dtype=torch.int32), non_blocking=True)
        if not on_chain:
            for k, r in enumerate(path[1:], 1):
                if r != k:
                    src = (base_pos + r) % kv_cap
                    dst = (base_pos + k) % kv_cap
                    kv_src_buf.copy_(kv_pos_table[src:src + 1])
                    kv_dst_buf.copy_(kv_pos_table[dst:dst + 1])
                    kv_commit_compiled(ks_full, vs_full, kv_src_buf, kv_dst_buf)

    def run_batch(n_steps, nt_buf, h_buf, cur_tree, depth, topk_k, gather_idx, ancestor_indices, cur_sampling,
                  pos=0, record_timer_fn=None):
        """Run n_steps (1..MAX_BATCH_STEPS) complete draft+verify+accept cycles on device with 1 D2H readback."""
        comfy.model_management.throw_exception_if_processing_interrupted()
        cur_rows = cur_tree.rows
        if cur_sampling is not None:
            u_cpu = torch.stack(
                [torch.rand((cur_rows, 1), dtype=torch.float32, generator=cpu_gen) for _ in range(n_steps)],
                dim=0,
            )
            u_batch_buf[:n_steps].copy_(u_cpu, non_blocking=True)
        ctl_tup = (state["conv_src_i32"], state_i32)
        td = state["tree_doublings"]
        t_pos_i32 = state["tree_pos_i32"]
        t_par_i32 = state["tree_parent_i32"]
        if state["last_gather"][0] is gather_idx:
            g_idx_i32 = state["last_gather"][1]
        else:
            g_idx_i32 = gather_idx.to(torch.int32)
            state["last_gather"] = (gather_idx, g_idx_i32)
        if state["last_anc"][0] is ancestor_indices:
            anc_i32 = state["last_anc"][1]
        else:
            anc_i32 = ancestor_indices.to(torch.int32)
            state["last_anc"] = (ancestor_indices, anc_i32)

        # Slice KV caches once for the whole batch using conservative upper bound on sequence growth.
        ks, vs = _slice_kv(kv_caches, pos + n_steps * (depth + 1) + cur_rows)
        mk_l, mv_l = _slice_kv((mtp_kv,), state["mtp_base"] + (n_steps + 1) * depth)
        mk, mv = mk_l[0], mv_l[0]

        do_samp = cur_sampling is not None
        temp = cur_sampling["temperature"] if do_samp else 1.0
        tk = cur_sampling["top_k"] if do_samp else 0
        tp = cur_sampling["top_p"] if do_samp else 1.0
        mp = cur_sampling["min_p"] if do_samp else 0.0
        rp = cur_sampling["repetition_penalty"] if do_samp else 1.0
        pp = cur_sampling["presence_penalty"] if do_samp else 0.0

        for s in range(n_steps):
            if record_timer_fn is not None and s == 0:
                record_timer_fn(0)
            verify_tokens_i32, tops_i32 = draft_compiled(
                embed_w, mtp_w, lm_w, lm_s,
                nt_buf, h_buf, None,
                mk, mv, state_i32[18:19], g_idx_i32, divisor, dummy_bf, dummy_u8, dummy_i32,
                attn_cfg, depth, topk_k, shared_quant, native_rope,
                use_tpc_rms, use_tpc_topk, use_tpc_qk, use_tpc_sig, use_tpc_silu,
                rot_tab, state_i32, rot_half, True,
            )
            if record_timer_fn is not None and s == 0:
                record_timer_fn(1)

            x, rotary, write_pos, prefix_len, sdpa_mask = prep_ondev_compiled(
                ks[0], rot_tab, embed_w, verify_tokens_i32, state_i32, t_pos_i32, cur_tree.mask,
                cur_rows, rot_half, use_fused_sdpa, use_tpc_qk,
            )
            x = _run_verify_layers(
                x, rotary, ks, vs, write_pos, prefix_len, cur_tree,
                sdpa_mask if use_fused_sdpa else None, ctl_tup, 2, td,
            )
            u_arg = u_batch_buf[s] if do_samp else dummy_f32_2d
            old_pos, path_slots = head_accept_compiled(
                x, final_nw, lm_w, lm_s, verify_tokens_i32, tops_i32,
                pen_mask_u8 if use_pen_update else dummy_u8,
                anc_i32, u_arg,
                state_i32, t_par_i32, nt_buf, h_buf,
                ring_toks[s], ring_acc[s:s + 1], ring_path[s], ring_ranks[s, :depth],
                divisor, dummy_bf, dummy_u8, dummy_i32, dummy_f32_2d,
                eps, add_one, do_samp, temp, tk, tp, mp, rp, pp,
                depth, adaptive, use_pen_update, use_tpc_rms, use_tpc_topk, use_tpc_accept,
            )
            kv_compact_compiled(ks, vs, old_pos, path_slots, depth)
            state["parity"] ^= 1
            if record_timer_fn is not None and s == 0:
                record_timer_fn(2)

        # Single batched D2H readback for all n_steps.
        if adaptive:
            pack = torch.cat((
                ring_acc[:n_steps],
                ring_toks[:n_steps].reshape(-1),
                ring_path[:n_steps].reshape(-1),
                ring_ranks[:n_steps, :depth].reshape(-1),
            )).tolist()
            accs = pack[:n_steps]
            o1 = n_steps
            o2 = o1 + n_steps * 8
            o3 = o2 + n_steps * 8
            toks_flat = pack[o1:o2]
            path_flat = pack[o2:o3]
            ranks_flat = pack[o3:]
        else:
            pack = torch.cat((
                ring_acc[:n_steps],
                ring_toks[:n_steps].reshape(-1),
                ring_path[:n_steps].reshape(-1),
            )).tolist()
            accs = pack[:n_steps]
            o1 = n_steps
            o2 = o1 + n_steps * 8
            toks_flat = pack[o1:o2]
            path_flat = pack[o2:]
            ranks_flat = None

        results = []
        for s in range(n_steps):
            al = accs[s]
            commit = tuple(toks_flat[s * 8:s * 8 + al])
            p_row = path_flat[s * 8:s * 8 + al]
            on_chain = (p_row[-1] == al - 1) if p_row else True
            state["mtp_base"] += min(al if on_chain else al - 1, depth)
            if adaptive:
                r_row = ranks_flat[s * depth:(s + 1) * depth]
                ranks = []
                for node in p_row:
                    if node >= depth:
                        break
                    ranks.append(r_row[node])
            else:
                ranks = []
            results.append((commit, ranks))
        return results

    return {
        "run_draft": run_draft,
        "run_verify": run_verify,
        "record_commit": record_commit,
        "sync_tree": sync_tree,
        "run_batch": run_batch if can_batch else None,
        "batch_steps": min(batch_steps, MAX_BATCH_STEPS) if can_batch else 1,
        "use_tpc_gdn": use_tpc_gdn,
        "use_tpc_rms": use_tpc_rms,
        "use_tpc_topk": use_tpc_topk,
        "use_tpc_gated": use_tpc_gated,
        "use_tpc_silu": use_tpc_silu,
        "use_tpc_sig": use_tpc_sig,
        "use_tpc_qk": use_tpc_qk,
        "use_tpc_accept": use_tpc_accept,
    }
