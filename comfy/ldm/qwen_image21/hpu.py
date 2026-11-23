"""FP8 and pure-BF16 projection + RMW attention regions for target rows with a cached prefix."""
import contextlib
from functools import lru_cache

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules import hpu_attention
from comfy.ldm.modules.attention import AttentionTensorContainer

_hpu_tpc = None
if hasattr(torch, "hpu"):
    try:
        from comfy.text_encoders import hpu_qwen35_tpc as _hpu_tpc
    except Exception:
        _hpu_tpc = None


def _bf16(value):
    # Bitcasts keep the bridge from folding away the eager BF16 rounding point.
    return value.float().view(torch.int32).view(torch.float32).to(torch.bfloat16).view(torch.int16).view(torch.bfloat16)


def _quantize(x, divisor):
    flat = _bf16(x).flatten(0, 1)
    maximum = _bf16(flat.abs().amax(-1, keepdim=True))
    # A tensor divisor preserves eager division. A constant is compiled as a
    # reciprocal multiply, which can change the final BF16 projection by 1 ULP.
    scale = (maximum.float().clamp_min(1e-12) / divisor).view(torch.int32).view(torch.float32)
    scaled = _bf16(flat * _bf16(scale.reciprocal()))
    return torch.ops.hpu.cast_to_fp8_v2(scaled, None, False, False, torch.float8_e4m3fn)[0], scale


def _project(qx, scale, weight, weight_scale):
    value = torch.ops.hpu.fp8_gemm_v2(qx, False, weight, True, None, torch.bfloat16, None, None, None, False)
    return _bf16(value * _bf16(scale * weight_scale))


def _qkv(x, scale1, zero, eps, divisor, wq, sq, wk, sk, wv, sv, nq, nk, norm_eps):
    x = comfy.quant_ops.ck.adaln(x, scale1, zero, eps)
    qx, scale = _quantize(x, divisor)
    shape = (x.shape[0], x.shape[1], 32, 128)
    q = _project(qx, scale, wq, sq).view(shape)
    k = _project(qx, scale, wk, sk).view(shape)
    v = _project(qx, scale, wv, sv).view(shape)
    return F.rms_norm(q, (128,), nq, norm_eps), F.rms_norm(k, (128,), nk, norm_eps), v


def _prepare(q, k, v, pe, pk, pv):
    cos = torch.stack((pe[..., 0, 0], pe[..., 1, 1]), -1).flatten(-2)
    sin = torch.stack((-pe[..., 0, 1], pe[..., 1, 0]), -1).flatten(-2)
    q = torch.ops.hpu.rotary_pos_embedding(q.float(), sin, cos, None, 0, 1).to(q.dtype)
    k = torch.ops.hpu.rotary_pos_embedding(k.float(), sin, cos, None, 0, 1).to(k.dtype)
    return q.flatten(2), torch.cat((pk, k), 1).flatten(2), torch.cat((pv, v), 1).flatten(2)


def _post(x, output, gate1, scale2, zero, eps, divisor, weight, weight_scale):
    qx, scale = _quantize(output, divisor)
    projected = _project(qx, scale, weight, weight_scale).view_as(x)
    mid = _bf16(x + _bf16(projected * gate1))
    return mid, comfy.quant_ops.ck.adaln(mid, scale2, zero, eps)


@lru_cache(maxsize=1)
def _regions():
    return tuple(torch.compile(f, backend="hpu_backend", dynamic=None, options={"force_static_compile": True})
                 for f in (_qkv, _prepare, _post))


@lru_cache(maxsize=1)
def _divisor(device):
    return torch.tensor(240.0, dtype=torch.float32, device=device)


def _dynamic_fp8(linear):
    return (getattr(linear, "layout_type", None) == "HPUFP8E4M3Layout" and not linear._full_precision_mm
            and linear.bias is None and not linear.weight_function and not linear.bias_function
            and getattr(linear, "input_scale", None) is None and getattr(linear, "pre_quant_scale", None) is None)


def prefix_attention(block, x, mod, pe, pk, pv, options, attention, pending):
    attn = block.attn
    projections = (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0])
    if not all(_dynamic_fp8(layer) for layer in projections):
        return None
    # Retire the oldest region before its buffers overlap a third block.
    if len(pending) == 2:
        completed = pending.pop(0)
        completed.synchronize()
    else:
        completed = torch.hpu.Event()
    qkv, prepare, post = _regions()
    scale1, gate1, scale2, _, zero = mod
    divisor = _divisor(x.device)
    with contextlib.ExitStack() as stack:
        weights = []
        for linear in projections:
            weight, _ = stack.enter_context(comfy.ops.CastBiasWeightContext(
                linear, x, dtype=linear.weight.dtype, bias_dtype=x.dtype, compute_dtype=x.dtype,
                offloadable=True, want_requant=True))
            weights.extend((weight._qdata, weight._params.scale))
        norms = [stack.enter_context(comfy.ops.CastBiasWeightContext(norm, x, offloadable=True))[0]
                 for norm in (attn.norm_q, attn.norm_k)]
        q, k, v = qkv(x, scale1[1], zero, block.img_norm1.eps, divisor, *weights[:6], *norms, attn.norm_q.eps)
        torch._dynamo.mark_dynamic(pk, 1)
        torch._dynamo.mark_dynamic(pv, 1)
        tensors = (AttentionTensorContainer(t) for t in prepare(q, k, v, pe, pk, pv))
        output = attention(*tensors, attn.heads, transformer_options=options, preferred_attention=attn.comfy_attention)
        mid, normed = post(x, output, gate1[1], scale2[1], zero, block.img_norm2.eps, divisor, *weights[6:])
    return mid, normed, completed


def _bf16_pre_nopad(x, scale1, zero, norm1_eps, wq, wk, wv, nq, nk, qk_eps, cos, sin, pk, pv, sc_t):
    h = comfy.quant_ops.ck.adaln(x, scale1, zero, norm1_eps)
    B, Q, _ = h.shape
    q = F.rms_norm(F.linear(h, wq).view(B, Q, 32, 128), (128,), nq, qk_eps)
    k = F.rms_norm(F.linear(h, wk).view(B, Q, 32, 128), (128,), nk, qk_eps)
    v = F.linear(h, wv).view(B, Q, 32, 128)
    q = torch.ops.hpu.rotary_pos_embedding(q.float(), sin, cos, None, 0, 1).to(q.dtype)
    k = torch.ops.hpu.rotary_pos_embedding(k.float(), sin, cos, None, 0, 1).to(k.dtype)
    qh = (q[0].transpose(0, 1) * sc_t).contiguous()
    kh = (torch.cat((pk[0], k[0]), dim=0).transpose(0, 1) * sc_t).contiguous()
    vh = torch.cat((pv[0], v[0]), dim=0).transpose(0, 1).contiguous()
    return qh, kh, vh


def _bf16_pre_pad(x, scale1, zero, norm1_eps, wq, wk, wv, nq, nk, qk_eps, cos, sin, pk, pv, z_pad, sc_t):
    h = comfy.quant_ops.ck.adaln(x, scale1, zero, norm1_eps)
    B, Q, _ = h.shape
    q = F.rms_norm(F.linear(h, wq).view(B, Q, 32, 128), (128,), nq, qk_eps)
    k = F.rms_norm(F.linear(h, wk).view(B, Q, 32, 128), (128,), nk, qk_eps)
    v = F.linear(h, wv).view(B, Q, 32, 128)
    q = torch.ops.hpu.rotary_pos_embedding(q.float(), sin, cos, None, 0, 1).to(q.dtype)
    k = torch.ops.hpu.rotary_pos_embedding(k.float(), sin, cos, None, 0, 1).to(k.dtype)
    qh = (q[0].transpose(0, 1) * sc_t).contiguous()
    kh = (torch.cat((pk[0], k[0], z_pad), dim=0).transpose(0, 1) * sc_t).contiguous()
    vh = torch.cat((pv[0], v[0], z_pad), dim=0).transpose(0, 1).contiguous()
    return qh, kh, vh


def _bf16_post1(x, attn_out, sum_out, gate1, scale2, zero, norm2_eps, wo):
    Q = attn_out.shape[1]
    normed = (attn_out.float() / sum_out[:, :Q].unsqueeze(-1)).to(attn_out.dtype)
    nhd = normed.transpose(0, 1).reshape(1, Q, 4096)
    mid = x + F.linear(nhd, wo) * gate1
    return mid, comfy.quant_ops.ck.adaln(mid, scale2, zero, norm2_eps)


def _bf16_post2(x, o0, s0, m0, o1, s1, m1, gate1, scale2, zero, norm2_eps, wo):
    Q = o0.shape[1]
    s0 = s0[:, :Q]
    m0 = m0[:, :Q]
    s1 = s1[:, :Q]
    m1 = m1[:, :Q]
    mm = torch.maximum(m0, m1)
    w0 = torch.exp(m0 - mm)
    w1 = torch.exp(m1 - mm)
    denom = (w0 * s0 + w1 * s1).unsqueeze(-1)
    normed = ((o0.float() * w0.unsqueeze(-1) + o1.float() * w1.unsqueeze(-1)) / denom).to(o0.dtype)
    nhd = normed.transpose(0, 1).reshape(1, Q, 4096)
    mid = x + F.linear(nhd, wo) * gate1
    return mid, comfy.quant_ops.ck.adaln(mid, scale2, zero, norm2_eps)


def _bf16_ffn_res2(mid, h2, gate2, w_gu, w_out):
    g, u = F.linear(h2, w_gu).chunk(2, dim=-1)
    return mid + F.linear(F.silu(g) * u, w_out) * gate2


@lru_cache(maxsize=1)
def _bf16_regions():
    opts = {"force_static_compile": True}
    return tuple(
        torch.compile(fn, backend="hpu_backend", dynamic=None, options=opts)
        for fn in (_bf16_pre_nopad, _bf16_pre_pad, _bf16_post1, _bf16_post2, _bf16_ffn_res2)
    )


@lru_cache(maxsize=2)
def _bf16_qk_scale(device):
    return torch.tensor(hpu_attention.BF16_EXPMAX_QK_SCALE, dtype=torch.bfloat16, device=device)


@lru_cache(maxsize=2)
def _bf16_zero_pad(device):
    return torch.zeros((512, 32, 128), dtype=torch.bfloat16, device=device)


def _plain_bf16_linear(linear):
    return (getattr(linear, "layout_type", None) is None
            and not getattr(linear, "comfy_force_forward", False)
            and linear.bias is None and not linear.weight_function and not linear.bias_function
            and not isinstance(linear.weight, comfy.quant_ops.QuantizedTensor))


def pack_pe_cs(pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.stack((pe[..., 0, 0], pe[..., 1, 1]), dim=-1).flatten(-2).contiguous()
    sin = torch.stack((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1).flatten(-2).contiguous()
    return cos, sin


def prefix_attention_bf16(block, x, mod, pe, pk, pv, options, pending):
    if _hpu_tpc is None or not _hpu_tpc.available:
        return None
    if pk.shape[0] != 1 or pk.shape[1] < 2 or len(mod) != 5:
        return None
    queries = int(x.shape[1])
    if pe.device != x.device or tuple(pe.shape) != (1, queries, 1, 64, 2, 2):
        return None
    attn = block.attn
    mlp = block.img_mlp
    if not getattr(mlp, "fused", False) or attn.norm_q.eps != attn.norm_k.eps:
        return None
    if getattr(attn.norm_q, "weight_function", None) or getattr(attn.norm_k, "weight_function", None):
        return None
    attn_projs = (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0])
    mlp_projs = (mlp.gate_up, mlp.out)
    if not all(_plain_bf16_linear(layer) for layer in (*attn_projs, *mlp_projs)):
        return None
    if (any(tuple(layer.weight.shape) != (4096, 4096) for layer in attn_projs)
            or mlp.gate_up.weight.shape[1] != 4096 or mlp.out.weight.shape[0] != 4096
            or mlp.gate_up.weight.shape[0] != 2 * mlp.out.weight.shape[1]):
        return None
    kv_len = int(pk.shape[1]) + queries
    if not hpu_attention.rmw_bf16_expmax_supported(queries, kv_len, 32):
        return None

    _, total_k_pad, _ = hpu_attention.rmw_bf16_expmax_layout(kv_len)
    pad_len = total_k_pad - kv_len
    if pad_len == 1:
        # Rare 1-in-512 case where mark_dynamic on dim=1 would fail; fall back cleanly.
        return None

    # Bound in-flight block allocations to keep peak VRAM flat even at 4MP/5MP+.
    if len(pending) == 2:
        completed = pending.pop(0)
        completed.synchronize()
    else:
        completed = torch.hpu.Event()

    pre_nopad, pre_pad, post1, post2, ffn_res2 = _bf16_regions()
    scale1, gate1, scale2, gate2, zero = mod
    sc_t = _bf16_qk_scale(x.device)
    pe_cs = options.get("_qwen21_hpu_pe_cs")
    cos, sin = pe_cs if pe_cs is not None else pack_pe_cs(pe)

    with contextlib.ExitStack() as stack:
        weights = [
            stack.enter_context(comfy.ops.CastBiasWeightContext(layer, x, offloadable=True))[0]
            for layer in attn_projs
        ]
        norms = [
            stack.enter_context(comfy.ops.CastBiasWeightContext(norm, x, offloadable=True))[0]
            for norm in (attn.norm_q, attn.norm_k)
        ]
        torch._dynamo.mark_dynamic(pk, 1)
        torch._dynamo.mark_dynamic(pv, 1)
        if pad_len == 0:
            qh, kh, vh = pre_nopad(
                x, scale1[1], zero, block.img_norm1.eps,
                weights[0], weights[1], weights[2], norms[0], norms[1], attn.norm_q.eps,
                cos, sin, pk, pv, sc_t,
            )
        else:
            z_pad = _bf16_zero_pad(x.device)[:pad_len]
            torch._dynamo.mark_dynamic(z_pad, 0)
            qh, kh, vh = pre_pad(
                x, scale1[1], zero, block.img_norm1.eps,
                weights[0], weights[1], weights[2], norms[0], norms[1], attn.norm_q.eps,
                cos, sin, pk, pv, z_pad, sc_t,
            )
        attn_res = hpu_attention.rmw_bf16_expmax_attention_prepared(qh, kh, vh, kv_len)
        if len(attn_res) == 2:
            o0, s0 = attn_res
            mid, h2 = post1(x, o0, s0, gate1[1], scale2[1], zero, block.img_norm2.eps, weights[3])
        else:
            o0, s0, m0, o1, s1, m1 = attn_res
            mid, h2 = post2(x, o0, s0, m0, o1, s1, m1, gate1[1], scale2[1], zero, block.img_norm2.eps, weights[3])

    with contextlib.ExitStack() as stack:
        w_gu = stack.enter_context(comfy.ops.CastBiasWeightContext(mlp.gate_up, h2, offloadable=True))[0]
        w_out = stack.enter_context(comfy.ops.CastBiasWeightContext(mlp.out, h2, offloadable=True))[0]
        output = ffn_res2(mid, h2, gate2[1], w_gu, w_out)

    completed.record()
    pending.append(completed)
    return output
