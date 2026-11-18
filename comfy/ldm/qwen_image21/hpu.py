"""FP8 projection and RMW attention regions for target rows with a cached prefix."""
import contextlib
from functools import lru_cache

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer


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
