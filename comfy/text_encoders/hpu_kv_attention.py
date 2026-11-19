"""Regional compilation of fixed-cache decode attention on HPU."""
from functools import lru_cache

import torch
import comfy_kitchen
from comfy_kitchen.backends.eager.rope import apply_rope_split_half

import comfy.model_management

if comfy.model_management.hpu_available:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA


def single_decode(q, key, value, length):
    visible = torch.arange(key.shape[2], device=q.device).view(1, 1, 1, -1) < length.view(-1, 1, 1, 1)
    mask = torch.zeros(visible.shape, device=q.device, dtype=q.dtype).masked_fill(~visible, float("-inf"))
    output = FusedSDPA.apply(q, key, value, mask, 0.0, False, None, "fp32", False)
    return output.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def tree_decode_sdpa(q, key, value, prefix_len, tree_mask_bits):
    seq = q.shape[2]
    rel = torch.arange(key.shape[2], device=q.device).view(1, 1, 1, -1) - prefix_len.view(-1, 1, 1, 1)
    tree_vis = ((tree_mask_bits.view(1, 1, seq, 1) >> rel.clamp(0, seq - 1)) & 1).bool()
    visible = (rel < 0) | ((rel >= 0) & (rel < seq) & tree_vis)
    mask = torch.zeros(visible.shape, device=q.device, dtype=q.dtype).masked_fill(~visible, float("-inf"))
    output = FusedSDPA.apply(q, key, value, mask, 0.0, False, None, "fp32", False)
    return output.transpose(1, 2).reshape(q.shape[0], seq, -1)


def _decode(q, key, value, length):
    return comfy_kitchen.flash_attention_decode_gqa(q, key, value, length)


def _tree(q, key, value, length, k, v, mask):
    out, lse = comfy_kitchen.flash_attention_decode_gqa(q, key, value, length, return_lse=True, causal=False)
    return comfy_kitchen.flash_attention_decode_tree_merge(out, lse, q, k, v, mask, torch.empty_like(out))


@lru_cache(maxsize=1)
def compiled_ops():
    options = {"force_static_compile": True}
    return (torch.compile(_decode, backend="hpu_backend", dynamic=None, options=options),
            torch.compile(_tree, backend="hpu_backend", dynamic=None, options=options))


@lru_cache(maxsize=1)
def _compiled_rope():
    # Retain native addcmul rounding inside the existing kitchen operation.
    return torch.compile(_rope_forward, backend="hpu_backend", dynamic=None,
                         options={"force_static_compile": True, "decomposition_mode": "none"})


def _rotate_split_half(x, frequencies):
    # Functional form of kitchen RoPE avoids compiled writes through temporary views.
    diagonal = frequencies.diagonal(dim1=-2, dim2=-1).movedim(-1, -2)
    products = x.unflatten(-1, (2, -1)) * diagonal
    half = x.shape[-1] // 2
    first = torch.addcmul(products[..., 0, :], x[..., half:], frequencies[..., 0, 1])
    second = torch.addcmul(products[..., 1, :], x[..., :half], frequencies[..., 1, 0])
    return torch.cat((first, second), dim=-1).to(x.dtype)


def _rope_forward(xq, xk, freqs_cis):
    return _rotate_split_half(xq, freqs_cis), _rotate_split_half(xk, freqs_cis)


def _native_rope_forward(query, key, frequencies):
    cosine = torch.cat((frequencies[..., 0, 0], frequencies[..., 1, 1]), dim=-1)
    sine = torch.cat((-frequencies[..., 0, 1], frequencies[..., 1, 0]), dim=-1)
    query_out = torch.ops.hpu.rotary_pos_embedding(query.float(), sine, cosine, None, 0, 0)
    key_out = torch.ops.hpu.rotary_pos_embedding(key.float(), sine, cosine, None, 0, 0)
    return query_out.to(query.dtype), key_out.to(key.dtype)


@lru_cache(maxsize=1)
def _compiled_qwen_rope():
    return torch.compile(_native_rope_forward, backend="hpu_backend", fullgraph=True, dynamic=False,
                         options={"force_static_compile": True, "decomposition_mode": "none"})


def _rope_split_half(xq, xk, freqs_cis):
    if (xq.dtype == xk.dtype == torch.bfloat16 and freqs_cis.dtype == torch.float32
            and xq.ndim == xk.ndim == 4 and freqs_cis.ndim == 6
            and xq.shape[-1] == xk.shape[-1] == 64 and 0 < xq.shape[-2] <= 8
            and xq.shape[-2] == xk.shape[-2]
            and not comfy.model_management.args.disable_comfy_compiler
            and not comfy.model_management.in_training and not torch.is_grad_enabled()):
        function = _rope_forward if torch.compiler.is_compiling() else _compiled_rope()
        return function(xq, xk, freqs_cis)
    return apply_rope_split_half(xq, xk, freqs_cis)


if comfy.model_management.hpu_available:
    torch.library.register_kernel("comfy_kitchen::apply_rope_split_half", "hpu", _rope_split_half)

    @torch.library.custom_op("comfy_hpu::qwen35_rope", mutates_args=(), device_types="hpu")
    def qwen_rope(query: torch.Tensor, key: torch.Tensor, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate using the equal diagonals and opposite sine terms produced by Qwen."""
        return _compiled_qwen_rope()(query, key, frequencies)

    @qwen_rope.register_fake
    def _qwen_rope_meta(query, key, frequencies):
        return (torch.empty_like(query, memory_format=torch.contiguous_format),
                torch.empty_like(key, memory_format=torch.contiguous_format))
