"""Regional compilation of fixed-cache decode attention on HPU."""
from functools import lru_cache

import torch
import comfy_kitchen
from comfy_kitchen.backends.eager.rope import apply_rope_split_half

import comfy.model_management


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


def _rope_split_half(xq, xk, freqs_cis):
    if (xq.dtype == xk.dtype == torch.bfloat16 and freqs_cis.dtype == torch.float32
            and xq.ndim == xk.ndim == 4 and freqs_cis.ndim == 6
            and xq.shape[-1] == xk.shape[-1] == 64 and 0 < xq.shape[-2] <= 8
            and xq.shape[-2] == xk.shape[-2]
            and not comfy.model_management.args.disable_comfy_compiler
            and not comfy.model_management.in_training and not torch.is_grad_enabled()):
        return _compiled_rope()(xq, xk, freqs_cis)
    return apply_rope_split_half(xq, xk, freqs_cis)


if comfy.model_management.hpu_available:
    torch.library.register_kernel("comfy_kitchen::apply_rope_split_half", "hpu", _rope_split_half)
