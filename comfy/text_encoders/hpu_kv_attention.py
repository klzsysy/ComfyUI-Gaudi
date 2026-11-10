"""Regional compilation of fixed-cache decode attention on HPU."""
from functools import lru_cache

import torch
import comfy_kitchen


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
