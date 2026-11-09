"""Regional compilation of dense MLP calls on HPU."""
from functools import lru_cache

import torch
import comfy.ops


def _forward(module, x, norm):
    return module(x, norm=norm)


@lru_cache(maxsize=1)
def _compiled_forward():
    return torch.compile(_forward, backend="hpu_backend", dynamic=None, options={"force_static_compile": True})


def forward(module, x, norm):
    if comfy.ops.has_forward_hooks(module) or (norm is not None and comfy.ops.has_forward_hooks(norm)):
        return module(x, norm=norm)
    return _compiled_forward()(module, x, norm)
