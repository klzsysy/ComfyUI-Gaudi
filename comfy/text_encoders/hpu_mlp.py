"""Regional compilation of dense MLP calls on HPU."""
from functools import lru_cache

import torch
import comfy.ops


def _forward(module, x, norm, residual=False):
    output = module(x, norm=norm)
    if residual:
        torch.add(x, output, out=x)
        # Keep the BF16 projection materialized at the graph boundary.
        return x, output
    return output


@lru_cache(maxsize=1)
def _compiled_forward():
    return torch.compile(_forward, backend="hpu_backend", dynamic=None, options={"force_static_compile": True})


def forward(module, x, norm, residual=False):
    if comfy.ops.has_forward_hooks(module) or (norm is not None and comfy.ops.has_forward_hooks(norm)):
        output = module(x, norm=norm)
        return torch.add(x, output, out=x) if residual else output
    result = _compiled_forward()(module, x, norm, residual)
    return result[0] if residual else result
