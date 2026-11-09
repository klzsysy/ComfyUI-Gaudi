"""Regional compilation of Kitchen's deferred DeltaNet operations on HPU."""
from functools import lru_cache

import torch
import comfy_kitchen


@lru_cache(maxsize=1)
def compiled_ops():
    # Compile tensor operations only; cache contents and rollback stay with the caller.
    return (
        torch.compile(comfy_kitchen.deltanet_conv_step_deferred, backend="hpu_backend", fullgraph=True, dynamic=False),
        torch.compile(comfy_kitchen.gated_delta_decode_deferred, backend="hpu_backend", fullgraph=True, dynamic=False),
    )
