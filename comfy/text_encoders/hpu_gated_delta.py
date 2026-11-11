"""Regional compilation of DeltaNet tensor operations on HPU."""
from functools import lru_cache

import torch
import torch.nn.functional as F
import comfy_kitchen


@lru_cache(maxsize=1)
def compiled_prefill(function):
    # The chunk recurrence has length-dependent Python loops. Keep shapes static;
    # Habana 1.24.2 cannot lower the resulting symbolic slice bounds reliably.
    compiled = torch.compile(function, backend="hpu_backend", dynamic=False,
                             options={"force_static_compile": True})

    def prefill(query, key, value, g, beta, initial_state=None, output_final_state=False):
        length = query.shape[1]
        # Zero beta/value and zero log-decay make added chunks state-preserving.
        # Bucket prompt lengths so ordinary prompt edits reuse static recipes.
        padded_length = max(64, 1 << (length - 1).bit_length())
        padding = padded_length - length
        if padding:
            query, key, value = (F.pad(x, (0, 0, 0, 0, 0, padding)) for x in (query, key, value))
            g, beta = (F.pad(x, (0, 0, 0, padding)) for x in (g, beta))
        output, state = compiled(query, key, value, g=g, beta=beta, initial_state=initial_state, output_final_state=True)
        return output[:, :length].contiguous(), state if output_final_state else None

    return prefill


@lru_cache(maxsize=1)
def compiled_ops():
    # Compile tensor operations only; cache contents and rollback stay with the caller.
    return (
        torch.compile(comfy_kitchen.deltanet_conv_step_deferred, backend="hpu_backend", fullgraph=True, dynamic=False),
        torch.compile(comfy_kitchen.gated_delta_decode_deferred, backend="hpu_backend", fullgraph=True, dynamic=False),
    )
