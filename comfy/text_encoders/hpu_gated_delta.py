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
    compiled_delta = torch.compile(_decode_deferred, backend="hpu_backend", fullgraph=True, dynamic=False)

    def decode(x, *values, **kwargs):
        # Habana 1.24.2 can reject a four-row RMSNorm output reused by this
        # region. Materialize the activation before entering the compiled graph.
        if x.shape[1] == 4:
            x = x.clone()
        return compiled_delta(x, *values, **kwargs)

    return (
        torch.compile(comfy_kitchen.deltanet_conv_step_deferred, backend="hpu_backend", fullgraph=True, dynamic=False),
        decode,
    )


def _path_weights(decay, parents):
    rows = decay.shape[-1]
    identity = torch.eye(rows, dtype=decay.dtype, device=decay.device)
    positions = torch.arange(rows, device=parents.device)
    transition = decay.unsqueeze(-1) * (parents.unsqueeze(-1) == positions).to(decay.dtype)
    paths, power = identity, transition
    # A tree's transition is nilpotent. Doubling sums I + T + ... + T**(rows-1),
    # including zero-decay paths without division or logarithms.
    for _ in range((rows - 1).bit_length()):
        paths = paths + power @ paths
        power = power @ power
    initial = (paths @ (decay * (parents < 0)).unsqueeze(-1)).squeeze(-1)
    return paths, initial, identity


def _delta_updates(state, key, value, decay, beta, parents):
    keys = key.transpose(-1, -2)
    values = value.transpose(-1, -2)
    decay, beta = decay.transpose(1, 2), beta.transpose(1, 2)
    paths, initial, identity = _path_weights(decay, parents)
    dependencies = (keys @ key) * paths * beta.unsqueeze(-1) * (1 - identity)
    updates = beta.unsqueeze(-1) * (values - initial.unsqueeze(-1) * (keys @ state))
    # The delta corrections solve (I + dependencies) U = updates. This second
    # finite series removes the per-token dependency through the large state.
    power = -dependencies
    for _ in range((keys.shape[-2] - 1).bit_length()):
        updates = updates + power @ updates
        power = power @ power
    return updates, paths, initial


def _commit(state, key, value, decay, beta):
    parents = torch.arange(key.shape[-1], device=key.device) - 1
    updates, paths, initial = _delta_updates(state, key, value, decay, beta, parents)
    return state * initial[..., -1, None, None] + (key * paths[..., -1, None, :]) @ updates


def _tree_output(state, query, key, value, decay, beta, program):
    rows = program & 15
    committed = (program & 32) != 0
    positions = torch.arange(program.shape[0], device=program.device)
    previous = torch.where((positions.unsqueeze(0) < positions.unsqueeze(1)) & committed.unsqueeze(0), positions, -1).amax(-1)
    parent_order = torch.where(previous >= 0, rows.index_select(0, previous.clamp(min=0)), -1)
    parents = torch.empty_like(rows).scatter(0, rows, parent_order)
    updates, paths, initial = _delta_updates(state, key, value, decay, beta, parents)
    queries = query.transpose(-1, -2)
    output = initial.unsqueeze(-1) * (queries @ state) + ((queries @ key) * paths) @ updates
    return output.transpose(1, 2)


def _slab(buffer, side):
    return buffer.index_select(0, side.view(1)).squeeze(0)


def _decode_deferred(x, w_a, w_b, dt_bias, g_decay, state, key_dim, num_key_heads, scale,
                     z, norm_weight, eps, qkv_buf, gates_buf, sumsq_buf, ctl):
    # Kitchen has no native HPU deferred kernel. Keep its buffer and rounding
    # contract, with parallel FP32 state updates for the short verification tree.
    batch, seq, _ = x.shape
    heads, key_dim_head, value_dim = state.shape[1:]
    groups = heads // num_key_heads
    slots_max = comfy_kitchen.gated_delta_slot_max
    ctl = ctl.long()
    pending, parity = ctl[0], ctl[1]
    slots = ctl[2:2 + slots_max]
    program = ctl[2 + 2 * slots_max:2 + 3 * slots_max]

    beta = F.linear(x, w_b).sigmoid().float()
    decay = torch.exp(g_decay * F.softplus(F.linear(x, w_a).float() + dt_bias))
    current = _slab(qkv_buf, parity)[:, :, :seq]
    query = current[:, :key_dim].reshape(batch, num_key_heads, key_dim_head, seq).float()
    key = current[:, key_dim:2 * key_dim].reshape(batch, num_key_heads, key_dim_head, seq).float()
    value = current[:, 2 * key_dim:].reshape(batch, heads, value_dim, seq).float()
    sumsq = torch.stack([query.square().sum(2), key.square().sum(2)], dim=-1)
    gates_buf.index_copy_(0, parity.view(1), F.pad(
        torch.stack([decay, beta], dim=-1), (0, 0, 0, 0, 0, slots_max - seq)).unsqueeze(0))
    sumsq_buf.index_copy_(0, parity.view(1), F.pad(
        sumsq.transpose(1, 2), (0, 0, 0, 0, 0, slots_max - seq)).unsqueeze(0))

    live = pending > torch.arange(slots_max, device=ctl.device)
    previous = torch.where(live.view(1, 1, slots_max), _slab(qkv_buf, 1 - parity).index_select(2, slots), 0.0)
    previous_key = previous[:, key_dim:2 * key_dim].reshape(batch, num_key_heads, key_dim_head, slots_max).float()
    previous_value = previous[:, 2 * key_dim:].reshape(batch, heads, value_dim, slots_max).float()
    previous_gates = _slab(gates_buf, 1 - parity).index_select(1, slots)
    previous_sumsq = torch.where(live.view(1, 1, slots_max, 1),
                                _slab(sumsq_buf, 1 - parity).index_select(1, slots).transpose(1, 2), 1.0)
    previous_decay = torch.where(live.view(1, slots_max, 1), previous_gates[..., 0], 1.0)
    previous_beta = torch.where(live.view(1, slots_max, 1), previous_gates[..., 1], 0.0)
    previous_key = (previous_key / previous_sumsq[..., 1].sqrt().clamp(min=1e-12).unsqueeze(2)).repeat_interleave(groups, dim=1)
    committed = _commit(state, previous_key, previous_value, previous_decay, previous_beta)
    state.copy_(committed)

    norms = sumsq.sqrt().clamp(min=1e-12).unsqueeze(2)
    query = (query / norms[..., 0] * scale).repeat_interleave(groups, dim=1)
    key = (key / norms[..., 1]).repeat_interleave(groups, dim=1)
    output = _tree_output(committed, query, key, value, decay, beta, program[:seq])
    output = output.to(x.dtype).float()
    rstd = torch.rsqrt(output.square().mean(-1, keepdim=True) + eps)
    normalized = (output * rstd * norm_weight.float()).to(x.dtype)
    return normalized * F.silu(z.reshape(batch, seq, heads, value_dim))
