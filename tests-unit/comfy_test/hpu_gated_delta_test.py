import pytest
import torch
import comfy_kitchen

from comfy.text_encoders import hpu_gated_delta


@pytest.mark.parametrize("batch,rows", [(1, 1), (1, 3), (1, 8), (2, 8)])
def test_compiled_deferred_state_reuse(batch, rows):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("Requires an HPU")
    torch.manual_seed(41)
    device = "hpu"
    dtype = torch.bfloat16
    heads, key_heads, dim, hidden = 4, 2, 128, 256
    channels = (2 * key_heads + heads) * dim

    def rand(*shape, dt=dtype):
        return (torch.randn(*shape) * 0.1).to(device=device, dtype=dt)

    conv_weight = rand(channels, 1, 4)
    bias = rand(channels) if batch == 2 else None
    w_a, w_b = rand(heads, hidden), rand(heads, hidden)
    dt_bias = rand(heads, dt=torch.float32)
    decay = -torch.rand(heads, device=device)
    norm = torch.ones(dim, device=device, dtype=dtype)
    initial_conv = rand(batch, channels, 3)
    initial_state = rand(batch, heads, dim, dim, dt=torch.float32)

    def buffers():
        qkv, proj, gates, sumsq = comfy_kitchen.gated_delta_deferred_buffers(batch, channels, heads, key_heads, dtype, torch.device(device))
        for tensor in (qkv, proj, gates, sumsq):
            tensor.zero_()
        return initial_conv.clone(), initial_state.clone(), qkv, proj, gates, sumsq

    reference, candidate = buffers(), buffers()
    compiled_conv, compiled_delta = hpu_gated_delta.compiled_ops()
    # Eight rows include siblings; changing pending/slots exercises rejected
    # drafts, off-chain commits and alternating side buffers without recompiling.
    parents = [-1, 0, 1, 1, 1, 0, 0, -1] if rows == 8 else list(range(-1, rows - 1))
    program = [32, 5, 6, 33, 3, 4, 2, 7] if rows == 8 else [i | 32 for i in range(rows)]
    paths = [[], [0], [0, 5] if rows == 8 else list(range(rows)), [], [0]]
    for turn, path in enumerate(paths):
        # New weights with the same shapes must not reuse captured parameter values.
        if turn == 2:
            w_a, w_b = rand(heads, hidden), rand(heads, hidden)
            conv_weight = rand(channels, 1, 4)
        ctl = torch.tensor([len(path), turn % 2] + path + [0] * (8 - len(path)) + parents + [-1] * (8 - rows) + program + [0] * (8 - rows), device=device, dtype=torch.int32)
        projected = rand(batch, rows, channels)
        x, z = rand(batch, rows, hidden), rand(batch, rows, heads * dim)
        outputs = []
        for state, conv, delta in [(reference, comfy_kitchen.deltanet_conv_step_deferred, comfy_kitchen.gated_delta_decode_deferred), (candidate, compiled_conv, compiled_delta)]:
            cs, rs, qkv, proj, gates, sumsq = state
            conv(projected, cs, conv_weight, bias, proj, qkv, ctl)
            outputs.append(delta(x, w_a, w_b, dt_bias, decay, rs, key_heads * dim, key_heads, dim ** -0.5, z, norm, 1e-6, qkv, gates, sumsq, ctl))
        torch.hpu.synchronize()
        assert torch.isfinite(outputs[1]).all()
        torch.testing.assert_close(outputs[1].cpu(), outputs[0].cpu(), rtol=2e-2, atol=4e-3)
        torch.testing.assert_close(candidate[0].cpu(), reference[0].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(candidate[1].cpu(), reference[1].cpu(), rtol=5e-3, atol=5e-4)


def test_cpu_decode_does_not_compile_hpu(monkeypatch):
    from comfy.text_encoders.qwen35 import GatedDeltaNet, LinearKV, Qwen35Config
    from comfy import ops
    from comfy.cli_args import args

    monkeypatch.setattr(hpu_gated_delta, "compiled_ops", lambda: pytest.fail("HPU compiler used for CPU"))
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    config = Qwen35Config(hidden_size=32, linear_num_key_heads=1, linear_num_value_heads=1)
    layer = GatedDeltaNet(config, device="cpu", dtype=torch.float32, ops=ops.manual_cast)
    for param in layer.parameters():
        param.data.fill_(0.01)
    channels = layer.key_dim * 2 + layer.value_dim
    kv = LinearKV(torch.zeros(1, channels, 3), torch.zeros(1, 1, 128, 128), 1, None, None)
    kv.qkv_buf, kv.proj_buf, kv.gates_buf, kv.sumsq_buf = comfy_kitchen.gated_delta_deferred_buffers(1, channels, 1, 1, torch.float32, torch.device("cpu"))
    for value in (kv.qkv_buf, kv.proj_buf, kv.gates_buf, kv.sumsq_buf):
        value.zero_()
    tracker = LinearKV.deferred_tracker(torch.device("cpu"))
    kv.ctl, kv.ctl_tracker = tracker["ctl"], tracker
    kv.dt_bias = layer.dt_bias
    kv.g_decay = -layer.A_log.exp()
    kv.norm_weight = layer.norm.scale()
    kv.prepare(1)
    result, _ = layer(torch.ones(1, 1, 32), past_key_value=kv)
    assert result.shape == (1, 1, 32)
    assert torch.isfinite(result).all()
