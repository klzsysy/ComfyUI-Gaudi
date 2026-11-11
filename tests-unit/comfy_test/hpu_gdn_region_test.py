import pytest
import torch
import comfy_kitchen

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import qwen35


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_layer(device="hpu", dtype=torch.bfloat16, operations=comfy.ops.manual_cast):
    config = qwen35.Qwen35Config(hidden_size=64, linear_num_key_heads=2, linear_num_value_heads=4)
    layer = qwen35.GatedDeltaNet(config, device=device, dtype=dtype, ops=operations)
    for name, parameter in layer.named_parameters():
        parameter.data.copy_(torch.randn(parameter.shape, dtype=dtype) * 0.03 + (1 if name == "norm.weight" else 0))
    return layer


def make_cache(layer, batch=1):
    device, dtype = layer.in_proj_a.weight.device, layer.in_proj_a.weight.dtype
    channels = layer.key_dim * 2 + layer.value_dim
    cache = qwen35.LinearKV(torch.zeros(batch, channels, 3, device=device, dtype=dtype),
                           torch.zeros(batch, layer.num_value_heads, layer.key_head_dim, layer.value_head_dim,
                                       device=device, dtype=torch.float32), 17, None, None)
    cache.qkv_buf, cache.proj_buf, cache.gates_buf, cache.sumsq_buf = comfy_kitchen.gated_delta_deferred_buffers(
        batch, channels, layer.num_value_heads, layer.num_key_heads, dtype, device)
    for tensor in (cache.qkv_buf, cache.proj_buf, cache.gates_buf, cache.sumsq_buf):
        tensor.zero_()
    cache.ctl_tracker = qwen35.LinearKV.deferred_tracker(device)
    cache.ctl = cache.ctl_tracker["ctl"]
    cache.ctl_tracker["tree"] = qwen35.VerifyTree(qwen35.MTP_MODES["tree2s2"][1], 8, device)
    cache.g_decay = -layer.A_log.float().exp()
    cache.dt_bias = layer.dt_bias.float()
    cache.norm_weight = layer.norm.weight
    return cache


@pytest.mark.parametrize("batch", [1, 2])
def test_region_preserves_outputs_state_and_updated_weights(hpu_compiler, batch):
    torch.manual_seed(192)
    layer = make_layer()
    with torch.no_grad():
        eager, compiled = make_cache(layer, batch), make_cache(layer, batch)
        for turn in range(3):
            if turn == 1:
                layer.in_proj_z.weight.mul_(0.75)
                layer.out_proj.weight.mul_(0.75)
            for cache in (eager, compiled):
                cache.prepare(8)
            x = torch.randn(batch, 8, 64, dtype=torch.bfloat16, device="hpu")
            before = x.cpu().clone()
            expected, _ = layer._forward(x, past_key_value=eager)
            actual, state = layer(x, past_key_value=compiled)
            assert state is compiled
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.015, atol=0.005)
            torch.testing.assert_close(compiled.recurrent_state.cpu(), eager.recurrent_state.cpu(), rtol=0.015, atol=0.001)
            torch.testing.assert_close(compiled.conv_state.cpu(), eager.conv_state.cpu(), rtol=0.015, atol=0.005)
            torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)
            for cache in (eager, compiled):
                base = cache.index
                cache.advance(8)
                cache.rollback(6)
                cache.commit_path(base, [0, 5])


@pytest.mark.parametrize("target", ["module", "projection", "output", "global"])
def test_region_honors_hooks_added_after_compilation(hpu_compiler, monkeypatch, target):
    layer = make_layer()
    calls = []
    with torch.no_grad():
        cache = make_cache(layer)
        cache.prepare(8)
        x = torch.randn(1, 8, 64, dtype=torch.bfloat16, device="hpu")
        layer(x, past_key_value=cache)
        monkeypatch.setattr(qwen35, "_compiled_hpu_delta", lambda: pytest.fail("Hooked region compiled"))
        watched = layer if target == "module" else layer.in_proj_z if target == "projection" else layer.out_proj
        def hook(module, values, output):
            if module is watched:
                calls.append(True)
        handle = torch.nn.modules.module.register_module_forward_hook(hook) if target == "global" else watched.register_forward_hook(hook)
        try:
            cache = make_cache(layer)
            cache.prepare(8)
            output, _ = layer(x, past_key_value=cache)
            assert torch.isfinite(output).all() and calls == [True]
        finally:
            handle.remove()


@pytest.mark.parametrize("mode", ["cpu", "fp32", "four_rows", "single_row", "training", "grad", "disabled", "custom_ops", "no_cache", "prefill"])
def test_region_preserves_eager_entries(monkeypatch, mode):
    if mode != "cpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("HPU required")
    class CustomOps(comfy.ops.manual_cast):
        pass
    monkeypatch.setattr(qwen35, "_compiled_hpu_delta", lambda: pytest.fail("Unexpected region compilation"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    layer = make_layer(device, dtype, CustomOps if mode == "custom_ops" else comfy.ops.manual_cast)
    cache = None if mode == "no_cache" else make_cache(layer)
    if mode == "prefill":
        cache.index = 0
    monkeypatch.setattr(layer, "_forward", lambda x, past_key_value=None, **kwargs: (x, past_key_value))
    rows = 4 if mode == "four_rows" else 1 if mode == "single_row" else 8
    x = torch.ones(1, rows, 64, device=device, dtype=dtype)
    with torch.set_grad_enabled(mode == "grad"):
        output, state = layer(x, past_key_value=cache)
    assert output is x and state is cache
