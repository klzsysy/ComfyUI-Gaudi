import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import hpu_kv_attention, qwen35


@pytest.fixture
def hpu(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(args, "use_ck_attention", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def generator(device="hpu", dtype=torch.bfloat16):
    config = qwen35.Qwen35Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                num_attention_heads=2, num_key_value_heads=1, linear_num_key_heads=2,
                                linear_num_value_heads=4, layer_types=qwen35._qwen35_layer_types(4))
    result = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(result)
    result.model = qwen35.Qwen35Transformer(config, device=device, dtype=dtype, ops=comfy.ops.manual_cast)
    result.mtp = None
    for parameter in result.model.parameters():
        parameter.data.zero_()
    caches = result.init_kv_cache(1, 32, device, dtype)
    for cache in caches:
        cache.index = 17
    return result, caches


def test_generation_owns_cache_views_and_preserves_model_forward(hpu, monkeypatch):
    model, caches = generator()
    x = torch.ones(1, 1, 64, device="hpu", dtype=torch.bfloat16)
    observed = []
    original_calls = []

    def original(self, *values, **kwargs):
        original_calls.append(True)
        return kwargs["embeds"], None, kwargs["past_key_values"]

    def compiled(module, hidden, rotary, proxies):
        assert module is model.model
        assert all(cache.index == 1 for cache in proxies)
        assert type(proxies[0]) is qwen35.ImmediateLinearKV
        assert type(proxies[-1]) is qwen35.HPUSingleTokenKV
        observed.append(proxies)
        return hidden.add_(1)

    monkeypatch.setattr(qwen35.Qwen35Transformer, "forward", original)
    monkeypatch.setattr(qwen35, "_compiled_hpu_decode", lambda: compiled)
    with torch.no_grad():
        forward = model._generation_forward(x, caches)
        assert forward != model.model.forward
        assert "forward" not in model.model.__dict__
        for _ in range(2):
            output, _, returned = forward(None, embeds=x, past_key_values=caches)
            assert returned is caches
            torch.testing.assert_close(output.cpu(), torch.full_like(x.cpu(), 2), atol=0, rtol=0)
        assert observed[0] is observed[1]
        assert all(cache.index == 19 for cache in caches)
        assert observed[0][0].key is caches[0].key
        assert observed[0][0].value is caches[0].value
        assert observed[0][-1].position is caches[-1].position
        torch.testing.assert_close(x.cpu(), torch.ones_like(x.cpu()), atol=0, rtol=0)
        forward(None, embeds=x.expand(1, 3, 64), past_key_values=caches)
        assert original_calls == [True]
        second = model._generation_forward(x, caches)
        second(None, embeds=x, past_key_values=caches)
        assert observed[-1] is not observed[0]

        def fail(*values):
            raise RuntimeError("decode failed")

        monkeypatch.setattr(qwen35, "_compiled_hpu_decode", lambda: fail)
        with pytest.raises(RuntimeError, match="decode failed"):
            second(None, embeds=x, past_key_values=caches)
    assert "forward" not in model.model.__dict__


@pytest.mark.parametrize("mode", ["cpu", "fp32", "batch", "mtp", "disabled", "training", "grad",
                                  "hook", "patched", "lowvram", "vbar", "attention", "kitchen", "forward", "offloaded"])
def test_generation_retains_existing_routes(hpu, monkeypatch, mode):
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    model, caches = generator(device, dtype)
    x = torch.ones(2 if mode == "batch" else 1, 1, 64, device=device, dtype=dtype)
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(args, "use_ck_attention", mode == "kitchen")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    projection = model.model.layers[0].linear_attn.in_proj_qkv
    if mode == "mtp":
        model.mtp = torch.nn.Identity()
    if mode == "hook":
        projection.register_forward_hook(lambda *values: None)
    if mode == "patched":
        projection.weight_function = [lambda weight: weight]
    if mode == "lowvram":
        projection.weight_lowvram_function = object()
    if mode == "vbar":
        projection._v = object()
    if mode == "attention":
        model.model.layers[-1].self_attn.comfy_attention.method = "pytorch"
    if mode == "forward":
        model.model.forward = lambda *values, **kwargs: None
    if mode == "offloaded":
        projection.weight = torch.nn.Parameter(projection.weight.cpu())
    with torch.set_grad_enabled(mode == "grad"):
        assert model._generation_forward(x, caches) == model.model.forward


def test_single_decode_ignores_masked_tail(hpu):
    torch.manual_seed(82)
    q = torch.randn(1, 4, 1, 256, dtype=torch.bfloat16, device="hpu")
    key = torch.randn(1, 1, 32, 256, dtype=torch.bfloat16, device="hpu")
    value = torch.randn_like(key)
    length = torch.tensor([17], dtype=torch.int32, device="hpu")
    with torch.no_grad():
        output = hpu_kv_attention.single_decode(q, key, value, length).cpu()
        key[:, :, 17:].fill_(100)
        value[:, :, 17:].fill_(100)
        actual = hpu_kv_attention.single_decode(q, key, value, length).cpu()
    assert output.shape == (1, 1, 1024)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, output, atol=0, rtol=0)


def test_generation_checks_interruption_before_updating_state(hpu, monkeypatch):
    model, caches = generator()
    x = torch.ones(1, 1, 64, device="hpu", dtype=torch.bfloat16)

    def interrupted():
        raise RuntimeError("interrupted")

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", interrupted)
    with torch.no_grad():
        forward = model._generation_forward(x, caches)
        with pytest.raises(RuntimeError, match="interrupted"):
            forward(None, embeds=x, past_key_values=caches)
    assert all(cache.index == 17 for cache in caches)
