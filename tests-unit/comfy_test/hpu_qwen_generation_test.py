import json

import pytest
import torch
from types import SimpleNamespace
from torch._dynamo.backends.registry import lookup_backend

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import hpu_kv_attention, qwen35


def test_decode_preserves_optional_final_norm():
    x = torch.ones(1, 1, 64)
    model = SimpleNamespace(layers=[], norm=None)
    assert qwen35._hpu_decode(model, x, None, []) is x


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


def generator(device="hpu", dtype=torch.bfloat16, quantized=False, lm_head=False, weight_seed=None):
    config = qwen35.Qwen35Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                num_attention_heads=2, num_key_value_heads=1, linear_num_key_heads=2,
                                linear_num_value_heads=4, layer_types=qwen35._qwen35_layer_types(4), lm_head=lm_head)
    result = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(result)
    result.model = qwen35.Qwen35Transformer(config, device=device, dtype=dtype, ops=comfy.ops.manual_cast)
    result.mtp = None
    weight_rng = torch.Generator().manual_seed(weight_seed) if weight_seed is not None else None
    for parameter in result.model.parameters():
        if weight_rng is None:
            parameter.data.zero_()
        else:
            parameter.data.copy_(torch.randn(parameter.shape, generator=weight_rng) * .03)
    if quantized:
        state = result.model.state_dict()
        for name, module in result.model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.endswith(('in_proj_a', 'in_proj_b')):
                state[name + '.weight'] = state[name + '.weight'].cpu().to(torch.float8_e4m3fn)
                state[name + '.weight_scale'] = torch.ones(())
                state[name + '.comfy_quant'] = torch.tensor(list(json.dumps({'format': 'float8_e4m3fn'}).encode()), dtype=torch.uint8)
        operations = comfy.ops.mixed_precision_ops({'formats': ['float8_e4m3fn']}, dtype, load_device=torch.device('hpu'))
        result.model = qwen35.Qwen35Transformer(config, device='cpu', dtype=dtype, ops=operations)
        result.model.load_state_dict(state)
        result.model.to(device)
    caches = result.init_kv_cache(1, 32, device, dtype)
    for cache in caches:
        cache.index = 17
    return result, caches


@pytest.mark.parametrize("quantized", [False, True])
def test_generation_owns_cache_views_and_preserves_model_forward(hpu, monkeypatch, quantized):
    model, caches = generator(quantized=quantized)
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
        assert model._generation_readback_size(x, caches) == 4
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
@pytest.mark.parametrize("quantized", [False, True])
def test_generation_retains_existing_routes(hpu, monkeypatch, mode, quantized):
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    model, caches = generator(device, dtype, quantized)
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
        assert model._generation_readback_size(x, caches) == 1


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


@pytest.mark.parametrize("lm_head", [False, True])
def test_native_fp8_decoder_executes_and_advances(hpu, lm_head):
    model, caches = generator(quantized=True, lm_head=lm_head)
    x = torch.ones(1, 1, 64, device="hpu", dtype=torch.bfloat16)
    for cache in caches:
        cache.key.zero_()
        cache.value.zero_()
        if type(cache) is qwen35.LinearKV:
            cache.g_decay = torch.full((4,), -1., device="hpu")
            cache.dt_bias = torch.zeros(4, device="hpu")
            cache.norm_weight = torch.ones(128, device="hpu", dtype=torch.bfloat16)
    with torch.no_grad():
        forward = model._generation_forward(x, caches)
        assert forward != model.model.forward
        for index in (18, 19):
            output, _, returned = forward(None, embeds=x, past_key_values=caches)
            assert returned is caches
            assert output.shape == x.shape and output.dtype == x.dtype
            assert output.isfinite().all()
            assert all(cache.index == index for cache in caches)


def test_generation_precomputes_rope_and_skips_deferred_control(hpu, monkeypatch):
    model, caches = generator(quantized=True, lm_head=True)
    x = torch.ones(1, 1, 64, device='hpu', dtype=torch.bfloat16)
    matrices = []
    proxies_seen = []
    original = model.model.compute_freqs_cis
    positions_seen = []

    def frequencies(self, positions, device):
        positions_seen.append(positions.cpu())
        return original(positions, device)

    def compiled(module, hidden, rotary, proxies):
        matrices.append(rotary.cpu())
        proxies_seen.append(proxies)
        return hidden

    def unexpected_prepare(*values):
        raise AssertionError('Deferred control should not be used for immediate state')

    monkeypatch.setattr(qwen35.Qwen35Transformer, 'compute_freqs_cis', frequencies)
    monkeypatch.setattr(qwen35, '_QWEN35_FREQS', frequencies)
    monkeypatch.setattr(qwen35.LinearKV, 'prepare', unexpected_prepare)
    monkeypatch.setattr(qwen35, '_compiled_hpu_decode', lambda: compiled)
    with torch.no_grad():
        forward = model._generation_forward(x, caches)
        forward(None, embeds=x, past_key_values=caches)
        forward(None, embeds=x, past_key_values=caches)
        assert len(positions_seen) == 1 and positions_seen[0].shape[-1] == 32
        assert all(matrix.stride() == torch.empty(matrix.shape).stride() for matrix in matrices)
        for matrix, position in zip(matrices, (17, 18)):
            expected = qwen35.rope_matrix(original(torch.tensor([[position]], device='hpu'), x.device))
            torch.testing.assert_close(matrix, expected.cpu(), atol=0, rtol=0)
        second = model._generation_forward(x, caches)
        second(None, embeds=x, past_key_values=caches)
        assert proxies_seen[0] is not proxies_seen[-1]
        explicit = torch.tensor([[7]], device='hpu')
        second(None, embeds=x, past_key_values=caches, position_ids=explicit)
        expected = qwen35.rope_matrix(original(explicit, x.device))
        torch.testing.assert_close(matrices[-1], expected.cpu(), atol=0, rtol=0)
        custom = torch.randn_like(expected)
        lengths = []

        def custom_frequencies(positions, device):
            lengths.append(positions.shape[-1])
            return custom

        monkeypatch.setattr(model.model, 'compute_freqs_cis', custom_frequencies)
        third = model._generation_forward(x, caches)
        third(None, embeds=x, past_key_values=caches)
        assert lengths == [1]
        assert not proxies_seen[-1][-1].native_rope
        torch.testing.assert_close(matrices[-1], custom.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize('quantized', [False, True])
def test_generation_reuses_decoder_across_kv_capacities(hpu, monkeypatch, quantized):
    model, _ = generator(quantized=quantized, lm_head=True)
    backend = lookup_backend('hpu_backend')
    compile_count = 0

    def counted_backend(graph, inputs, **kwargs):
        nonlocal compile_count
        compile_count += 1
        return backend(graph, inputs, **kwargs)

    compile_function = torch.compile

    def counted_compile(function, **kwargs):
        if function is qwen35._hpu_decode:
            kwargs['backend'] = counted_backend
        return compile_function(function, **kwargs)

    qwen35._compiled_hpu_decode.cache_clear()

    monkeypatch.setattr(torch, 'compile', counted_compile)
    x = torch.ones(1, 1, 64, device='hpu', dtype=torch.bfloat16)
    with torch.no_grad():
        for capacity in (32, 33, 127, 256, 257, 6897, 6899, 6900, 6901, 8192, 8193, 32):
            caches = model.init_kv_cache(1, capacity, 'hpu', torch.bfloat16)
            for cache in caches:
                cache.index = 17
                if type(cache) is qwen35.LinearKV:
                    cache.g_decay = torch.full((4,), -1., device='hpu')
                    cache.dt_bias = torch.zeros(4, device='hpu')
                    cache.norm_weight = torch.ones(128, device='hpu', dtype=torch.bfloat16)
            forward = model._generation_forward(x, caches)
            for step in (18, 19):
                output, _, returned = forward(None, embeds=x, past_key_values=caches)
                assert returned is caches
                assert all(cache.index == step for cache in caches)
                torch.testing.assert_close(output.cpu(), x.cpu(), atol=.01, rtol=.01)
            assert compile_count == 1
            assert caches[-1].key.shape[2] == capacity
            assert caches[-1].key._dynamo_dynamic_indices == {2}
    qwen35._compiled_hpu_decode.cache_clear()


@pytest.mark.parametrize('quantized', [False, True])
def test_generation_isolates_same_length_prompts_and_recomputed_requests(hpu, monkeypatch, quantized):
    model, _ = generator(quantized=quantized, lm_head=True, weight_seed=82)
    initial_cache = model.init_kv_cache
    logits = model.logits
    requests = []
    scores = []

    def fresh_cache(*values):
        caches = initial_cache(*values)
        for cache in caches:
            assert cache.index == 0
            assert torch.count_nonzero(cache.key.cpu()) == 0
            assert torch.count_nonzero(cache.value.cpu()) == 0
        for previous in requests:
            assert all(current.key.data_ptr() != old.key.data_ptr() for current, old in zip(caches, previous))
            assert all(current.value.data_ptr() != old.value.data_ptr() for current, old in zip(caches, previous))
        requests.append(caches)
        return caches

    def capture_logits(hidden):
        result = logits(hidden)
        scores.append(result[:, -1].cpu().clone())
        return result

    monkeypatch.setattr(model, 'init_kv_cache', fresh_cache)
    monkeypatch.setattr(model, 'logits', capture_logits)
    prompt_a = torch.arange(1, 13, device='hpu').unsqueeze(0)
    prompt_b = prompt_a.clone()
    prompt_b[:, -1] = 24

    def generate(prompt):
        scores.clear()
        ids = model.generate(embeds=model.model.embed_tokens(prompt), max_length=8, stop_tokens=[],
                             execution_dtype=torch.bfloat16, seed=17, top_k=5, presence_penalty=.5)
        assert len(ids) == len(scores) == 8
        return ids, torch.stack(scores)

    with torch.no_grad():
        expected_a = generate(prompt_a)
        expected_b = generate(prompt_b)
        assert not torch.equal(expected_a[1][0], expected_b[1][0])
        assert any(torch.count_nonzero(cache.key.cpu()) > 0 for cache in requests[0])
        assert any(torch.count_nonzero(cache.value.cpu()) > 0 for cache in requests[0])
        actual_a = generate(prompt_a)
        assert actual_a[0] == expected_a[0]
        torch.testing.assert_close(actual_a[1], expected_a[1], atol=0, rtol=0)

        # A fresh compile after B checks the reused graph against a new trace.
        torch._dynamo.reset()
        qwen35._compiled_hpu_decode.cache_clear()
        actual_b = generate(prompt_b)
        actual_a = generate(prompt_a)
        assert actual_b[0] == expected_b[0]
        assert actual_a[0] == expected_a[0]
        torch.testing.assert_close(actual_b[1], expected_b[1], atol=0, rtol=0)
        torch.testing.assert_close(actual_a[1], expected_a[1], atol=0, rtol=0)
    assert len(requests) == 5
    qwen35._compiled_hpu_decode.cache_clear()
