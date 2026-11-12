import pytest
import torch

import comfy.ops
import comfy.model_management
from comfy.cli_args import args
from comfy.ldm.moss_image import model as moss
from comfy.ldm.modules.attention import attention_pytorch
from comfy.ldm.modules import attention as attention_module


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    # CPU-only test modules can select the global sub-quadratic backend during
    # collection. Exercise the PyTorch backend used by the HPU launch path.
    monkeypatch.setattr(moss, "optimized_attention", attention_pytorch)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def attention_inputs(length, device="hpu", dtype=torch.bfloat16):
    heads, dim = 4, 32
    q, k, v = torch.randn(2, length, 3 * heads * dim, dtype=dtype).to(device).chunk(3, dim=-1)
    norms = [comfy.ops.manual_cast.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device) for _ in range(2)]
    for norm in norms:
        norm.weight.data.copy_(torch.randn(dim, dtype=dtype).to(device) * 0.1 + 1)
    angle = torch.randn(1, length, 1, dim // 2)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, length, 1, dim // 2, 2, 2).to(device)
    return (q, k, v, *norms, pe, heads, moss.ComfyAttention(), {})


@pytest.mark.parametrize("mode", ["plain", "mask", "references"])
def test_moss_attention_dynamic_inputs_and_reference_cache(hpu_compiler, mode):
    torch.manual_seed(25)
    lengths = (31, 37, 43, 53, 61, 67, 73, 83, 89, 97, 31) if mode == "plain" else (17, 43, 17)
    with torch.no_grad():
        for length in lengths:
            values = attention_inputs(length)
            extra = {}
            if mode == "mask":
                extra['mask'] = torch.randn(2, 1, 1, length).to(device="hpu", dtype=torch.bfloat16) * 0.2
            elif mode == "references":
                extra['ref_kv'] = torch.randn(2, 2, 5, 128, dtype=torch.bfloat16).to("hpu")
                extra['kv_out'] = torch.empty(2, 2, length, 128, device="hpu", dtype=torch.bfloat16)
            expected = moss._attention(*values, **extra)
            expected_kv = extra['kv_out'].clone() if mode == "references" else None
            actual = moss.attention(*values, **extra)
            assert actual.device.type == "hpu" and actual.dtype == torch.bfloat16
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.01)
            if expected_kv is not None:
                torch.testing.assert_close(extra['kv_out'].cpu(), expected_kv.cpu(), atol=0, rtol=0)


def test_moss_cpu_attention_does_not_compile_hpu(monkeypatch):
    monkeypatch.setattr(moss, "_compiled_attention", lambda: pytest.fail("HPU compiler used for CPU"))
    values = attention_inputs(17, "cpu", torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(moss.attention(*values), moss._attention(*values))


@pytest.mark.parametrize("mode", ["disabled", "training", "override"])
def test_moss_attention_preserves_eager_paths(hpu_compiler, monkeypatch, mode):
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    monkeypatch.setattr(moss, "_compiled_attention", lambda: pytest.fail("compiler should be bypassed"))
    values = attention_inputs(17)
    calls = []
    if mode == "override":
        def override(function, *inputs, **kwargs):
            calls.append(True)
            return function(*inputs, **kwargs)
        values[-1]['optimized_attention_override'] = override
    with torch.no_grad():
        result = moss.attention(*values)
    assert torch.isfinite(result).all()
    if mode == "override":
        assert calls == [True]


def test_head_attention_preserves_order_and_dispatch(monkeypatch):
    q, k, v = torch.randn(2, 7, 3 * 128).chunk(3, dim=-1)
    preferred = moss.ComfyAttention()
    options = {"block_index": 3}
    calls = []

    def attention(query, key, value, heads, transformer_options, preferred_attention):
        calls.append((query.take(), key.take(), value.take()))
        assert heads == 1 and transformer_options is options and preferred_attention is preferred
        return calls[-1][2]

    monkeypatch.setattr(moss, "optimized_attention", attention)
    actual = moss._attention_heads(q, k, v, 4, preferred, options)
    torch.testing.assert_close(actual, v, atol=0, rtol=0)
    for index, (query, key, value) in enumerate(calls):
        for actual, original in zip((query, key, value), (q, k, v)):
            torch.testing.assert_close(actual, original[..., index * 32:(index + 1) * 32], atol=0, rtol=0)


@pytest.mark.parametrize("length", [4119, 4403])
def test_compiled_head_attention_matches_together(hpu_compiler, length):
    torch.manual_seed(62)
    q, k, v = torch.randn(1, length, 3 * 3072, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    preferred = moss.ComfyAttention()

    def together(q, k, v):
        tensors = (moss.AttentionTensorContainer(tensor) for tensor in (q, k, v))
        return moss.optimized_attention(*tensors, 24, transformer_options={}, preferred_attention=preferred)

    compiled = torch.compile(moss._attention_heads, backend="hpu_backend", dynamic=False, options={"force_static_compile": True})
    reference = torch.compile(together, backend="hpu_backend", dynamic=False, options={"force_static_compile": True})
    with torch.no_grad():
        expected = reference(q, k, v).cpu()
        actual = compiled(q, k, v, 24, preferred, {}).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("length", [4119, 4403])
def test_fp8_av_uses_native_path_through_attention_dispatch(hpu_compiler, monkeypatch, length):
    torch.manual_seed(72)
    q, k, v = torch.randn(1, length, 3 * 128, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    preferred = moss.ComfyAttention()

    def dispatched(q, k, v):
        tensors = (moss.AttentionTensorContainer(tensor) for tensor in (q, k, v))
        return moss.optimized_attention(*tensors, 1, transformer_options={"fp8_attention": True}, preferred_attention=preferred)

    with torch.no_grad():
        expected = attention_pytorch(q, k, v, 1).float()
        monkeypatch.setattr(comfy.ops, "scaled_dot_product_attention", lambda *a, **kw: pytest.fail("FP8 AV path was bypassed"))
        actual = torch.compile(dispatched, backend="hpu_backend", dynamic=False)(q, k, v)
        relative_error = (actual.float() - expected).norm() / expected.norm()
        assert actual.device.type == "hpu" and actual.dtype == torch.bfloat16
        # Gaudi2 fused FP8 softmax has additional error on diffuse probabilities.
        assert torch.isfinite(actual).all() and relative_error < 0.08
        scale = q.shape[-1] ** -0.25
        scores = (q[0] * scale) @ (k[0] * scale).T
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0).float().cpu() / 128
        values = v[0].float().cpu()
        value_scale = values.abs().amax() / 240
        values = (values / value_scale).to(torch.float8_e4m3fn).float() * value_scale
        reference = probabilities @ values
        assert (actual[0].float().cpu() - reference).norm() / reference.norm() < 0.005


@pytest.mark.parametrize("mode", ["disabled", "compiler-disabled", "precision", "mask", "scale", "short", "grad"])
def test_fp8_av_preserves_attention_fallbacks(hpu_compiler, monkeypatch, mode):
    length = 31 if mode == "short" else 4096
    q, k, v = torch.randn(1, length, 3 * 128, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    options = {"fp8_attention": mode != "disabled"}
    extra = {}
    if mode == "precision":
        extra["attn_precision"] = torch.float32
    if mode == "mask":
        extra["mask"] = torch.zeros(1, 1, 1, length, dtype=q.dtype, device=q.device)
    if mode == "scale":
        extra["scale"] = 0.125
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "compiler-disabled")
    monkeypatch.setattr(attention_module, "_hpu_fp8_attention_av", lambda *a: pytest.fail("FP8 AV path should be bypassed"))

    def dispatched(q, k, v):
        return attention_pytorch(q, k, v, 1, transformer_options=options, **extra)

    with torch.set_grad_enabled(mode == "grad"):
        expected = attention_pytorch(q, k, v, 1, **extra)
        actual = torch.compile(dispatched, backend="hpu_backend", dynamic=False)(q, k, v)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.01)


def test_fp8_av_zero_values_and_cpu_dispatch(hpu_compiler, monkeypatch):
    q, k, v = torch.randn(1, 1, 4096, 3 * 128, dtype=torch.bfloat16).to("hpu").chunk(3, dim=-1)
    with torch.no_grad():
        output = torch.compile(attention_module._hpu_fp8_attention_av, backend="hpu_backend", dynamic=False)(q, k, torch.zeros_like(v))
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    monkeypatch.setattr(attention_module, "_hpu_fp8_attention_av", lambda *a: pytest.fail("HPU function called for CPU"))
    q, k, v = torch.randn(1, 17, 3 * 128).chunk(3, dim=-1)
    expected = attention_pytorch(q, k, v, 1, transformer_options=None)
    torch.testing.assert_close(attention_pytorch(q, k, v, 1, transformer_options={"fp8_attention": True}), expected)


@pytest.mark.parametrize("full_precision,disabled,wanted", [(False, [], True), (True, [], False), (False, ["float8_e4m3fn"], False)])
def test_moss_fp8_attention_capability(full_precision, disabled, wanted):
    operations = comfy.ops.mixed_precision_ops({"formats": ["float8_e4m3fn"]}, full_precision_mm=full_precision,
                                               disabled=disabled, load_device=torch.device("hpu"))
    model = moss.MOSSImage(dim=128, ffn_dim=128, num_heads=4, num_layers=0, text_dim=128, operations=operations)
    assert model.hpu_fp8_attention is wanted


@pytest.mark.parametrize("enabled", [True, False])
def test_moss_fp8_attention_options_preserve_explicit_choice(monkeypatch, enabled):
    model = moss.MOSSImage(dim=128, ffn_dim=128, num_heads=4, num_layers=0, text_dim=128,
                          dtype=torch.float32, operations=comfy.ops.manual_cast)
    model.hpu_fp8_attention = True
    for parameter in model.parameters():
        parameter.data.fill_(0.01)
    observed = []
    original = model.text_embedder.forward
    def forward(x, pe, transformer_options, mask=None):
        observed.append(transformer_options["fp8_attention"])
        return original(x, pe, transformer_options, mask)
    monkeypatch.setattr(model.text_embedder, "forward", forward)
    options = {} if enabled else {"fp8_attention": False}
    before = options.copy()
    with torch.no_grad():
        model(torch.randn(1, 128, 2, 2), torch.ones(1), torch.randn(1, 3, 128), transformer_options=options)
    assert observed == [enabled] and options == before
