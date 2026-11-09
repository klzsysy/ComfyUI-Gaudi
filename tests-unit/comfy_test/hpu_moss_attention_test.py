import pytest
import torch

import comfy.ops
import comfy.model_management
from comfy.cli_args import args
from comfy.ldm.moss_image import model as moss


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
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
