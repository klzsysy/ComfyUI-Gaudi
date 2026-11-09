import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.moss_image import model as moss


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_block(fused=True, device="hpu", dtype=torch.bfloat16):
    block = moss.Block(128, 192, 4, fused_qkv=fused, dtype=dtype, device=device, operations=comfy.ops.manual_cast)
    for name, value in block.named_parameters():
        value.data.copy_(torch.randn(value.shape, dtype=dtype).to(device) * 0.03 + (1 if "norm" in name else 0))
    return block


def inputs(length, device="hpu", dtype=torch.bfloat16):
    x = torch.randn(2, length, 128, dtype=dtype).to(device)
    mod = tuple(torch.randn(2, 1, 128, dtype=dtype).to(device) * 0.1 for _ in range(6))
    angle = torch.randn(1, length, 1, 16)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, length, 1, 16, 2, 2).to(device)
    return x, mod, pe


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("mode", ["plain", "mask", "kv", "reference_tokens"])
def test_compiled_moss_block_contract(hpu_compiler, monkeypatch, fused, mode):
    torch.manual_seed(52)
    block = make_block(fused)
    with torch.no_grad():
        for turn, length in enumerate((17, 31, 17)):
            x, mod, pe = inputs(length)
            extra = {}
            if turn == 2:
                for parameter in block.proj['o'].parameters():
                    parameter.copy_(torch.randn_like(parameter) * 0.03)
            if mode == "mask":
                extra['mask'] = torch.randn(2, 1, 1, length, dtype=torch.bfloat16).to("hpu") * 0.2
            elif mode == "kv":
                extra['ref_kv'] = torch.randn(2, 2, 5, 128, dtype=torch.bfloat16).to("hpu")
                extra['kv_out'] = torch.empty(2, 2, length, 128, device="hpu", dtype=torch.bfloat16)
            elif mode == "reference_tokens":
                extra['ref'] = (length - 5, tuple(t * 0.7 for t in mod))
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected = block(x.clone(), mod, pe, {}, **extra).cpu()
            expected_kv = extra['kv_out'].cpu().clone() if mode == "kv" else None
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            working = x.clone()
            actual = block(working, mod, pe, {}, **extra)
            assert actual.data_ptr() == working.data_ptr()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)
            if expected_kv is not None:
                torch.testing.assert_close(extra['kv_out'].cpu(), expected_kv, rtol=0.01, atol=0.01)


def test_moss_block_many_lengths(hpu_compiler, monkeypatch):
    block = make_block()
    with torch.no_grad():
        for length in (17, 23, 31, 43, 53, 61, 71, 83, 97, 107, 17):
            x, mod, pe = inputs(length)
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected = block(x.clone(), mod, pe).cpu()
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            actual = block(x.clone(), mod, pe)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)


def test_cpu_moss_block_does_not_compile_hpu(monkeypatch):
    monkeypatch.setattr(moss, "_compiled_block_ops", lambda: pytest.fail("HPU compiler used for CPU"))
    block = make_block(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        result = block(*inputs(17, "cpu", torch.float32))
    assert result.shape == (2, 17, 128) and torch.isfinite(result).all()


@pytest.mark.parametrize("mode", ["disabled", "training", "override"])
def test_moss_block_eager_paths(hpu_compiler, monkeypatch, mode):
    monkeypatch.setattr(moss, "_compiled_block_ops", lambda: pytest.fail("block compiler should be bypassed"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    block = make_block()
    options, calls = {}, []
    if mode == "override":
        def override(function, *values, **kwargs):
            calls.append(True)
            return function(*values, **kwargs)
        options['optimized_attention_override'] = override
    with torch.no_grad():
        result = block(*inputs(17), transformer_options=options)
    assert torch.isfinite(result).all()
    if mode == "override":
        assert calls == [True]
