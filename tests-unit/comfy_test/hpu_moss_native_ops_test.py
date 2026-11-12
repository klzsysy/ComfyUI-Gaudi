import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.ldm.moss_image import hpu, model as moss


@pytest.fixture
def device():
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    return "hpu"


@pytest.mark.parametrize("batch,length,width", [(1, 23, 128), (2, 31, 128), (1, 307, 3072)])
def test_native_modulation_matches_reference(device, batch, length, width):
    torch.manual_seed(127)
    x = torch.randn(batch, length, width, dtype=torch.bfloat16).to(device)
    scale = torch.randn(batch, 1, width, dtype=torch.bfloat16).to(device) * 0.2
    shift = torch.randn_like(scale) * 0.2
    expected = torch.nn.functional.rms_norm(x, (width,), eps=1e-6) * (1 + scale) + shift
    actual = hpu.rms_adaln(x, scale, shift, 1e-6)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.01, atol=0.02)
    assert actual.dtype == x.dtype and actual.device == x.device


@pytest.mark.parametrize("batch,length,heads,width", [(1, 23, 4, 32), (2, 31, 4, 128), (1, 307, 24, 128)])
@pytest.mark.parametrize("general_matrix", [False, True])
@pytest.mark.parametrize("verify_blockwise", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_native_rope_layout_and_coefficients(device, batch, length, heads, width, general_matrix, verify_blockwise, compiled):
    torch.manual_seed(128)
    packed = torch.randn(batch, length, 3 * heads * width, dtype=torch.bfloat16).to(device)
    q, k, _ = (t.reshape(batch, length, heads, width) for t in packed.chunk(3, dim=-1))
    saved = packed.cpu().clone()
    q_scale = (1 + torch.randn(width, dtype=torch.bfloat16) * 0.1).to(device)
    k_scale = (1 + torch.randn(width, dtype=torch.bfloat16) * 0.1).to(device)
    angle = torch.randn(1, length, 1, width // 2)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, length, 1, width // 2, 2, 2)
    if general_matrix:
        pe = pe + torch.randn_like(pe) * 0.1
    pe = pe.to(device)
    blockwise = verify_blockwise and hpu.blockwise_coefficients(pe)
    assert blockwise == (verify_blockwise and not general_matrix)
    function = hpu.rms_rope_split_half
    if compiled:
        torch._dynamo.reset()
        function = torch.compile(function, backend="hpu_backend", dynamic=False)
    actual = function(q, k, pe, q_scale, k_scale, 1e-6, blockwise=blockwise)
    for value, source, weight in zip(actual, (q, k), (q_scale, k_scale)):
        normalized = torch.nn.functional.rms_norm(source, (width,), weight, 1e-6)
        expected = moss.rope_split_half(normalized, pe)
        torch.testing.assert_close(value.cpu(), expected.cpu(), rtol=0.01, atol=0.02)
        assert value.dtype == source.dtype and value.shape == source.shape
    torch.testing.assert_close(packed.cpu(), saved, rtol=0, atol=0)
    if compiled:
        torch._dynamo.reset()


@pytest.mark.parametrize("mode", ["cpu", "fp32", "grad", "training"])
def test_modulation_preserves_other_paths(monkeypatch, mode):
    monkeypatch.setattr(hpu, "rms_adaln", lambda *args: pytest.fail("native HPU inference path selected"))
    device = "cpu" if mode == "cpu" else "hpu"
    if device == "hpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("HPU required")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    dtype = torch.float32 if mode in ("cpu", "fp32") else torch.bfloat16
    norm = comfy.ops.manual_cast.RMSNorm(32, eps=1e-6, dtype=dtype, device=device)
    with torch.no_grad():
        norm.weight.fill_(1)
    x = torch.randn(1, 7, 32, dtype=dtype, device=device)
    scale, shift = torch.zeros_like(x[:, :1]), torch.zeros_like(x[:, :1])
    with torch.set_grad_enabled(mode == "grad"):
        result = moss.modulated_rms_norm(x, norm, scale, shift)
    assert result.shape == x.shape and torch.isfinite(result).all()
