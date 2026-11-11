import pytest
import torch

import comfy.model_management
from comfy.cli_args import args
from comfy.text_encoders import qwen35


@pytest.fixture
def hpu_norm(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    compiled = qwen35._compiled_hpu_rms_norm
    compiled.cache_clear()
    yield
    torch._dynamo.reset()
    compiled.cache_clear()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("add", [False, True])
def test_norm_shapes_weights_and_reference(hpu_norm, dtype, weight_dtype, add):
    torch.manual_seed(731)
    module = qwen35.RMSNorm(256, eps=1e-6, add=add, device="hpu", dtype=weight_dtype)
    module.weight.data.copy_(torch.randn(256, device="hpu") * 0.1 + (0 if add else 1))
    with torch.no_grad():
        for shape in ((1, 1, 256), (1, 8, 256), (1, 23, 256), (1, 307, 256), (1, 8, 4, 256), (1, 1, 256)):
            x = torch.randn(shape, dtype=dtype).to("hpu")
            if len(shape) == 4:
                x = torch.randn((*shape[:-1], 512), dtype=dtype).to("hpu")[..., :256]
            before = x.cpu()
            module.weight.add_(0.005)
            expected = qwen35.LlamaRMSNorm.forward(module, x).cpu()
            actual = module(x).cpu()
            scale = module.scale().to(dtype).cpu().double()
            reference = before.double() * torch.rsqrt(before.double().square().mean(-1, keepdim=True) + module.eps) * scale
            tol = 0.02 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(actual.double(), reference, atol=tol, rtol=tol)
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
            torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)
    assert list(module.state_dict()) == ["weight"]


@pytest.mark.parametrize("target", ["module", "global"])
def test_norm_hooks_added_after_compile(hpu_norm, target):
    module = qwen35.RMSNorm(256, add=True, device="hpu", dtype=torch.float32)
    module.weight.data.fill_(0.01)
    x = torch.randn(1, 8, 256, dtype=torch.bfloat16).to("hpu")
    with torch.no_grad():
        module(x)
        def hook(child, values, output):
            return output * 0.5 if child is module else None
        handle = (module.register_forward_hook(hook) if target == "module"
                  else torch.nn.modules.module.register_module_forward_hook(hook))
        try:
            expected = qwen35.LlamaRMSNorm.forward(module, x) * 0.5
            torch.testing.assert_close(module(x).cpu(), expected.cpu(), atol=0, rtol=0)
        finally:
            handle.remove()


@pytest.mark.parametrize("mode", ["disabled", "training", "grad"])
def test_norm_hpu_eager_routes(hpu_norm, monkeypatch, mode):
    monkeypatch.setattr(qwen35, "_compiled_hpu_rms_norm", lambda: pytest.fail("compiler must be bypassed"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    module = qwen35.RMSNorm(32, device="hpu", dtype=torch.float32)
    module.weight.data.fill_(1)
    with torch.set_grad_enabled(mode == "grad"):
        x = torch.randn(1, 8, 32, device="hpu", requires_grad=mode == "grad")
        result = module(x)
        torch.testing.assert_close(result.cpu(), qwen35.LlamaRMSNorm.forward(module, x).cpu(), atol=0, rtol=0)
        if mode == "grad":
            result.sum().backward()
            assert torch.isfinite(x.grad).all()


def test_norm_cpu_preserves_gradients(monkeypatch):
    monkeypatch.setattr(qwen35, "_compiled_hpu_rms_norm", lambda: pytest.fail("HPU compiler used for CPU"))
    module = qwen35.RMSNorm(32, add=True)
    module.weight.data.fill_(0.01)
    x = torch.randn(1, 8, 32, requires_grad=True)
    result = module(x)
    torch.testing.assert_close(result, qwen35.LlamaRMSNorm.forward(module, x), atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
