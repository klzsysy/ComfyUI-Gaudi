import pytest
import torch

import comfy.ops
import comfy.model_management
from comfy.cli_args import args
from comfy.text_encoders import hpu_mlp
from comfy.text_encoders.llama import Llama2Config, MLP, RMSNorm
from comfy.text_encoders.qwen35 import Qwen35Config, Qwen35TransformerBlock


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_mlp(merged=False):
    config = Llama2Config(hidden_size=32, intermediate_size=64)
    config.merged_mlp = merged
    module = MLP(config, device="hpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    for parameter in module.parameters():
        parameter.data.copy_(torch.randn_like(parameter) * 0.05)
    return module


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("add", [False, True])
def test_hpu_mlp_inputs_and_weights(hpu_compiler, merged, add):
    torch.manual_seed(63)
    module = make_mlp(merged)
    norm = RMSNorm(32, eps=1e-6, add=add, dtype=torch.float32, device="hpu")
    norm.weight.data.fill_(0.01 if add else 1.01)
    compiled = hpu_mlp.forward
    with torch.no_grad():
        for turn, length in enumerate((1, 8, 4, 2, 1)):
            x = torch.randn(2, length, 32, dtype=torch.bfloat16).to("hpu")
            before = x.clone()
            if turn == 4:
                module.down_proj.weight.copy_(torch.randn_like(module.down_proj.weight) * 0.05)
                norm.weight.add_(0.01)
            expected = module(x, norm=norm)
            actual = compiled(module, x, norm)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.01, rtol=0.01)
            torch.testing.assert_close(x.cpu(), before.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("target", ["module", "linear", "norm", "global"])
def test_hpu_mlp_preserves_module_hooks(hpu_compiler, target):
    module = make_mlp()
    norm = RMSNorm(32, eps=1e-6, add=True, dtype=torch.float32, device="hpu")
    norm.weight.data.fill_(0.01)
    x = torch.randn(1, 8, 32, dtype=torch.bfloat16).to("hpu")
    compiled = hpu_mlp.forward
    with torch.no_grad():
        original = compiled(module, x, norm)
        if target == "global":
            hook = torch.nn.modules.module.register_module_forward_hook(lambda child, values, output: output * 0.5 if child is module else None)
        else:
            owner = {"module": module, "linear": module.down_proj, "norm": norm}[target]
            hook = owner.register_forward_hook(lambda child, values, output: output * 0.5)
        try:
            expected = module(x, norm=norm)
            torch.testing.assert_close(compiled(module, x, norm).cpu(), expected.cpu(), atol=0.01, rtol=0.01)
        finally:
            hook.remove()
        torch.testing.assert_close(compiled(module, x, norm).cpu(), original.cpu(), atol=0.01, rtol=0.01)


class ZeroAttention(torch.nn.Module):
    def forward(self, x, past_key_value=None, **kwargs):
        return torch.zeros_like(x), past_key_value


class CustomOps(comfy.ops.manual_cast):
    pass


@pytest.mark.parametrize("mode", ["disabled", "training", "custom_ops", "prefill", "no_cache"])
def test_qwen_mlp_eager_paths(hpu_compiler, monkeypatch, mode):
    monkeypatch.setattr(hpu_mlp, "forward", lambda *values: pytest.fail("MLP compiler should be bypassed"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    config = Qwen35Config(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=["full_attention"])
    block = Qwen35TransformerBlock(config, 0, device="hpu", dtype=torch.bfloat16, ops=CustomOps if mode == "custom_ops" else comfy.ops.manual_cast)
    block.self_attn = ZeroAttention()
    for parameter in block.parameters():
        parameter.data.fill_(0.01)
    x = torch.ones(1, 17 if mode == "prefill" else 1, 32, device="hpu", dtype=torch.bfloat16)
    with torch.no_grad():
        out, _ = block(x, past_key_value=None if mode == "no_cache" else object())
    assert torch.isfinite(out).all()


def test_qwen_cpu_mlp_does_not_compile_hpu(monkeypatch):
    monkeypatch.setattr(hpu_mlp, "forward", lambda *values: pytest.fail("HPU compiler used for CPU"))
    config = Qwen35Config(hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=["full_attention"])
    block = Qwen35TransformerBlock(config, 0, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)
    block.self_attn = ZeroAttention()
    for parameter in block.parameters():
        parameter.data.fill_(0.01)
    with torch.no_grad():
        out, _ = block(torch.ones(1, 1, 32), past_key_value=object())
    assert torch.isfinite(out).all()
