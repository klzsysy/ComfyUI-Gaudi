import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.modules.attention import attention_pytorch
from comfy.ldm.qwen_image21 import model as qwen


@pytest.fixture
def compiler(monkeypatch):
    if not hasattr(torch, 'hpu') or not torch.hpu.is_available():
        pytest.skip('HPU required')
    monkeypatch.setattr(comfy.model_management, 'in_training', False)
    monkeypatch.setattr(args, 'disable_comfy_compiler', False)
    monkeypatch.setattr(qwen, 'optimized_attention', attention_pytorch)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_inputs(device='hpu', length=17, fused=True):
    torch.manual_seed(83)
    dtype = torch.bfloat16 if device == 'hpu' else torch.float32
    block = qwen.QwenImage21TransformerBlock(128, 4, 32, fused_mlp=fused, dtype=dtype,
                                            device=device, operations=comfy.ops.manual_cast)
    for name, p in block.named_parameters():
        p.data.copy_((torch.randn(p.shape) * .03 + (1 if 'norm' in name else 0)).to(device=device, dtype=dtype))
    x = torch.randn(2, length, 128).to(device=device, dtype=dtype)
    mod = tuple((torch.randn(1, 1, 128).to(x) * .1, torch.randn(2, 1, 128).to(x) * .1) for _ in range(4))
    mod += (torch.zeros(1, 1, 128).to(x),)
    angle = torch.randn(1, length, 1, 16)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1).reshape(1, length, 1, 16, 2, 2).to(device)
    return block, x, mod, pe


@pytest.mark.parametrize('fused', [True, False])
@pytest.mark.parametrize('mode', ['causal', 'cached'])
def test_compiled_block_prefix_and_weight_changes(compiler, monkeypatch, fused, mode):
    with torch.no_grad():
        block, _, mod, _ = make_inputs(fused=fused)
        for length in [17, 31, 17]:
            _, x, _, pe = make_inputs(length=length)
            if mode == 'causal':
                prefix_len = 5
                fn = qwen.block_causal_attention([(0,5,torch.ones(5,5,device='hpu',dtype=torch.bool).tril()),(5,length,None)])
            else:
                prefix_len = 0
                kv = torch.randn(2, 5, 4, 32).to(x)
                fn = qwen.prefix_cached_attention(kv, kv * .5)
            monkeypatch.setattr(args, 'disable_comfy_compiler', True)
            expected = block(x.clone(), mod, pe, fn, prefix_len).cpu()
            monkeypatch.setattr(args, 'disable_comfy_compiler', False)
            working = x.clone()
            actual = block(working, mod, pe, fn, prefix_len)
            assert actual.data_ptr() == working.data_ptr()
            torch.testing.assert_close(actual.cpu(), expected, rtol=.015, atol=.015)
            block.attn.to_out[0].weight.mul_(.9)


@pytest.mark.parametrize('mode', ['disabled', 'training', 'grad', 'hook'])
def test_compilation_bypass_contracts(compiler, monkeypatch, mode):
    block, x, mod, pe = make_inputs()
    options = {}
    with torch.no_grad():
        block(x.clone(), mod, pe, qwen.block_causal_attention([(0,17,None)]), 0)
    if mode == 'hook':
        block.img_mlp.out.register_forward_hook(lambda module, inputs, out: out * .5)
    monkeypatch.setattr(args, 'disable_comfy_compiler', mode == 'disabled')
    monkeypatch.setattr(comfy.model_management, 'in_training', mode == 'training')
    monkeypatch.setattr(qwen, '_compiled_ffn', lambda: pytest.fail('compiler must be bypassed'))
    with torch.set_grad_enabled(mode == 'grad'):
        out = block(x.clone(), mod, pe, qwen.block_causal_attention([(0,17,None)],options), 0, options)
    assert torch.isfinite(out.cpu()).all()


def test_cpu_block_bypasses_hpu_compiler(monkeypatch):
    monkeypatch.setattr(qwen, '_compiled_ffn', lambda: pytest.fail('HPU compiler on CPU'))
    block,x,mod,pe = make_inputs(device='cpu')
    with torch.no_grad():
        out = block(x,mod,pe,qwen.block_causal_attention([(0,17,None)]),0)
    assert torch.isfinite(out).all()


def test_custom_operations_keep_eager_ffn(compiler, monkeypatch):
    class CustomOps(comfy.ops.manual_cast):
        pass

    monkeypatch.setattr(qwen, '_compiled_ffn', lambda: pytest.fail('custom operations must remain eager'))
    module = qwen.SwiGLUFeedForward(128, 256, device='hpu', dtype=torch.bfloat16, operations=CustomOps)
    for parameter in module.parameters():
        parameter.data.fill_(.01)
    with torch.no_grad():
        output = module(torch.ones(1, 17, 128, device='hpu', dtype=torch.bfloat16))
    assert torch.isfinite(output.cpu()).all()
