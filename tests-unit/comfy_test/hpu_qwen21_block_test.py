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


@pytest.mark.parametrize('general', [False, True])
@pytest.mark.parametrize('batch,length', [(1, 17), (2, 31), (1, 4096)])
def test_native_rms_rope_preserves_coefficients(compiler, general, batch, length):
    torch.manual_seed(106)
    with torch.no_grad():
        q, k = torch.randn(2, batch, length, 4, 128, dtype=torch.bfloat16).to('hpu').unbind(0)
        q_scale, k_scale = (torch.randn(128, dtype=torch.bfloat16).to('hpu') * .1 + 1 for _ in range(2))
        angle = torch.randn(1, length, 1, 64)
        pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1).reshape(1, length, 1, 64, 2, 2)
        if general:
            pe = torch.randn_like(pe)
        pe = pe.to('hpu')
        expected = comfy.ops.quant_ops.ck.rms_rope(q, k, pe, q_scale, k_scale, 1e-6)
        actual = qwen._hpu_rms_rope(q, k, pe, q_scale, k_scale, 1e-6)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=0)


def test_cached_attention_switches_prefix_lengths(compiler, monkeypatch):
    torch.manual_seed(107)
    with torch.no_grad():
        for length, prefix in [(31, 5), (43, 7), (31, 5)]:
            query, key, value = torch.randn(3, 2, length, 4, 32, dtype=torch.bfloat16).to('hpu').unbind(0)
            prefix_k, prefix_v = torch.randn(2, 2, prefix, 4, 32, dtype=torch.bfloat16).to('hpu').unbind(0)
            function = qwen.prefix_cached_attention(prefix_k, prefix_v)
            def run():
                containers = (qwen.AttentionTensorContainer(t) for t in (query, key, value))
                return function(*containers, 4, preferred_attention=qwen.ComfyAttention())
            monkeypatch.setattr(args, 'disable_comfy_compiler', True)
            expected = run().cpu()
            monkeypatch.setattr(args, 'disable_comfy_compiler', False)
            actual = run().cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_attention_override_bypasses_compilation(compiler, monkeypatch):
    monkeypatch.setattr(qwen, '_compiled_attention', lambda: pytest.fail('override must bypass compilation'))
    calls = []
    def override(function, q, k, v, heads, **kwargs):
        calls.append(heads)
        return v
    tensors = [torch.randn(1, 17, 128, dtype=torch.bfloat16).to('hpu') for _ in range(3)]
    with torch.no_grad():
        output = qwen.attention(*(qwen.AttentionTensorContainer(t) for t in tensors), 4,
                                transformer_options={'optimized_attention_override': override})
    torch.testing.assert_close(output.cpu(), tensors[2].cpu(), rtol=0, atol=0)
    assert calls == [4]


@pytest.mark.parametrize('additive', [False, True])
def test_masked_prefix_keeps_eager_rounding(compiler, monkeypatch, additive):
    monkeypatch.setattr(qwen, '_compiled_attention', lambda: pytest.fail('masked prefix must remain eager'))
    q, k, v = torch.randn(3, 1, 17, 128, dtype=torch.bfloat16).to('hpu').unbind(0)
    mask = torch.ones(17, 17, device='hpu', dtype=torch.bool).tril()
    if additive:
        mask = torch.where(mask, 0.0, -10000.0)
    with torch.no_grad():
        containers = (qwen.AttentionTensorContainer(t) for t in (q, k, v))
        result = qwen.attention(*containers, 4, mask=mask)
    assert torch.isfinite(result.cpu()).all()
