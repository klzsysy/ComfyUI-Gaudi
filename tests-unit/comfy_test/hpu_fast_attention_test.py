import pytest
import torch
from torch._dynamo.backends.registry import lookup_backend

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.modules import attention
from comfy.ldm.qwen_image21 import model as qwen


def unsplit_fast_attention(q, k, v):
    k = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    return torch.ops.hpu.sdpa_recomp_fwd(q.contiguous(), k.contiguous(), v.contiguous(), None, 0.0,
                                        q.shape[-1] ** -.5, False, True, 'fast', None, 'right')[0]


@pytest.fixture
def hpu(monkeypatch):
    if not hasattr(torch, 'hpu') or not torch.hpu.is_available():
        pytest.skip('HPU required')
    monkeypatch.setattr(args, 'disable_comfy_compiler', False)
    monkeypatch.setattr(comfy.model_management, 'in_training', False)
    monkeypatch.setattr(qwen, 'optimized_attention', attention.attention_pytorch)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize('length', [128, 129, 255, 256, 257, 1023, 8193])
def test_split_attention_matches_reference_and_repeated_inputs(hpu, length):
    generator = torch.Generator().manual_seed(412)
    query = torch.randn(1, 37, 2, 128, generator=generator, dtype=torch.bfloat16).transpose(1, 2)
    key = torch.randn(1, length, 2, 128, generator=generator, dtype=torch.bfloat16).transpose(1, 2)
    value = torch.randn(1, length, 2, 128, generator=generator, dtype=torch.bfloat16).transpose(1, 2)
    reference = ((query.double() @ key.double().transpose(-1, -2)) / 128 ** .5).softmax(-1) @ value.double()
    compiled = torch.compile(attention._hpu_fast_attention, backend='hpu_backend', dynamic=None,
                             options={'force_static_compile': True})
    unsplit = torch.compile(unsplit_fast_attention, backend='hpu_backend', dynamic=None,
                            options={'force_static_compile': True})
    with torch.no_grad():
        q, k, v = (tensor.to('hpu') for tensor in (query, key, value))
        first = compiled(q, k, v).cpu()
        changed = compiled(q, k, v * -.5).cpu()
        repeated = compiled(q, k, v).cpu()
        vendor = unsplit(q, k, v).cpu().double()
    assert first.isfinite().all()
    # Fast softmax has a larger approximation error on diffuse random attention.
    assert float((vendor - reference).norm() / reference.norm()) < .08
    assert float((first.double() - reference).norm() / reference.norm()) < .08
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    assert not torch.equal(changed, first)


@pytest.mark.parametrize('length', [128, 129, 255, 8193])
def test_split_merge_against_independent_softmax(monkeypatch, length):
    generator = torch.Generator().manual_seed(417)
    q = torch.randn(1, 2, 5, 128, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(1, 2, length, 128, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(1, 2, length, 128, generator=generator, dtype=torch.bfloat16)

    def exact_sdpa(query, key, value, mask, dropout, scale, *args):
        scores = query.double() @ key.double().transpose(-1, -2) * scale
        maximum = scores.amax(-1, keepdim=True)
        probabilities = (scores - maximum).exp()
        inverse = probabilities.sum(-1, keepdim=True).reciprocal()
        return (probabilities * inverse) @ value.double(), maximum, inverse

    monkeypatch.setattr(torch.ops.hpu, 'sdpa_recomp_fwd', exact_sdpa, raising=False)
    centered = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    reference = ((q.double() @ centered.double().transpose(-1, -2)) / 128 ** .5).softmax(-1) @ v.double()
    actual = attention._hpu_fast_attention(q, k, v)
    assert float((actual - reference).norm() / reference.norm()) < 1e-6


@pytest.mark.parametrize('case', ['enabled', 'disabled', 'mask', 'scale', 'gqa', 'fp32', 'short_query',
                                  'short_key', 'heads', 'batch', 'grad', 'training', 'compiler_off',
                                  'eager', 'precision'])
def test_fast_attention_dispatch_contract(hpu, monkeypatch, case):
    batch = 2 if case == 'batch' else 1
    heads = 16 if case == 'heads' else 32
    dtype = torch.float32 if case == 'fp32' else torch.bfloat16
    qlength = 4095 if case == 'short_query' else 4096
    klength = 4095 if case == 'short_key' else 4225
    q = torch.zeros(batch, heads, qlength, 128, device='hpu', dtype=dtype)
    k = torch.zeros(batch, heads, klength, 128, device='hpu', dtype=dtype)
    v = torch.ones_like(k)
    calls = []

    def fast(q, k, v):
        calls.append('fast')
        return q

    def fallback(q, k, v, **kwargs):
        calls.append('fallback')
        return q

    monkeypatch.setattr(attention, '_hpu_fast_attention', fast)
    monkeypatch.setattr(comfy.ops, 'scaled_dot_product_attention', fallback)
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: case != 'eager')
    monkeypatch.setattr(args, 'disable_comfy_compiler', case == 'compiler_off')
    monkeypatch.setattr(comfy.model_management, 'in_training', case == 'training')
    options = dict(hpu_fast_attention=case != 'disabled', skip_reshape=True)
    if case == 'mask':
        options['mask'] = torch.ones(1, 1, 1, klength, device='hpu', dtype=torch.bool)
    if case == 'scale':
        options['scale'] = .1
    if case == 'gqa':
        options['enable_gqa'] = True
    if case == 'precision':
        options['attn_precision'] = torch.float32
    with torch.set_grad_enabled(case == 'grad'):
        result = attention.attention_pytorch(q, k, v, heads, **options)
    assert result.shape == (batch, qlength, heads * 128)
    assert calls == ['fast' if case == 'enabled' else 'fallback']


@pytest.mark.parametrize('enabled,index,expected', [(True, 0, False), (True, 1, True), (True, 31, True),
                                                  (False, 1, False)])
def test_qwen_keeps_first_layer_and_explicit_opt_out(hpu, monkeypatch, enabled, index, expected):
    calls = []

    def compiled(q, k, v, heads, mask, options, preferred):
        calls.append(options['hpu_fast_attention'])
        return v

    monkeypatch.setattr(qwen, '_compiled_attention', lambda: compiled)
    tensor = torch.ones(1, 17, 128, device='hpu', dtype=torch.bfloat16)
    options = dict(hpu_fast_attention=enabled, block_index=index)
    before = options.copy()
    with torch.no_grad():
        qwen.attention(*(qwen.AttentionTensorContainer(tensor) for _ in range(3)), 4, transformer_options=options)
    assert calls == [expected]
    assert options == before


def test_cpu_attention_ignores_hpu_fast_request(monkeypatch):
    monkeypatch.setattr(attention, '_hpu_fast_attention', lambda *args: pytest.fail('HPU path on CPU'))
    tensors = [torch.randn(1, 17, 128) for _ in range(3)]
    expected = attention.attention_pytorch(*tensors, 4)
    actual = attention.attention_pytorch(*tensors, 4, hpu_fast_attention=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen_normal_entry_recomputes_same_shape_inputs(hpu):
    generator = torch.Generator().manual_seed(413)
    query = torch.randn(1, 4096, 4096, generator=generator, dtype=torch.bfloat16)
    key = torch.randn(1, 4225, 4096, generator=generator, dtype=torch.bfloat16)
    value = torch.randn(1, 4225, 4096, generator=generator, dtype=torch.bfloat16)
    rows = torch.tensor([0, 127, 2048, 4095])
    qref = query.reshape(1, 4096, 32, 128)[0, rows].transpose(0, 1).double()
    kref, vref = (tensor.reshape(1, 4225, 32, 128)[0].transpose(0, 1).double() for tensor in (key, value))
    reference = ((qref @ kref.transpose(-1, -2)) / 128 ** .5).softmax(-1) @ vref
    unsplit = torch.compile(unsplit_fast_attention, backend='hpu_backend', dynamic=None,
                            options={'force_static_compile': True})
    with torch.no_grad():
        q, k, v = (tensor.to('hpu') for tensor in (query, key, value))

        def run(value):
            containers = (qwen.AttentionTensorContainer(tensor) for tensor in (q, k, value))
            return qwen.attention(*containers, 32, transformer_options={'hpu_fast_attention': True, 'block_index': 1}).cpu()

        first = run(v)
        changed = run(v * -.5)
        repeated = run(v)
        vendor = unsplit(*(tensor.reshape(1, tensor.shape[1], 32, 128).transpose(1, 2)
                           for tensor in (q, k, v))).cpu()[0, :, rows].double()
    sample = first.reshape(1, 4096, 32, 128)[0, rows].transpose(0, 1).double()
    assert float((vendor - reference).norm() / reference.norm()) < .08
    assert float((sample - reference).norm() / reference.norm()) < .08
    assert float((sample - vendor).norm() / vendor.norm()) < .01
    assert not torch.equal(changed, first)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)


def test_qwen_reuses_attention_graphs_across_kv_boundaries(hpu, monkeypatch):
    backend = lookup_backend('hpu_backend')
    graphs = []

    def counted_backend(graph, inputs, **kwargs):
        graphs.append(graph)
        return backend(graph, inputs, **kwargs)

    compiled = torch.compile(qwen._attention, backend=counted_backend, dynamic=None,
                             options={'force_static_compile': True})
    monkeypatch.setattr(qwen, '_compiled_attention', lambda: compiled)
    lengths = (8191, 8192, 8193, 8256, 8257, 8319, 8320, 8321, 9445, 9400, 8191)
    with torch.no_grad():
        query = torch.zeros(1, 8160, 4096, device='hpu', dtype=torch.bfloat16)
        first = None
        for index, length in enumerate(lengths):
            value = .25 if index == len(lengths) - 1 else .25 + index / 8
            key = torch.full((1, length, 4096), .25, device='hpu', dtype=torch.bfloat16)
            values = torch.full_like(key, value)
            containers = (qwen.AttentionTensorContainer(tensor) for tensor in (query, key, values))
            output = qwen.attention(*containers, 32, transformer_options={'hpu_fast_attention': True, 'block_index': 1}).cpu()
            torch.testing.assert_close(output, torch.full_like(output, value), rtol=.02, atol=.01)
            if first is None:
                first = output
        torch.testing.assert_close(output, first, rtol=0, atol=0)
    assert len(graphs) <= 4
