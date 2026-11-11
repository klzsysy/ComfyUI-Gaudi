import json

import pytest
import torch

from comfy import ops, quant_ops, sd


def layer_state(rows=32, cols=64):
    torch.manual_seed(42)
    weight = (torch.randn(rows, cols) * 80).clamp(-448, 448).to(torch.float8_e4m3fn)
    weight[0, 0] = 448
    return {'weight': weight, 'weight_scale': torch.tensor(0.002),
            'bias': torch.randn(rows, dtype=torch.bfloat16),
            'comfy_quant': torch.tensor(list(json.dumps({'format': 'float8_e4m3fn'}).encode()), dtype=torch.uint8)}


def load_layer(state, full_precision=False):
    factory = ops.mixed_precision_ops(compute_dtype=torch.bfloat16, full_precision_mm=full_precision,
                                      load_device=torch.device('hpu'))
    layer = factory.Linear(state['weight'].shape[1], state['weight'].shape[0], device='cpu')
    result = layer.load_state_dict(state, strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    return layer


def test_repack_and_reload():
    state = layer_state()
    reference = state['weight'].float() * state['weight_scale']
    layer = load_layer(state)
    assert layer.layout_type == 'HPUFP8E4M3Layout'
    assert layer.weight._qdata.float().abs().max() <= 240
    torch.testing.assert_close(layer.weight.dequantize().float(), reference, rtol=.005, atol=1e-5)
    reloaded = load_layer(layer.state_dict())
    torch.testing.assert_close(reloaded.weight._qdata.view(torch.uint8), layer.weight._qdata.view(torch.uint8))
    torch.testing.assert_close(reloaded.weight._params.scale, layer.weight._params.scale)


def test_explicit_full_precision():
    layer = load_layer(layer_state(), full_precision=True)
    assert layer.layout_type == 'TensorCoreFP8E4M3Layout'
    assert layer._full_precision_mm


def test_weight_update_uses_safe_range():
    layer = load_layer(layer_state())
    replacement = torch.randn(32, 64, dtype=torch.bfloat16) * 1000
    layer.set_weight(replacement, seed=123)
    assert layer.weight._qdata.float().abs().max() <= 240
    assert layer.weight.dequantize().isfinite().all()


def test_prefixed_quantization_metadata(monkeypatch):
    metadata = {'_quantization_metadata': json.dumps({'layers': {'model.diffusion_model.linear': {'format': 'float8_e4m3fn'}}})}
    state = {'model.diffusion_model.linear.weight': torch.zeros(4, 4, dtype=torch.float8_e4m3fn),
             'model.diffusion_model.linear.weight_scale': torch.ones(())}
    captured = {}
    def detect(state, prefix, **kwargs):
        captured.update(state)
        return None
    monkeypatch.setattr(sd.model_detection, 'unet_prefix_from_state_dict', lambda state: 'model.diffusion_model.')
    monkeypatch.setattr(sd.model_detection, 'model_config_from_unet', detect)
    monkeypatch.setattr(sd.model_detection, 'convert_diffusers_mmdit', lambda *args: None)
    monkeypatch.setattr(sd.model_detection, 'model_config_from_diffusers_unet', lambda state: None)
    assert sd.load_diffusion_model_state_dict(state, metadata=metadata) is None
    assert set(captured) == {'linear.weight', 'linear.weight_scale', 'linear.comfy_quant'}


@pytest.mark.skipif(not hasattr(torch, 'hpu') or not torch.hpu.is_available(), reason='Requires an HPU')
@pytest.mark.parametrize('shape', [(64,), (7, 64), (2, 7, 64)])
@pytest.mark.parametrize('compiled', [False, True])
def test_native_linear(shape, compiled):
    layer = load_layer(layer_state()).to('hpu')
    x = torch.randn(shape, dtype=torch.bfloat16, device='hpu')
    reference = torch.nn.functional.linear(x, layer.weight.dequantize(), layer.bias)
    function = torch.compile(layer, backend='hpu_backend', dynamic=False) if compiled else layer
    actual = function(x)
    assert actual.shape == reference.shape and actual.dtype == x.dtype
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert relative.item() < .05
    assert actual.isfinite().all()
    # Dynamic patches use the existing compute-dtype cast path.
    layer.weight_function = [lambda w: w * 1.01]
    patched = function(x)
    assert patched.isfinite().all()


def test_cpu_execution_and_gradients():
    layer = load_layer(layer_state())
    x = torch.randn(3, 64, dtype=torch.bfloat16, requires_grad=True)
    reference = torch.nn.functional.linear(x, layer.weight.dequantize(), layer.bias)
    actual = layer(x)
    torch.testing.assert_close(actual, reference)
    actual.float().sum().backward()
    assert x.grad is not None and x.grad.isfinite().all()


@pytest.mark.skipif(not hasattr(torch, 'hpu') or not torch.hpu.is_available(), reason='Requires an HPU')
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_native_zero_and_static_scale(dtype):
    layer = load_layer(layer_state()).to('hpu')
    layer.input_scale = torch.nn.Parameter(torch.tensor(.1,device='hpu'),requires_grad=False)
    x = torch.zeros((2, 64),dtype=dtype,device='hpu')
    actual = layer(x)
    torch.testing.assert_close(actual.cpu(), layer.bias.cpu().to(dtype).expand(2,32))


def test_legacy_byte_weights():
    state = layer_state()
    original = state['weight'].clone()
    state['weight'] = state['weight'].view(torch.uint8)
    layer = load_layer(state)
    torch.testing.assert_close(layer.weight._qdata.float(), (original.float() * .5).to(torch.float8_e4m3fn).float())


def test_text_encoder_quantization_policy():
    assert 'float8_e4m3fn' in ops.get_disabled_quant_formats(torch.device('hpu'))
    factory = ops.mixed_precision_ops(full_precision_mm=True)
    layer = factory.Linear(64, 32, device='cpu')
    layer.load_state_dict(layer_state(), strict=False)
    with ops.use_quantized_matmul(layer, torch.device('hpu')):
        assert layer._full_precision_mm


def test_expert_policy():
    factory = ops.mixed_precision_ops(load_device=torch.device('hpu'))
    experts = factory.MoEExperts(2, 64, 32, device='cpu')
    state = layer_state()
    state['weight'] = state['weight'].unsqueeze(0).expand(2, -1, -1).contiguous()
    state['bias'] = state['bias'].unsqueeze(0).expand(2, -1).contiguous()
    experts.load_state_dict(state, strict=False)
    assert experts._full_precision_mm
    assert experts.layout_type == 'TensorCoreFP8E4M3Layout'
