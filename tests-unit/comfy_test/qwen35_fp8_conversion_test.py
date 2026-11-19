import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE, dequantize_int8_convrot_weight_dtype
from scripts.gaudi.convert_qwen35_fp8 import convert


def test_conversion_preserves_sensitive_weights_and_metadata(tmp_path):
    prefix = 'model.language_model.layers.0.'
    state = {
        prefix + 'linear_attn.in_proj_qkv.weight': torch.randn(32, 64, dtype=torch.bfloat16),
        prefix + 'linear_attn.in_proj_a.weight': torch.randn(4, 64, dtype=torch.bfloat16),
        prefix + 'linear_attn.A_log': torch.randn(4),
        prefix + 'mlp.up_proj.weight': torch.zeros(64, 64, dtype=torch.bfloat16),
        'lm_head.weight': torch.randn(128, 64, dtype=torch.bfloat16),
        'mtp.fc.weight': torch.randn(64, 128, dtype=torch.bfloat16),
        'model.language_model.embed_tokens.weight': torch.randn(128, 64, dtype=torch.bfloat16),
    }
    source, destination = tmp_path / 'source.safetensors', tmp_path / 'fp8.safetensors'
    save_file(state, str(source), metadata={'conversion': 'int8_convrot_to_bf16'})
    report = convert(source, destination)
    assert report['converted_linears'] == 3
    with safe_open(str(destination), framework='pt') as handle:
        metadata = handle.metadata()
        assert metadata['source_conversion'] == 'int8_convrot_to_bf16'
        layers = json.loads(metadata['_quantization_metadata'])['layers']
        for key, expected in state.items():
            actual = handle.get_tensor(key)
            if key.removesuffix('.weight') in layers:
                assert actual.dtype == torch.float8_e4m3fn
                assert actual.float().abs().max() <= 224
                scale = handle.get_tensor(key.removesuffix('.weight') + '.weight_scale')
                relative = (actual.float() * scale - expected.float()).norm() / expected.float().norm().clamp_min(1)
                assert relative < .04
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        convert(source, destination)
    with pytest.raises(FileExistsError):
        convert(source, source)


def test_invalid_source_is_not_published(tmp_path):
    source, destination = tmp_path / 'source.safetensors', tmp_path / 'fp8.safetensors'
    save_file({'model.language_model.layers.0.linear_attn.in_proj_qkv.weight': torch.full((2, 2), float('nan'))}, str(source))
    with pytest.raises(ValueError, match='Nonfinite'):
        convert(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('*.part-*'))


def test_int8_convrot_uses_fp32_until_fp8_quantization(tmp_path):
    torch.manual_seed(23)
    prefix = 'model.language_model.layers.0.linear_attn.'
    state = {}
    for name, rows in [('in_proj_qkv', 32), ('in_proj_a', 4)]:
        key = prefix + name
        state[key + '.weight'] = torch.randint(-127, 128, (rows, 64), dtype=torch.int8)
        state[key + '.weight_scale'] = torch.rand(rows, 1) * .003 + .001
        config = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16}
        state[key + '.comfy_quant'] = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
    state[prefix + 'dt_bias'] = torch.randn(4, dtype=torch.bfloat16)
    source, destination = tmp_path / 'int8.safetensors', tmp_path / 'fp8.safetensors'
    save_file(state, str(source))
    with pytest.raises(ValueError, match='from-int8-convrot'):
        convert(source, destination)
    report = convert(source, destination, from_int8_convrot=True)
    assert report['converted_linears'] == 1
    with safe_open(str(destination), framework='pt') as handle:
        assert handle.metadata()['source_conversion'] == 'int8_convrot_to_fp32'
        assert not any(key.endswith('.comfy_quant') for key in handle.keys())
        for name in ('in_proj_qkv', 'in_proj_a'):
            key = prefix + name
            reference = dequantize_int8_convrot_weight_dtype(
                state[key + '.weight'], state[key + '.weight_scale'], 16, DTYPE_TO_CODE[torch.float32])
            if name == 'in_proj_qkv':
                scale = reference.abs().amax().clamp_min(1e-12) / 224.0
                actual = handle.get_tensor(key + '.weight')
                torch.testing.assert_close(handle.get_tensor(key + '.weight_scale'), scale, rtol=0, atol=0)
                assert torch.equal(actual.view(torch.uint8), (reference/scale).to(torch.float8_e4m3fn).view(torch.uint8))
                rounded_scale = reference.bfloat16().float().abs().amax().clamp_min(1e-12) / 224.0
                assert scale != rounded_scale
            else:
                torch.testing.assert_close(handle.get_tensor(key + '.weight'), reference.bfloat16(), rtol=0, atol=0)
                assert key + '.weight_scale' not in handle.keys()
        torch.testing.assert_close(handle.get_tensor(prefix + 'dt_bias'), state[prefix + 'dt_bias'], rtol=0, atol=0)


@pytest.mark.parametrize('config', [
    {'format': 'float8_e4m3fn'},
    {'format': 'int8_tensorwise', 'convrot': False},
])
def test_direct_conversion_rejects_other_quantization(tmp_path, config):
    prefix = 'model.language_model.layers.0.linear_attn.in_proj_qkv'
    source, destination = tmp_path / 'source.safetensors', tmp_path / 'fp8.safetensors'
    save_file({prefix + '.weight': torch.zeros(4, 16, dtype=torch.int8),
               prefix + '.comfy_quant': torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)}, str(source))
    with pytest.raises(ValueError, match='Unsupported quantization'):
        convert(source, destination, from_int8_convrot=True)
    assert not destination.exists()


@pytest.mark.parametrize('quantize_mtp', [False, True])
def test_attach_mtp_from_int8_convrot_and_optional_fp8_quantization(tmp_path, quantize_mtp):
    torch.manual_seed(41)
    prefix = 'model.language_model.layers.0.linear_attn.'
    source_state = {
        prefix + 'in_proj_qkv.weight': torch.randn(32, 64, dtype=torch.bfloat16),
        'lm_head.weight': torch.randn(64, 32, dtype=torch.bfloat16),
    }
    conv_cfg = {'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16}
    mtp_state = {
        'mtp.fc.weight': torch.randn(32, 64, dtype=torch.bfloat16),
        'mtp.norm.weight': torch.randn(32, dtype=torch.bfloat16),
        'mtp.layers.0.mlp.gate_proj.weight': torch.randint(-127, 128, (64, 32), dtype=torch.int8),
        'mtp.layers.0.mlp.gate_proj.weight_scale': torch.rand(64, 1) * .003 + .001,
        'mtp.layers.0.mlp.gate_proj.comfy_quant': torch.tensor(list(json.dumps(conv_cfg).encode()), dtype=torch.uint8),
        'unrelated.weight': torch.randn(8, 8, dtype=torch.bfloat16),
    }
    source = tmp_path / 'pe.safetensors'
    mtp_src = tmp_path / 'base_int8.safetensors'
    destination = tmp_path / f'pe_mtp_q{int(quantize_mtp)}.safetensors'
    save_file(source_state, str(source))
    save_file(mtp_state, str(mtp_src))

    report = convert(source, destination, attach_mtp=mtp_src, quantize_mtp=quantize_mtp)
    assert report['mtp_tensors'] == 3
    assert report['mtp_converted_linears'] == (2 if quantize_mtp else 0)
    assert report['converted_linears'] == (4 if quantize_mtp else 2)
    with safe_open(str(destination), framework='pt') as handle:
        metadata = handle.metadata()
        assert metadata['mtp_source'] == mtp_src.name
        assert metadata['mtp_source_conversion'] == 'int8_convrot_to_fp32'
        layers = json.loads(metadata['_quantization_metadata'])['layers']
        assert 'unrelated.weight' not in handle.keys()
        assert not any(k.endswith('.comfy_quant') for k in handle.keys())
        torch.testing.assert_close(handle.get_tensor('mtp.norm.weight'), mtp_state['mtp.norm.weight'], rtol=0, atol=0)
        gate_ref = dequantize_int8_convrot_weight_dtype(
            mtp_state['mtp.layers.0.mlp.gate_proj.weight'],
            mtp_state['mtp.layers.0.mlp.gate_proj.weight_scale'],
            16, DTYPE_TO_CODE[torch.float32])
        if quantize_mtp:
            assert metadata['mtp_quantization'] == 'e4m3_range224'
            assert 'mtp.fc' in layers and 'mtp.layers.0.mlp.gate_proj' in layers
            scale = gate_ref.abs().amax().clamp_min(1e-12) / 224.0
            torch.testing.assert_close(handle.get_tensor('mtp.layers.0.mlp.gate_proj.weight_scale'), scale, rtol=0, atol=0)
            assert torch.equal(
                handle.get_tensor('mtp.layers.0.mlp.gate_proj.weight').view(torch.uint8),
                (gate_ref / scale).to(torch.float8_e4m3fn).view(torch.uint8))
        else:
            assert 'mtp.fc' not in layers and 'mtp.layers.0.mlp.gate_proj' not in layers
            torch.testing.assert_close(handle.get_tensor('mtp.fc.weight'), mtp_state['mtp.fc.weight'], rtol=0, atol=0)
            torch.testing.assert_close(handle.get_tensor('mtp.layers.0.mlp.gate_proj.weight'), gate_ref.bfloat16(), rtol=0, atol=0)


def test_attach_mtp_rejects_duplicate_or_missing_mtp(tmp_path):
    prefix = 'model.language_model.layers.0.linear_attn.'
    with_mtp = tmp_path / 'with_mtp.safetensors'
    no_mtp = tmp_path / 'no_mtp.safetensors'
    save_file({prefix + 'in_proj_qkv.weight': torch.randn(4, 16, dtype=torch.bfloat16),
               'mtp.fc.weight': torch.randn(16, 32, dtype=torch.bfloat16)}, str(with_mtp))
    save_file({prefix + 'in_proj_qkv.weight': torch.randn(4, 16, dtype=torch.bfloat16)}, str(no_mtp))
    with pytest.raises(ValueError, match='already contains mtp'):
        convert(with_mtp, tmp_path / 'out1.safetensors', attach_mtp=with_mtp)
    with pytest.raises(ValueError, match='Expected mtp'):
        convert(no_mtp, tmp_path / 'out2.safetensors', attach_mtp=no_mtp)
