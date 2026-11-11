import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
