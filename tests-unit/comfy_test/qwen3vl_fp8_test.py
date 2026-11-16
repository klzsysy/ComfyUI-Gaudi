import pytest
import torch

import comfy.ops
from comfy import sd1_clip
from comfy.text_encoders import qwen3vl


@pytest.mark.parametrize('device,formats,native', [
    ('hpu', ['float8_e4m3fn'], True),
    ('cpu', ['float8_e4m3fn'], False),
    ('hpu', ['int8_tensorwise'], False),
])
def test_native_fp8_loader_selection(monkeypatch, device, formats, native):
    captured = {}
    monkeypatch.setattr(sd1_clip.SDClipModel, '__init__', lambda self, **kwargs: captured.update(kwargs))
    options = {'load_device': torch.device(device), 'quantization_metadata': {'formats': formats}}
    qwen3vl.Qwen3VLClipModel(dtype=torch.bfloat16, model_options=options)
    operations = captured['model_options'].get('custom_operations')
    assert (operations is not None) == native
    if native:
        assert operations.hpu_fp8
        assert not operations._full_precision_mm
    assert 'custom_operations' not in options


def test_custom_operations_preserved(monkeypatch):
    captured = {}
    monkeypatch.setattr(sd1_clip.SDClipModel, '__init__', lambda self, **kwargs: captured.update(kwargs))
    options = {'load_device': torch.device('hpu'), 'quantization_metadata': {'formats': ['float8_e4m3fn']},
               'custom_operations': comfy.ops.manual_cast}
    qwen3vl.Qwen3VLClipModel(dtype=torch.bfloat16, model_options=options)
    assert captured['model_options'] is options
