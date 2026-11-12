import pytest
import torch
import comfy_kitchen

import comfy.model_management
from comfy.text_encoders import hpu_kv_attention
from comfy_kitchen.backends.eager.rope import apply_rope_split_half


@pytest.fixture
def hpu_rope(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(comfy.model_management.args, "disable_comfy_compiler", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    compiled_rope = hpu_kv_attention._compiled_rope
    torch._dynamo.reset()
    compiled_rope.cache_clear()
    yield
    torch._dynamo.reset()
    compiled_rope.cache_clear()


def inputs(length, dtype=torch.bfloat16):
    q = torch.randn(2, length, 16, 256, dtype=dtype, device="hpu").transpose(1, 2)[..., :64]
    k = torch.randn(2, length, 4, 256, dtype=dtype, device="hpu").transpose(1, 2)[..., :64]
    frequencies = torch.randn(1, 1, length, 32, 2, 2, device="hpu")
    return q, k, frequencies


def test_short_rope_preserves_noncontiguous_inputs_and_dynamic_lengths(hpu_rope):
    with torch.no_grad():
        for length in (1, 8, 4, 2, 1):
            values = inputs(length)
            before = [value.cpu() for value in values]
            expected = [value.cpu() for value in apply_rope_split_half(*values)]
            actual = comfy_kitchen.apply_rope_split_half(*values)
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0)
            for value, reference in zip(values, before):
                torch.testing.assert_close(value.cpu(), reference, atol=0, rtol=0)
    assert hpu_kv_attention._compiled_rope.cache_info().currsize == 1


@pytest.mark.parametrize("mode", ["disabled", "training", "grad", "prefill", "float32"])
def test_rope_preserves_eager_paths(hpu_rope, monkeypatch, mode):
    monkeypatch.setattr(comfy.model_management.args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    monkeypatch.setattr(hpu_kv_attention, "_compiled_rope", lambda: pytest.fail("Rope compiler should be bypassed"))
    values = inputs(17 if mode == "prefill" else 4, torch.float32 if mode == "float32" else torch.bfloat16)
    with torch.set_grad_enabled(mode == "grad"):
        expected = apply_rope_split_half(*values)
        actual = comfy_kitchen.apply_rope_split_half(*values)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), reference.cpu(), atol=0, rtol=0)
