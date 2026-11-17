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


def test_qwen_native_rope_matches_standard_rotations(hpu_rope):
    from comfy.text_encoders.llama import precompute_freqs_cis, rope_matrix
    with torch.no_grad():
        for position in (0, 17, 2803, 4712):
            q, k, _ = inputs(1)
            matrix = rope_matrix(precompute_freqs_cis(64, torch.tensor([[position]], device='hpu'), 10000000, device='hpu'))
            actual = hpu_kv_attention.qwen_rope(q, k, matrix)
            for source, output in zip((q, k), actual):
                values = source.cpu().double()
                frequencies = matrix.cpu().double()
                expected = torch.cat((values[..., :32] * frequencies[..., 0, 0] + values[..., 32:] * frequencies[..., 0, 1],
                                      values[..., 32:] * frequencies[..., 1, 1] + values[..., :32] * frequencies[..., 1, 0]), dim=-1)
                torch.testing.assert_close(output.cpu().double(), expected, atol=1e-5, rtol=.004)


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
