import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.text_encoders import hpu_gated_delta
from comfy.text_encoders.qwen35 import GatedDeltaNet, Qwen35Config, torch_chunk_gated_delta_rule


def test_prefill_shapes_and_recurrent_state():
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("Requires an HPU")
    torch.manual_seed(74)
    compiled = hpu_gated_delta.compiled_prefill(torch_chunk_gated_delta_rule)
    # Return to the first shape with new data to expose captured-input reuse.
    with torch.no_grad():
        for length, final_state in [(1970, True), (23, False), (307, False), (65, True),
                                    (17, False), (31, False), (47, False), (63, False),
                                    (129, False), (255, False), (511, False), (23, False)]:
            batch = 2 if length == 65 else 1
            dtype = torch.float32 if length in (307, 511) else torch.bfloat16
            q, k, v = [torch.randn(batch, length, 2, 128, device="hpu", dtype=dtype) * 0.1 for _ in range(3)]
            g = -torch.rand(batch, length, 2, device="hpu")
            beta = torch.rand(batch, length, 2, device="hpu", dtype=dtype)
            initial = torch.randn(batch, 2, 128, 128, device="hpu") * 0.01 if final_state else None
            saved = initial.cpu().clone() if initial is not None else None
            kwargs = dict(g=g, beta=beta, initial_state=initial, output_final_state=final_state)
            reference = torch_chunk_gated_delta_rule(q, k, v, **kwargs)
            actual = compiled(q, k, v, **kwargs)
            for a, b in zip(actual, reference):
                if b is None:
                    assert a is None
                    continue
                assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
                a, b = a.cpu().float(), b.cpu().float()
                assert a.isfinite().all()
                tolerance = 1e-5 if dtype == torch.float32 else 1e-3
                assert (a - b).square().mean().sqrt() / b.square().mean().sqrt() < tolerance
            if saved is not None:
                torch.testing.assert_close(initial.cpu(), saved, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["cpu", "grad", "training", "disabled"])
def test_prefill_keeps_eager_paths(monkeypatch, mode):
    device = "cpu" if mode == "cpu" else "hpu"
    if device == "hpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("Requires an HPU")
    monkeypatch.setattr(hpu_gated_delta, "compiled_prefill", lambda *args: pytest.fail("Unexpected prefill compile"))
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    monkeypatch.setattr(comfy.model_management.args, "disable_comfy_compiler", mode == "disabled")
    dtype = torch.float32 if mode == "cpu" else torch.bfloat16
    layer = GatedDeltaNet(Qwen35Config(hidden_size=32, linear_num_key_heads=1, linear_num_value_heads=1),
                         device=device, dtype=dtype, ops=comfy.ops.manual_cast)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.01)
    with torch.set_grad_enabled(mode == "grad"):
        value, state = layer(torch.ones(1, 23, 32, device=device, dtype=dtype))
    assert value.shape == (1, 23, 32) and value.isfinite().all() and state is None
