import pytest
import torch

from comfy.text_encoders.llama import BaseGenerate


@pytest.mark.parametrize("device", ["cpu", "hpu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("top_k", [0, 1, 64])
def test_sample_token_vocabulary_indices(device, dtype, top_k):
    if device == "hpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("Requires an HPU")
    # Qwen3.5 vocabulary size, with different allowed tokens in each batch row.
    logits = torch.full((2, 248320), -1000.0, dtype=dtype)
    allowed = torch.tensor([[100003, 200007], [170009, 248319]])
    logits.scatter_(1, allowed, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=dtype))
    sampler = BaseGenerate()
    logits = logits.to(device)

    def sample():
        generator = torch.Generator(device=device).manual_seed(42)
        return sampler.sample_token(logits, 0.7, top_k, 0.95, 0.05, 1.0, [], generator)

    result = sample()
    assert result.device.type == device
    assert result.dtype == torch.long
    assert result.shape == (2, 1)
    assert (result.cpu() == allowed).any(dim=1).all()
    torch.testing.assert_close(result.cpu(), sample().cpu(), rtol=0, atol=0)
    if top_k == 1:
        torch.testing.assert_close(result.cpu(), allowed[:, 1:], rtol=0, atol=0)
