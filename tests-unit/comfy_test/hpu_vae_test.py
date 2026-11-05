import pytest
import torch

from comfy.ldm.wan import vae2_2


@pytest.mark.parametrize("device", ["cpu", "hpu"])
@pytest.mark.parametrize("strip_elems", [64, 2 ** 24])
def test_wan_upsample_residual_after_convolution(device, strip_elems, monkeypatch):
    if device == "hpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("Requires an HPU")
    monkeypatch.setattr(vae2_2, "STRIP_ELEMS", strip_elems)
    x = torch.arange(64, dtype=torch.float32).reshape(1, 4, 1, 4, 4)
    shortcut = vae2_2.DupUp3D(4, 4, factor_t=2, factor_s=2)
    expected = 1 + shortcut(x, first_chunk=True)
    # A convolution result has HPU layout metadata that a plain tensor lacks.
    weight = torch.eye(4).reshape(4, 4, 1, 1).to(device)
    main = torch.nn.functional.conv2d(torch.ones(1, 4, 8, 8, device=device), weight).unsqueeze(2)
    actual = vae2_2.strip_apply(lambda s: shortcut(s, first_chunk=True), x.to(device), scale=2, halo=0, out=main)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
