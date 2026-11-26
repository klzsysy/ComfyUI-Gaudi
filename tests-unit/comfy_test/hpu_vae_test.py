import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.wan import vae as wan_vae
from comfy.ldm.wan import vae2_2


def _has_hpu():
    return hasattr(torch, "hpu") and torch.hpu.is_available()


def _naive_dup_up_3d(x: torch.Tensor, in_c: int, out_c: int, ft: int, fs: int, first_chunk: bool) -> torch.Tensor:
    repeats = out_c * (ft * fs * fs) // in_c
    x = x.repeat_interleave(repeats, dim=1)
    x = x.view(x.size(0), out_c, ft, fs, fs, x.size(2), x.size(3), x.size(4))
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(x.size(0), out_c, x.size(2) * ft, x.size(4) * fs, x.size(6) * fs)
    if first_chunk:
        x = x[:, :, ft - 1:, :, :]
    return x


def _naive_avg_down_3d(x: torch.Tensor, in_c: int, out_c: int, ft: int, fs: int) -> torch.Tensor:
    factor = ft * fs * fs
    group_size = in_c * factor // out_c
    pad_t = (ft - x.shape[2] % ft) % ft
    x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
    B, C, T, H, W = x.shape
    x = x.view(B, C, T // ft, ft, H // fs, fs, W // fs, fs)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    x = x.view(B, out_c, group_size, T // ft, H // fs, W // fs)
    return x.mean(dim=2)


@pytest.mark.parametrize("device", ["cpu", "hpu"])
@pytest.mark.parametrize("strip_elems", [64, 2 ** 24])
def test_wan_upsample_residual_after_convolution(device, strip_elems, monkeypatch):
    if device == "hpu" and not _has_hpu():
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


@pytest.mark.parametrize("device", ["cpu", "hpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dup_up_3d_bit_exact(device, dtype):
    if device == "hpu" and not _has_hpu():
        pytest.skip("Requires an HPU")
    torch.manual_seed(0)
    configs = [
        (16, 16, 2, 2),
        (32, 16, 2, 2),
        (32, 16, 1, 2),
        (64, 16, 2, 2),
        (16, 32, 1, 2),
        (16, 16, 1, 2),
        (16, 16, 1, 1),
    ]
    with torch.inference_mode():
        for in_c, out_c, ft, fs in configs:
            mod = vae2_2.DupUp3D(in_c, out_c, ft, fs).eval()
            for B in (1, 2):
                for T in (1, 2, 3):
                    for first_chunk in (True, False):
                        x = torch.randn(B, in_c, T, 5, 7, dtype=dtype, device=device)
                        ref = _naive_dup_up_3d(x.cpu(), in_c, out_c, ft, fs, first_chunk)
                        got = mod(x, first_chunk=first_chunk).cpu()
                        assert torch.equal(got, ref), f"Mismatch {(device, dtype, in_c, out_c, ft, fs, B, T, first_chunk)}"


@pytest.mark.parametrize("device", ["cpu", "hpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_avg_down_3d_exact(device, dtype):
    if device == "hpu" and not _has_hpu():
        pytest.skip("Requires an HPU")
    torch.manual_seed(1)
    configs = [
        (16, 16, 1, 2),
        (16, 32, 1, 2),
        (16, 32, 2, 2),
        (32, 32, 2, 2),
        (32, 32, 1, 1),
    ]
    with torch.inference_mode():
        for in_c, out_c, ft, fs in configs:
            mod = vae2_2.AvgDown3D(in_c, out_c, ft, fs).eval()
            for T in (1, 2, 4):
                x = torch.randn(1, in_c, T, 8, 8, dtype=dtype, device=device)
                ref = _naive_avg_down_3d(x.cpu(), in_c, out_c, ft, fs).float()
                got = mod(x).cpu().float()
                atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(got, ref, rtol=1e-2 if dtype == torch.bfloat16 else 1e-5, atol=atol)


def test_hpu_tpc_vae_rmsnorm_silu():
    if not _has_hpu() or vae2_2._hpu_tpc is None or not vae2_2._hpu_tpc.available:
        pytest.skip("Requires HPU with compiled TPC kernels")
    tpc = vae2_2._hpu_tpc
    torch.manual_seed(2)
    cases = [
        (1, 144, 1, 0, 1, 1),
        (1, 96, 7 * 13, 2 * 13, 3 * 13, 1),
        (1, 96, 7 * 13, 1 * 13, 5 * 13, 0),
        (2, 288, 19 * 37, 3 * 37, 11 * 37, 1),
        (1, 1152, 120 * 68, 0, 120 * 68, 1),
        (1, 1152, 120 * 68, 0, 120 * 68, 0),
    ]
    with torch.inference_mode():
        for B, C, n_full, noff, nlen, do_silu in cases:
            x = torch.randn(B, C, n_full, dtype=torch.bfloat16, device="hpu") * 2.0
            gamma = torch.linspace(-1.5, 2.5, C, dtype=torch.bfloat16, device="hpu")
            scale = math.sqrt(C)
            got = tpc.vae_rmsnorm_silu_bf16(x, gamma, noff, nlen, do_silu, scale).cpu().float()
            xs = x[:, :, noff:noff + nlen].cpu().float()
            ref = F.normalize(xs, dim=1) * scale * gamma.cpu().float().view(1, C, 1)
            if do_silu:
                ref = F.silu(ref)
            ref = ref.to(torch.bfloat16).float()
            assert torch.isfinite(got).all()
            assert (got - ref).abs().mean().item() < 1e-2
            assert (got - ref).abs().max().item() < 0.1
        # Zero input stays finite zero
        xz = torch.zeros(1, 144, 300, dtype=torch.bfloat16, device="hpu")
        gz = torch.ones(144, dtype=torch.bfloat16, device="hpu")
        oz = tpc.vae_rmsnorm_silu_bf16(xz, gz, 0, 300, 1, math.sqrt(144)).cpu()
        assert torch.equal(oz, torch.zeros_like(oz))


def test_hpu_tpc_vae_dupup2d_add_and_conv_layout():
    if not _has_hpu() or vae2_2._hpu_tpc is None or not vae2_2._hpu_tpc.available:
        pytest.skip("Requires HPU with compiled TPC kernels")
    tpc = vae2_2._hpu_tpc
    torch.manual_seed(3)
    cases = [
        (1, 16, 16, 2, 0, 5, 1),
        (1, 16, 16, 2, 0, 7, 3),
        (1, 32, 16, 2, 1, 9, 17),
        (1, 32, 16, 1, 2, 9, 83),
        (1, 64, 16, 2, 3, 5, 68),
        (2, 48, 24, 1, 2, 11, 136),
    ]
    with torch.inference_mode():
        dummy = torch.empty(1, 1, 1, 1, dtype=torch.bfloat16, device="hpu")
        for B, cin, cout, ft, mode, H, W in cases:
            x = torch.randn(B, cin, 1, H, W, dtype=torch.bfloat16, device="hpu")
            ref_dup = _naive_dup_up_3d(x.cpu(), cin, cout, ft, 2, True).squeeze(2)
            out0 = tpc.vae_dupup2d_add_bf16(x.squeeze(2), dummy, cout, mode, 0).cpu()
            assert torch.equal(out0, ref_dup)
            # Also validate with a Conv2d output as main_t (HPU internal conv layout metadata).
            w_conv = torch.randn(cout, cout, 1, 1, dtype=torch.bfloat16, device="hpu") * 0.25
            main_conv = F.conv2d(torch.randn(B, cout, 2 * H, 2 * W, dtype=torch.bfloat16, device="hpu"), w_conv)
            out1 = tpc.vae_dupup2d_add_bf16(x.squeeze(2), main_conv, cout, mode, 1).cpu()
            assert torch.equal(out1, main_conv.cpu() + ref_dup)


def _init_small_vae(model: nn.Module):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "gamma" in name:
                param.fill_(1.0)
            elif "bias" in name:
                param.zero_()
            else:
                param.normal_(0.0, 0.05)


def test_wan_vae_hpu_tpc_parity_and_safety_gates(monkeypatch):
    if not _has_hpu() or vae2_2._hpu_tpc is None or not vae2_2._hpu_tpc.available:
        pytest.skip("Requires HPU with compiled TPC kernels")
    torch.manual_seed(4)
    # Image-mode VAE (matches Qwen-Image 2.1 architecture: patch_size=1, temporal_kernel=1)
    model_img = vae2_2.WanVAE(
        dim=8,
        dec_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        image_channels=4,
        patch_size=1,
        temporal_kernel=1,
    )
    _init_small_vae(model_img)
    model_img = model_img.to(device="hpu", dtype=torch.bfloat16).eval()

    # Video-mode Wan 2.2 VAE (temporal_kernel=3 for multi-frame causal cache)
    model_vid = vae2_2.WanVAE(
        dim=8,
        dec_dim=8,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        image_channels=3,
        patch_size=2,
        temporal_kernel=3,
    )
    _init_small_vae(model_vid)
    model_vid = model_vid.to(device="hpu", dtype=torch.bfloat16).eval()

    # Force multi-strip execution even on a small spatial size.
    monkeypatch.setattr(vae2_2, "STRIP_ELEMS", 4096)

    z_img = torch.randn(1, 4, 6, 7, dtype=torch.bfloat16, device="hpu")
    x_img = torch.randn(1, 4, 48, 56, dtype=torch.bfloat16, device="hpu")
    z_vid = torch.randn(1, 4, 2, 4, 4, dtype=torch.bfloat16, device="hpu")
    x_vid = torch.randn(1, 3, 5, 64, 64, dtype=torch.bfloat16, device="hpu")

    with torch.inference_mode():
        dec_tpc = model_img.decode(z_img).cpu().float()
        enc_tpc = model_img.encode(x_img).cpu().float()
        dec_vid_tpc = model_vid.decode(z_vid).cpu().float()
        enc_vid_tpc = model_vid.encode(x_vid).cpu().float()

    # Compare against TPC-disabled fallback path.
    monkeypatch.setattr(vae2_2._hpu_tpc, "available", False)
    if wan_vae._hpu_tpc is not None:
        monkeypatch.setattr(wan_vae._hpu_tpc, "available", False)
    with torch.inference_mode():
        dec_fb = model_img.decode(z_img).cpu().float()
        enc_fb = model_img.encode(x_img).cpu().float()
        dec_vid_fb = model_vid.decode(z_vid).cpu().float()
        enc_vid_fb = model_vid.encode(x_vid).cpu().float()

    assert torch.isfinite(dec_tpc).all() and (dec_tpc - dec_fb).abs().mean().item() < 1e-2
    assert torch.isfinite(enc_tpc).all() and (enc_tpc - enc_fb).abs().mean().item() < 1e-2
    assert torch.isfinite(dec_vid_tpc).all() and (dec_vid_tpc - dec_vid_fb).abs().mean().item() < 1e-2
    assert torch.isfinite(enc_vid_tpc).all() and (enc_vid_tpc - enc_vid_fb).abs().mean().item() < 1e-2

    # Verify forward hooks on bypassed submodules still fire when TPC is re-enabled.
    monkeypatch.setattr(vae2_2._hpu_tpc, "available", True)
    if wan_vae._hpu_tpc is not None:
        monkeypatch.setattr(wan_vae._hpu_tpc, "available", True)
    hook_hits = {"rms": 0, "dup": 0, "res_rms": 0}
    h1 = model_img.decoder.head[0].register_forward_hook(
        lambda m, i, o: hook_hits.__setitem__("rms", hook_hits["rms"] + 1)
    )
    h2 = model_img.decoder.upsamples[0].avg_shortcut.register_forward_hook(
        lambda m, i, o: hook_hits.__setitem__("dup", hook_hits["dup"] + 1)
    )
    h3 = model_img.decoder.middle[0].residual[0].register_forward_hook(
        lambda m, i, o: hook_hits.__setitem__("res_rms", hook_hits["res_rms"] + 1)
    )
    try:
        with torch.inference_mode():
            _ = model_img.decode(z_img)
    finally:
        h1.remove()
        h2.remove()
        h3.remove()
    assert hook_hits["rms"] == 1
    assert hook_hits["dup"] >= 1
    assert hook_hits["res_rms"] >= 1

    # Verify grad-enabled path works and produces finite gradients.
    z_grad = z_img.clone().detach().requires_grad_(True)
    out_grad = model_img.decode(z_grad)
    out_grad.sum().backward()
    assert z_grad.grad is not None and torch.isfinite(z_grad.grad.cpu()).all()
