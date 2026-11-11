import torch


def rms_adaln(x, scale, shift, eps):
    weight = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    normalized = torch.ops.hpu.rms_norm(x, weight, eps)[0]
    return normalized * (1 + scale) + shift


def blockwise_coefficients(pe):
    # BLOCKWISE shares both coefficients between the two halves.
    return bool(((pe[..., 0, 0] == pe[..., 1, 1]) & (pe[..., 0, 1] == -pe[..., 1, 0])).all().item())


def _rope(x, pe, blockwise=False):
    if blockwise:
        cos = torch.cat((pe[..., 0, 0], pe[..., 1, 1]), dim=-1)
        sin = torch.cat((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1)
        return torch.ops.hpu.rotary_pos_embedding(x.to(pe.dtype), sin, cos, None, 0, 0).to(x.dtype)
    # Pairwise mode preserves all four coefficients; BLOCKWISE reuses the first half.
    cos = torch.stack((pe[..., 0, 0], pe[..., 1, 1]), dim=-1).flatten(-2)
    sin = torch.stack((-pe[..., 0, 1], pe[..., 1, 0]), dim=-1).flatten(-2)
    paired = x.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2)
    out = torch.ops.hpu.rotary_pos_embedding(paired.to(pe.dtype), sin, cos, None, 0, 1).to(x.dtype)
    return out.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)


def rms_rope_split_half(q, k, pe, q_scale, k_scale, eps, blockwise=False):
    # Retain the existing Q/K normalization and BF16 rounding before FP32 RoPE.
    q = torch.nn.functional.rms_norm(q, (q.shape[-1],), q_scale, eps)
    k = torch.nn.functional.rms_norm(k, (k.shape[-1],), k_scale, eps)
    return _rope(q, pe, blockwise), _rope(k, pe, blockwise)
