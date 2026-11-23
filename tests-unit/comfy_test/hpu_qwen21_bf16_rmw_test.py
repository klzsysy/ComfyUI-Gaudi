import math

import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.modules import hpu_attention
from comfy.ldm.modules.attention import attention_pytorch
from comfy.ldm.qwen_image21 import hpu as qwen_hpu
from comfy.ldm.qwen_image21 import model as qwen


@pytest.fixture
def hpu_bf16_env(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    from comfy.text_encoders import hpu_qwen35_tpc

    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 TPC library not built")
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(qwen, "optimized_attention", attention_pytorch)
    torch._dynamo.reset()
    yield
    hpu_attention.release_inflight()
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "kv_len,expected_blocks",
    [
        (1, 1),
        (512, 1),
        (4096, 1),
        (8607, 1),
        (9310, 1),
        (10240, 1),
        (10241, 2),
        (12731, 2),
        (13434, 2),
        (16767, 2),
        (17470, 2),
        (20480, 2),
        (20481, 3),
        (21587, 3),
        (30720, 3),
        (30721, 4),
    ],
)
def test_bf16_expmax_layout_invariants(kv_len, expected_blocks):
    num_blocks, total_k_pad, blocks = hpu_attention.rmw_bf16_expmax_layout(kv_len)
    assert num_blocks == expected_blocks
    assert len(blocks) == expected_blocks
    assert total_k_pad == ((kv_len + 511) // 512) * 512
    assert 0 <= total_k_pad - kv_len <= 511
    assert sum(valid_b for _, valid_b, _ in blocks) == kv_len
    offset = 0
    for idx, (off, valid_b, kb_pad) in enumerate(blocks):
        assert off == offset
        assert 1 <= valid_b <= kb_pad <= 10240
        assert kb_pad % 512 == 0
        if idx < len(blocks) - 1:
            assert valid_b == kb_pad
        else:
            assert 0 <= kb_pad - valid_b <= 511
        tile = hpu_attention._rmw_bf16_expmax_tile(4096, kb_pad)
        assert tile is not None and tile >= 384
        offset += kb_pad
    assert offset == total_k_pad
    assert hpu_attention.rmw_bf16_expmax_supported(4096, kv_len, 32)


def _merge_bf16_expmax_result(res, queries):
    if len(res) == 2:
        o0, s0 = res
        return (o0.float() / s0[:, :queries].unsqueeze(-1)).to(o0.dtype)
    o0, s0, m0, o1, s1, m1 = res
    s0, m0 = s0[:, :queries], m0[:, :queries]
    s1, m1 = s1[:, :queries], m1[:, :queries]
    mm = torch.maximum(m0, m1)
    w0 = torch.exp(m0 - mm)
    w1 = torch.exp(m1 - mm)
    denom = (w0 * s0 + w1 * s1).unsqueeze(-1)
    return ((o0.float() * w0.unsqueeze(-1) + o1.float() * w1.unsqueeze(-1)) / denom).to(o0.dtype)


@pytest.mark.parametrize(
    "heads,queries,kv_len,expected_blocks,shift_second_half",
    [
        (8, 4096, 5000, 1, False),
        (8, 4100, 11500, 2, False),
        (8, 4100, 11500, 2, True),
        (4, 4100, 21500, 3, True),
    ],
)
def test_bf16_expmax_attention_accuracy_and_workspace(
    hpu_bf16_env, heads, queries, kv_len, expected_blocks, shift_second_half
):
    torch.manual_seed(2026 + kv_len)
    num_blocks, total_k_pad, _ = hpu_attention.rmw_bf16_expmax_layout(kv_len)
    assert num_blocks == expected_blocks
    q = (torch.randn((heads, queries, 128), dtype=torch.bfloat16) * 0.9).to("hpu")
    k = (torch.randn((heads, kv_len, 128), dtype=torch.bfloat16) * 0.9).to("hpu")
    v = (torch.randn((heads, kv_len, 128), dtype=torch.bfloat16) * 0.5).to("hpu")
    if shift_second_half:
        # Inject a large cross-block logit peak in the later K-block to test exact max_out scaling.
        k[:, kv_len // 2 : kv_len // 2 + 16, :] = q[:, :16, :] * 2.5

    sc = torch.tensor(hpu_attention.BF16_EXPMAX_QK_SCALE, dtype=torch.bfloat16, device="hpu")
    qh = (q * sc).contiguous()
    pad = total_k_pad - kv_len
    k_pad = torch.cat((k, torch.zeros((heads, pad, 128), dtype=torch.bfloat16, device="hpu")), dim=1) if pad else k
    v_pad = torch.cat((v, torch.zeros((heads, pad, 128), dtype=torch.bfloat16, device="hpu")), dim=1) if pad else v
    kh = (k_pad * sc).contiguous()
    vh = v_pad.contiguous()
    qh_snap, kh_snap, vh_snap = qh.cpu().clone(), kh.cpu().clone(), vh.cpu().clone()

    with torch.no_grad():
        res = hpu_attention.rmw_bf16_expmax_attention_prepared(qh, kh, vh, kv_len)
        actual = _merge_bf16_expmax_result(res, queries).cpu().float()
        rt = hpu_attention._bf16_expmax_runtime()
        assert rt.last_num_blocks == expected_blocks
        assert rt.last_workspace <= 256
        # Inputs must remain completely untouched.
        assert torch.equal(qh.cpu(), qh_snap)
        assert torch.equal(kh.cpu(), kh_snap)
        assert torch.equal(vh.cpu(), vh_snap)

        # Reference in FP64 on CPU on a deterministic slice of queries to keep unit test fast.
        idx = torch.linspace(0, queries - 1, 64).long()
        q64 = q[:, idx].cpu().double()
        k64 = k.cpu().double()
        v64 = v.cpu().double()
        scores = torch.matmul(q64, k64.transpose(-1, -2)) / math.sqrt(128.0)
        ref = torch.matmul(torch.softmax(scores, dim=-1), v64).float()
        got = actual[:, idx]
        assert torch.isfinite(actual).all()
        rrmse = (got - ref).norm() / ref.norm()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        assert rrmse < 0.015, f"rrmse={rrmse.item():.4f}"
        assert cos > 0.9998, f"cos={cos.item():.6f}"


def test_bf16_expmax_isolation_and_async_stress(hpu_bf16_env):
    torch.manual_seed(77)
    heads, queries, kv_len = 8, 4096, 5000
    _, total_k_pad, _ = hpu_attention.rmw_bf16_expmax_layout(kv_len)
    pad = total_k_pad - kv_len
    sc = torch.tensor(hpu_attention.BF16_EXPMAX_QK_SCALE, dtype=torch.bfloat16, device="hpu")

    def make_set(seed):
        g = torch.Generator().manual_seed(seed)
        q = (torch.randn((heads, queries, 128), generator=g, dtype=torch.bfloat16) * 0.7).to("hpu")
        k = (torch.randn((heads, kv_len, 128), generator=g, dtype=torch.bfloat16) * 0.7).to("hpu")
        v = (torch.randn((heads, kv_len, 128), generator=g, dtype=torch.bfloat16) * 0.5).to("hpu")
        z = torch.zeros((heads, pad, 128), dtype=torch.bfloat16, device="hpu")
        return (q * sc).contiguous(), (torch.cat((k, z), dim=1) * sc).contiguous(), torch.cat((v, z), dim=1).contiguous()

    set_a = make_set(1)
    set_b = make_set(2)
    with torch.no_grad():
        out_a1 = _merge_bf16_expmax_result(hpu_attention.rmw_bf16_expmax_attention_prepared(*set_a, kv_len), queries).cpu()
        out_b1 = _merge_bf16_expmax_result(hpu_attention.rmw_bf16_expmax_attention_prepared(*set_b, kv_len), queries).cpu()
        assert not torch.equal(out_a1, out_b1)
        out_a2 = _merge_bf16_expmax_result(hpu_attention.rmw_bf16_expmax_attention_prepared(*set_a, kv_len), queries).cpu()
        assert torch.equal(out_a1, out_a2)

        # 24-iteration alternating async stress to verify _InflightTracker safety.
        outs = []
        for i in range(24):
            inp = set_a if (i % 2 == 0) else set_b
            res = hpu_attention.rmw_bf16_expmax_attention_prepared(*inp, kv_len)
            outs.append(_merge_bf16_expmax_result(res, queries))
        hpu_attention.release_inflight()
        for i, out in enumerate(outs):
            got = out.cpu()
            assert torch.isfinite(got).all()
            assert torch.equal(got, out_a1 if (i % 2 == 0) else out_b1)


def _make_full_bf16_block():
    torch.manual_seed(91)
    block = qwen.QwenImage21TransformerBlock(
        4096, 32, 128, mlp_ratio=3, fused_mlp=True, dtype=torch.bfloat16, device="hpu", operations=comfy.ops.manual_cast
    )
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "norm" in name:
                p.copy_((torch.randn(p.shape) * 0.05 + 1.0).to(device="hpu", dtype=torch.bfloat16))
            else:
                p.copy_((torch.randn(p.shape) * 0.015).to(device="hpu", dtype=torch.bfloat16))
    return block


def _make_block_inputs(queries, prefix_len, seed=101):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((1, queries, 4096), generator=g, dtype=torch.bfloat16) * 0.5).to("hpu")
    pk = (torch.randn((1, prefix_len, 32, 128), generator=g, dtype=torch.bfloat16) * 0.5).to("hpu")
    pv = (torch.randn((1, prefix_len, 32, 128), generator=g, dtype=torch.bfloat16) * 0.5).to("hpu")
    mod = tuple(
        (
            (torch.randn((1, 1, 4096), generator=g, dtype=torch.bfloat16) * 0.1).to("hpu"),
            (torch.randn((1, 1, 4096), generator=g, dtype=torch.bfloat16) * 0.1).to("hpu"),
        )
        for _ in range(4)
    ) + (torch.zeros((1, 1, 4096), dtype=torch.bfloat16, device="hpu"),)
    angle = torch.randn((1, queries, 1, 64), generator=g)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, queries, 1, 64, 2, 2).to("hpu")
    return x, mod, pe, pk, pv


@pytest.mark.parametrize(
    "queries,prefix_len",
    [
        (4096, 512),   # pad_len == 0 (B_k = 1)
        (4096, 447),   # pad_len > 1  (B_k = 1)
        (4100, 6500),  # non-32-multiple Q + B_k = 2 (_bf16_post2)
    ],
)
def test_qwen21_bf16_block_matches_baseline(hpu_bf16_env, queries, prefix_len):
    block = _make_full_bf16_block()
    x, mod, pe, pk, pv = _make_block_inputs(queries, prefix_len)
    with torch.no_grad():
        opts_ref = {"hpu_fast_attention": False, "hpu_bf16_rmw": False}
        ref = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts_ref, []).cpu().float()

        opts_opt = {"hpu_fast_attention": False, "hpu_bf16_rmw": True, "_qwen21_hpu_pe_cs": qwen_hpu.pack_pe_cs(pe)}
        pending = []
        actual = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts_opt, pending).cpu().float()
        assert len(pending) == 1
        assert torch.isfinite(actual).all()
        rrmse = (actual - ref).norm() / ref.norm()
        cos = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
        assert rrmse < 0.015, f"rrmse={rrmse.item():.4f}"
        assert cos > 0.9998, f"cos={cos.item():.6f}"


def test_qwen21_bf16_block_dynamic_prefix_no_recompile_cliff(hpu_bf16_env):
    block = _make_full_bf16_block()
    queries = 4096
    x, mod, pe, _, _ = _make_block_inputs(queries, 447)
    pe_cs = qwen_hpu.pack_pe_cs(pe)
    # 10 distinct prefix lengths (> default dynamo recompile_limit=8) must succeed via mark_dynamic.
    pending = []
    with torch.no_grad():
        for plen in (447, 1150, 300, 611, 777, 888, 999, 256, 512, 447):
            pk = (torch.randn((1, plen, 32, 128), dtype=torch.bfloat16) * 0.2).to("hpu")
            pv = (torch.randn((1, plen, 32, 128), dtype=torch.bfloat16) * 0.2).to("hpu")
            opts = {"hpu_fast_attention": False, "hpu_bf16_rmw": True, "_qwen21_hpu_pe_cs": pe_cs}
            out = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts, pending)
            assert torch.isfinite(out.cpu()).all()
    assert len(pending) <= 2


@pytest.mark.parametrize(
    "reason",
    ["pad1", "pk1", "small_q", "opt_out", "disabled_compiler", "lora_attn", "lora_mlp", "hook", "unfused_mlp", "force_forward"],
)
def test_qwen21_bf16_block_clean_fallback(hpu_bf16_env, monkeypatch, reason):
    block = _make_full_bf16_block()
    queries = 2048 if reason == "small_q" else 4096
    prefix_len = 511 if reason == "pad1" else (1 if reason == "pk1" else 447)
    x, mod, pe, pk, pv = _make_block_inputs(queries, prefix_len)
    options = {"hpu_fast_attention": False, "hpu_bf16_rmw": reason != "opt_out"}

    if reason == "disabled_compiler":
        monkeypatch.setattr(args, "disable_comfy_compiler", True)
    elif reason == "lora_attn":
        block.attn.to_q.weight_function = [lambda w: w]
    elif reason == "lora_mlp":
        block.img_mlp.gate_up.weight_function = [lambda w: w]
    elif reason == "hook":
        block.img_mlp.out.register_forward_hook(lambda m, inp, out: out)
    elif reason == "unfused_mlp":
        block.img_mlp.fused = False
    elif reason == "force_forward":
        block.attn.to_k.comfy_force_forward = True

    called = {"fallback": False}
    monkeypatch.setattr(hpu_attention, "rmw_bf16_expmax_attention_prepared", lambda *a, **kw: pytest.fail("must fall back"))
    monkeypatch.setattr(block, "forward", lambda inp, *a, **kw: (called.__setitem__("fallback", True), inp)[1])

    with torch.no_grad():
        pending = []
        out = block._forward_with_prefix(x, mod, pe, pk, pv, options, pending)
        assert out is x
        assert called["fallback"]
        assert pending == []
