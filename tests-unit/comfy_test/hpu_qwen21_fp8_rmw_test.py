import json
import math

import pytest
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy.cli_args import args
from comfy.ldm.modules import hpu_attention
from comfy.ldm.modules.attention import attention_pytorch
from comfy.ldm.qwen_image21 import hpu as qwen_hpu
from comfy.ldm.qwen_image21 import model as qwen


@pytest.fixture
def hpu_fp8_env(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    from comfy.text_encoders import hpu_qwen35_tpc

    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 TPC library not built")
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(qwen, "optimized_attention", attention_pytorch)
    torch._dynamo.reset()
    yield hpu_qwen35_tpc
    hpu_attention.release_inflight()
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "kv_len,expected_blocks",
    [
        (1, 1),
        (256, 1),
        (512, 1),
        (4096, 1),
        (5246, 1),
        (8607, 1),
        (9310, 1),
        (12731, 1),
        (13390, 1),
        (14336, 1),
        (14337, 2),
        (16767, 2),
        (17470, 2),
        (20847, 2),
        (21550, 2),
        (28672, 2),
        (28673, 3),
        (30720, 3),
        (43008, 3),
        (43009, 4),
    ],
)
def test_qwen_fp8_expmax_layout_invariants(kv_len, expected_blocks):
    num_blocks, total_k_pad, blocks = hpu_attention.rmw_qwen_fp8_expmax_layout(kv_len)
    assert num_blocks == expected_blocks
    assert len(blocks) == expected_blocks
    assert total_k_pad == ((kv_len + 255) // 256) * 256
    assert 0 <= total_k_pad - kv_len <= 255
    assert sum(valid_b for _, valid_b, _ in blocks) == kv_len
    offset = 0
    for idx, (off, valid_b, kb_pad) in enumerate(blocks):
        assert off == offset
        assert 1 <= valid_b <= kb_pad <= 14336
        assert kb_pad % 256 == 0
        if idx < len(blocks) - 1:
            assert valid_b == kb_pad
        else:
            assert 0 <= kb_pad - valid_b <= 255
        tile = hpu_attention._rmw_qwen_fp8_expmax_tile(4096, kb_pad)
        assert tile is not None and tile >= 288
        offset += kb_pad
    assert offset == total_k_pad
    assert hpu_attention.rmw_qwen_fp8_expmax_supported(4096, kv_len, 32)


def _permute_v_all_lanes_cpu(v_fp8: torch.Tensor) -> torch.Tensor:
    heads, keys, dim = v_fp8.shape
    return v_fp8.view(heads, keys // 256, 2, 128, dim).permute(0, 1, 3, 2, 4).reshape(heads, keys, dim).contiguous()


def _merge_fp8_expmax_result(res, v_scales: torch.Tensor, queries: int) -> torch.Tensor:
    vs = v_scales.view(-1, 1, 1).float()
    if len(res) == 2:
        o0, s0 = res
        return (o0.float() * (vs / s0[:, :queries].unsqueeze(-1))).to(o0.dtype)
    o0, s0, m0, o1, s1, m1 = res
    s0, m0 = s0[:, :queries], m0[:, :queries]
    s1, m1 = s1[:, :queries], m1[:, :queries]
    mm = torch.maximum(m0, m1)
    w0 = torch.exp(m0 - mm)
    w1 = torch.exp(m1 - mm)
    denom = (w0 * s0 + w1 * s1).unsqueeze(-1)
    return ((o0.float() * w0.unsqueeze(-1) + o1.float() * w1.unsqueeze(-1)) * (vs / denom)).to(o0.dtype)


def test_qwen_fp8_tpc_kernels_accuracy(hpu_fp8_env):
    tpc = hpu_fp8_env
    torch.manual_seed(2026)
    queries, prefix = 1025, 311
    kv_len = queries + prefix
    k_pad = ((kv_len + 255) // 256) * 256

    q_raw = (torch.randn((queries, 4096), dtype=torch.bfloat16) * 0.7).to("hpu")
    k_raw = (torch.randn((queries, 4096), dtype=torch.bfloat16) * 0.7).to("hpu")
    v_raw = (torch.randn((queries, 4096), dtype=torch.bfloat16) * 0.6).to("hpu")
    pk = (torch.randn((prefix, 32, 128), dtype=torch.bfloat16) * 0.5).to("hpu")
    pv = (torch.randn((prefix, 32, 128), dtype=torch.bfloat16) * 0.5).to("hpu")
    nq = (torch.randn((128,), dtype=torch.bfloat16) * 0.05 + 1.0).to("hpu")
    nk = (torch.randn((128,), dtype=torch.bfloat16) * 0.05 + 1.0).to("hpu")
    angle = torch.randn((1, queries, 1, 64))
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, queries, 1, 64, 2, 2).to("hpu")
    pe_cs = qwen_hpu.pack_pe_cs_fp8(pe)
    q_scale = hpu_attention.QWEN_FP8_EXPMAX_Q_SCALE

    with torch.no_grad():
        qh, kh = tpc.qwen_qk_rope_fp8(q_raw, k_raw, pk, nq, nk, pe_cs, 3, 1e-6, q_scale, 1.0)
        cos_r, sin_r = qwen_hpu.pack_pe_cs(pe)
        q_ref = torch.ops.hpu.rotary_pos_embedding(F.rms_norm(q_raw.view(1, queries, 32, 128), (128,), nq, 1e-6).float(), sin_r, cos_r, None, 0, 1).to(torch.bfloat16)
        k_ref = torch.ops.hpu.rotary_pos_embedding(F.rms_norm(k_raw.view(1, queries, 32, 128), (128,), nk, 1e-6).float(), sin_r, cos_r, None, 0, 1).to(torch.bfloat16)
        qh_ref = (q_ref[0].transpose(0, 1).float() * q_scale).to(torch.bfloat16)
        kh_ref = torch.cat((pk, k_ref[0]), dim=0).transpose(0, 1)
        assert ((qh.cpu().float() - qh_ref.cpu().float()).norm() / qh_ref.cpu().float().norm()) < 0.005
        assert ((kh[:, :kv_len].cpu().float() - kh_ref.cpu().float()).norm() / kh_ref.cpu().float().norm()) < 0.005
        assert kh[:, kv_len:].abs().max().item() == 0.0

        vh_ref = torch.cat((pv, v_raw.view(queries, 32, 128)), dim=0).transpose(0, 1)
        vs_1d = vh_ref.abs().amax(dim=(-2, -1)).float().clamp_min(1e-12) / 240.0
        vs_64 = vs_1d.unsqueeze(-1).expand(32, 64).contiguous()
        vh_q256 = tpc.qwen_v_quant_fp8(v_raw, pv, vs_64, 3)
        vq_unp = vh_q256.cpu().view(32, k_pad // 256, 128, 2, 128).permute(0, 1, 3, 2, 4).reshape(32, k_pad, 128)
        v_deq = vq_unp[:, :kv_len].float() * vs_1d.cpu().view(32, 1, 1)
        assert ((v_deq - vh_ref.cpu().float()).norm() / vh_ref.cpu().float().norm()) < 0.035
        assert (vq_unp[:, kv_len:].view(torch.uint8) != 0).sum().item() == 0

        # res_adaln_quant (has_res=0 and has_res=1)
        x = (torch.randn((queries, 4096), dtype=torch.bfloat16) * 0.5).to("hpu")
        res = (torch.randn((queries, 4096), dtype=torch.bfloat16) * 0.4).to("hpu")
        gate = (torch.randn((4096,), dtype=torch.bfloat16) * 0.2).to("hpu")
        sc = (torch.randn((4096,), dtype=torch.bfloat16) * 0.2).to("hpu")
        sp1 = (1.0 + sc).contiguous()
        zero = torch.zeros((1, 1, 4096), dtype=torch.bfloat16, device="hpu")

        _, q0, s0 = tpc.qwen_res_adaln_quant(x, x, gate, sp1, 0, 1e-6)
        h0_ref = comfy.quant_ops.ck.adaln(x.unsqueeze(0), sc.view(1, 1, 4096), zero, 1e-6)[0].cpu().float()
        assert ((q0.cpu().float() * s0.cpu().unsqueeze(-1) - h0_ref).norm() / h0_ref.norm()) < 0.035

        mid, q1, s1 = tpc.qwen_res_adaln_quant(x, res, gate, sp1, 1, 1e-6)
        mid_ref = (x + res * gate).cpu().float()
        h1_ref = comfy.quant_ops.ck.adaln((x + res * gate).unsqueeze(0), sc.view(1, 1, 4096), zero, 1e-6)[0].cpu().float()
        assert ((mid.cpu().float() - mid_ref).norm() / mid_ref.norm()) < 0.002
        assert ((q1.cpu().float() * s1.cpu().unsqueeze(-1) - h1_ref).norm() / h1_ref.norm()) < 0.035

        # hnd_scale_sum_quant
        attn_t = (torch.randn((32, queries, 128), dtype=torch.bfloat16) * 20.0).to("hpu")
        stat_s = ((queries + 31) // 32) * 32
        sum_t = (torch.rand((32, stat_s), dtype=torch.float32) * 30.0 + 1.0).to("hpu")
        oq, os_ = tpc.qwen_hnd_scale_sum_quant(attn_t, vs_64, sum_t)
        hnd_ref = (attn_t.float() * (vs_1d.unsqueeze(-1) / sum_t[:, :queries]).unsqueeze(-1)).transpose(0, 1).reshape(queries, 4096).cpu()
        assert ((oq.cpu().float() * os_.cpu().unsqueeze(-1) - hnd_ref).norm() / hnd_ref.norm()) < 0.035


@pytest.mark.parametrize(
    "heads,queries,kv_len,expected_blocks,shift_second_half",
    [
        (8, 4096, 5000, 1, False),    # 5120 (tail_512)
        (8, 4096, 5246, 1, False),    # 5376 (tail_256)
        (8, 4096, 9310, 1, False),    # 9472 (2MP pos, tail_256, 27K VLM)
        (8, 4096, 13390, 1, False),   # 13568 (3MP pos, 1-block, tail_256, 27K VLM)
        (8, 4100, 15000, 2, False),   # 2-block (7680 + 7424, tail_256)
        (8, 4100, 15000, 2, True),    # 2-block cross-block max shift
        (4, 4100, 29000, 3, True),    # 3-block incremental merge
    ],
)
def test_qwen_fp8_expmax_attention_accuracy_and_multiblock(
    hpu_fp8_env, heads, queries, kv_len, expected_blocks, shift_second_half
):
    torch.manual_seed(3026 + kv_len)
    num_blocks, total_k_pad, _ = hpu_attention.rmw_qwen_fp8_expmax_layout(kv_len)
    assert num_blocks == expected_blocks
    q = (torch.randn((heads, queries, 128), dtype=torch.bfloat16) * 0.8).to("hpu")
    k = (torch.randn((heads, kv_len, 128), dtype=torch.bfloat16) * 0.8).to("hpu")
    v = (torch.randn((heads, kv_len, 128), dtype=torch.bfloat16) * 0.5).to("hpu")
    if shift_second_half:
        k[:, kv_len // 2 : kv_len // 2 + 16, :] = q[:, :16, :] * 2.5

    qh = (q.float() * hpu_attention.QWEN_FP8_EXPMAX_Q_SCALE).to(torch.bfloat16).contiguous()
    pad = total_k_pad - kv_len
    kh = torch.cat((k, torch.zeros((heads, pad, 128), dtype=torch.bfloat16, device="hpu")), dim=1).contiguous() if pad else k.contiguous()
    v_cpu = v.cpu().float()
    vs = v_cpu.abs().amax(dim=(-2, -1)).clamp_min(1e-12) / 240.0
    v_pad = torch.cat((v_cpu, torch.zeros((heads, pad, 128), dtype=torch.float32)), dim=1) if pad else v_cpu
    v_q = (v_pad / vs.view(heads, 1, 1)).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
    vh_fp8 = _permute_v_all_lanes_cpu(v_q).to("hpu").contiguous()

    qh_snap, kh_snap, vh_snap = qh.cpu().clone(), kh.cpu().clone(), vh_fp8.view(torch.uint8).cpu().clone()
    with torch.no_grad():
        res = hpu_attention.rmw_qwen_fp8_expmax_attention_prepared(qh, kh, vh_fp8, kv_len)
        actual = _merge_fp8_expmax_result(res, vs.to("hpu"), queries).cpu().float()
        rt = hpu_attention._qwen_fp8_expmax_runtime()
        assert rt.last_num_blocks == expected_blocks
        assert torch.equal(qh.cpu(), qh_snap)
        assert torch.equal(kh.cpu(), kh_snap)
        assert torch.equal(vh_fp8.view(torch.uint8).cpu(), vh_snap)

        idx = torch.linspace(0, queries - 1, 64).long()
        q64 = q[:, idx].cpu().double()
        k64 = k.cpu().double()
        v64 = v.cpu().double()
        ref = torch.matmul(torch.softmax(torch.matmul(q64, k64.transpose(-1, -2)) / math.sqrt(128.0), dim=-1), v64).float()
        got = actual[:, idx]
        assert torch.isfinite(actual).all()
        rrmse = (got - ref).norm() / ref.norm()
        cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        assert rrmse < (0.035 if shift_second_half else 0.055), f"rrmse={rrmse.item():.4f}"
        assert cos > 0.9985, f"cos={cos.item():.6f}"


def test_qwen_fp8_expmax_isolation_and_async_stress(hpu_fp8_env):
    torch.manual_seed(88)
    heads, queries, kv_len = 8, 4096, 5000
    _, total_k_pad, _ = hpu_attention.rmw_qwen_fp8_expmax_layout(kv_len)
    pad = total_k_pad - kv_len

    def make_set(seed):
        g = torch.Generator().manual_seed(seed)
        q = (torch.randn((heads, queries, 128), generator=g, dtype=torch.bfloat16) * 0.7).to("hpu")
        k = (torch.randn((heads, kv_len, 128), generator=g, dtype=torch.bfloat16) * 0.7).to("hpu")
        v = (torch.randn((heads, kv_len, 128), generator=g, dtype=torch.float32) * 0.5)
        vs = v.abs().amax(dim=(-2, -1)).clamp_min(1e-12) / 240.0
        vp = torch.cat((v, torch.zeros((heads, pad, 128))), dim=1)
        vq = _permute_v_all_lanes_cpu((vp / vs.view(heads, 1, 1)).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)).to("hpu")
        qh = (q.float() * hpu_attention.QWEN_FP8_EXPMAX_Q_SCALE).to(torch.bfloat16).contiguous()
        kh = torch.cat((k, torch.zeros((heads, pad, 128), dtype=torch.bfloat16, device="hpu")), dim=1).contiguous()
        return qh, kh, vq, vs.to("hpu")

    set_a = make_set(1)
    set_b = make_set(2)
    with torch.no_grad():
        out_a1 = _merge_fp8_expmax_result(hpu_attention.rmw_qwen_fp8_expmax_attention_prepared(*set_a[:3], kv_len), set_a[3], queries).cpu()
        out_b1 = _merge_fp8_expmax_result(hpu_attention.rmw_qwen_fp8_expmax_attention_prepared(*set_b[:3], kv_len), set_b[3], queries).cpu()
        assert not torch.equal(out_a1, out_b1)
        out_a2 = _merge_fp8_expmax_result(hpu_attention.rmw_qwen_fp8_expmax_attention_prepared(*set_a[:3], kv_len), set_a[3], queries).cpu()
        assert torch.equal(out_a1, out_a2)

        outs = []
        for i in range(24):
            inp = set_a if (i % 2 == 0) else set_b
            res = hpu_attention.rmw_qwen_fp8_expmax_attention_prepared(*inp[:3], kv_len)
            outs.append(_merge_fp8_expmax_result(res, inp[3], queries))
        hpu_attention.release_inflight()
        for i, out in enumerate(outs):
            got = out.cpu()
            assert torch.isfinite(got).all()
            assert torch.equal(got, out_a1 if (i % 2 == 0) else out_b1)


def _make_full_fp8_block():
    torch.manual_seed(191)
    operations = comfy.ops.mixed_precision_ops({"formats": ["float8_e4m3fn"]}, torch.bfloat16, load_device=torch.device("hpu"))
    block = qwen.QwenImage21TransformerBlock(
        4096, 32, 128, mlp_ratio=3, fused_mlp=True, dtype=torch.bfloat16, device="cpu", operations=operations
    )
    state = {}
    qmeta = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
    for name, mod in block.named_modules():
        if isinstance(mod, operations.Linear):
            w = (torch.randn((mod.out_features, mod.in_features)) * 18.0).to(torch.float8_e4m3fn)
            state[f"{name}.weight"] = w
            state[f"{name}.weight_scale"] = torch.tensor(0.0008, dtype=torch.float32)
            state[f"{name}.comfy_quant"] = qmeta.clone()
        elif isinstance(mod, operations.RMSNorm):
            state[f"{name}.weight"] = (torch.randn(mod.normalized_shape) * 0.05 + 1.0).to(torch.bfloat16)
    block.load_state_dict(state, strict=False)
    block.to("hpu")
    return block


def _make_block_inputs(queries, prefix_len, seed=202):
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
        (4096, 256),    # pad_len == 0 on 256-boundary (B_k = 1)
        (4096, 255),    # pad_len == 1 on 256-boundary (B_k = 1, supported natively!)
        (4096, 512),    # pad_len == 0 on 512-boundary (B_k = 1)
        (4096, 447),    # pad_len > 1  (B_k = 1)
        (4100, 10500),  # non-32-multiple Q + B_k = 2 (_fp8_v2_post2)
    ],
)
def test_qwen21_fp8_block_matches_baseline(hpu_fp8_env, queries, prefix_len):
    block = _make_full_fp8_block()
    x, mod, pe, pk, pv = _make_block_inputs(queries, prefix_len)
    with torch.no_grad():
        opts_ref = {"hpu_fast_attention": False, "hpu_bf16_rmw": False, "hpu_fp8_rmw": False}
        ref = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts_ref, []).cpu().float()

        opts_opt = {
            "hpu_fast_attention": True,
            "hpu_fp8_rmw": True,
            "_qwen21_hpu_pe_cs_fp8": qwen_hpu.pack_pe_cs_fp8(pe),
        }
        pending = []
        actual = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts_opt, pending).cpu().float()
        assert len(pending) == 1
        assert torch.isfinite(actual).all()
        rrmse = (actual - ref).norm() / ref.norm()
        cos = F.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
        assert rrmse < 0.030, f"rrmse={rrmse.item():.4f}"
        assert cos > 0.9994, f"cos={cos.item():.6f}"


def test_qwen21_fp8_block_dynamic_prefix_no_recompile_cliff(hpu_fp8_env):
    block = _make_full_fp8_block()
    queries = 4096
    x, mod, pe, _, _ = _make_block_inputs(queries, 447)
    pe_cs = qwen_hpu.pack_pe_cs_fp8(pe)
    pending = []
    with torch.no_grad():
        for plen in (447, 1150, 300, 611, 777, 888, 999, 256, 512, 511, 447):
            pk = (torch.randn((1, plen, 32, 128), dtype=torch.bfloat16) * 0.2).to("hpu")
            pv = (torch.randn((1, plen, 32, 128), dtype=torch.bfloat16) * 0.2).to("hpu")
            opts = {"hpu_fast_attention": True, "hpu_fp8_rmw": True, "_qwen21_hpu_pe_cs_fp8": pe_cs}
            out = block._forward_with_prefix(x.clone(), mod, pe, pk, pv, opts, pending)
            assert torch.isfinite(out.cpu()).all()
    assert len(pending) <= 2


@pytest.mark.parametrize(
    "reason",
    ["pk1", "small_q", "opt_out_fp8_rmw", "opt_out_fast", "disabled_compiler", "lora_attn", "lora_mlp", "hook", "unfused_mlp", "tpc_unavailable"],
)
def test_qwen21_fp8_block_clean_fallback(hpu_fp8_env, monkeypatch, reason):
    block = _make_full_fp8_block()
    queries = 2048 if reason == "small_q" else 4096
    prefix_len = 1 if reason == "pk1" else 447
    x, mod, pe, pk, pv = _make_block_inputs(queries, prefix_len)
    options = {
        "hpu_fast_attention": reason != "opt_out_fast",
        "hpu_fp8_rmw": reason != "opt_out_fp8_rmw",
        "hpu_bf16_rmw": False,
    }

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
    elif reason == "tpc_unavailable":
        monkeypatch.setattr(qwen_hpu, "_hpu_tpc", None)

    monkeypatch.setattr(hpu_attention, "rmw_qwen_fp8_expmax_attention_prepared", lambda *a, **kw: pytest.fail("must fall back"))
    fallback_called = {"legacy": False, "eager": False}
    monkeypatch.setattr(qwen_hpu, "prefix_attention", lambda *a, **kw: (fallback_called.__setitem__("legacy", True), None)[1])
    monkeypatch.setattr(block, "forward", lambda inp, *a, **kw: (fallback_called.__setitem__("eager", True), inp)[1])

    with torch.no_grad():
        pending = []
        out = block._forward_with_prefix(x, mod, pe, pk, pv, options, pending)
        assert out is x
        assert fallback_called["eager"]
        assert pending == []
