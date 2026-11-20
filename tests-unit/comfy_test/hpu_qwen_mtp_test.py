import json
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import hpu_kv_attention, hpu_qwen35, hpu_qwen35_tpc, qwen35
from comfy.text_encoders.llama import FixedKVCache


@pytest.fixture
def hpu_env(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(args, "use_ck_attention", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    hpu_qwen35.reset_compile_caches()
    qwen35._compiled_hpu_decode.cache_clear()
    yield
    torch._dynamo.reset()
    hpu_qwen35.reset_compile_caches()
    qwen35._compiled_hpu_decode.cache_clear()


def make_mtp_model(device="hpu", dtype=torch.bfloat16, quantized=True, quantize_mtp=True, weight_seed=99):
    cfg = qwen35.Qwen35Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=256,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=qwen35._qwen35_layer_types(4), lm_head=True, mtp=True,
    )
    base = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(base)
    base.model = qwen35.Qwen35Transformer(cfg, device="cpu", dtype=dtype, ops=comfy.ops.manual_cast)
    base.mtp = qwen35.MTPHead(cfg, device="cpu", dtype=dtype, ops=comfy.ops.manual_cast)
    rng = torch.Generator().manual_seed(weight_seed)
    for p in list(base.model.parameters()) + list(base.mtp.parameters()):
        p.data.copy_(torch.randn(p.shape, generator=rng) * 0.04)

    if not quantized:
        return base.to(device)

    model_sd = base.model.state_dict()
    mtp_sd = base.mtp.state_dict()
    for mod, sd, do_quant in [(base.model, model_sd, True), (base.mtp, mtp_sd, quantize_mtp)]:
        if not do_quant:
            continue
        for name, sub in mod.named_modules():
            if isinstance(sub, torch.nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b")):
                w = sd[name + ".weight"].float()
                scale = w.abs().amax().clamp_min(1e-12) / 224.0
                sd[name + ".weight"] = (w / scale).to(torch.float8_e4m3fn)
                sd[name + ".weight_scale"] = scale
                sd[name + ".comfy_quant"] = torch.tensor(
                    list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)

    ops = comfy.ops.mixed_precision_ops({"formats": ["float8_e4m3fn"]}, dtype, load_device=torch.device(device))
    model = qwen35.Qwen35.__new__(qwen35.Qwen35)
    torch.nn.Module.__init__(model)
    model.model = qwen35.Qwen35Transformer(cfg, device="cpu", dtype=dtype, ops=ops)
    model.mtp = qwen35.MTPHead(cfg, device="cpu", dtype=dtype, ops=ops)
    model.model.load_state_dict(model_sd)
    model.mtp.load_state_dict(mtp_sd)
    model.to(device)
    return model


@pytest.mark.parametrize("mode,rows", [("2", 4), ("3", 4), ("tree2s2", 8), ("tree3s1", 8), ("tree4s1m8", 8)])
def test_tree_decode_sdpa_matches_fp64_reference(hpu_env, mode, rows):
    torch.manual_seed(51)
    cap = 131
    tree = qwen35.VerifyTree(qwen35.MTP_MODES[mode][1], rows, "hpu")
    key = torch.randn(1, 2, cap, 256, device="hpu", dtype=torch.bfloat16) * 0.25
    val = torch.randn_like(key)
    q = torch.randn(1, 8, rows, 256, device="hpu", dtype=torch.bfloat16) * 0.25
    for prefix in (0, 1, 17, 64, cap - rows):
        seqlen = torch.tensor([prefix], device="hpu", dtype=torch.int32)
        with torch.no_grad():
            actual = hpu_kv_attention.tree_decode_sdpa(q, key, val, seqlen, tree.mask).cpu().float()
        visible = torch.zeros(rows, cap, dtype=torch.bool)
        for r in range(rows):
            visible[r, :prefix] = True
            a = r
            while a >= 0:
                visible[r, prefix + a] = True
                a = tree.parent[a]
        expected = F.scaled_dot_product_attention(
            q.cpu().double(), key.cpu().double(), val.cpu().double(),
            attn_mask=visible, enable_gqa=True,
        ).transpose(1, 2).reshape(1, rows, -1).float()
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.02)


@pytest.mark.parametrize("mode", ["2", "tree2s2", "tree3s1", "tree4s1m8"])
def test_vectorized_tree_penalty_mask_matches_reference(hpu_env, mode):
    torch.manual_seed(61)
    cap, spec = qwen35.MTP_MODES[mode]
    tree = qwen35.VerifyTree(spec, cap, "hpu")
    vocab = 128
    verify_tokens = torch.randint(0, vocab, (1, cap), device="hpu", dtype=torch.long)
    pen_mask = torch.zeros((vocab,), device="hpu", dtype=torch.bool)
    pen_mask[torch.tensor([3, 17, 42], device="hpu")] = True
    pen_mask.index_fill_(0, verify_tokens[:, 0], 1)

    logits = torch.randn(cap, vocab, device="hpu", dtype=torch.bfloat16)
    anc_idx = hpu_qwen35.tree_ancestor_indices(tree, torch.device("hpu"))
    with torch.no_grad():
        actual_probs, actual_idx = hpu_qwen35._sample_prep(
            logits, verify_tokens, pen_mask, anc_idx,
            1.0, 10, 0.95, 0.0, 1.0, 1.5,
        )

    # Reference loop computed on CPU (Habana 1.24.2 eager drops overlapping same-storage row copy_).
    ref_mask = pen_mask.cpu().unsqueeze(0).repeat(cap, 1)
    vt_cpu = verify_tokens.cpu()
    for r in range(1, cap):
        ref_mask[r].copy_(ref_mask[tree.parent[r]])
        ref_mask[r].index_fill_(0, vt_cpu[:, r], 1)
    ref_rows = torch.where(ref_mask.to("hpu"), logits - 1.5, logits)
    ref_top, ref_idx = torch.topk(ref_rows, 10, dim=-1)
    ref_top = ref_top.float()
    sorted_l, sorted_i = torch.sort(ref_top, descending=True)
    cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
    rem = cum >= 0.95
    rem = torch.cat((torch.zeros_like(rem[..., :1]), rem[..., :-1]), dim=-1)
    rm = torch.zeros_like(ref_top, dtype=torch.bool)
    rm.scatter_(1, sorted_i, rem)
    ref_probs = F.softmax(ref_top.masked_fill(rm, torch.finfo(ref_top.dtype).min), dim=-1)

    torch.testing.assert_close(actual_idx.cpu(), ref_idx.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(actual_probs.cpu(), ref_probs.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("quantize_mtp", [True, False])
@pytest.mark.parametrize("mode", ["2", "tree3s1", True])
def test_mtp_compiled_prompt_isolation_and_consistency(hpu_env, quantize_mtp, mode):
    model = make_mtp_model(quantized=True, quantize_mtp=quantize_mtp, weight_seed=99)
    prompt_a = torch.arange(1, 21, device="hpu").unsqueeze(0)
    prompt_b = prompt_a.clone()
    prompt_b[:, -1] = 77

    with torch.no_grad():
        g1 = model.generate(embeds=model.model.embed_tokens(prompt_a), do_sample=False, max_length=16, stop_tokens=[], seed=7, mtp=mode)
        g2 = model.generate(embeds=model.model.embed_tokens(prompt_a), do_sample=False, max_length=16, stop_tokens=[], seed=7, mtp=mode)
        assert g1 == g2 and len(g1) == 16

        a1 = model.generate(embeds=model.model.embed_tokens(prompt_a), do_sample=True, max_length=16,
                            temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
        b1 = model.generate(embeds=model.model.embed_tokens(prompt_b), do_sample=True, max_length=16,
                            temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
        a2 = model.generate(embeds=model.model.embed_tokens(prompt_a), do_sample=True, max_length=16,
                            temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=42, mtp=mode)
        assert a1 == a2 and len(a1) == 16
        assert a1 != b1


@pytest.mark.parametrize("use_tpc", [True, False])
def test_mtp_reuses_compiled_graphs_across_kv_capacities(hpu_env, monkeypatch, use_tpc):
    if use_tpc and not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 Qwen3.5 TPC kernels not built or not on GC_KERNEL_PATH")
    model = make_mtp_model(quantized=True, quantize_mtp=True, weight_seed=101)
    orig_builder = hpu_qwen35.build_mtp_runner
    monkeypatch.setattr(
        hpu_qwen35, "build_mtp_runner",
        lambda *a, **kw: orig_builder(*a, **{**kw, "use_tpc": use_tpc}),
    )
    backend = lookup_backend("hpu_backend")
    if use_tpc:
        counts = {"gdn3": 0, "attn": 0, "prep": 0, "head_accept": 0, "draft": 0, "kv_compact": 0}
        target_map = {
            hpu_qwen35._gdn3_stage_fn: "gdn3",
            hpu_qwen35._attn_stage_fn: "attn",
            hpu_qwen35._prepare_verify_ondev_fn: "prep",
            hpu_qwen35._verify_head_accept_fn: "head_accept",
            hpu_qwen35._mtp_draft_chain_fn: "draft",
            hpu_qwen35._kv_compact_ondev_fn: "kv_compact",
        }
    else:
        counts = {"gdn3": 0, "attn": 0, "mask": 0, "head": 0, "draft": 0}
        target_map = {
            hpu_qwen35._gdn3_stage_fn: "gdn3",
            hpu_qwen35._attn_stage_fn: "attn",
            hpu_qwen35._build_tree_sdpa_mask_fn: "mask",
            hpu_qwen35._verify_head_fn: "head",
            hpu_qwen35._mtp_draft_chain_fn: "draft",
        }

    def wrap_compile(fn, **kwargs):
        key = target_map.get(fn)
        if key is not None:
            def counted(graph, inputs, _key=key, **kw):
                counts[_key] += 1
                return backend(graph, inputs, **kw)
            kwargs["backend"] = counted
        return orig_compile(fn, **kwargs)

    orig_compile = torch.compile
    monkeypatch.setattr(torch, "compile", wrap_compile)

    with warnings.catch_warnings(record=True) as caught, torch.no_grad():
        warnings.simplefilter("always")
        for plen, mlen in [(12, 12), (13, 12), (19, 16), (31, 20), (64, 16), (12, 12)]:
            prompt = torch.arange(1, plen + 1, device="hpu").unsqueeze(0)
            out = model.generate(
                embeds=model.model.embed_tokens(prompt), do_sample=True, max_length=mlen,
                temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5,
                stop_tokens=[], seed=42, mtp="tree3s1",
            )
            assert len(out) == mlen
        for key, count in counts.items():
            assert count == 1, f"Expected 1 compile for {key} across capacities, got {counts}"
        assert not [w for w in caught if "batch_as_strided" in str(w.message)]


@pytest.mark.parametrize("mode", ["disabled", "training", "grad", "hook", "mtp_hook", "patched", "lowvram", "vbar", "attention", "kitchen", "forward", "offloaded", "cpu", "fp32", "batch"])
def test_mtp_can_compile_fallback_routes(hpu_env, monkeypatch, mode):
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    model = make_mtp_model(device=device, dtype=dtype, quantized=(device == "hpu" and dtype == torch.bfloat16))
    x = torch.ones(2 if mode == "batch" else 1, 4, 64, device=device, dtype=dtype)
    pkv = model.init_kv_cache(x.shape[0], 32, device, dtype)
    mtp_kv = model.mtp.layers[0].self_attn.init_kv_cache(x.shape[0], 16, device, dtype, {})
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(args, "use_ck_attention", mode == "kitchen")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    proj = model.model.layers[0].linear_attn.in_proj_qkv
    if mode == "hook":
        proj.register_forward_hook(lambda *a: None)
    if mode == "mtp_hook":
        model.mtp.fc.register_forward_hook(lambda *a: None)
    if mode == "patched":
        proj.weight_function = [lambda w: w]
    if mode == "lowvram":
        proj.weight_lowvram_function = object()
    if mode == "vbar":
        proj._v = object()
    if mode == "attention":
        model.model.layers[-1].self_attn.comfy_attention.method = "pytorch"
    if mode == "forward":
        model.model.forward = lambda *a, **kw: None
    if mode == "offloaded":
        proj.weight = torch.nn.Parameter(proj.weight.cpu())
    with torch.set_grad_enabled(mode == "grad"):
        assert not hpu_qwen35.can_compile_mtp(model, x, pkv, mtp_kv, qwen35.Qwen35Transformer, qwen35.LinearKV)


def test_mtp_off_on_mtp_checkpoint_uses_compiled_single_token_path(hpu_env):
    model = make_mtp_model(quantized=True, quantize_mtp=True, weight_seed=77)
    x = torch.ones(1, 1, 64, device="hpu", dtype=torch.bfloat16)
    caches = model.init_kv_cache(1, 32, "hpu", torch.bfloat16)
    with torch.no_grad():
        assert model._can_compile_generation(x, caches)
        prompt = torch.arange(1, 13, device="hpu").unsqueeze(0)
        a1 = model.generate(embeds=model.model.embed_tokens(prompt), do_sample=True, max_length=10,
                            temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=17, mtp=False)
        a2 = model.generate(embeds=model.model.embed_tokens(prompt), do_sample=True, max_length=10,
                            temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5, stop_tokens=[], seed=17, mtp=False)
        assert a1 == a2 and len(a1) == 10


def _rrmse(a, b):
    af = a.float().cpu()
    bf = b.float().cpu()
    return float((af - bf).pow(2).sum().sqrt() / bf.pow(2).sum().sqrt().clamp_min(1e-12))


@pytest.mark.parametrize("pen_cfg", [(0, 0.0, 1.0), (1, 1.5, 1.0), (1, 0.0, 1.2), (1, 1.0, 1.15)])
def test_tpc_topk_matches_reference(hpu_env, pen_cfg):
    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 Qwen3.5 TPC kernels not available")
    has_pen, pres_pen, rep_pen = pen_cfg
    torch.manual_seed(111)
    rows, vocab, k = 8, 4099, 20
    logits = torch.randn(rows, vocab, device="hpu", dtype=torch.bfloat16) * 3.0
    pen_mask = (torch.rand(vocab, device="hpu") < 0.15).to(torch.uint8)
    anc = torch.randint(0, vocab, (rows, 7), device="hpu", dtype=torch.int32)
    with torch.no_grad():
        got_v, got_i = hpu_qwen35_tpc.topk(logits, pen_mask, anc, k, has_pen, pres_pen, rep_pen)
        got_v2, got_i2 = hpu_qwen35_tpc.topk(logits, pen_mask, anc, k, has_pen, pres_pen, rep_pen)
    assert torch.equal(got_v.cpu(), got_v2.cpu()) and torch.equal(got_i.cpu(), got_i2.cpu())

    ref_lg = logits.float().cpu().clone()
    if has_pen:
        pm = pen_mask.cpu().bool().unsqueeze(0).repeat(rows, 1)
        for r in range(rows):
            pm[r, anc[r].cpu().long()] = True
        pen = ref_lg.clone()
        if rep_pen != 1.0:
            pen = torch.where(pen < 0, pen * rep_pen, pen / rep_pen)
        if pres_pen != 0.0:
            pen = pen - pres_pen
        ref_lg = torch.where(pm, pen, ref_lg)
    ref_v, ref_i = torch.topk(ref_lg, k, dim=-1)
    gv_cpu, gi_cpu = got_v.cpu(), got_i.cpu().long()
    torch.testing.assert_close(gv_cpu, ref_v, atol=1e-5, rtol=1e-5)
    assert int(gi_cpu.min()) >= 0 and int(gi_cpu.max()) < vocab
    for r in range(rows):
        assert len(set(gi_cpu[r].tolist())) == k
    # Every returned index maps to the exact sorted top-k value, and non-tied ranks match identically.
    torch.testing.assert_close(ref_lg.gather(1, gi_cpu), ref_v, atol=1e-5, rtol=1e-5)
    for r in range(rows):
        for j in range(k):
            if int((ref_lg[r] == ref_v[r, j]).sum().item()) == 1:
                assert int(gi_cpu[r, j]) == int(ref_i[r, j])


@pytest.mark.parametrize("pending", [0, 1, 2, 3])
def test_tpc_gdn_conv_and_rec_match_fp64_reference(hpu_env, pending):
    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 Qwen3.5 TPC kernels not available")
    torch.manual_seed(123 + pending)
    bs, kh, vh, k_hd, v_hd, rows = 1, 2, 4, 128, 128, 8
    channels = kh * k_hd * 2 + vh * v_hd
    conv_st = torch.randn(bs, 3, channels, dtype=torch.bfloat16) * 0.4
    prev_proj = torch.randn(bs, rows, channels, dtype=torch.bfloat16) * 0.4
    cur_proj = torch.randn(bs, rows, channels, dtype=torch.bfloat16) * 0.4
    conv_w = torch.randn(4, channels, dtype=torch.float32) * 0.3
    conv_b = torch.randn(channels, dtype=torch.float32) * 0.1
    tree = qwen35.VerifyTree(qwen35.MTP_MODES["tree3s1"][1], rows, "cpu")
    _, conv_src_i32, _, _, prog32, _ = hpu_qwen35._tree_gdn_tables(tree, 3, 4, torch.device("cpu"))
    slots = [0, 2, 5, 1, 0, 0, 0, 0]
    ctl = torch.cat((prog32, torch.tensor([pending] + slots, dtype=torch.int32)))

    with torch.no_grad():
        got_cst, got_cc = hpu_qwen35_tpc.gdn_conv(
            conv_st.to("hpu"), prev_proj.to("hpu"), cur_proj.to("hpu"),
            conv_w.to("hpu"), conv_b.to("hpu"), conv_src_i32.to("hpu"), ctl.to("hpu"), 1,
        )
    cat_p = torch.cat((conv_st, prev_proj[:, slots[:pending], :]), dim=1) if pending else conv_st
    ref_cst = cat_p[:, -3:, :]
    win = torch.cat((ref_cst, cur_proj), dim=1)
    gath = win[:, conv_src_i32.long(), :].view(bs, rows, 4, channels).float()
    acc = (gath * conv_w.view(1, 1, 4, channels)).sum(2) + conv_b
    y = acc.to(torch.bfloat16).float()
    ref_cc = (y / (1.0 + torch.exp(-y))).to(torch.bfloat16)
    assert torch.equal(got_cst.cpu(), ref_cst)
    assert _rrmse(got_cc, ref_cc) < 5e-3

    state = torch.randn(bs, vh, k_hd, v_hd, dtype=torch.float32) * 0.2
    prev_conv = torch.randn(bs, rows, channels, dtype=torch.bfloat16) * 0.4
    cur_conv = ref_cc
    prev_ab = torch.randn(bs, rows, 2 * vh, dtype=torch.bfloat16) * 0.4
    cur_ab = torch.randn(bs, rows, 2 * vh, dtype=torch.bfloat16) * 0.4
    g_decay = -torch.rand(vh, dtype=torch.float32) * 0.4 - 0.1
    dt_bias = torch.randn(vh, dtype=torch.float32) * 0.2

    with torch.no_grad():
        got_st_pk, got_out = hpu_qwen35_tpc.gdn_rec(
            hpu_qwen35_tpc.pack_gdn_state(state).to("hpu"),
            prev_conv.to("hpu"), cur_conv.to("hpu"),
            prev_ab.to("hpu"), cur_ab.to("hpu"),
            g_decay.to("hpu"), dt_bias.to("hpu"), ctl.to("hpu"), kh, vh, v_hd,
        )
    got_st = hpu_qwen35_tpc.unpack_gdn_state(got_st_pk.cpu(), vh, v_hd)

    s = state.double().clone()
    groups = vh // kh
    kd = kh * 128

    def gates_fp64(ab):
        return (
            torch.exp(g_decay.double() * F.softplus(ab[:, :, :vh].double() + dt_bias.double())),
            torch.sigmoid(ab[:, :, vh:].double()),
        )

    pd, pb = gates_fp64(prev_ab)
    cd, cb = gates_fp64(cur_ab)
    for i in range(pending):
        sl = slots[i]
        k = prev_conv[:, sl, kd:2 * kd].view(bs, kh, 128).double()
        k = (k / k.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)).repeat_interleave(groups, 1)
        v = prev_conv[:, sl, 2 * kd:].view(bs, vh, v_hd).double()
        d, bt = pd[:, sl].unsqueeze(-1), pb[:, sl].unsqueeze(-1)
        upd = (v - (k.unsqueeze(-1) * s).sum(-2) * d) * bt
        s = s * d.unsqueeze(-1) + k.unsqueeze(-1) * upd.unsqueeze(-2)
    ref_st = s.clone()
    ref_out = torch.zeros(bs, rows, vh, v_hd, dtype=torch.float64)
    for instr in prog32.tolist():
        r = instr & 15
        q = cur_conv[:, r, :kd].view(bs, kh, 128).double()
        q = (q / q.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12) * (128.0 ** -0.5)).repeat_interleave(groups, 1)
        k = cur_conv[:, r, kd:2 * kd].view(bs, kh, 128).double()
        k = (k / k.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)).repeat_interleave(groups, 1)
        v = cur_conv[:, r, 2 * kd:].view(bs, vh, v_hd).double()
        d, bt = cd[:, r].unsqueeze(-1), cb[:, r].unsqueeze(-1)
        upd = (v - (k.unsqueeze(-1) * s).sum(-2) * d) * bt
        ref_out[:, r] = d * (q.unsqueeze(-1) * s).sum(-2) + (q * k).sum(-1, keepdim=True) * upd
        if instr & 32:
            s = s * d.unsqueeze(-1) + k.unsqueeze(-1) * upd.unsqueeze(-2)
    assert _rrmse(got_st, ref_st) < 1e-5
    assert _rrmse(got_out, ref_out.reshape(bs, rows, vh * v_hd)) < 1e-5


def test_tpc_norm_act_rope_and_accept_kernels(hpu_env):
    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 Qwen3.5 TPC kernels not available")
    torch.manual_seed(145)
    dev = torch.device("hpu")
    rows = 8

    # rmsnorm_quant with residual + store_norm
    x = torch.randn(rows, 512, device=dev, dtype=torch.bfloat16)
    res = torch.randn_like(x)
    w = torch.randn(512, device=dev, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        ox, oy, oq, os_ = hpu_qwen35_tpc.rmsnorm_quant(x, res, w, 1, 1, 1, 1e-6)
        rx = x + res
        ry = hpu_qwen35._rms_norm(rx, w, 1e-6, True)
        rq, rs = hpu_qwen35._quantize_fp8(ry, hpu_qwen35._divisor(dev))
    assert torch.equal(ox.cpu(), rx.cpu())
    assert _rrmse(oy, ry) < 1e-3
    assert _rrmse(oq.float() * os_, rq.float() * rs) < 3e-2
    assert _rrmse(oq.float() * os_, ry.float()) <= _rrmse(rq.float() * rs, ry.float()) + 2e-3

    # gated_rmsnorm_quant, sig_mul_quant, silu_mul_quant
    out_f32 = torch.randn(rows, 512, device=dev, dtype=torch.float32)
    z = torch.randn(rows, 512, device=dev, dtype=torch.bfloat16)
    gnw = torch.randn(128, device=dev, dtype=torch.float32)
    with torch.no_grad():
        gq, gs = hpu_qwen35_tpc.gated_rmsnorm_quant(out_f32, z, gnw, 4, 1e-6)
        ov = out_f32.view(rows, 4, 128)
        ref_g = ((ov * torch.rsqrt(ov.square().mean(-1, keepdim=True) + 1e-6) * gnw).to(torch.bfloat16) * F.silu(z.view(rows, 4, 128))).reshape(rows, 512)
        rq, rs = hpu_qwen35._quantize_fp8(ref_g, hpu_qwen35._divisor(dev))
        assert _rrmse(gq.float() * gs, rq.float() * rs) < 3e-2

        sq, ss = hpu_qwen35_tpc.sig_mul_quant(x, z)
        rq, rs = hpu_qwen35._quantize_fp8(x * z.sigmoid(), hpu_qwen35._divisor(dev))
        assert _rrmse(sq.float() * ss, rq.float() * rs) < 3e-2

        for rr in (1, 8):
            g12 = torch.randn(rr, 12288, device=dev, dtype=torch.bfloat16)
            u12 = torch.randn(rr, 12288, device=dev, dtype=torch.bfloat16)
            dq, ds = hpu_qwen35_tpc.silu_mul_quant(g12, u12)
            rq, rs = hpu_qwen35._quantize_fp8(F.silu(g12) * u12, hpu_qwen35._divisor(dev))
            assert _rrmse(dq.float() * ds, rq.float() * rs) < 3e-2

        w2 = torch.randn(512, device=dev, dtype=torch.bfloat16) * 0.1
        fcq, fcs = hpu_qwen35_tpc.dual_rmsnorm_cat_quant(x[:1], res[:1], w, w2, 1, 1e-6)
        ref_fc = torch.cat((hpu_qwen35._rms_norm(x[:1], w, 1e-6, True), hpu_qwen35._rms_norm(res[:1], w2, 1e-6, True)), dim=-1)
        rq, rs = hpu_qwen35._quantize_fp8(ref_fc, hpu_qwen35._divisor(dev))
        assert _rrmse(fcq.float() * fcs, rq.float() * rs) < 3e-2
        assert _rrmse(fcq.float() * fcs, ref_fc.float()) <= _rrmse(rq.float() * rs, ref_fc.float()) + 2e-3

    # qk_norm_rope
    nh, nkh, hd = 4, 2, 256
    qg = torch.randn(rows, nh * 2 * hd, device=dev, dtype=torch.bfloat16)
    xk = torch.randn(rows, nkh * hd, device=dev, dtype=torch.bfloat16)
    qnw = torch.randn(hd, device=dev, dtype=torch.bfloat16) * 0.1
    knw = torch.randn(hd, device=dev, dtype=torch.bfloat16) * 0.1
    ang = torch.randn(rows, 32, device=dev, dtype=torch.float32)
    c32, s32 = torch.cos(ang), torch.sin(ang)
    rot = torch.stack((torch.stack((c32, -s32), -1), torch.stack((s32, c32), -1)), -2).view(1, 1, rows, 32, 2, 2)
    cs = torch.cat((c32, c32, -s32, s32), dim=-1).contiguous()
    with torch.no_grad():
        gq, gk, gg = hpu_qwen35_tpc.qk_norm_rope(qg, xk, qnw, knw, cs, nh, nkh, 1, 1e-6)
        qgv = qg.view(1, rows, nh, hd * 2)
        rq, rg = qgv[..., :hd], qgv[..., hd:].reshape(rows, -1)
        rq = hpu_qwen35._rms_norm(rq, qnw, 1e-6, True).transpose(1, 2)
        rk = hpu_qwen35._rms_norm(xk.view(1, rows, nkh, hd), knw, 1e-6, True).transpose(1, 2)
        rq, rk = hpu_qwen35._apply_partial_rope(rq, rk, rot, 64, True)
    assert _rrmse(gq, rq) < 5e-3 and _rrmse(gk, rk) < 5e-3 and torch.equal(gg.cpu(), rg.cpu())

    # sample_tree_accept: on-chain, off-chain, and reject cases
    k, depth, topk_k = 20, 2, 7
    tv = torch.sort(torch.randn(rows, k, dtype=torch.float32) * 3.0, dim=-1, descending=True).values.to(dev)
    ti = torch.randint(10, 50000, (rows, k), dtype=torch.int32, device=dev)
    ru = torch.rand(rows, 1, dtype=torch.float32, device=dev)
    probs = hpu_qwen35._finish_topk_probs(tv, 1.0, 0.95, 0.05)
    ref_toks = ti.gather(1, (probs.cumsum(-1) <= ru).sum(-1, keepdim=True).clamp(max=k - 1)).squeeze(-1).cpu().tolist()
    tpar = torch.tensor([-1, 0, 1, 0, 0, 1, 1, 1], dtype=torch.int32, device=dev)
    tops = torch.full((depth, 1, topk_k), 99999, dtype=torch.int32, device=dev)
    tops[0, 0, 3] = ref_toks[0]
    tops[1, 0, 2] = ref_toks[1]
    st_in = torch.tensor([0] * 17 + [100, 50], dtype=torch.int32, device=dev)
    # On-chain depth-2 accept: 0 -> 1 -> 2 (cur == acc_len - 1 -> kept = min(3, depth=2) = 2)
    v_on = torch.tensor([[999, ref_toks[0], ref_toks[1], 3, 4, 5, 6, 7]], dtype=torch.int32, device=dev)
    st_o, ct_o, _, meta_o, rk_o = hpu_qwen35_tpc.sample_tree_accept(tv, ti, v_on, tops, ru, st_in, tpar, depth, 1, 1, 1.0, 0.95, 0.05)
    assert meta_o.cpu().tolist() == [2, ref_toks[2], 100, 3]
    assert ct_o.cpu().tolist()[:3] == [ref_toks[0], ref_toks[1], ref_toks[2]]
    assert st_o.cpu().tolist()[17:19] == [103, 52]
    assert rk_o.cpu().tolist()[:2] == [3, 2]
    # Off-chain depth-1 accept: 0 -> 3 (cur=3 != acc_len-1=1 -> kept = acc_len - 1 = 1)
    v_off = torch.tensor([[999, 1, 2, ref_toks[0], 4, 5, 6, 7]], dtype=torch.int32, device=dev)
    st_f, ct_f, _, meta_f, _ = hpu_qwen35_tpc.sample_tree_accept(tv, ti, v_off, tops, ru, st_in, tpar, depth, 1, 1, 1.0, 0.95, 0.05)
    assert meta_f.cpu().tolist() == [3, ref_toks[3], 100, 2]
    assert ct_f.cpu().tolist()[:2] == [ref_toks[0], ref_toks[3]]
    assert st_f.cpu().tolist()[17:19] == [102, 51]
    # Reject at root: path [0] (cur=0 == acc_len-1=0 -> kept = 1)
    v_rej = torch.full((1, rows), 88888, dtype=torch.int32, device=dev)
    st_r, ct_r, _, meta_r, _ = hpu_qwen35_tpc.sample_tree_accept(tv, ti, v_rej, tops, ru, st_in, tpar, depth, 1, 1, 1.0, 0.95, 0.05)
    assert meta_r.cpu().tolist() == [0, ref_toks[0], 100, 1]
    assert ct_r.cpu().tolist()[0] == ref_toks[0]
    assert st_r.cpu().tolist()[17:19] == [101, 51]


def test_mtp_batch_steps_and_kv_bucket_exact_equivalence(hpu_env, monkeypatch):
    if not hpu_qwen35_tpc.available:
        pytest.skip("Gaudi2 Qwen3.5 TPC kernels not available")
    model = make_mtp_model(quantized=True, quantize_mtp=True, weight_seed=107)
    prompt = torch.arange(1, 21, device="hpu").unsqueeze(0)
    orig_builder = hpu_qwen35.build_mtp_runner
    outputs = []
    for bs, kvb in [(1, 0), (4, 0), (8, 0), (8, 32)]:
        monkeypatch.setattr(
            hpu_qwen35, "build_mtp_runner",
            lambda *a, _bs=bs, _kvb=kvb, **kw: orig_builder(*a, **{**kw, "batch_steps": _bs, "kv_bucket": _kvb}),
        )
        with torch.no_grad():
            ids = model.generate(
                embeds=model.model.embed_tokens(prompt), do_sample=True, max_length=24,
                temperature=1.0, top_k=10, top_p=0.95, presence_penalty=1.5,
                stop_tokens=[], seed=42, mtp=True,
            )
        outputs.append(ids)
    for other in outputs[1:]:
        assert other == outputs[0], f"Batch/bucket mismatch: {other} != {outputs[0]}"


def test_tpc_build_script_fingerprint_and_slim_runtime_reuse(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import scripts.gaudi.build_qwen35_tpc as builder

    root = Path(builder.__file__).resolve().parents[2]
    fake_build = tmp_path / "build"
    fake_build.mkdir()

    fp_full = builder.compute_fingerprint(root, compiler="/fake/tpc-clang", gxx="/fake/g++", habana_inc="/fake/inc")
    fp_rt = builder.compute_fingerprint(root)
    assert fp_full["sources_sha256"] == fp_rt["sources_sha256"]

    # Missing outputs -> not ready
    assert not builder._outputs_ready(fake_build, fp_full)
    (fake_build / "libcomfy_qwen35_tpc.so").write_bytes(b"lib")
    (fake_build / "comfy_qwen35_tpc_bridge.so").write_bytes(b"bridge")
    assert not builder._outputs_ready(fake_build, fp_full)

    # Matching full fingerprint -> ready for both full and slim runtime-only checks
    (fake_build / builder.FINGERPRINT_NAME).write_text(json.dumps(fp_full))
    assert builder._outputs_ready(fake_build, fp_full, runtime_only=False)
    assert builder._outputs_ready(fake_build, fp_rt, runtime_only=True)

    # Stale source hash -> rejected for both full and slim runtime-only checks
    stale = dict(fp_full, sources_sha256="0" * 64)
    (fake_build / builder.FINGERPRINT_NAME).write_text(json.dumps(stale))
    assert not builder._outputs_ready(fake_build, fp_full, runtime_only=False)
    assert not builder._outputs_ready(fake_build, fp_rt, runtime_only=True)

    # Missing compiler with no valid prebuilt outputs -> returns non-zero cleanly
    monkeypatch.setenv("TPC_CLANG", str(tmp_path / "nonexistent-tpc-clang"))
    assert builder._resolve_tpc_clang() is None


def test_hpu_init_and_start_scripts_separation(tmp_path):
    import json
    import os
    from pathlib import Path
    import shutil
    import stat
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    work = tmp_path / "checkout"
    (work / "scripts" / "gaudi").mkdir(parents=True)
    init_dst = work / "init-hpu.sh"
    start_root_dst = work / "start-hpu.sh"
    start_dst = work / "scripts" / "gaudi" / "start.sh"
    shutil.copy2(repo_root / "init-hpu.sh", init_dst)
    shutil.copy2(repo_root / "start-hpu.sh", start_root_dst)
    shutil.copy2(repo_root / "scripts" / "gaudi" / "start.sh", start_dst)
    shutil.copy2(repo_root / "uv.toml", work / "uv.toml")
    (work / "scripts" / "gaudi" / "build_qwen35_tpc.py").write_text("# stub\n")
    (work / "main.py").write_text("# stub\n")

    base_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}

    # 1. Fresh clone without .venv -> both start-hpu.sh and scripts/gaudi/start.sh fail fast & prompt init-hpu.sh
    for cmd in (["bash", str(start_dst)], ["bash", str(start_root_dst)]):
        proc = subprocess.run(cmd, cwd=str(work), env=base_env, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "init-hpu.sh" in proc.stderr

    # 2. Install fixed stub .venv/bin/python (start scripts must strictly use .venv/bin/python)
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    stub_py = venv_bin / "python"
    stub_py.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1].endswith('build_qwen35_tpc.py'):\n"
        "    assert '--check' in sys.argv\n"
        "    raise SystemExit(0 if os.environ.get('STUB_CHECK_OK', '1') == '1' else 1)\n"
        "assert sys.argv[1] == 'main.py'\n"
        "Path(os.environ['STUB_OUT']).write_text(json.dumps({\n"
        "    'gc_kernel_path': os.environ.get('GC_KERNEL_PATH', ''),\n"
        "    'uv_config_file': os.environ.get('UV_CONFIG_FILE', ''),\n"
        "    'lazy': os.environ.get('PT_HPU_LAZY_MODE', ''),\n"
        "    'args': sys.argv[1:],\n"
        "}))\n"
    )
    stub_py.chmod(stub_py.stat().st_mode | stat.S_IXUSR)

    # 3. Missing build/hpu/*.so -> fails fast prompting init-hpu.sh (never compiles in start.sh)
    proc_missing = subprocess.run(["bash", str(start_dst)], cwd=str(work), env=base_env, capture_output=True, text=True)
    assert proc_missing.returncode != 0
    assert "init-hpu.sh" in proc_missing.stderr

    # 4. Disabled via COMFY_HPU_DISABLE_QWEN35_TPC=1 -> runs .venv/bin/python without build/hpu/*.so
    out_dis = tmp_path / "out_dis.json"
    env_dis = {**base_env, "COMFY_HPU_DISABLE_QWEN35_TPC": "1", "STUB_OUT": str(out_dis)}
    subprocess.run(["bash", str(start_dst), "--port", "8199"], cwd=str(work), env=env_dis, check=True)
    data_dis = json.loads(out_dis.read_text())
    expected_lib = str(work.resolve() / "build" / "hpu" / "libcomfy_qwen35_tpc.so")
    assert data_dis["lazy"] == "0"
    assert expected_lib not in data_dis["gc_kernel_path"].split(":")
    assert data_dis["uv_config_file"] == str(work.resolve() / "uv.toml")
    assert "--enable-manager" in data_dis["args"]
    assert "--disable-all-custom-nodes" not in data_dis["args"]
    assert data_dis["args"][-2:] == ["--port", "8199"]
    uv_toml_text = (repo_root / "uv.toml").read_text()
    for pkg in ("torch", "torchvision", "torchaudio", "triton", "xformers"):
        assert f'"{pkg}"' in uv_toml_text
    assert "manager_requirements.txt" in (repo_root / "init-hpu.sh").read_text()

    # 5. Stale build/hpu/*.so (--check fails) -> fails fast prompting init-hpu.sh
    (work / "build" / "hpu").mkdir(parents=True)
    (work / "build" / "hpu" / "libcomfy_qwen35_tpc.so").write_bytes(b"old")
    (work / "build" / "hpu" / "comfy_qwen35_tpc_bridge.so").write_bytes(b"old")
    proc_stale = subprocess.run(
        ["bash", str(start_dst)], cwd=str(work), env={**base_env, "STUB_CHECK_OK": "0"},
        capture_output=True, text=True,
    )
    assert proc_stale.returncode != 0
    assert "out of date" in proc_stale.stderr and "init-hpu.sh" in proc_stale.stderr

    # 6. Ready build/hpu/*.so -> appends libcomfy_qwen35_tpc.so to GC_KERNEL_PATH (without duplicating)
    out_ok = tmp_path / "out_ok.json"
    subprocess.run(
        ["bash", str(start_dst)], cwd=str(work), env={**base_env, "STUB_OUT": str(out_ok)}, check=True,
    )
    assert expected_lib in json.loads(out_ok.read_text())["gc_kernel_path"].split(":")

    out_dedup = tmp_path / "out_dedup.json"
    subprocess.run(
        ["bash", str(start_dst)], cwd=str(work),
        env={**base_env, "STUB_OUT": str(out_dedup), "GC_KERNEL_PATH": f"/usr/lib/habanalabs/libtpc_kernels.so:{expected_lib}"},
        check=True,
    )
    assert json.loads(out_dedup.read_text())["gc_kernel_path"].split(":").count(expected_lib) == 1

    # 7. init-hpu.sh discovers newest /opt/gaudi-install/venv-* with habana_frameworks, links gaudi-base.pth, builds TPC
    install_root = tmp_path / "gaudi-install"
    py_ver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    for name, has_habana in (("venv-1.22.0-100-ubuntu2204", True), ("venv-1.24.2-481-ubuntu2204", True), ("venv-9.99.0-broken", False)):
        vdir = install_root / name
        (vdir / "bin").mkdir(parents=True)
        (vdir / "bin" / "python").symlink_to(sys.executable)
        (vdir / "pyvenv.cfg").write_text(f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n")
        sp = vdir / "lib" / py_ver / "site-packages"
        sp.mkdir(parents=True)
        if has_habana:
            (sp / "habana_frameworks").mkdir()

    init_work = tmp_path / "init_checkout"
    (init_work / "scripts" / "gaudi").mkdir(parents=True)
    shutil.copy2(repo_root / "init-hpu.sh", init_work / "init-hpu.sh")
    tpc_log = tmp_path / "tpc_build_args.json"
    (init_work / "scripts" / "gaudi" / "build_qwen35_tpc.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(tpc_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )

    # Missing install root fails cleanly
    proc_no_base = subprocess.run(
        ["bash", str(init_work / "init-hpu.sh"), "--skip-deps", "--skip-tpc"],
        cwd=str(init_work), env={**base_env, "GAUDI_INSTALL_ROOT": str(tmp_path / "empty")},
        capture_output=True, text=True,
    )
    assert proc_no_base.returncode != 0
    assert "venv-*" in proc_no_base.stderr

    # Valid multi-version install root selects venv-1.24.2-481-ubuntu2204, creates .venv + gaudi-base.pth, invokes TPC build
    subprocess.run(
        ["bash", str(init_work / "init-hpu.sh"), "--skip-deps"],
        cwd=str(init_work), env={**base_env, "GAUDI_INSTALL_ROOT": str(install_root)},
        check=True, capture_output=True, text=True,
    )
    pth_files = list((init_work / ".venv" / "lib").glob("python*/site-packages/gaudi-base.pth"))
    assert len(pth_files) == 1
    assert "venv-1.24.2-481-ubuntu2204" in pth_files[0].read_text()
    assert json.loads(tpc_log.read_text()) == ["--ensure"]

    # --rebuild-tpc passes --force
    subprocess.run(
        ["bash", str(init_work / "init-hpu.sh"), "--skip-deps", "--rebuild-tpc"],
        cwd=str(init_work), env={**base_env, "GAUDI_INSTALL_ROOT": str(install_root)},
        check=True, capture_output=True, text=True,
    )
    assert json.loads(tpc_log.read_text()) == ["--force"]
