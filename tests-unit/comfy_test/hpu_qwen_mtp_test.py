import json
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import hpu_kv_attention, hpu_qwen35, qwen35
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


def test_mtp_reuses_compiled_graphs_across_kv_capacities(hpu_env, monkeypatch):
    model = make_mtp_model(quantized=True, quantize_mtp=True, weight_seed=101)
    backend = lookup_backend("hpu_backend")
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
