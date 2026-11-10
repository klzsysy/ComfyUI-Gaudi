import pytest
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.text_encoders import hpu_kv_attention, qwen35
from comfy.text_encoders.llama import FixedKVCache, precompute_freqs_cis, rope_matrix
from comfy.text_encoders.qwen35 import MTP_MODES, VerifyTree


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    monkeypatch.setattr(args, "use_ck_attention", False)
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def reference(q, key, value, index, tree):
    rows = q.shape[2]
    visible = torch.zeros(rows, key.shape[2], dtype=torch.bool)
    for row in range(rows):
        visible[row, :index] = True
        if tree is None:
            visible[row, index:index + row + 1] = True
        else:
            ancestor = row
            while ancestor >= 0:
                visible[row, index + ancestor] = True
                ancestor = tree.parent[ancestor]
    out = F.scaled_dot_product_attention(q.cpu().double(), key.cpu().double(), value.cpu().double(),
                                         attn_mask=visible, enable_gqa=q.shape[1] != key.shape[1])
    return out.transpose(1, 2).reshape(q.shape[0], rows, -1).float()


@pytest.mark.parametrize("mode,rows", [(None, 1), (None, 8), ("3", 4), ("tree2s2", 8)])
@pytest.mark.parametrize("batch,heads,kv_heads,dim", [(1, 4, 1, 64), (2, 4, 2, 256)])
def test_hpu_cache_prefix_and_masks(hpu_compiler, mode, rows, batch, heads, kv_heads, dim):
    torch.manual_seed(71)
    cache = FixedKVCache.zeros(batch, kv_heads, 617, dim, "hpu", torch.bfloat16, FixedKVCache.shared(batch, "hpu"))
    tree = VerifyTree(MTP_MODES[mode][1], rows, "hpu") if mode else None
    cache.tracker["tree"] = tree
    with torch.no_grad():
        for index in (0, 1, 249, 255, 256, 503, 609, 1):
            cache.key.copy_(torch.randn(cache.key.shape, dtype=torch.bfloat16))
            cache.value.copy_(torch.randn(cache.value.shape, dtype=torch.bfloat16))
            cache.index = index
            cache.prepare(rows)
            q = torch.randn(batch, heads, rows, dim, dtype=torch.bfloat16).to("hpu")
            k = torch.randn(batch, kv_heads, rows, dim, dtype=torch.bfloat16).to("hpu")
            v = torch.randn_like(k)
            cache.write(k, v)
            expected = reference(q, cache.key, cache.value, index, tree)
            before = [x.cpu().clone() for x in (q, cache.key, cache.value, cache.seqlen)]
            actual = cache.attend(q, k, v)
            assert actual.device.type == "hpu" and actual.dtype == torch.bfloat16
            torch.testing.assert_close(actual.cpu().float(), expected, atol=0.012, rtol=0.015)
            for tensor, saved in zip((q, cache.key, cache.value, cache.seqlen), before):
                torch.testing.assert_close(tensor.cpu(), saved, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kv_heads,dim", [(1, 64), (4, 256)])
def test_hpu_tree_rollback_and_topology_update(hpu_compiler, monkeypatch, compiled, kv_heads, dim):
    monkeypatch.setattr(args, "disable_comfy_compiler", not compiled)
    torch.manual_seed(73)
    cache = FixedKVCache.zeros(1, kv_heads, 1025, dim, "hpu", torch.bfloat16, FixedKVCache.shared(1, "hpu"))
    tree = VerifyTree(MTP_MODES["tree2s2"][1], 8, "hpu")
    cache.tracker["tree"] = tree
    cache.index = 254
    with torch.no_grad():
        cache.key.copy_(torch.randn(cache.key.shape, dtype=torch.bfloat16))
        cache.value.copy_(torch.randn(cache.value.shape, dtype=torch.bfloat16))
        for turn in range(3):
            cache.prepare(8)
            q = torch.randn(1, 4 * kv_heads, 8, dim, dtype=torch.bfloat16).to("hpu")
            k = torch.randn(1, kv_heads, 8, dim, dtype=torch.bfloat16).to("hpu")
            v = torch.randn_like(k)
            cache.write(k, v)
            expected = reference(q, cache.key, cache.value, cache.index, tree)
            torch.testing.assert_close(cache.attend(q, k, v).cpu().float(), expected, atol=0.012, rtol=0.015)
            base = cache.index
            path = [0, tree.children[0][-1]]
            cache.advance(8)
            cache.rollback(8 - len(path))
            cache.commit_path(base, path)
            torch.testing.assert_close(cache.key[:, :, base + 1].cpu(), k[:, :, path[1]].cpu(), atol=0, rtol=0)
            torch.testing.assert_close(cache.value[:, :, base + 1].cpu(), v[:, :, path[1]].cpu(), atol=0, rtol=0)
            tree.update(MTP_MODES["tree4s1m8" if turn == 0 else "tree2s2"][1])


@pytest.mark.parametrize("mode", ["disabled", "training", "fp32", "cpu"])
def test_cache_preserves_eager_paths(monkeypatch, mode):
    if mode != "cpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("HPU required")
    monkeypatch.setattr(hpu_kv_attention, "compiled_ops", lambda: pytest.fail("Unexpected HPU compilation"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    cache = FixedKVCache.zeros(1, 1, 19, 64, device, dtype, FixedKVCache.shared(1, device))
    cache.prepare(1)
    k = torch.ones(1, 1, 1, 64, device=device, dtype=dtype)
    cache.write(k, k)
    actual = cache.attend(k, k, k)
    torch.testing.assert_close(actual.cpu(), torch.ones(1, 1, 64, dtype=dtype), atol=0, rtol=0)


def make_attention(device="hpu", dtype=torch.bfloat16, operations=comfy.ops.manual_cast):
    config = qwen35.Qwen35Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32)
    module = qwen35.GatedAttention(config, device=device, dtype=dtype, ops=operations)
    for parameter in module.parameters():
        parameter.data.copy_(torch.randn(parameter.shape, dtype=dtype) * 0.04)
    return module


def make_cache(batch=1, device="hpu", dtype=torch.bfloat16):
    return FixedKVCache.zeros(batch, 2, 128, 32, device, dtype, FixedKVCache.shared(batch, device))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("matrix", [False, True])
def test_hpu_attention_decode_weights_and_state(hpu_compiler, batch, matrix):
    torch.manual_seed(79)
    module = make_attention()
    eager, compiled = make_cache(batch), make_cache(batch)
    with torch.no_grad():
        for turn, rows in enumerate((1, 8, 4, 2, 1)):
            tree = VerifyTree(MTP_MODES["tree2s2"][1], 8, "hpu") if rows == 8 else None
            for cache in (eager, compiled):
                cache.index = 17 + turn * 8
                cache.tracker["tree"] = tree
                cache.prepare(rows)
            eager.key.copy_(torch.randn(eager.key.shape, dtype=torch.bfloat16))
            eager.value.copy_(torch.randn(eager.value.shape, dtype=torch.bfloat16))
            compiled.key.copy_(eager.key)
            compiled.value.copy_(eager.value)
            if turn == 4:
                module.o_proj.weight.mul_(0.75)
            x = torch.randn(batch, rows, 64, dtype=torch.bfloat16).to("hpu")
            before = x.cpu().clone()
            positions = torch.arange(rows, device="hpu").unsqueeze(0) + eager.index
            freqs = precompute_freqs_cis(module.rotary_dim, positions, 10000, device="hpu")
            if matrix:
                freqs = rope_matrix(freqs)
            expected, _ = module._forward(x, freqs_cis=freqs, past_key_value=eager)
            actual, state = module(x, freqs_cis=freqs, past_key_value=compiled)
            assert state is compiled
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.005, rtol=0.015)
            torch.testing.assert_close(compiled.key.cpu(), eager.key.cpu(), atol=0.015, rtol=0.015)
            torch.testing.assert_close(compiled.value.cpu(), eager.value.cpu(), atol=0.005, rtol=0.015)
            torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)


@pytest.mark.parametrize("target", ["module", "projection", "norm", "global"])
def test_hpu_attention_hooks_after_compilation(hpu_compiler, monkeypatch, target):
    module = make_attention()
    cache = make_cache()
    cache.prepare(1)
    x = torch.randn(1, 1, 64, dtype=torch.bfloat16).to("hpu")
    freqs = precompute_freqs_cis(module.rotary_dim, torch.zeros(1, 1, device="hpu"), 10000, device="hpu")
    with torch.no_grad():
        module(x, freqs_cis=freqs, past_key_value=cache)
        if target == "global":
            hook = torch.nn.modules.module.register_module_forward_hook(lambda child, values, output: output * 0.5 if child is module.o_proj else None)
        elif target == "module":
            hook = module.register_forward_hook(lambda child, values, output: (output[0] * 0.5, output[1]))
        else:
            child = module.o_proj if target == "projection" else module.q_norm
            hook = child.register_forward_hook(lambda child, values, output: output * 0.5)
        try:
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected, _ = module(x, freqs_cis=freqs, past_key_value=cache)
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            actual, _ = module(x, freqs_cis=freqs, past_key_value=cache)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.005, rtol=0.015)
        finally:
            hook.remove()


@pytest.mark.parametrize("mode", ["disabled", "training", "fp32", "cpu", "prefill", "no_cache", "custom_ops", "override", "ck"])
def test_attention_preserves_eager_paths(monkeypatch, mode):
    if mode != "cpu" and (not hasattr(torch, "hpu") or not torch.hpu.is_available()):
        pytest.skip("HPU required")
    monkeypatch.setattr(qwen35, "_compiled_hpu_attention", lambda: pytest.fail("Unexpected attention compilation"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(args, "use_ck_attention", mode == "ck")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    class CustomOps(comfy.ops.manual_cast):
        pass
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    module = make_attention(device, dtype, CustomOps if mode == "custom_ops" else comfy.ops.manual_cast)
    module.comfy_attention.method = "override" if mode == "override" else None
    monkeypatch.setattr(module, "_forward", lambda x, attention_mask, freqs, attention, cache: (x, cache))
    cache = None if mode == "no_cache" else make_cache(device=device, dtype=dtype)
    x = torch.ones(1, 17 if mode == "prefill" else 1, 64, device=device, dtype=dtype)
    actual, state = module(x, past_key_value=cache)
    assert actual is x and state is cache
