import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

import comfy.model_base
import comfy.model_management
import comfy.ops
import comfy.supported_models
from comfy.cli_args import args


@pytest.fixture
def hpu(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def model(device="hpu", dtype=torch.bfloat16):
    config = comfy.supported_models.MOSSImage(dict(dim=64, ffn_dim=128, num_heads=2, num_layers=2,
                                                 text_dim=32, latent_channels=128, fused_qkv=True, dtype=dtype))
    config.custom_operations = comfy.ops.manual_cast
    module = comfy.model_base.MOSSImage(config, device=device)
    for name, parameter in module.named_parameters():
        parameter.data.copy_(torch.randn(parameter.shape, dtype=dtype) * 0.03 + (1 if "norm" in name and name.endswith("weight") else 0))
    return module


def process(conditions, batch=1):
    return {name: condition.process_cond(batch_size=batch).concat([]) for name, condition in conditions.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_preparation_reuses_static_inputs(hpu, monkeypatch, batch, weighted):
    torch.manual_seed(81)
    module = model()
    x = torch.randn(batch, 128, 4, 5, device="hpu", dtype=torch.float32)
    context = torch.randn(batch, 7, 32, device="hpu", dtype=torch.bfloat16)
    bias = torch.randn(batch, 7, device="hpu", dtype=torch.bfloat16) * 0.1 if weighted else None
    options = {"rope_options": {"scale_x": 1.2, "shift_y": 0.5}}
    module.current_patcher = SimpleNamespace(model_options={"transformer_options": options}, hook_patches={}, is_dynamic=lambda: False)
    with torch.inference_mode():
        expected = [module.apply_model(x, torch.full((batch,), t, device="hpu"), c_crossattn=context,
                                       context_attention_bias=bias, transformer_options=options) for t in (0.9, 0.5, 0.1)]
        counts = {"rope": 0, "text": 0}
        rope, text = module.diffusion_model.rope, module.diffusion_model.text_embedder.forward
        def count_rope(*values, **kwargs):
            counts["rope"] += 1
            return rope(*values, **kwargs)
        def count_text(*values, **kwargs):
            counts["text"] += 1
            return text(*values, **kwargs)
        monkeypatch.setattr(module.diffusion_model, "rope", count_rope)
        monkeypatch.setattr(module.diffusion_model.text_embedder, "forward", count_text)
        prepared = process(module.extra_conds(noise=x, device=x.device, cross_attn=context, context_attention_bias=bias, latent_shapes=[list(x.shape)]), batch)
        assert prepared['prepared_rope']['pe'].dtype == torch.float32
        before = prepared['prepared_context'].clone()
        for index, t in enumerate((0.9, 0.5, 0.1)):
            actual = module.apply_model(x, torch.full((batch,), t, device="hpu"), transformer_options=options, **prepared)
            torch.testing.assert_close(actual.cpu(), expected[index].cpu(), atol=0, rtol=0)
        assert counts == {"rope": 1, "text": 1}
        torch.testing.assert_close(prepared['prepared_context'].cpu(), before.cpu(), atol=0, rtol=0)


@pytest.mark.parametrize("change", ["shape", "rope_options", "hook", "references"])
def test_changed_forward_inputs_use_original_path(hpu, change):
    module = model()
    x = torch.randn(1, 128, 4, 5, device="hpu")
    context = torch.randn(1, 7, 32, dtype=torch.bfloat16, device="hpu")
    with torch.inference_mode():
        prepared = process(module.extra_conds(noise=x, device=x.device, cross_attn=context))
        options, calls, hook, extra = {}, [], None, {}
        if change == "shape":
            x = x[:, :, :3]
        elif change == "rope_options":
            options = {"rope_options": {"shift_y": 0.7}}
        elif change == "references":
            extra['ref_latents'] = [torch.randn(1, 128, 2, 2, device="hpu", dtype=torch.bfloat16)]
        else:
            hook = module.diffusion_model.text_embedder.proj.register_forward_hook(lambda child, values, output: calls.append(True) or output * 0.5)
        try:
            expected = module.apply_model(x, torch.ones(1, device="hpu"), c_crossattn=context, transformer_options=options, **extra)
            actual = module.apply_model(x, torch.ones(1, device="hpu"), transformer_options=options, **prepared, **extra)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
            if hook is not None:
                assert len(calls) == 2
        finally:
            if hook is not None:
                hook.remove()


@pytest.mark.parametrize("mode", ["cpu", "fp32", "training", "grad", "hooks", "area", "control", "packed", "options"])
def test_preparation_keeps_unsupported_paths_eager(hpu, monkeypatch, mode):
    device = "cpu" if mode == "cpu" else "hpu"
    dtype = torch.float32 if mode in ("cpu", "fp32") else torch.bfloat16
    module = model(device, dtype)
    monkeypatch.setattr(module.diffusion_model, "preprocess_text", lambda *values: pytest.fail("Unexpected preparation"))
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    extra = {}
    if mode in ("hooks", "control"):
        extra[mode] = object()
    elif mode == "area":
        extra['area'] = (3, 4, 0, 0)
    elif mode == "packed":
        extra['latent_shapes'] = [(4, 5)]
    elif mode == "options":
        module.current_patcher = SimpleNamespace(model_options={'transformer_options': {'optimized_attention_override': object()}}, hook_patches={})
    with torch.inference_mode(mode != "grad"):
        out = module.extra_conds(noise=torch.zeros(1, 128, 4, 5, device=device), device=torch.device(device),
                                 cross_attn=torch.ones(1, 7, 32, device=device, dtype=dtype), **extra)
    assert 'prepared_context' not in out and 'prepared_rope' not in out


def test_prepared_tensors_are_owned_by_conditioning(hpu):
    module = model()
    with torch.inference_mode():
        out = module.extra_conds(noise=torch.zeros(1, 128, 4, 5, device="hpu"), device=torch.device("hpu"),
                                 cross_attn=torch.ones(1, 7, 32, dtype=torch.bfloat16, device="hpu"))
    refs = [weakref.ref(out['prepared_context'].cond), weakref.ref(out['prepared_rope'].cond['pe'])]
    del out
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_prepared_conditions_can_batch(hpu):
    module = model()
    x = torch.randn(2, 128, 4, 5, device="hpu")
    context = torch.randn(2, 7, 32, dtype=torch.bfloat16, device="hpu")
    with torch.inference_mode():
        first = module.extra_conds(noise=x[:1], device=x.device, cross_attn=context[:1])
        second = module.extra_conds(noise=x[1:], device=x.device, cross_attn=context[1:])
        assert all(first[name].can_concat(second[name]) for name in first)
        merged = {name: first[name].concat([second[name]]) for name in first}
        expected = module.apply_model(x, torch.ones(2, device="hpu"), c_crossattn=context)
        actual = module.apply_model(x, torch.ones(2, device="hpu"), **merged)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.005, rtol=0.01)
