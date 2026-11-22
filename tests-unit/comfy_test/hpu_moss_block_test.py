import pytest
import torch

import comfy.model_management
import comfy.ops
from comfy.cli_args import args
from comfy.ldm.moss_image import model as moss


@pytest.fixture
def hpu_compiler(monkeypatch):
    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        pytest.skip("HPU required")
    monkeypatch.setattr(comfy.model_management, "in_training", False)
    monkeypatch.setattr(args, "disable_comfy_compiler", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_block(fused=True, device="hpu", dtype=torch.bfloat16):
    block = moss.Block(128, 192, 4, fused_qkv=fused, dtype=dtype, device=device, operations=comfy.ops.manual_cast)
    for name, value in block.named_parameters():
        value.data.copy_(torch.randn(value.shape, dtype=dtype).to(device) * 0.03 + (1 if "norm" in name else 0))
    return block


def inputs(length, device="hpu", dtype=torch.bfloat16):
    x = torch.randn(2, length, 128, dtype=dtype).to(device)
    mod = tuple(torch.randn(2, 1, 128, dtype=dtype).to(device) * 0.1 for _ in range(6))
    angle = torch.randn(1, length, 1, 16)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, length, 1, 16, 2, 2).to(device)
    return x, mod, pe


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("fp8_regions", [False, True])
@pytest.mark.parametrize("mode", ["plain", "mask", "kv", "reference_tokens"])
def test_compiled_moss_block_contract(hpu_compiler, monkeypatch, fused, fp8_regions, mode):
    torch.manual_seed(52)
    block = make_block(fused)
    block.hpu_fp8 = fp8_regions
    with torch.no_grad():
        for turn, length in enumerate((17, 31, 17)):
            x, mod, pe = inputs(length)
            extra = {}
            if turn == 2:
                for parameter in block.proj['o'].parameters():
                    parameter.copy_(torch.randn_like(parameter) * 0.03)
            if mode == "mask":
                extra['mask'] = torch.randn(2, 1, 1, length, dtype=torch.bfloat16).to("hpu") * 0.2
            elif mode == "kv":
                extra['ref_kv'] = torch.randn(2, 2, 5, 128, dtype=torch.bfloat16).to("hpu")
                extra['kv_out'] = torch.empty(2, 2, length, 128, device="hpu", dtype=torch.bfloat16)
            elif mode == "reference_tokens":
                extra['ref'] = (length - 5, tuple(t * 0.7 for t in mod))
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected = block(x.clone(), mod, pe, {}, **extra).cpu()
            expected_kv = extra['kv_out'].cpu().clone() if mode == "kv" else None
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            working = x.clone()
            actual = block(working, mod, pe, {}, **extra)
            assert actual.data_ptr() == working.data_ptr()
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)
            if expected_kv is not None:
                torch.testing.assert_close(extra['kv_out'].cpu(), expected_kv, rtol=0.01, atol=0.01)


def test_moss_block_many_lengths(hpu_compiler, monkeypatch):
    block = make_block()
    with torch.no_grad():
        for length in (17, 23, 31, 43, 53, 61, 71, 83, 97, 107, 17):
            x, mod, pe = inputs(length)
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected = block(x.clone(), mod, pe).cpu()
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            actual = block(x.clone(), mod, pe)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.015, atol=0.015)


@pytest.mark.parametrize("fp8_regions", [False, True])
def test_moss_block_honors_hooks_added_after_compile(hpu_compiler, monkeypatch, fp8_regions):
    block = make_block()
    block.hpu_fp8 = fp8_regions
    x, mod, pe = inputs(17)
    with torch.no_grad():
        block(x.clone(), mod, pe)
        hook = block.proj['o'].register_forward_hook(lambda module, values, output: output * 0.5)
        try:
            monkeypatch.setattr(args, "disable_comfy_compiler", True)
            expected = block(x.clone(), mod, pe)
            monkeypatch.setattr(args, "disable_comfy_compiler", False)
            actual = block(x.clone(), mod, pe)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        finally:
            hook.remove()


def test_cpu_moss_block_does_not_compile_hpu(monkeypatch):
    monkeypatch.setattr(moss, "_compiled_block_ops", lambda: pytest.fail("HPU compiler used for CPU"))
    block = make_block(device="cpu", dtype=torch.float32)
    with torch.no_grad():
        result = block(*inputs(17, "cpu", torch.float32))
    assert result.shape == (2, 17, 128) and torch.isfinite(result).all()


@pytest.mark.parametrize("mode", ["disabled", "training", "override"])
def test_moss_block_eager_paths(hpu_compiler, monkeypatch, mode):
    monkeypatch.setattr(moss, "_compiled_block_ops", lambda: pytest.fail("block compiler should be bypassed"))
    monkeypatch.setattr(moss, "_compiled_block_forward", lambda: pytest.fail("FP8 block compiler should be bypassed"))
    monkeypatch.setattr(args, "disable_comfy_compiler", mode == "disabled")
    monkeypatch.setattr(comfy.model_management, "in_training", mode == "training")
    block = make_block()
    block.hpu_fp8 = True
    options, calls = {}, []
    if mode == "override":
        def override(function, *values, **kwargs):
            calls.append(True)
            return function(*values, **kwargs)
        options['optimized_attention_override'] = override
    with torch.no_grad():
        result = block(*inputs(17), transformer_options=options)
    assert torch.isfinite(result).all()
    if mode == "override":
        assert calls == [True]


def _make_hpu_fp8_linear(in_features, out_features, device="hpu"):
    from comfy.quant_ops import QuantizedTensor, HPUFP8E4M3Layout

    ops = comfy.ops.mixed_precision_ops({"formats": ["float8_e4m3fn"]}, load_device=torch.device(device))
    lin = ops.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device) * 0.03
    w_scale = w_bf16.abs().amax().float().clamp_min(1e-12) / 224.0
    w_fp8 = torch.ops.hpu.cast_to_fp8_v2(w_bf16, w_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    w_qt = QuantizedTensor(
        w_fp8, "HPUFP8E4M3Layout",
        HPUFP8E4M3Layout.Params(scale=w_scale, orig_dtype=torch.bfloat16, orig_shape=(out_features, in_features)),
    )
    lin.weight = torch.nn.Parameter(w_qt, requires_grad=False)
    lin.layout_type = "HPUFP8E4M3Layout"
    lin.quant_format = "float8_e4m3fn"
    return lin


@pytest.mark.parametrize("use_tpc", [False, True])
def test_linear_input_act_hpu_fp8_no_graph_breaks(hpu_compiler, monkeypatch, use_tpc):
    from comfy.text_encoders import hpu_qwen35_tpc

    if use_tpc and not hpu_qwen35_tpc.available:
        pytest.skip("HPU TPC extension not active in GC_KERNEL_PATH")
    if not use_tpc:
        monkeypatch.setattr(comfy.ops, "_hpu_tpc", None)
    torch.manual_seed(111)
    lin = _make_hpu_fp8_linear(512, 256)
    y = torch.randn(1, 131, 1024, dtype=torch.bfloat16, device="hpu")

    def fn(inp):
        return comfy.ops.linear_input_act(lin, inp, "situ")

    with torch.no_grad():
        explanation = torch._dynamo.explain(fn)(y)
        assert explanation.graph_break_count == 0
        expected = lin(moss.situ(*y.chunk(2, dim=-1))).cpu()
        actual = torch.compile(fn, backend="hpu_backend", dynamic=False)(y).cpu()
        w_ref = lin.weight.dequantize().float().cpu()
        y_fp32 = moss.situ(*y.cpu().float().chunk(2, dim=-1))
        exact_fp32 = torch.nn.functional.linear(y_fp32, w_ref)
        rel_vs_pt = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
        rel_actual_fp32 = (actual.float() - exact_fp32).norm() / exact_fp32.norm().clamp_min(1e-12)
        rel_expected_fp32 = (expected.float() - exact_fp32).norm() / exact_fp32.norm().clamp_min(1e-12)
        assert torch.isfinite(actual).all() and rel_vs_pt < 0.03
        assert rel_actual_fp32 <= rel_expected_fp32 + 0.005


def test_moss_tpc_kernels_accuracy_and_shapes(hpu_compiler):
    from comfy.ldm.moss_image import hpu as moss_hpu
    from comfy.text_encoders import hpu_qwen35_tpc

    if not hpu_qwen35_tpc.available:
        pytest.skip("HPU TPC extension not active in GC_KERNEL_PATH")
    torch.manual_seed(123)
    with torch.no_grad():
        # 1. moss_situ_quant
        y = torch.randn(257, 16384, dtype=torch.bfloat16, device="hpu") * 2.0
        q, sc = torch.compile(hpu_qwen35_tpc.moss_situ_quant, backend="hpu_backend", dynamic=False)(y)
        assert q.shape == (257, 8192) and q.dtype == torch.float8_e4m3fn
        assert sc.shape == (257,) and sc.dtype == torch.float32
        deq = q.cpu().float() * sc.cpu().unsqueeze(-1)
        h64, g64 = y.cpu().double().chunk(2, dim=-1)
        ref = (4.0 * torch.tanh(g64 / 4.0) * torch.sigmoid(g64)) * (25.0 * torch.tanh(h64 / 25.0))
        assert (deq - ref.float()).norm() / ref.float().norm() < 0.035

        # 2. moss_qkv_prep (aligned=1 at 256, aligned=0 at 261)
        qw = (torch.randn(128, dtype=torch.bfloat16, device="hpu") * 0.1 + 1.0).contiguous()
        kw = (torch.randn(128, dtype=torch.bfloat16, device="hpu") * 0.1 + 1.0).contiguous()
        sc25 = 128.0 ** -0.25
        for rows in (256, 261):
            aligned = 1 if (rows % 128) == 0 else 0
            k_pad = ((rows + 127) // 128) * 128
            qk_dim = 128 if aligned else 192
            qkv = torch.randn(rows, 9216, dtype=torch.bfloat16, device="hpu")
            angle = torch.randn(1, rows, 1, 64)
            pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, rows, 1, 64, 2, 2).to("hpu")
            pe_cs = moss_hpu.pack_blockwise_pe_cs(pe)
            qh, kh, vh, out_sc = torch.compile(
                lambda a, b, c, d: hpu_qwen35_tpc.moss_qkv_prep(a, b, c, d, aligned, 1e-6, sc25, sc25),
                backend="hpu_backend", dynamic=False,
            )(qkv, qw, kw, pe_cs)
            assert qh.shape == (24, rows, qk_dim) and kh.shape == (24, k_pad, qk_dim)
            assert vh.shape == (24, k_pad, 128) and out_sc.shape == (24, 64)
            qr, kr, vr = qkv.unsqueeze(0).chunk(3, dim=-1)
            qr_ref, kr_ref = moss_hpu.rms_rope_split_half(
                qr.view(1, rows, 24, 128), kr.view(1, rows, 24, 128), pe, qw, kw, 1e-6, blockwise=True
            )
            qh_ref = (qr_ref.transpose(1, 2)[0] * sc25).cpu()
            kh_ref = (kr_ref.transpose(1, 2)[0] * sc25).cpu()
            vh_ref = vr.view(1, rows, 24, 128).transpose(1, 2)[0].cpu()
            assert (qh[:, :, :128].cpu().float() - qh_ref.float()).norm() / qh_ref.float().norm() < 0.01
            assert (kh[:, :rows, :128].cpu().float() - kh_ref.float()).norm() / kh_ref.float().norm() < 0.01
            v_deq = vh[:, :rows, :].cpu().float() * (out_sc[:, 0].cpu().view(24, 1, 1) * 128.0)
            assert (v_deq - vh_ref.float()).norm() / vh_ref.float().norm() < 0.035
            if not aligned:
                assert (qh[:, :, 128].cpu() == 1.0).all() and (qh[:, :, 129:].cpu() == 0.0).all()
                assert (kh[:, :rows, 128:].cpu() == 0.0).all()
                assert (kh[:, rows:, :128].cpu() == 0.0).all()
                assert (kh[:, rows:, 128].cpu() == -8192.0).all()
                assert (vh[:, rows:, :].cpu().float() == 0.0).all()


@pytest.mark.parametrize("length", [4096, 4119])
def test_moss_fp8_block_rmw_tpc_fast_path(hpu_compiler, monkeypatch, length):
    from comfy.ldm.modules import hpu_attention
    from comfy.ldm.moss_image import hpu as moss_hpu
    from comfy.text_encoders import hpu_qwen35_tpc

    if not hpu_qwen35_tpc.available:
        pytest.skip("HPU TPC extension not active in GC_KERNEL_PATH")
    torch.manual_seed(141 + length)
    block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, device="hpu", operations=comfy.ops.manual_cast)
    for name, value in block.named_parameters():
        value.data.copy_(torch.randn(value.shape, dtype=torch.bfloat16, device="hpu") * 0.02 + (1 if "norm" in name else 0))
    for pname, in_f, out_f in (("qkv", 3072, 9216), ("o", 3072, 3072), ("ffn_up_gate", 3072, 16384), ("ffn_down", 8192, 3072)):
        block.proj[pname] = _make_hpu_fp8_linear(in_f, out_f)
    block.hpu_fp8 = True

    x = torch.randn(1, length, 3072, dtype=torch.bfloat16, device="hpu")
    mod = tuple(torch.randn(1, 1, 3072, dtype=torch.bfloat16, device="hpu") * 0.1 for _ in range(6))
    angle = torch.randn(1, length, 1, 64)
    pe = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).reshape(1, length, 1, 64, 2, 2).to("hpu")
    pe_cs = moss_hpu.pack_blockwise_pe_cs(pe)
    opts = {"fp8_attention": True, "_moss_hpu_blockwise_rope": True, "_moss_hpu_pe_cs": pe_cs}

    with torch.no_grad():
        monkeypatch.setattr(args, "disable_comfy_compiler", True)
        expected = block(x.clone(), mod, pe, {"fp8_attention": False}).cpu()
        monkeypatch.setattr(args, "disable_comfy_compiler", False)
        working = x.clone()
        actual = block(working, mod, pe, opts)
        assert actual.data_ptr() == working.data_ptr()
        assert hpu_attention._fp8_runtime().last_workspace <= 256
        rel = (actual.cpu().float() - expected.float()).norm() / expected.float().norm()
        assert torch.isfinite(actual).all() and rel < 0.04

        # Verify fallback when TPC extension is disabled (still uses RMW FP8 attention cleanly).
        monkeypatch.setattr(moss, "_hpu_tpc", None)
        monkeypatch.setattr(comfy.ops, "_hpu_tpc", None)
        working_notpc = x.clone()
        actual_notpc = block(working_notpc, mod, pe, opts)
        assert actual_notpc.data_ptr() == working_notpc.data_ptr()
        rel_notpc = (actual_notpc.cpu().float() - expected.float()).norm() / expected.float().norm()
        assert torch.isfinite(actual_notpc).all() and rel_notpc < 0.04

        # Verify explicit fp8_attention=False bypasses RMW FP8 attention completely.
        monkeypatch.setattr(hpu_attention, "rmw_fp8_attention_prepared", lambda *a: pytest.fail("RMW prepared should be bypassed"))
        monkeypatch.setattr(hpu_attention, "rmw_fp8_attention", lambda *a: pytest.fail("RMW should be bypassed"))
        out_no_fp8 = block(x.clone(), mod, pe, {"fp8_attention": False, "_moss_hpu_pe_cs": pe_cs})
        assert torch.isfinite(out_no_fp8).all()
