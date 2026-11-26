"""Optional Gaudi2 TPC-C kernels for Qwen3.5 MTP decoding.

Built during environment setup by ``./init-hpu.sh`` (via
``scripts/gaudi/build_qwen35_tpc.py``) and added to ``GC_KERNEL_PATH`` by
``scripts/gaudi/start.sh``. Falls back cleanly when unavailable.
"""
import logging
import os
from pathlib import Path

import torch


build_dir = Path(__file__).resolve().parents[3] / "build" / "hpu"
kernel_library = build_dir / "libcomfy_qwen35_tpc.so"
bridge_library = build_dir / "comfy_qwen35_tpc_bridge.so"
_kernel_paths = os.environ.get("GC_KERNEL_PATH", "").split(":")
available = False

if (hasattr(torch, "hpu")
        and os.environ.get("COMFY_HPU_DISABLE_QWEN35_TPC", "0") != "1"
        and kernel_library.is_file()
        and bridge_library.is_file()
        and any(Path(p).resolve() == kernel_library.resolve() for p in _kernel_paths if p)):
    try:
        torch.ops.load_library(str(bridge_library))
        available = True
    except OSError as error:
        logging.warning(
            "HPU Qwen3.5 TPC extension could not be loaded; re-run ./init-hpu.sh --rebuild-tpc. %s",
            error,
        )
elif hasattr(torch, "hpu") and os.environ.get("COMFY_HPU_DISABLE_QWEN35_TPC", "0") != "1":
    if not (kernel_library.is_file() and bridge_library.is_file()):
        logging.info(
            "Optional HPU Qwen3.5 TPC kernels not built yet; run ./init-hpu.sh to build them."
        )
    else:
        logging.info(
            "Optional HPU Qwen3.5 TPC kernels are built at %s but not in GC_KERNEL_PATH; "
            "start via ./start-hpu.sh (or scripts/gaudi/start.sh) to enable them.",
            kernel_library,
        )


def _silu_chunks(dim, rows):
    return (12 if rows <= 2 else 3) if dim >= 6144 else 1


def pack_gdn_state(state):
    """Pack [B, vh, k_hd=128, v_hd] FP32 into 4x16 TPC layout [B, vh*(v_hd//16), 32, 64]."""
    bs, vh, k_hd, v_hd = state.shape
    v_blks = v_hd // 16
    return (
        state.view(bs, vh, 2, 4, 16, v_blks, 16)
        .permute(0, 1, 5, 2, 4, 3, 6)
        .reshape(bs, vh * v_blks, 32, 64)
        .contiguous()
    )


def unpack_gdn_state(packed, vh, v_hd):
    """Unpack 4x16 TPC layout [B, vh*(v_hd//16), 32, 64] back to [B, vh, 128, v_hd] FP32."""
    bs = packed.shape[0]
    v_blks = v_hd // 16
    return (
        packed.view(bs, vh, v_blks, 2, 16, 4, 16)
        .permute(0, 1, 3, 5, 4, 2, 6)
        .reshape(bs, vh, 128, v_hd)
        .contiguous()
    )


if available:
    @torch.library.register_fake("custom_op::comfy_qwen35_topk_s1")
    def _fake_topk_s1(logits, pen_mask, anc_toks, has_pen, pres_pen, rep_pen):
        return logits.new_empty((logits.shape[0], (logits.shape[1] + 127) // 128), dtype=torch.float32)

    @torch.library.register_fake("custom_op::comfy_qwen35_topk_s2")
    def _fake_topk_s2(bmax, k):
        return bmax.new_empty((bmax.shape[0], k), dtype=torch.int32)

    @torch.library.register_fake("custom_op::comfy_qwen35_topk_s3")
    def _fake_topk_s3(logits, pen_mask, anc_toks, top_blocks, k, has_pen, pres_pen, rep_pen):
        return (
            logits.new_empty((logits.shape[0], k), dtype=torch.float32),
            logits.new_empty((logits.shape[0], k), dtype=torch.int32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_gdn_rec")
    def _fake_gdn_rec(state, prev_conv, cur_conv, prev_ab, cur_ab, g_decay, dt_bias, ctl, kh, vh, v_hd):
        return (
            torch.empty_like(state),
            cur_conv.new_empty((cur_conv.shape[0], cur_conv.shape[1], vh * v_hd), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_gdn_conv")
    def _fake_gdn_conv(conv_st, prev_proj, cur_proj, conv_w, conv_bias, conv_src, ctl, has_bias):
        return torch.empty_like(conv_st), torch.empty_like(cur_proj)

    @torch.library.register_fake("custom_op::comfy_qwen35_rmsnorm_quant")
    def _fake_rmsnorm_quant(x, res, weight, has_res, add_one, store_norm, eps):
        ox = torch.empty_like(x) if has_res else x.new_empty((1, 1))
        oy = torch.empty_like(x) if store_norm else x.new_empty((1, 1))
        return (
            ox,
            oy,
            x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
            x.new_empty((x.shape[0], 1), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_gated_rmsnorm_quant")
    def _fake_gated_rms_quant(out_f32, z, gdn_nw, vh, eps):
        return (
            z.new_empty(z.shape, dtype=torch.float8_e4m3fn),
            z.new_empty((z.shape[0], 1), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_sig_mul_quant")
    def _fake_sig_mul_quant(x, gate):
        return (
            x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
            x.new_empty((x.shape[0], 1), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_silu_mul_quant_s1")
    def _fake_silu_mul_s1(gate, up):
        ch = _silu_chunks(gate.shape[1], gate.shape[0])
        return (
            torch.empty_like(gate),
            gate.new_empty((gate.shape[0], ch, 64), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_silu_mul_quant_s2")
    def _fake_silu_mul_s2(y_in, cmax):
        return (
            y_in.new_empty(y_in.shape, dtype=torch.float8_e4m3fn),
            y_in.new_empty((y_in.shape[0], 1), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_qk_norm_rope")
    def _fake_qk_norm_rope(qg, xk, qn_w, kn_w, cs, num_heads, num_kv_heads, add_one, eps):
        rows = qg.shape[0]
        return (
            qg.new_empty((1, num_heads, rows, 256), dtype=torch.bfloat16),
            qg.new_empty((1, num_kv_heads, rows, 256), dtype=torch.bfloat16),
            qg.new_empty((rows, num_heads * 256), dtype=torch.bfloat16),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_sample_tree_accept")
    def _fake_sample_accept(top_vals, top_idx, verify_tokens, tops, rand_u, state_in, tree_parent,
                            depth, do_sample, adaptive, temperature, top_p, min_p):
        return (
            state_in.new_empty((19,), dtype=torch.int32),
            state_in.new_empty((8,), dtype=torch.int32),
            state_in.new_empty((8,), dtype=torch.int32),
            state_in.new_empty((4,), dtype=torch.int32),
            state_in.new_empty((7,), dtype=torch.int32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen35_dual_rmsnorm_cat_quant")
    def _fake_dual_rms_cat(emb, hid, w_emb, w_hid, add_one, eps):
        return (
            emb.new_empty((emb.shape[0], emb.shape[1] * 2), dtype=torch.float8_e4m3fn),
            emb.new_empty((emb.shape[0], 1), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_situ_quant")
    def _fake_moss_situ_quant(y):
        return (
            y.new_empty((y.shape[0], y.shape[1] // 2), dtype=torch.float8_e4m3fn),
            y.new_empty((y.shape[0],), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_qkv_prep")
    def _fake_moss_qkv_prep(qkv, qn_w, kn_w, pe_cs, aligned, eps, q_scale, k_scale):
        rows = qkv.shape[0]
        if aligned == 2:
            k_pad = ((rows + 511) // 512) * 512
            return (
                qkv.new_empty((24, rows, 128), dtype=torch.bfloat16),
                qkv.new_empty((24, k_pad, 128), dtype=torch.bfloat16),
                qkv.new_empty((24, k_pad // 2, 256), dtype=torch.float8_e4m3fn),
                qkv.new_empty((24, 64), dtype=torch.float32),
            )
        k_pad = ((rows + 127) // 128) * 128
        qk_dim = 128 if aligned else 192
        return (
            qkv.new_empty((24, rows, qk_dim), dtype=torch.bfloat16),
            qkv.new_empty((24, k_pad, qk_dim), dtype=torch.bfloat16),
            qkv.new_empty((24, k_pad, 128), dtype=torch.float8_e4m3fn),
            qkv.new_empty((24, 64), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_res_adaln_quant")
    def _fake_moss_res_adaln_quant(x, res, gate, eff_scale, shift, eps):
        return (
            x.new_empty(x.shape, dtype=torch.bfloat16),
            x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
            x.new_empty((x.shape[0],), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_hnd_scale_quant")
    def _fake_moss_hnd_scale_quant(attn, scales):
        rows = attn.shape[1]
        return (
            attn.new_empty((rows, 3072), dtype=torch.float8_e4m3fn),
            attn.new_empty((rows,), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_hnd_scale_sum_quant")
    def _fake_moss_hnd_scale_sum_quant(attn, scales, sum_in):
        rows = attn.shape[1]
        return (
            attn.new_empty((rows, 3072), dtype=torch.float8_e4m3fn),
            attn.new_empty((rows,), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_moss_expmax_fp8")
    def _fake_moss_expmax_fp8(scores, valid_keys):
        return (
            scores.new_empty(scores.shape, dtype=torch.float8_e4m3fn),
            scores.new_empty((scores.shape[0],), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_vae_rmsnorm_silu_bf16")
    def _fake_vae_rmsnorm_silu_bf16(x, gamma, n_offset, n_len, do_silu, scale):
        return x.new_empty((x.shape[0], x.shape[1], n_len), dtype=torch.bfloat16)

    @torch.library.register_fake("custom_op::comfy_vae_dupup2d_add_bf16")
    def _fake_vae_dupup2d_add_bf16(x, main_t, c_out, mode, has_main):
        return x.new_empty((x.shape[0], c_out, x.shape[2] * 2, x.shape[3] * 2), dtype=torch.bfloat16)

    @torch.library.register_fake("custom_op::comfy_qwen_expmax_fp8")
    def _fake_qwen_expmax_fp8(scores, valid_keys):
        return (
            scores.new_empty(scores.shape, dtype=torch.float8_e4m3fn),
            scores.new_empty((scores.shape[0],), dtype=torch.float32),
            scores.new_empty((scores.shape[0],), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen_qk_rope_fp8")
    def _fake_qwen_qk_rope_fp8(q, k, pk, qn_w, kn_w, pe_cs, chunks, eps, q_scale, k_scale):
        queries = q.shape[0]
        k_pad = ((pk.shape[0] + queries + 255) // 256) * 256
        return (
            q.new_empty((32, queries, 128), dtype=torch.bfloat16),
            q.new_empty((32, k_pad, 128), dtype=torch.bfloat16),
        )

    @torch.library.register_fake("custom_op::comfy_qwen_v_quant_fp8")
    def _fake_qwen_v_quant_fp8(v, pv, v_scales, chunks):
        k_pad = ((pv.shape[0] + v.shape[0] + 255) // 256) * 256
        return v.new_empty((32, k_pad // 2, 256), dtype=torch.float8_e4m3fn)

    @torch.library.register_fake("custom_op::comfy_qwen_hnd_scale_sum_quant")
    def _fake_qwen_hnd_scale_sum_quant(attn, scales, sum_in):
        rows = attn.shape[1]
        return (
            attn.new_empty((rows, 4096), dtype=torch.float8_e4m3fn),
            attn.new_empty((rows,), dtype=torch.float32),
        )

    @torch.library.register_fake("custom_op::comfy_qwen_res_adaln_quant")
    def _fake_qwen_res_adaln_quant(x, res, gate, scale_p1, has_res, eps):
        ox = torch.empty_like(x) if has_res else x.new_empty((1, 1))
        return (
            ox,
            x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
            x.new_empty((x.shape[0],), dtype=torch.float32),
        )


def topk(logits, pen_mask, anc_toks, k, has_pen, pres_pen, rep_pen):
    bmax = torch.ops.custom_op.comfy_qwen35_topk_s1(logits, pen_mask, anc_toks, has_pen, pres_pen, rep_pen)
    blocks = torch.ops.custom_op.comfy_qwen35_topk_s2(bmax, k)
    return torch.ops.custom_op.comfy_qwen35_topk_s3(logits, pen_mask, anc_toks, blocks, k, has_pen, pres_pen, rep_pen)


def gdn_rec(state, prev_conv, cur_conv, prev_ab, cur_ab, g_decay, dt_bias, ctl, kh, vh, v_hd):
    return torch.ops.custom_op.comfy_qwen35_gdn_rec(
        state, prev_conv, cur_conv, prev_ab, cur_ab, g_decay, dt_bias, ctl, kh, vh, v_hd,
    )


def gdn_conv(conv_st, prev_proj, cur_proj, conv_w, conv_bias, conv_src, ctl, has_bias):
    return torch.ops.custom_op.comfy_qwen35_gdn_conv(conv_st, prev_proj, cur_proj, conv_w, conv_bias, conv_src, ctl, has_bias)


def rmsnorm_quant(x, res, weight, has_res, add_one, store_norm, eps):
    return torch.ops.custom_op.comfy_qwen35_rmsnorm_quant(x, res, weight, has_res, add_one, store_norm, eps)


def gated_rmsnorm_quant(out_f32, z, gdn_nw, vh, eps):
    return torch.ops.custom_op.comfy_qwen35_gated_rmsnorm_quant(out_f32, z, gdn_nw, vh, eps)


def sig_mul_quant(x, gate):
    return torch.ops.custom_op.comfy_qwen35_sig_mul_quant(x, gate)


def silu_mul_quant(gate, up):
    y_in, cmax = torch.ops.custom_op.comfy_qwen35_silu_mul_quant_s1(gate, up)
    return torch.ops.custom_op.comfy_qwen35_silu_mul_quant_s2(y_in, cmax)


def qk_norm_rope(qg, xk, qn_w, kn_w, cs, num_heads, num_kv_heads, add_one, eps):
    return torch.ops.custom_op.comfy_qwen35_qk_norm_rope(
        qg, xk, qn_w, kn_w, cs, num_heads, num_kv_heads, add_one, eps,
    )


def sample_tree_accept(top_vals, top_idx, verify_tokens, tops, rand_u, state_in, tree_parent,
                       depth, do_sample, adaptive, temperature, top_p, min_p):
    return torch.ops.custom_op.comfy_qwen35_sample_tree_accept(
        top_vals, top_idx, verify_tokens, tops, rand_u, state_in, tree_parent,
        depth, do_sample, adaptive, temperature, top_p, min_p,
    )


def dual_rmsnorm_cat_quant(emb, hid, w_emb, w_hid, add_one, eps):
    return torch.ops.custom_op.comfy_qwen35_dual_rmsnorm_cat_quant(emb, hid, w_emb, w_hid, add_one, eps)


def moss_situ_quant(y):
    return torch.ops.custom_op.comfy_moss_situ_quant(y)


def moss_qkv_prep(qkv, qn_w, kn_w, pe_cs, aligned, eps, q_scale, k_scale):
    qh, kh, vh, out_scales = torch.ops.custom_op.comfy_moss_qkv_prep(
        qkv, qn_w, kn_w, pe_cs, aligned, eps, q_scale, k_scale,
    )
    if aligned == 2:
        vh = vh.view(24, vh.shape[1] * 2, 128)
    return qh, kh, vh, out_scales


def moss_res_adaln_quant(x, res, gate, eff_scale, shift, eps):
    return torch.ops.custom_op.comfy_moss_res_adaln_quant(x, res, gate, eff_scale, shift, eps)


def moss_hnd_scale_quant(attn, scales):
    return torch.ops.custom_op.comfy_moss_hnd_scale_quant(attn, scales)


def moss_hnd_scale_sum_quant(attn, scales, sum_in):
    return torch.ops.custom_op.comfy_moss_hnd_scale_sum_quant(attn, scales, sum_in)


def moss_expmax_fp8(scores, valid_keys):
    return torch.ops.custom_op.comfy_moss_expmax_fp8(scores, valid_keys)


def vae_rmsnorm_silu_bf16(x, gamma, n_offset, n_len, do_silu, scale):
    return torch.ops.custom_op.comfy_vae_rmsnorm_silu_bf16(x, gamma, n_offset, n_len, do_silu, scale)


def vae_dupup2d_add_bf16(x, main_t, c_out, mode, has_main):
    b, c_in, h, w = x.shape
    # Viewing through 3D strips HPU internal 4D Conv2d layout metadata (zero-copy when already NCHW).
    x4 = x.reshape(b, c_in, h * w).view(b, c_in, h, w)
    if has_main:
        m4 = main_t.reshape(b, c_out, (h * 2) * (w * 2)).view(b, c_out, h * 2, w * 2)
    else:
        m4 = main_t
    return torch.ops.custom_op.comfy_vae_dupup2d_add_bf16(x4, m4, c_out, mode, has_main)


def qwen_expmax_fp8(scores, valid_keys):
    return torch.ops.custom_op.comfy_qwen_expmax_fp8(scores, valid_keys)


def qwen_qk_rope_fp8(q, k, pk, qn_w, kn_w, pe_cs, chunks, eps, q_scale, k_scale):
    return torch.ops.custom_op.comfy_qwen_qk_rope_fp8(q, k, pk, qn_w, kn_w, pe_cs, chunks, eps, q_scale, k_scale)


def qwen_v_quant_fp8(v, pv, v_scales, chunks):
    return torch.ops.custom_op.comfy_qwen_v_quant_fp8(v, pv, v_scales, chunks)


def qwen_hnd_scale_sum_quant(attn, scales, sum_in):
    return torch.ops.custom_op.comfy_qwen_hnd_scale_sum_quant(attn, scales, sum_in)


def qwen_res_adaln_quant(x, res, gate, scale_p1, has_res, eps):
    return torch.ops.custom_op.comfy_qwen_res_adaln_quant(x, res, gate, scale_p1, has_res, eps)
