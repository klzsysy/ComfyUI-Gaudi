// Fused MOSS QKV split + FP32 Q/K RMSNorm + blockwise split-half RoPE + Q/K scaling
// + HND transpose + key alignment padding + per-head dynamic FP8 E4M3 V quantization
// for Gaudi2 RMW SRAM attention.
//
// 24 TPC cores == 24 heads! Each core h owns all rows of head h.
// Modes (passed via `mode`):
//   mode == 0 : legacy unaligned RMW softmax (qk_dim=192, k_pad=ceil_128, linear V, out_scales=vscale/128)
//   mode == 1 : legacy aligned   RMW softmax (qk_dim=128, k_pad=rows,     linear V, out_scales=vscale/128)
//   mode == 2 : P1 expmax        RMW attention (qk_dim=128 always, k_pad=ceil_512, non-linear Q/K since Q@K^T
//               is channel-permutation-invariant, paired 256B all_lanes-permuted V_fp8 [256, k_pad/2, 24],
//               out_scales=vscale)
//
// Inputs:
//   qkv   : [9216, rows] BF16 (Q = 0..3071, K = 3072..6143, V = 6144..9215)
//   qn_w  : [128] BF16
//   kn_w  : [128] BF16
//   pe_cs : [128, rows] FP32 (0..63 = cos64, 64..127 = sin64 = pe[..., 1, 0])

static inline float64 ld_bf16_lin(int5 p, tensor t) {
    bfloat128 u = v_bf16_ld_tnsr_b(p, t, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
    return v_convert_bf16_to_f32_b(u, 0, 0, 1, 0);
}

void main(tensor qkv, tensor qn_w, tensor kn_w, tensor pe_cs,
          tensor qh_out, tensor kh_out, tensor vh_q_out, tensor out_scales,
          int rows, int k_pad, int qk_dim, int mode, float eps, float q_scale, float k_scale) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_128 = 1.0f / 128.0f;
    const float inv_240 = 1.0f / 240.0f;
    const bfloat c240 = (bfloat)240.0f;
    const bfloat cn240 = (bfloat)-240.0f;
    const bfloat128 zero_bf = (bfloat128)0.0f;
    const minifloat256 zero_f8 = (minifloat256)0;
    const shuffle_mask_results_t m32 = gen_shuffle_map_32bit_nolut();

    float64 qw0 = ld_bf16_lin((int5){0, 0, 0, 0, 0}, qn_w) * (float64)q_scale;
    float64 qw1 = ld_bf16_lin((int5){64, 0, 0, 0, 0}, qn_w) * (float64)q_scale;
    float64 kw0 = ld_bf16_lin((int5){0, 0, 0, 0, 0}, kn_w) * (float64)k_scale;
    float64 kw1 = ld_bf16_lin((int5){64, 0, 0, 0, 0}, kn_w) * (float64)k_scale;

    if (mode == 2) {
        for (int h = begin[0]; h < end[0] && h < 24; ++h) {
            const int q_col = h * 128;
            const int k_col = 3072 + q_col;
            const int v_col = 6144 + q_col;

            bfloat128 vmax_b0 = (bfloat128)0.0f;
            bfloat128 vmax_b1 = (bfloat128)0.0f;
            int r = 0;
            for (; r + 1 < rows; r += 2) {
                float64 cA = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
                float64 sA = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
                float64 cB = v_f32_ld_tnsr_b((int5){0, r + 1, 0, 0, 0}, pe_cs);
                float64 sB = v_f32_ld_tnsr_b((int5){64, r + 1, 0, 0, 0}, pe_cs);

                float64 q0A = ld_bf16_lin((int5){q_col, r, 0, 0, 0}, qkv);
                float64 q1A = ld_bf16_lin((int5){q_col + 64, r, 0, 0, 0}, qkv);
                float64 q0B = ld_bf16_lin((int5){q_col, r + 1, 0, 0, 0}, qkv);
                float64 q1B = ld_bf16_lin((int5){q_col + 64, r + 1, 0, 0, 0}, qkv);

                float64 k0A = ld_bf16_lin((int5){k_col, r, 0, 0, 0}, qkv);
                float64 k1A = ld_bf16_lin((int5){k_col + 64, r, 0, 0, 0}, qkv);
                float64 k0B = ld_bf16_lin((int5){k_col, r + 1, 0, 0, 0}, qkv);
                float64 k1B = ld_bf16_lin((int5){k_col + 64, r + 1, 0, 0, 0}, qkv);

                bfloat128 vbA = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
                bfloat128 vbB = v_bf16_ld_tnsr_b((int5){v_col, r + 1, 0, 0, 0}, qkv);

                float64 qrA = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0A * q0A + q1A * q1A, m32) * inv_128 + eps);
                float64 qrB = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0B * q0B + q1B * q1B, m32) * inv_128 + eps);
                float64 krA = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0A * k0A + k1A * k1A, m32) * inv_128 + eps);
                float64 krB = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0B * k0B + k1B * k1B, m32) * inv_128 + eps);

                float64 qf0A = q0A * qrA * qw0, qf1A = q1A * qrA * qw1;
                float64 qf0B = q0B * qrB * qw0, qf1B = q1B * qrB * qw1;
                float64 kf0A = k0A * krA * kw0, kf1A = k1A * krA * kw1;
                float64 kf0B = k0B * krB * kw0, kf1B = k1B * krB * kw1;

                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                    v_convert_f32_to_bf16_all_b((float128){qf0A * cA - qf1A * sA, qf1A * cA + qf0A * sA}, SW_RHNE, (bfloat128){0}, 1, 0));
                v_bf16_st_tnsr((int5){0, r + 1, h, 0, 0}, qh_out,
                    v_convert_f32_to_bf16_all_b((float128){qf0B * cB - qf1B * sB, qf1B * cB + qf0B * sB}, SW_RHNE, (bfloat128){0}, 1, 0));
                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, kh_out,
                    v_convert_f32_to_bf16_all_b((float128){kf0A * cA - kf1A * sA, kf1A * cA + kf0A * sA}, SW_RHNE, (bfloat128){0}, 1, 0));
                v_bf16_st_tnsr((int5){0, r + 1, h, 0, 0}, kh_out,
                    v_convert_f32_to_bf16_all_b((float128){kf0B * cB - kf1B * sB, kf1B * cB + kf0B * sB}, SW_RHNE, (bfloat128){0}, 1, 0));

                vmax_b0 = v_bf16_max_b(vmax_b0, v_bf16_abs_b(vbA, 0, vbA, 1, 0), 0, vmax_b0, 1, 0);
                vmax_b1 = v_bf16_max_b(vmax_b1, v_bf16_abs_b(vbB, 0, vbB, 1, 0), 0, vmax_b1, 1, 0);
            }
            for (; r < rows; ++r) {
                float64 c64 = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
                float64 s64 = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
                float64 q0 = ld_bf16_lin((int5){q_col, r, 0, 0, 0}, qkv);
                float64 q1 = ld_bf16_lin((int5){q_col + 64, r, 0, 0, 0}, qkv);
                float64 qr = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0 * q0 + q1 * q1, m32) * inv_128 + eps);
                float64 qf0 = q0 * qr * qw0, qf1 = q1 * qr * qw1;
                float64 k0 = ld_bf16_lin((int5){k_col, r, 0, 0, 0}, qkv);
                float64 k1 = ld_bf16_lin((int5){k_col + 64, r, 0, 0, 0}, qkv);
                float64 kr = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0 * k0 + k1 * k1, m32) * inv_128 + eps);
                float64 kf0 = k0 * kr * kw0, kf1 = k1 * kr * kw1;
                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                    v_convert_f32_to_bf16_all_b((float128){qf0 * c64 - qf1 * s64, qf1 * c64 + qf0 * s64}, SW_RHNE, (bfloat128){0}, 1, 0));
                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, kh_out,
                    v_convert_f32_to_bf16_all_b((float128){kf0 * c64 - kf1 * s64, kf1 * c64 + kf0 * s64}, SW_RHNE, (bfloat128){0}, 1, 0));
                bfloat128 vb = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
                vmax_b0 = v_bf16_max_b(vmax_b0, v_bf16_abs_b(vb, 0, vb, 1, 0), 0, vmax_b0, 1, 0);
            }
            for (int rp = rows; rp < k_pad; ++rp) {
                v_bf16_st_tnsr((int5){0, rp, h, 0, 0}, kh_out, zero_bf);
            }

            bfloat128 vmax_b = v_bf16_max_b(vmax_b0, vmax_b1, 0, vmax_b0, 1, 0);
            float128 vmax_f = v_convert_bf16_to_f32_all_b(vmax_b, 0, (float128){0}, 1, 0);
            float64 vamax = v_f32_reduce_nolut_noshuffle_max(v_f32_max_b(vmax_f.v1, vmax_f.v2, 0, vmax_f.v1, 1, 0), m32);
            float64 vscale = v_f32_max_b(vamax, 1e-12f, 0, vamax, 1, 0) * inv_240;
            v_f32_st_tnsr((int5){0, h, 0, 0, 0}, out_scales, vscale);
            float64 inv_vs = (float64)1.0f / vscale;
            bfloat128 inv_vsb = v_convert_f32_to_bf16_all_b((float128){inv_vs, inv_vs}, SW_RHNE, (bfloat128){0}, 1, 0);

            // Paired 256B coalesced V_fp8 quant + all_lanes permute:
            // vh_q_out has TPC geometry [256, k_pad/2, 24] -> pair (blk*128 + j) writes rows
            // (blk*256 + 2*j, blk*256 + 2*j + 1) from src rows (blk*256 + j, blk*256 + 128 + j).
            const int full_blks = rows >> 8;
            int blk = 0;
            for (; blk < full_blks; ++blk) {
                const int src_base = blk << 8;
                const int dst_base = blk << 7;
                for (int j = 0; j < 128; j += 2) {
                    bfloat128 lo0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, src_base + j, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, src_base + 128 + j, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 lo1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, src_base + j + 1, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, src_base + 129 + j, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0);
                    v_f8_st_tnsr((int5){0, dst_base + j, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo0, hi0}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                    v_f8_st_tnsr((int5){0, dst_base + j + 1, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo1, hi1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                }
            }
            const int total_blks = k_pad >> 8;
            for (; blk < total_blks; ++blk) {
                const int src_base = blk << 8;
                const int dst_base = blk << 7;
                for (int j = 0; j < 128; ++j) {
                    const int sl = src_base + j;
                    const int sh = src_base + 128 + j;
                    bfloat128 lo = (sl < rows) ? v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, sl, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0) : zero_bf;
                    bfloat128 hi = (sh < rows) ? v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){v_col, sh, 0, 0, 0}, qkv), inv_vsb, 0, inv_vsb, 1, 0) : zero_bf;
                    v_f8_st_tnsr((int5){0, dst_base + j, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo, hi}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                }
            }
        }
        return;
    }

    // Legacy modes 0 (qk_dim=192) and 1 (qk_dim=128) for vendor SOFTMAX_HF8_1B fallback.
    const int padded = (qk_dim == 192);
    const int64 lane_i = (int64)read_lane_id_4b_b();
    const bool64 is_lane0 = v_i32_cmp_eq_b(lane_i, 0, 0, (bool64){0}, 1, 0);
    const bfloat128 q_pad_vec = v_convert_f32_to_bf16_all_b(
        (float128){v_f32_mov_vb(1.0f, 0, 0.0f, is_lane0, 0), 0.0f}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0);
    const bfloat128 k_pad_vec = v_convert_f32_to_bf16_all_b(
        (float128){v_f32_mov_vb(-8192.0f, 0, 0.0f, is_lane0, 0), 0.0f}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0);

    for (int h = begin[0]; h < end[0] && h < 24; ++h) {
        const int q_col = h * 128;
        const int k_col = 3072 + q_col;
        const int v_col = 6144 + q_col;

        bfloat128 vmax_b0 = (bfloat128)0.0f;
        bfloat128 vmax_b1 = (bfloat128)0.0f;
        int r = 0;
        for (; r + 1 < rows; r += 2) {
            float64 cA = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
            float64 sA = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
            float64 cB = v_f32_ld_tnsr_b((int5){0, r + 1, 0, 0, 0}, pe_cs);
            float64 sB = v_f32_ld_tnsr_b((int5){64, r + 1, 0, 0, 0}, pe_cs);

            float64 q0A = ld_bf16_lin((int5){q_col, r, 0, 0, 0}, qkv);
            float64 q1A = ld_bf16_lin((int5){q_col + 64, r, 0, 0, 0}, qkv);
            float64 q0B = ld_bf16_lin((int5){q_col, r + 1, 0, 0, 0}, qkv);
            float64 q1B = ld_bf16_lin((int5){q_col + 64, r + 1, 0, 0, 0}, qkv);

            float64 k0A = ld_bf16_lin((int5){k_col, r, 0, 0, 0}, qkv);
            float64 k1A = ld_bf16_lin((int5){k_col + 64, r, 0, 0, 0}, qkv);
            float64 k0B = ld_bf16_lin((int5){k_col, r + 1, 0, 0, 0}, qkv);
            float64 k1B = ld_bf16_lin((int5){k_col + 64, r + 1, 0, 0, 0}, qkv);

            bfloat128 vbA = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
            bfloat128 vbB = v_bf16_ld_tnsr_b((int5){v_col, r + 1, 0, 0, 0}, qkv);

            float64 qrA = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0A * q0A + q1A * q1A, m32) * inv_128 + eps);
            float64 qrB = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0B * q0B + q1B * q1B, m32) * inv_128 + eps);
            float64 krA = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0A * k0A + k1A * k1A, m32) * inv_128 + eps);
            float64 krB = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0B * k0B + k1B * k1B, m32) * inv_128 + eps);

            float64 qf0A = q0A * qrA * qw0, qf1A = q1A * qrA * qw1;
            float64 qf0B = q0B * qrB * qw0, qf1B = q1B * qrB * qw1;
            float64 kf0A = k0A * krA * kw0, kf1A = k1A * krA * kw1;
            float64 kf0B = k0B * krB * kw0, kf1B = k1B * krB * kw1;

            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){qf0A * cA - qf1A * sA, qf1A * cA + qf0A * sA}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));
            v_bf16_st_tnsr((int5){0, r + 1, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){qf0B * cB - qf1B * sB, qf1B * cB + qf0B * sB}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));
            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){kf0A * cA - kf1A * sA, kf1A * cA + kf0A * sA}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));
            v_bf16_st_tnsr((int5){0, r + 1, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){kf0B * cB - kf1B * sB, kf1B * cB + kf0B * sB}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));

            if (padded) {
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, qh_out, q_pad_vec);
                v_bf16_st_tnsr((int5){128, r + 1, h, 0, 0}, qh_out, q_pad_vec);
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, kh_out, zero_bf);
                v_bf16_st_tnsr((int5){128, r + 1, h, 0, 0}, kh_out, zero_bf);
            }

            vmax_b0 = v_bf16_max_b(vmax_b0, v_bf16_abs_b(vbA, 0, vbA, 1, 0), 0, vmax_b0, 1, 0);
            vmax_b1 = v_bf16_max_b(vmax_b1, v_bf16_abs_b(vbB, 0, vbB, 1, 0), 0, vmax_b1, 1, 0);
        }
        for (; r < rows; ++r) {
            float64 c64 = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
            float64 s64 = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
            float64 q0 = ld_bf16_lin((int5){q_col, r, 0, 0, 0}, qkv);
            float64 q1 = ld_bf16_lin((int5){q_col + 64, r, 0, 0, 0}, qkv);
            float64 qr = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(q0 * q0 + q1 * q1, m32) * inv_128 + eps);
            float64 qf0 = q0 * qr * qw0, qf1 = q1 * qr * qw1;
            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){qf0 * c64 - qf1 * s64, qf1 * c64 + qf0 * s64}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));
            float64 k0 = ld_bf16_lin((int5){k_col, r, 0, 0, 0}, qkv);
            float64 k1 = ld_bf16_lin((int5){k_col + 64, r, 0, 0, 0}, qkv);
            float64 kr = rsqrt_f32(v_f32_reduce_nolut_noshuffle_add(k0 * k0 + k1 * k1, m32) * inv_128 + eps);
            float64 kf0 = k0 * kr * kw0, kf1 = k1 * kr * kw1;
            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){kf0 * c64 - kf1 * s64, kf1 * c64 + kf0 * s64}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0));
            if (padded) {
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, qh_out, q_pad_vec);
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, kh_out, zero_bf);
            }
            bfloat128 vb = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
            vmax_b0 = v_bf16_max_b(vmax_b0, v_bf16_abs_b(vb, 0, vb, 1, 0), 0, vmax_b0, 1, 0);
        }

        for (int rp = rows; rp < k_pad; ++rp) {
            v_bf16_st_tnsr((int5){0, rp, h, 0, 0}, kh_out, zero_bf);
            v_bf16_st_tnsr((int5){128, rp, h, 0, 0}, kh_out, k_pad_vec);
            v_f8_st_tnsr((int5){0, rp, h, 0, 0}, vh_q_out, zero_f8);
        }

        bfloat128 vmax_b = v_bf16_max_b(vmax_b0, vmax_b1, 0, vmax_b0, 1, 0);
        float128 vmax_f = v_convert_bf16_to_f32_all_b(vmax_b, 0, (float128){0}, 1, 0);
        float64 vamax = v_f32_reduce_nolut_noshuffle_max(v_f32_max_b(vmax_f.v1, vmax_f.v2, 0, vmax_f.v1, 1, 0), m32);
        float64 vscale = v_f32_max_b(vamax, 1e-12f, 0, vamax, 1, 0) * inv_240;
        v_f32_st_tnsr((int5){0, h, 0, 0, 0}, out_scales, vscale * inv_128);
        float64 inv_vs = (float64)1.0f / vscale;
        bfloat128 inv_vsb = v_convert_f32_to_bf16_all_b((float128){inv_vs, inv_vs}, SW_RHNE, (bfloat128){0}, 1, 0);

        r = 0;
        for (; r + 3 < rows; r += 4) {
            bfloat128 v0 = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
            bfloat128 v1 = v_bf16_ld_tnsr_b((int5){v_col, r + 1, 0, 0, 0}, qkv);
            bfloat128 v2 = v_bf16_ld_tnsr_b((int5){v_col, r + 2, 0, 0, 0}, qkv);
            bfloat128 v3 = v_bf16_ld_tnsr_b((int5){v_col, r + 3, 0, 0, 0}, qkv);
            v0 = v_bf16_min_b(v_bf16_max_b(v_bf16_mul_b(v0, inv_vsb, 0, v0, 1, 0), cn240, 0, v0, 1, 0), c240, 0, v0, 1, 0);
            v1 = v_bf16_min_b(v_bf16_max_b(v_bf16_mul_b(v1, inv_vsb, 0, v1, 1, 0), cn240, 0, v1, 1, 0), c240, 0, v1, 1, 0);
            v2 = v_bf16_min_b(v_bf16_max_b(v_bf16_mul_b(v2, inv_vsb, 0, v2, 1, 0), cn240, 0, v2, 1, 0), c240, 0, v2, 1, 0);
            v3 = v_bf16_min_b(v_bf16_max_b(v_bf16_mul_b(v3, inv_vsb, 0, v3, 1, 0), cn240, 0, v3, 1, 0), c240, 0, v3, 1, 0);
            v_f8_st_tnsr((int5){0, r, h, 0, 0}, vh_q_out, v_convert_bf16_to_f8_all_b((bfloat256){v0, zero_bf}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
            v_f8_st_tnsr((int5){0, r + 1, h, 0, 0}, vh_q_out, v_convert_bf16_to_f8_all_b((bfloat256){v1, zero_bf}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
            v_f8_st_tnsr((int5){0, r + 2, h, 0, 0}, vh_q_out, v_convert_bf16_to_f8_all_b((bfloat256){v2, zero_bf}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
            v_f8_st_tnsr((int5){0, r + 3, h, 0, 0}, vh_q_out, v_convert_bf16_to_f8_all_b((bfloat256){v3, zero_bf}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
        }
        for (; r < rows; ++r) {
            bfloat128 vb = v_bf16_ld_tnsr_b((int5){v_col, r, 0, 0, 0}, qkv);
            bfloat128 qbf = v_bf16_min_b(v_bf16_max_b(v_bf16_mul_b(vb, inv_vsb, 0, vb, 1, 0), cn240, 0, vb, 1, 0), c240, 0, vb, 1, 0);
            v_f8_st_tnsr((int5){0, r, h, 0, 0}, vh_q_out, v_convert_bf16_to_f8_all_b((bfloat256){qbf, zero_bf}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
        }
    }
}
