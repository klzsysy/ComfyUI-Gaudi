// Fused Qwen-Image 2.1 optional gated residual (mid = x + res * gate) + single-pass FP32 LayerNorm (no affine)
// + *(1 + scale) + per-row dynamic FP8 E4M3 quantization.
// Supports dim % 256 == 0, dim <= 4096 (row_vlm[32] = 8 KiB <= 16 KiB small VLM, 0 VRF spills).
//
// Inputs:
//   x        : [dim, rows] BF16
//   res      : [dim, rows] BF16 (ignored when has_res == 0)
//   gate     : [dim] BF16 (ignored when has_res == 0)
//   scale_p1 : [dim] BF16 (precomputed 1 + scale)
// Outputs:
//   out_x    : [dim, rows] BF16 when has_res != 0, else [1, 1] BF16 dummy
//   out_q    : [dim, rows] FP8 E4M3
//   out_scale: [rows] FP32

__local__ bfloat128 row_vlm[32];

static inline float64 rsqrt_nolut_f32(float64 x) {
    float64 half_x = 0.5f * x;
    float64 y = rsqrt_quake_f32(x);
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return y;
}

void main(tensor x, tensor res, tensor gate, tensor scale_p1,
          tensor out_x, tensor out_q, tensor out_scale,
          int dim, int rows, int has_res, float inv_dim, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int n128 = dim >> 7;
    const float inv_240 = 1.0f / 240.0f;
    const shuffle_mask_results_t m32 = gen_shuffle_map_32bit_nolut();

    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;
    if (r0 >= r1) return;

    for (int r = r0; r < r1; ++r) {
        float64 s0 = 0.0f, s1 = 0.0f, q0 = 0.0f, q1 = 0.0f;
        if (has_res) {
            for (int j = 0; j < n128; j += 2) {
                const int c0 = j * 128, c1 = c0 + 128;
                int5 p0 = {c0, r, 0, 0, 0}, p1 = {c1, r, 0, 0, 0};
                bfloat128 x0 = v_bf16_mac_b(v_bf16_ld_tnsr_b(p0, res), v_bf16_ld_tnsr_b((int5){c0, 0, 0, 0, 0}, gate), v_bf16_ld_tnsr_b(p0, x), 0, 1, 0);
                bfloat128 x1 = v_bf16_mac_b(v_bf16_ld_tnsr_b(p1, res), v_bf16_ld_tnsr_b((int5){c1, 0, 0, 0, 0}, gate), v_bf16_ld_tnsr_b(p1, x), 0, 1, 0);
                v_bf16_st_tnsr(p0, out_x, x0);
                v_bf16_st_tnsr(p1, out_x, x1);
                row_vlm[j] = x0;
                row_vlm[j + 1] = x1;
                float128 f0 = v_convert_bf16_to_f32_all_b(x0, 0, (float128){0}, 1, 0);
                float128 f1 = v_convert_bf16_to_f32_all_b(x1, 0, (float128){0}, 1, 0);
                s0 = s0 + (f0.v1 + f0.v2);
                s1 = s1 + (f1.v1 + f1.v2);
                q0 = v_f32_mac_b(f0.v1, f0.v1, q0, 0, 1, 0);
                q0 = v_f32_mac_b(f0.v2, f0.v2, q0, 0, 1, 0);
                q1 = v_f32_mac_b(f1.v1, f1.v1, q1, 0, 1, 0);
                q1 = v_f32_mac_b(f1.v2, f1.v2, q1, 0, 1, 0);
            }
        } else {
            for (int j = 0; j < n128; j += 2) {
                const int c0 = j * 128, c1 = c0 + 128;
                bfloat128 x0 = v_bf16_ld_tnsr_b((int5){c0, r, 0, 0, 0}, x);
                bfloat128 x1 = v_bf16_ld_tnsr_b((int5){c1, r, 0, 0, 0}, x);
                row_vlm[j] = x0;
                row_vlm[j + 1] = x1;
                float128 f0 = v_convert_bf16_to_f32_all_b(x0, 0, (float128){0}, 1, 0);
                float128 f1 = v_convert_bf16_to_f32_all_b(x1, 0, (float128){0}, 1, 0);
                s0 = s0 + (f0.v1 + f0.v2);
                s1 = s1 + (f1.v1 + f1.v2);
                q0 = v_f32_mac_b(f0.v1, f0.v1, q0, 0, 1, 0);
                q0 = v_f32_mac_b(f0.v2, f0.v2, q0, 0, 1, 0);
                q1 = v_f32_mac_b(f1.v1, f1.v1, q1, 0, 1, 0);
                q1 = v_f32_mac_b(f1.v2, f1.v2, q1, 0, 1, 0);
            }
        }
        float64 mean = v_f32_reduce_nolut_noshuffle_add(s0 + s1, m32) * inv_dim;
        float64 mean_sq = v_f32_reduce_nolut_noshuffle_add(q0 + q1, m32) * inv_dim;
        float64 var = v_f32_max_b(mean_sq - mean * mean, 0.0f, 0, mean, 1, 0);
        float64 rstd = rsqrt_nolut_f32(var + eps);

        bfloat128 amax_b0 = (bfloat128)0.0f, amax_b1 = (bfloat128)0.0f;
        for (int j = 0; j < n128; j += 2) {
            const int c0 = j * 128, c1 = c0 + 128;
            float128 f0 = v_convert_bf16_to_f32_all_b(row_vlm[j], 0, (float128){0}, 1, 0);
            float128 f1 = v_convert_bf16_to_f32_all_b(row_vlm[j + 1], 0, (float128){0}, 1, 0);
            bfloat128 n0 = v_convert_f32_to_bf16_all_b((float128){(f0.v1 - mean) * rstd, (f0.v2 - mean) * rstd}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 n1 = v_convert_f32_to_bf16_all_b((float128){(f1.v1 - mean) * rstd, (f1.v2 - mean) * rstd}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 y0 = v_bf16_mul_b(n0, v_bf16_ld_tnsr_b((int5){c0, 0, 0, 0, 0}, scale_p1), 0, n0, 1, 0);
            bfloat128 y1 = v_bf16_mul_b(n1, v_bf16_ld_tnsr_b((int5){c1, 0, 0, 0, 0}, scale_p1), 0, n1, 1, 0);
            row_vlm[j] = y0;
            row_vlm[j + 1] = y1;
            amax_b0 = v_bf16_max_b(amax_b0, v_bf16_abs_b(y0, 0, y0, 1, 0), 0, amax_b0, 1, 0);
            amax_b1 = v_bf16_max_b(amax_b1, v_bf16_abs_b(y1, 0, y1, 1, 0), 0, amax_b1, 1, 0);
        }
        bfloat128 amax_b = v_bf16_max_b(amax_b0, amax_b1, 0, amax_b0, 1, 0);
        float128 amax_f = v_convert_bf16_to_f32_all_b(amax_b, 0, (float128){0}, 1, 0);
        float64 amax = v_f32_reduce_nolut_noshuffle_max(v_f32_max_b(amax_f.v1, amax_f.v2, 0, amax_f.v1, 1, 0), m32);
        float64 sc = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, sc, 0, 0);
        float64 inv_s = (float64)1.0f / sc;
        bfloat128 inv_sb = v_convert_f32_to_bf16_all_b((float128){inv_s, inv_s}, SW_RHNE, (bfloat128){0}, 1, 0);

        for (int j = 0; j < n128; j += 2) {
            bfloat128 z0 = v_bf16_mul_b(row_vlm[j], inv_sb, 0, row_vlm[j], 1, 0);
            bfloat128 z1 = v_bf16_mul_b(row_vlm[j + 1], inv_sb, 0, row_vlm[j + 1], 1, 0);
            v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q,
                v_convert_bf16_to_f8_all_b((bfloat256){z0, z1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0));
        }
    }
}
