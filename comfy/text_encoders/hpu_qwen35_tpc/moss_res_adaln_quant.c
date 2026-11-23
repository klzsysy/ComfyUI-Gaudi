// Fused MOSS gated residual (x_mid = x + res * gate) + FP32-accumulated RMSNorm + AdaLN (*eff_scale + shift)
// + per-row dynamic FP8 E4M3 quantization for Gaudi2.
// Supports dim % 256 == 0, dim <= 3072 (24 bfloat128 row + 24 bfloat128 scale = 12 KiB <= 16 KiB small VLM).
__local__ bfloat128 row_vlm[24];
__local__ bfloat128 scale_vlm[24];

void main(tensor x, tensor res, tensor gate, tensor eff_scale, tensor shift,
          tensor out_x, tensor out_q, tensor out_scale,
          int dim, int rows, float inv_dim, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int n128 = dim >> 7;
    const float inv_240 = 1.0f / 240.0f;
    const shuffle_mask_results_t m32 = gen_shuffle_map_32bit_nolut();

    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;
    if (r0 >= r1) return;

    for (int j = 0; j < n128; ++j) {
        scale_vlm[j] = v_bf16_ld_tnsr_b((int5){j * 128, 0, 0, 0, 0}, eff_scale);
    }

    for (int r = r0; r < r1; ++r) {
        float64 sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
        for (int j = 0; j < n128; j += 2) {
            const int c0 = j * 128;
            const int c1 = c0 + 128;
            int5 p0 = {c0, r, 0, 0, 0};
            int5 p1 = {c1, r, 0, 0, 0};
            bfloat128 x0 = v_bf16_mac_b(v_bf16_ld_tnsr_b(p0, res), v_bf16_ld_tnsr_b((int5){c0, 0, 0, 0, 0}, gate), v_bf16_ld_tnsr_b(p0, x), 0, 1, 0);
            bfloat128 x1 = v_bf16_mac_b(v_bf16_ld_tnsr_b(p1, res), v_bf16_ld_tnsr_b((int5){c1, 0, 0, 0, 0}, gate), v_bf16_ld_tnsr_b(p1, x), 0, 1, 0);
            v_bf16_st_tnsr(p0, out_x, x0);
            v_bf16_st_tnsr(p1, out_x, x1);
            row_vlm[j] = x0;
            row_vlm[j + 1] = x1;
            float128 f0 = v_convert_bf16_to_f32_all_b(x0, 0, (float128){0}, 1, 0);
            float128 f1 = v_convert_bf16_to_f32_all_b(x1, 0, (float128){0}, 1, 0);
            sq0 = v_f32_mac_b(f0.v1, f0.v1, sq0, 0, 1, 0);
            sq1 = v_f32_mac_b(f0.v2, f0.v2, sq1, 0, 1, 0);
            sq2 = v_f32_mac_b(f1.v1, f1.v1, sq2, 0, 1, 0);
            sq3 = v_f32_mac_b(f1.v2, f1.v2, sq3, 0, 1, 0);
        }
        float64 sum_sq = v_f32_reduce_nolut_noshuffle_add((sq0 + sq1) + (sq2 + sq3), m32);
        float64 rstd = rsqrt_f32(sum_sq * inv_dim + eps);
        bfloat128 rstd_b = v_convert_f32_to_bf16_all_b((float128){rstd, rstd}, SW_RHNE, (bfloat128){0}, 1, 0);

        bfloat128 amax_b0 = (bfloat128)0.0f;
        bfloat128 amax_b1 = (bfloat128)0.0f;
        for (int j = 0; j < n128; j += 2) {
            const int c0 = j * 128;
            const int c1 = c0 + 128;
            bfloat128 n0 = v_bf16_mul_b(row_vlm[j], rstd_b, 0, rstd_b, 1, 0);
            bfloat128 n1 = v_bf16_mul_b(row_vlm[j + 1], rstd_b, 0, rstd_b, 1, 0);
            bfloat128 y0 = v_bf16_mac_b(n0, scale_vlm[j], v_bf16_ld_tnsr_b((int5){c0, 0, 0, 0, 0}, shift), 0, 1, 0);
            bfloat128 y1 = v_bf16_mac_b(n1, scale_vlm[j + 1], v_bf16_ld_tnsr_b((int5){c1, 0, 0, 0, 0}, shift), 0, 1, 0);
            row_vlm[j] = y0;
            row_vlm[j + 1] = y1;
            amax_b0 = v_bf16_max_b(amax_b0, v_bf16_abs_b(y0, 0, y0, 1, 0), 0, amax_b0, 1, 0);
            amax_b1 = v_bf16_max_b(amax_b1, v_bf16_abs_b(y1, 0, y1, 1, 0), 0, amax_b1, 1, 0);
        }
        bfloat128 amax_b = v_bf16_max_b(amax_b0, amax_b1, 0, amax_b0, 1, 0);
        float128 amax_f = v_convert_bf16_to_f32_all_b(amax_b, 0, (float128){0}, 1, 0);
        float64 amax = v_f32_reduce_nolut_noshuffle_max(v_f32_max_b(amax_f.v1, amax_f.v2, 0, amax_f.v1, 1, 0), m32);
        float64 scale = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, scale, 0, 0);
        float64 inv_s = (float64)1.0f / scale;
        bfloat128 inv_sb = v_convert_f32_to_bf16_all_b((float128){inv_s, inv_s}, SW_RHNE, (bfloat128){0}, 1, 0);

        for (int j = 0; j < n128; j += 2) {
            bfloat128 q0 = v_bf16_mul_b(row_vlm[j], inv_sb, 0, row_vlm[j], 1, 0);
            bfloat128 q1 = v_bf16_mul_b(row_vlm[j + 1], inv_sb, 0, row_vlm[j + 1], 1, 0);
            minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){q0, q1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0);
            v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q, q8);
        }
    }
}
