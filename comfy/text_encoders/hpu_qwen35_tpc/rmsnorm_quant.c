// Fused optional residual-add + RMSNorm + optional BF16 normed output + per-row dynamic FP8 E4M3 quantization.
// Supports dim % 256 == 0 and dim <= 4096 (32 bfloat128 = 8 KiB VLM).
__local__ bfloat128 row_vlm[32];

void main(tensor x, tensor res, tensor weight,
          tensor out_x, tensor out_y, tensor out_q, tensor out_scale,
          int dim, int rows, int has_res, int add_one, int store_norm, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_dim = 1.0f / (float)dim;
    const float inv_240 = 1.0f / 240.0f;
    const int n128 = dim >> 7;

    for (int r = begin[0]; r < end[0] && r < rows; ++r) {
        float64 sq0 = 0.0f, sq1 = 0.0f;
        for (int j = 0; j < n128; ++j) {
            int5 p = {j * 128, r, 0, 0, 0};
            bfloat128 xb = v_bf16_ld_tnsr_b(p, x);
            if (has_res) {
                bfloat128 rb = v_bf16_ld_tnsr_b(p, res);
                xb = v_bf16_add_b(xb, rb, 0, xb, 1, 0);
                v_bf16_st_tnsr(p, out_x, xb);
            }
            row_vlm[j] = xb;
            float128 xf = v_convert_bf16_to_f32_all_b(xb, 0, (float128){0}, 1, 0);
            sq0 += xf.v1 * xf.v1;
            sq1 += xf.v2 * xf.v2;
        }
        float64 sum_sq = v_f32_reduce_nolut_add(sq0 + sq1);
        float64 rstd = rsqrt_f32(sum_sq * inv_dim + eps);

        float64 amax0 = 0.0f, amax1 = 0.0f;
        for (int j = 0; j < n128; ++j) {
            float128 xf = v_convert_bf16_to_f32_all_b(row_vlm[j], 0, (float128){0}, 1, 0);
            bfloat128 nb = v_convert_f32_to_bf16_all_b((float128){xf.v1 * rstd, xf.v2 * rstd}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 wb = v_bf16_ld_tnsr_b((int5){j * 128, 0, 0, 0, 0}, weight);
            if (add_one) {
                wb = v_bf16_add_b(wb, (bfloat)1.0f, 0, wb, 1, 0);
            }
            bfloat128 yb = v_bf16_mul_b(nb, wb, 0, nb, 1, 0);
            if (store_norm) {
                v_bf16_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_y, yb);
            }
            row_vlm[j] = yb;
            float128 yf = v_convert_bf16_to_f32_all_b(yb, 0, (float128){0}, 1, 0);
            amax0 = v_f32_max_b(amax0, v_f32_abs_b(yf.v1, 0, yf.v1, 1, 0), 0, amax0, 1, 0);
            amax1 = v_f32_max_b(amax1, v_f32_abs_b(yf.v2, 0, yf.v2, 1, 0), 0, amax1, 1, 0);
        }
        float64 amax = v_f32_reduce_nolut_max(v_f32_max_b(amax0, amax1, 0, amax0, 1, 0));
        float64 scale = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){0, r, 0, 0, 0}, out_scale, scale, 0, 0);
        float64 inv_s = (float64)1.0f / scale;

        for (int j = 0; j < n128; j += 2) {
            float128 y0 = v_convert_bf16_to_f32_all_b(row_vlm[j], 0, (float128){0}, 1, 0);
            float128 y1 = v_convert_bf16_to_f32_all_b(row_vlm[j + 1], 0, (float128){0}, 1, 0);
            bfloat128 qbf0 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y0.v1 * inv_s, -240.0f, 0, y0.v1, 1, 0), 240.0f, 0, y0.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y0.v2 * inv_s, -240.0f, 0, y0.v2, 1, 0), 240.0f, 0, y0.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 qbf1 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y1.v1 * inv_s, -240.0f, 0, y1.v1, 1, 0), 240.0f, 0, y1.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y1.v2 * inv_s, -240.0f, 0, y1.v2, 1, 0), 240.0f, 0, y1.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){qbf0, qbf1}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0);
            v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q, q8);
        }
    }
}
