// Fused GDN per-head RMSNorm(out_f32, gdn_nw) * SiLU(z) + per-row dynamic FP8 E4M3 quantization.
// Separates rsqrt_f32 and tanh_f32 passes to avoid Gaudi2 special-function LUT cache thrashing.
// Supports v_hd == 128 and vh * 128 <= 4096 (32 bfloat128 = 8 KiB VLM).
__local__ bfloat128 row_vlm[32];

static inline float64 silu_f32(float64 x) {
    return (0.5f * x) * (1.0f + tanh_f32(0.5f * x));
}

void main(tensor out_f32, tensor z, tensor gdn_nw,
          tensor out_q, tensor out_scale,
          int vh, int rows, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_128 = 1.0f / 128.0f;
    const float inv_240 = 1.0f / 240.0f;

    float64 nw0 = v_f32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, gdn_nw);
    float64 nw1 = v_f32_ld_tnsr_b((int5){64, 0, 0, 0, 0}, gdn_nw);

    for (int r = begin[0]; r < end[0] && r < rows; ++r) {
        // Pass 1a: per-head RMSNorm using only rsqrt_f32 LUT.
        for (int h = 0; h < vh; ++h) {
            const int col = h * 128;
            float64 o0 = v_f32_ld_tnsr_b((int5){col, r, 0, 0, 0}, out_f32);
            float64 o1 = v_f32_ld_tnsr_b((int5){col + 64, r, 0, 0, 0}, out_f32);
            float64 ss = v_f32_reduce_nolut_add(o0 * o0 + o1 * o1);
            float64 rstd = rsqrt_f32(ss * inv_128 + eps);
            row_vlm[h] = v_convert_f32_to_bf16_all_b(
                (float128){o0 * rstd * nw0, o1 * rstd * nw1}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0);
        }

        // Pass 1b: SiLU(z) * norm + row amax using only tanh_f32 LUT (1 load of z per head!).
        float64 amax0 = 0.0f, amax1 = 0.0f;
        for (int h = 0; h < vh; ++h) {
            bfloat128 zb = v_bf16_ld_tnsr_b((int5){h * 128, r, 0, 0, 0}, z);
            float128 zf = v_convert_bf16_to_f32_all_b(zb, 0, (float128){0}, 1, 0);
            bfloat128 sz = v_convert_f32_to_bf16_all_b(
                (float128){silu_f32(zf.v1), silu_f32(zf.v2)}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 yb = v_bf16_mul_b(row_vlm[h], sz, 0, sz, 1, 0);
            row_vlm[h] = yb;
            float128 yf = v_convert_bf16_to_f32_all_b(yb, 0, (float128){0}, 1, 0);
            amax0 = v_f32_max_b(amax0, v_f32_abs_b(yf.v1, 0, yf.v1, 1, 0), 0, amax0, 1, 0);
            amax1 = v_f32_max_b(amax1, v_f32_abs_b(yf.v2, 0, yf.v2, 1, 0), 0, amax1, 1, 0);
        }

        float64 amax = v_f32_reduce_nolut_max(v_f32_max_b(amax0, amax1, 0, amax0, 1, 0));
        float64 scale = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){0, r, 0, 0, 0}, out_scale, scale, 0, 0);
        // Reuse rsqrt_f32 LUT instead of rcp_f32 so only 2 special LUTs (rsqrt + tanh) are used.
        float64 inv_s = rsqrt_f32(scale * scale);

        for (int h = 0; h < vh; h += 2) {
            float128 y0 = v_convert_bf16_to_f32_all_b(row_vlm[h], 0, (float128){0}, 1, 0);
            float128 y1 = v_convert_bf16_to_f32_all_b(row_vlm[h + 1], 0, (float128){0}, 1, 0);
            bfloat128 qbf0 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y0.v1 * inv_s, -240.0f, 0, y0.v1, 1, 0), 240.0f, 0, y0.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y0.v2 * inv_s, -240.0f, 0, y0.v2, 1, 0), 240.0f, 0, y0.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 qbf1 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y1.v1 * inv_s, -240.0f, 0, y1.v1, 1, 0), 240.0f, 0, y1.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y1.v2 * inv_s, -240.0f, 0, y1.v2, 1, 0), 240.0f, 0, y1.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){qbf0, qbf1}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0);
            v_f8_st_tnsr((int5){h * 128, r, 0, 0, 0}, out_q, q8);
        }
    }
}
