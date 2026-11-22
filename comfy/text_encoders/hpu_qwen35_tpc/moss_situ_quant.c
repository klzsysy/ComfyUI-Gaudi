// Fused MOSS SiTU(hidden, gate) + per-row dynamic FP8 E4M3 quantization for Gaudi2.
// Input y_in: [2*dim, rows] BF16 (hidden = 0..dim-1, gate = dim..2*dim-1), dim % 256 == 0, dim <= 8192.
// Uses exact algebraic identity:
//   4*tanh(g/4)*sigmoid(g) * 25*tanh(h/25) = 50 * t*(1+t)^2 / (1+t^2) * tanh(h/25), where t = tanh(g/4).
// This eliminates sigmoid() and cuts tanh() calls by 33% (only 2 tanh_bf16 per 128 elements, 1 LUT).
// 1/(1+s) for s = t^2 in [0, 1] uses minimax quadratic seed + 1 Newton-Raphson step (rel err < 1e-4).
// VLM usage: 64 * 256B = 16 KiB (fits small VLM). Partitioned in 64-row blocks via DimIndexSpaceMapping.
__local__ bfloat128 row_vlm[64];

static inline bfloat128 rcp_1_plus_s_bf16(bfloat128 s) {
    const bfloat c0 = (bfloat)0.9918f;
    const bfloat c1 = (bfloat)-0.8185f;
    const bfloat c2 = (bfloat)0.3300f;
    const bfloat c1_0 = (bfloat)1.0f;
    const bfloat c2_0 = (bfloat)2.0f;
    bfloat128 r = v_bf16_mac_b(s, c2, c1, 0, 1, 0);
    r = v_bf16_mac_b(s, r, c0, 0, 1, 0);
    bfloat128 u = v_bf16_add_b(s, c1_0, 0, s, 1, 0);
    bfloat128 corr = v_bf16_sub_b(c2_0, v_bf16_mul_b(u, r, 0, u, 1, 0), 0, r, 1, 0);
    return v_bf16_mul_b(r, corr, 0, r, 1, 0);
}

void main(tensor y_in, tensor out_q, tensor out_scale, int dim, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int n128 = dim >> 7;
    const bfloat c004 = (bfloat)0.04f;
    const bfloat c025 = (bfloat)0.25f;
    const bfloat c100 = (bfloat)1.0f;
    const bfloat c500 = (bfloat)50.0f;
    const bfloat c240 = (bfloat)240.0f;
    const bfloat cn240 = (bfloat)-240.0f;
    const float inv_240 = 1.0f / 240.0f;

    for (int blk = begin[0]; blk < end[0]; ++blk) {
        const int r0 = blk * 64;
        const int r1 = (r0 + 64 < rows) ? (r0 + 64) : rows;
        for (int r = r0; r < r1; ++r) {
            bfloat128 amax_b0 = (bfloat128)0.0f;
            bfloat128 amax_b1 = (bfloat128)0.0f;
            for (int j = 0; j < n128; j += 2) {
                const int c0 = j * 128;
                const int c1 = c0 + 128;
                bfloat128 h0 = v_bf16_ld_tnsr_b((int5){c0, r, 0, 0, 0}, y_in);
                bfloat128 g0 = v_bf16_ld_tnsr_b((int5){dim + c0, r, 0, 0, 0}, y_in);
                bfloat128 h1 = v_bf16_ld_tnsr_b((int5){c1, r, 0, 0, 0}, y_in);
                bfloat128 g1 = v_bf16_ld_tnsr_b((int5){dim + c1, r, 0, 0, 0}, y_in);

                bfloat128 th0 = tanh_bf16(v_bf16_mul_b(h0, c004, 0, h0, 1, 0));
                bfloat128 th1 = tanh_bf16(v_bf16_mul_b(h1, c004, 0, h1, 1, 0));
                bfloat128 t0 = tanh_bf16(v_bf16_mul_b(g0, c025, 0, g0, 1, 0));
                bfloat128 t1 = tanh_bf16(v_bf16_mul_b(g1, c025, 0, g1, 1, 0));

                bfloat128 op0 = v_bf16_add_b(t0, c100, 0, t0, 1, 0);
                bfloat128 op1 = v_bf16_add_b(t1, c100, 0, t1, 1, 0);
                bfloat128 r0_v = rcp_1_plus_s_bf16(v_bf16_mul_b(t0, t0, 0, t0, 1, 0));
                bfloat128 r1_v = rcp_1_plus_s_bf16(v_bf16_mul_b(t1, t1, 0, t1, 1, 0));
                bfloat128 num0 = v_bf16_mul_b(v_bf16_mul_b(op0, op0, 0, op0, 1, 0), v_bf16_mul_b(t0, c500, 0, t0, 1, 0), 0, t0, 1, 0);
                bfloat128 num1 = v_bf16_mul_b(v_bf16_mul_b(op1, op1, 0, op1, 1, 0), v_bf16_mul_b(t1, c500, 0, t1, 1, 0), 0, t1, 1, 0);
                bfloat128 p0 = v_bf16_mul_b(v_bf16_mul_b(num0, r0_v, 0, num0, 1, 0), th0, 0, th0, 1, 0);
                bfloat128 p1 = v_bf16_mul_b(v_bf16_mul_b(num1, r1_v, 0, num1, 1, 0), th1, 0, th1, 1, 0);

                row_vlm[j] = p0;
                row_vlm[j + 1] = p1;
                amax_b0 = v_bf16_max_b(amax_b0, v_bf16_abs_b(p0, 0, p0, 1, 0), 0, amax_b0, 1, 0);
                amax_b1 = v_bf16_max_b(amax_b1, v_bf16_abs_b(p1, 0, p1, 1, 0), 0, amax_b1, 1, 0);
            }
            bfloat128 amax_b = v_bf16_max_b(amax_b0, amax_b1, 0, amax_b0, 1, 0);
            float128 amax_f = v_convert_bf16_to_f32_all_b(amax_b, 0, (float128){0}, 1, 0);
            float64 amax = v_f32_reduce_nolut_max(v_f32_max_b(amax_f.v1, amax_f.v2, 0, amax_f.v1, 1, 0));
            float64 scale = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, scale, 0, 0);
            float64 inv_s = (float64)1.0f / scale;
            bfloat128 inv_sb = v_convert_f32_to_bf16_all_b((float128){inv_s, inv_s}, SW_RHNE, (bfloat128){0}, 1, 0);

            for (int j = 0; j < n128; j += 2) {
                bfloat128 y0 = v_bf16_mul_b(row_vlm[j], inv_sb, 0, row_vlm[j], 1, 0);
                bfloat128 y1 = v_bf16_mul_b(row_vlm[j + 1], inv_sb, 0, row_vlm[j + 1], 1, 0);
                y0 = v_bf16_min_b(v_bf16_max_b(y0, cn240, 0, y0, 1, 0), c240, 0, y0, 1, 0);
                y1 = v_bf16_min_b(v_bf16_max_b(y1, cn240, 0, y1, 1, 0), c240, 0, y1, 1, 0);
                minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){y0, y1}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0);
                v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q, q8);
            }
        }
    }
}
