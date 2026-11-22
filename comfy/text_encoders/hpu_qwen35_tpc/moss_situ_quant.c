// Fused MOSS SiTU(hidden, gate) + per-row dynamic FP8 E4M3 quantization for Gaudi2.
// Input y_in: [2*dim, rows] BF16 (hidden = 0..dim-1, gate = dim..2*dim-1), dim % 256 == 0, dim <= 8192.
// Uses exact algebraic identity:
//   4*tanh(g/4)*sigmoid(g) * 25*tanh(h/25) = 50 * (t + 2*t^2/(1+t^2)) * tanh(h/25), where t = tanh(g/4).
// 2*s/(1+s) on s = t^2 in [0, 1] uses degree-3 Chebyshev polynomial (max err 1.7e-3, zero division).
// VLM usage: 64 * 256B = 16 KiB (fits small VLM). Row-wise stride=1 index space keeps all 24 TPC cores
// balanced even when Synapse SRAM bundler slices rows across producer/consumer GEMMs.
__local__ bfloat128 row_vlm[64];

void main(tensor y_in, tensor out_q, tensor out_scale, int dim, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int n128 = dim >> 7;
    const bfloat c004 = (bfloat)0.04f;
    const bfloat c025 = (bfloat)0.25f;
    const bfloat c500 = (bfloat)50.0f;
    const bfloat p0_c = (bfloat)1.996534f;
    const bfloat p1_c = (bfloat)-1.885615f;
    const bfloat p2_c = (bfloat)1.331023f;
    const bfloat p3_c = (bfloat)-0.443674f;
    const float inv_240 = 1.0f / 240.0f;

    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;

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

            bfloat128 s0 = v_bf16_mul_b(t0, t0, 0, t0, 1, 0);
            bfloat128 s1 = v_bf16_mul_b(t1, t1, 0, t1, 1, 0);
            bfloat128 q0 = v_bf16_mac_b(s0, p3_c, p2_c, 0, 1, 0);
            bfloat128 q1 = v_bf16_mac_b(s1, p3_c, p2_c, 0, 1, 0);
            q0 = v_bf16_mac_b(s0, q0, p1_c, 0, 1, 0);
            q1 = v_bf16_mac_b(s1, q1, p1_c, 0, 1, 0);
            q0 = v_bf16_mac_b(s0, q0, p0_c, 0, 1, 0);
            q1 = v_bf16_mac_b(s1, q1, p0_c, 0, 1, 0);
            bfloat128 fg0 = v_bf16_mac_b(s0, q0, t0, 0, 1, 0);
            bfloat128 fg1 = v_bf16_mac_b(s1, q1, t1, 0, 1, 0);
            bfloat128 p0 = v_bf16_mul_b(v_bf16_mul_b(fg0, c500, 0, fg0, 1, 0), th0, 0, th0, 1, 0);
            bfloat128 p1 = v_bf16_mul_b(v_bf16_mul_b(fg1, c500, 0, fg1, 1, 0), th1, 0, th1, 1, 0);

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
            minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){y0, y1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0);
            v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q, q8);
        }
    }
}
