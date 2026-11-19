// Fused dual RMSNorm + concat + dynamic FP8 E4M3 quantization for MTP draft pre_fc.
// Computes cat(RMSNorm(emb, w_emb), RMSNorm(hid, w_hid)) -> FP8 [rows, 2*dim] + scale [rows, 1].
// Uses zero VLM arrays and only rsqrt_f32 LUT (reusing rsqrt(scale^2) for 1/scale).
void main(tensor emb, tensor hid, tensor w_emb, tensor w_hid,
          tensor out_q, tensor out_scale,
          int dim, int rows, int add_one, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_dim = 1.0f / (float)dim;
    const float inv_240 = 1.0f / 240.0f;
    const int n128 = dim >> 7;

    for (int r = begin[0]; r < end[0] && r < rows; ++r) {
        float64 sq_e0 = 0.0f, sq_e1 = 0.0f, sq_h0 = 0.0f, sq_h1 = 0.0f;
        for (int j = 0; j < n128; ++j) {
            int5 p = {j * 128, r, 0, 0, 0};
            float128 ef = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, emb), 0, (float128){0}, 1, 0);
            float128 hf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, hid), 0, (float128){0}, 1, 0);
            sq_e0 += ef.v1 * ef.v1;
            sq_e1 += ef.v2 * ef.v2;
            sq_h0 += hf.v1 * hf.v1;
            sq_h1 += hf.v2 * hf.v2;
        }
        float64 rstd_e = rsqrt_f32(v_f32_reduce_nolut_add(sq_e0 + sq_e1) * inv_dim + eps);
        float64 rstd_h = rsqrt_f32(v_f32_reduce_nolut_add(sq_h0 + sq_h1) * inv_dim + eps);

        float64 amax0 = 0.0f, amax1 = 0.0f;
        for (int j = 0; j < n128; ++j) {
            int5 p = {j * 128, r, 0, 0, 0};
            int5 pw = {j * 128, 0, 0, 0, 0};
            float128 ef = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, emb), 0, (float128){0}, 1, 0);
            bfloat128 ne = v_convert_f32_to_bf16_all_b((float128){ef.v1 * rstd_e, ef.v2 * rstd_e}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 we = v_bf16_ld_tnsr_b(pw, w_emb);
            if (add_one) we = v_bf16_add_b(we, (bfloat)1.0f, 0, we, 1, 0);
            float128 ye = v_convert_bf16_to_f32_all_b(v_bf16_mul_b(ne, we, 0, ne, 1, 0), 0, (float128){0}, 1, 0);
            amax0 = v_f32_max_b(amax0, v_f32_abs_b(ye.v1, 0, ye.v1, 1, 0), 0, amax0, 1, 0);
            amax1 = v_f32_max_b(amax1, v_f32_abs_b(ye.v2, 0, ye.v2, 1, 0), 0, amax1, 1, 0);

            float128 hf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p, hid), 0, (float128){0}, 1, 0);
            bfloat128 nh = v_convert_f32_to_bf16_all_b((float128){hf.v1 * rstd_h, hf.v2 * rstd_h}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 wh = v_bf16_ld_tnsr_b(pw, w_hid);
            if (add_one) wh = v_bf16_add_b(wh, (bfloat)1.0f, 0, wh, 1, 0);
            float128 yh = v_convert_bf16_to_f32_all_b(v_bf16_mul_b(nh, wh, 0, nh, 1, 0), 0, (float128){0}, 1, 0);
            amax0 = v_f32_max_b(amax0, v_f32_abs_b(yh.v1, 0, yh.v1, 1, 0), 0, amax0, 1, 0);
            amax1 = v_f32_max_b(amax1, v_f32_abs_b(yh.v2, 0, yh.v2, 1, 0), 0, amax1, 1, 0);
        }
        float64 amax = v_f32_reduce_nolut_max(v_f32_max_b(amax0, amax1, 0, amax0, 1, 0));
        float64 scale = v_f32_max_b(amax, 1e-12f, 0, amax, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){0, r, 0, 0, 0}, out_scale, scale, 0, 0);
        float64 inv_s = rsqrt_f32(scale * scale);

        for (int j = 0; j < n128; j += 2) {
            bfloat128 qbf[2];
            for (int k = 0; k < 2; ++k) {
                int col = (j + k) * 128;
                float128 ef = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, emb), 0, (float128){0}, 1, 0);
                bfloat128 ne = v_convert_f32_to_bf16_all_b((float128){ef.v1 * rstd_e, ef.v2 * rstd_e}, SW_RHNE, (bfloat128){0}, 1, 0);
                bfloat128 we = v_bf16_ld_tnsr_b((int5){col, 0, 0, 0, 0}, w_emb);
                if (add_one) we = v_bf16_add_b(we, (bfloat)1.0f, 0, we, 1, 0);
                float128 y = v_convert_bf16_to_f32_all_b(v_bf16_mul_b(ne, we, 0, ne, 1, 0), 0, (float128){0}, 1, 0);
                qbf[k] = v_convert_f32_to_bf16_all_b((float128){
                    v_f32_min_b(v_f32_max_b(y.v1 * inv_s, -240.0f, 0, y.v1, 1, 0), 240.0f, 0, y.v1, 1, 0),
                    v_f32_min_b(v_f32_max_b(y.v2 * inv_s, -240.0f, 0, y.v2, 1, 0), 240.0f, 0, y.v2, 1, 0)},
                    SW_RHNE, (bfloat128){0}, 1, 0);
            }
            v_f8_st_tnsr((int5){j * 128, r, 0, 0, 0}, out_q,
                         v_convert_bf16_to_f8_all_b((bfloat256){qbf[0], qbf[1]}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));

            for (int k = 0; k < 2; ++k) {
                int col = (j + k) * 128;
                float128 hf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, hid), 0, (float128){0}, 1, 0);
                bfloat128 nh = v_convert_f32_to_bf16_all_b((float128){hf.v1 * rstd_h, hf.v2 * rstd_h}, SW_RHNE, (bfloat128){0}, 1, 0);
                bfloat128 wh = v_bf16_ld_tnsr_b((int5){col, 0, 0, 0, 0}, w_hid);
                if (add_one) wh = v_bf16_add_b(wh, (bfloat)1.0f, 0, wh, 1, 0);
                float128 y = v_convert_bf16_to_f32_all_b(v_bf16_mul_b(nh, wh, 0, nh, 1, 0), 0, (float128){0}, 1, 0);
                qbf[k] = v_convert_f32_to_bf16_all_b((float128){
                    v_f32_min_b(v_f32_max_b(y.v1 * inv_s, -240.0f, 0, y.v1, 1, 0), 240.0f, 0, y.v1, 1, 0),
                    v_f32_min_b(v_f32_max_b(y.v2 * inv_s, -240.0f, 0, y.v2, 1, 0), 240.0f, 0, y.v2, 1, 0)},
                    SW_RHNE, (bfloat128){0}, 1, 0);
            }
            v_f8_st_tnsr((int5){dim + j * 128, r, 0, 0, 0}, out_q,
                         v_convert_bf16_to_f8_all_b((bfloat256){qbf[0], qbf[1]}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0));
        }
    }
}
