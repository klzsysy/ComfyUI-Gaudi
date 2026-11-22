// Fused MOSS attention output scaling + HND->NHD transpose + per-row dynamic FP8 E4M3 quantization for Gaudi2.
// Inputs: attn_in [24, rows, 128] BF16 (TPC dims [128, rows, 24]), head_scales [24, 64] FP32 (broadcast per head).
// Outputs: out_q [rows, 3072] FP8 E4M3, out_scale [rows] FP32.
// Processes 2 contiguous rows per head for better DRAM row-buffer locality (48 bfloat128 = 12 KiB <= 16 KiB small VLM).
__local__ bfloat128 row0_vlm[24];
__local__ bfloat128 row1_vlm[24];

void main(tensor attn_in, tensor head_scales, tensor out_q, tensor out_scale, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_240 = 1.0f / 240.0f;

    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;

    int r = r0;
    for (; r + 1 < r1; r += 2) {
        bfloat128 amax0 = (bfloat128)0.0f;
        bfloat128 amax1 = (bfloat128)0.0f;
        for (int h = 0; h < 24; ++h) {
            float64 sv = v_f32_ld_tnsr_b((int5){0, h, 0, 0, 0}, head_scales);
            bfloat128 hs = v_convert_f32_to_bf16_all_b((float128){sv, sv}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 v0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, r, h, 0, 0}, attn_in), hs, 0, hs, 1, 0);
            bfloat128 v1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, r + 1, h, 0, 0}, attn_in), hs, 0, hs, 1, 0);
            row0_vlm[h] = v0;
            row1_vlm[h] = v1;
            amax0 = v_bf16_max_b(amax0, v_bf16_abs_b(v0, 0, v0, 1, 0), 0, amax0, 1, 0);
            amax1 = v_bf16_max_b(amax1, v_bf16_abs_b(v1, 0, v1, 1, 0), 0, amax1, 1, 0);
        }
        float128 af0 = v_convert_bf16_to_f32_all_b(amax0, 0, (float128){0}, 1, 0);
        float128 af1 = v_convert_bf16_to_f32_all_b(amax1, 0, (float128){0}, 1, 0);
        float64 sc0 = v_f32_max_b(v_f32_reduce_nolut_max(v_f32_max_b(af0.v1, af0.v2, 0, af0.v1, 1, 0)), 1e-12f, 0, af0.v1, 1, 0) * inv_240;
        float64 sc1 = v_f32_max_b(v_f32_reduce_nolut_max(v_f32_max_b(af1.v1, af1.v2, 0, af1.v1, 1, 0)), 1e-12f, 0, af1.v1, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, sc0, 0, 0);
        v_f32_st_tnsr_partial((int5){r + 1, 0, 0, 0, 0}, out_scale, sc1, 0, 0);
        float64 is0 = (float64)1.0f / sc0;
        float64 is1 = (float64)1.0f / sc1;
        bfloat128 isb0 = v_convert_f32_to_bf16_all_b((float128){is0, is0}, SW_RHNE, (bfloat128){0}, 1, 0);
        bfloat128 isb1 = v_convert_f32_to_bf16_all_b((float128){is1, is1}, SW_RHNE, (bfloat128){0}, 1, 0);
        for (int h = 0; h < 24; h += 2) {
            bfloat128 a0 = v_bf16_mul_b(row0_vlm[h], isb0, 0, isb0, 1, 0);
            bfloat128 a1 = v_bf16_mul_b(row0_vlm[h + 1], isb0, 0, isb0, 1, 0);
            bfloat128 b0 = v_bf16_mul_b(row1_vlm[h], isb1, 0, isb1, 1, 0);
            bfloat128 b1 = v_bf16_mul_b(row1_vlm[h + 1], isb1, 0, isb1, 1, 0);
            v_f8_st_tnsr((int5){h * 128, r, 0, 0, 0}, out_q, v_convert_bf16_to_f8_all_b((bfloat256){a0, a1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0));
            v_f8_st_tnsr((int5){h * 128, r + 1, 0, 0, 0}, out_q, v_convert_bf16_to_f8_all_b((bfloat256){b0, b1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0));
        }
    }
    if (r < r1) {
        bfloat128 amax0 = (bfloat128)0.0f;
        for (int h = 0; h < 24; ++h) {
            float64 sv = v_f32_ld_tnsr_b((int5){0, h, 0, 0, 0}, head_scales);
            bfloat128 hs = v_convert_f32_to_bf16_all_b((float128){sv, sv}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 v0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, r, h, 0, 0}, attn_in), hs, 0, hs, 1, 0);
            row0_vlm[h] = v0;
            amax0 = v_bf16_max_b(amax0, v_bf16_abs_b(v0, 0, v0, 1, 0), 0, amax0, 1, 0);
        }
        float128 af0 = v_convert_bf16_to_f32_all_b(amax0, 0, (float128){0}, 1, 0);
        float64 sc0 = v_f32_max_b(v_f32_reduce_nolut_max(v_f32_max_b(af0.v1, af0.v2, 0, af0.v1, 1, 0)), 1e-12f, 0, af0.v1, 1, 0) * inv_240;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, sc0, 0, 0);
        float64 is0 = (float64)1.0f / sc0;
        bfloat128 isb0 = v_convert_f32_to_bf16_all_b((float128){is0, is0}, SW_RHNE, (bfloat128){0}, 1, 0);
        for (int h = 0; h < 24; h += 2) {
            bfloat128 a0 = v_bf16_mul_b(row0_vlm[h], isb0, 0, isb0, 1, 0);
            bfloat128 a1 = v_bf16_mul_b(row0_vlm[h + 1], isb0, 0, isb0, 1, 0);
            v_f8_st_tnsr((int5){h * 128, r, 0, 0, 0}, out_q, v_convert_bf16_to_f8_all_b((bfloat256){a0, a1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0));
        }
    }
}
