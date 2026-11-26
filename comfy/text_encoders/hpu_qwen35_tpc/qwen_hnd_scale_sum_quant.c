// Fused Qwen-Image 2.1 attention output scaling (head_scale / row_sum) + HND->NHD transpose (32 heads -> 4096)
// + per-row dynamic FP8 E4M3 quantization for Gaudi2 RMW expmax FP8 attention.
//
// Inputs:
//   attn_in     : [32, rows, 128] BF16 (TPC dims [128, rows, 32])
//   head_scales : [32, 64] FP32 (element [h, 0] = v_scale[h])
//   sum_in      : [32, stat_stride] FP32 (TPC dims [stat_stride, 32], stat_stride >= rows)
// Outputs:
//   out_q       : [rows, 4096] FP8 E4M3 (TPC dims [4096, rows])
//   out_scale   : [rows] FP32
//
// VLM: 32*256B + 8*256B + 8*256B = 12 KiB <= 16 KiB small VLM (0 VRF spills).

__local__ bfloat128 row0_vlm[32];
__local__ float64 hs_vlm[8];
__local__ float64 eff_vlm[8];

#define APPLY_QWEN_HEAD(H_IDX, DG_IDX)                                                               \
    do {                                                                                             \
        float64 e0 = v_f32_mov_dual_group_all_b(s0, 0xFFFFFFFF, DG_IDX, DG_IDX, DG_IDX, DG_IDX,      \
                                                SW_MDG_ALL, s0, 1, 0);                               \
        bfloat128 hs0 = v_convert_f32_to_bf16_all_b((float128){e0, e0}, SW_RHNE, (bfloat128){0}, 1, 0); \
        bfloat128 v0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, r, (H_IDX), 0, 0}, attn_in), hs0, 0, hs0, 1, 0); \
        row0_vlm[(H_IDX)] = v0;                                                                      \
        amax0 = v_bf16_max_b(amax0, v_bf16_abs_b(v0, 0, v0, 1, 0), 0, amax0, 1, 0);                  \
    } while (0)

void main(tensor attn_in, tensor head_scales, tensor sum_in, tensor out_q, tensor out_scale, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float inv_240 = 1.0f / 240.0f;
    const shuffle_mask_results_t m32 = gen_shuffle_map_32bit_nolut();
    const int dg_all = SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;

    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;
    if (r0 >= r1) return;

    for (int g = 0; g < 8; ++g) {
        const int h0 = g << 2;
        float64 v = (float64)s_f32_ld_g(gen_addr((int5){0, h0, 0, 0, 0}, head_scales));
        v = v_f32_mov_dual_group_b((float64)s_f32_ld_g(gen_addr((int5){0, h0 + 1, 0, 0, 0}, head_scales)),
                                   0xFFFFFFFF, 0, 1, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, v, 1, 0);
        v = v_f32_mov_dual_group_b((float64)s_f32_ld_g(gen_addr((int5){0, h0 + 2, 0, 0, 0}, head_scales)),
                                   0xFFFFFFFF, 0, 2, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, v, 1, 0);
        v = v_f32_mov_dual_group_b((float64)s_f32_ld_g(gen_addr((int5){0, h0 + 3, 0, 0, 0}, head_scales)),
                                   0xFFFFFFFF, 0, 3, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, v, 1, 0);
        hs_vlm[g] = v;
    }

    for (int r_blk = r0; r_blk < r1; r_blk += 8) {
        const int r_end = (r_blk + 8 < r1) ? (r_blk + 8) : r1;
        for (int g = 0; g < 8; ++g) {
            const int h0 = g << 2;
            float64 sm = v_f32_ld_tnsr_b((int5){r_blk, h0, 0, 0, 0}, sum_in);
            sm = v_f32_mov_dual_group_b(v_f32_ld_tnsr_b((int5){r_blk, h0 + 1, 0, 0, 0}, sum_in),
                                        0xFFFFFFFF, 0, 1, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, sm, 1, 0);
            sm = v_f32_mov_dual_group_b(v_f32_ld_tnsr_b((int5){r_blk, h0 + 2, 0, 0, 0}, sum_in),
                                        0xFFFFFFFF, 0, 2, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, sm, 1, 0);
            sm = v_f32_mov_dual_group_b(v_f32_ld_tnsr_b((int5){r_blk, h0 + 3, 0, 0, 0}, sum_in),
                                        0xFFFFFFFF, 0, 3, SW_WR_LOWER_GROUP | SW_WR_UPPER_GROUP, sm, 1, 0);
            eff_vlm[g] = hs_vlm[g] / v_f32_max_b(sm, 1e-12f, 0, sm, 1, 0);
        }

        for (int k = 0; r_blk + k < r_end; ++k) {
            const int r = r_blk + k;
            const uchar256 lut0 = (uchar256)(uint64)(0x80808080u + (unsigned)k * 0x01010101u);
            bfloat128 amax0 = (bfloat128)0.0f;
            for (int g = 0; g < 8; ++g) {
                const int h0 = g << 2;
                float64 ef = eff_vlm[g];
                float64 s0 = v_f32_shuffle_b(ef, lut0, 0, ef, 1, 0);
                s0 = v_f32_mov_group_b(s0, 0xFFFFFFFF, dg_all, s0, 1, 0);
                APPLY_QWEN_HEAD(h0, 0);
                APPLY_QWEN_HEAD(h0 + 1, 1);
                APPLY_QWEN_HEAD(h0 + 2, 2);
                APPLY_QWEN_HEAD(h0 + 3, 3);
            }
            float128 af0 = v_convert_bf16_to_f32_all_b(amax0, 0, (float128){0}, 1, 0);
            float64 sc0 = v_f32_max_b(
                v_f32_reduce_nolut_noshuffle_max(v_f32_max_b(af0.v1, af0.v2, 0, af0.v1, 1, 0), m32),
                1e-12f, 0, af0.v1, 1, 0) * inv_240;
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, out_scale, sc0, 0, 0);
            float64 is0 = (float64)1.0f / sc0;
            bfloat128 isb0 = v_convert_f32_to_bf16_all_b((float128){is0, is0}, SW_RHNE, (bfloat128){0}, 1, 0);
            for (int h = 0; h < 32; h += 2) {
                bfloat128 a0 = v_bf16_mul_b(row0_vlm[h], isb0, 0, isb0, 1, 0);
                bfloat128 a1 = v_bf16_mul_b(row0_vlm[h + 1], isb0, 0, isb0, 1, 0);
                v_f8_st_tnsr((int5){h * 128, r, 0, 0, 0}, out_q,
                    v_convert_bf16_to_f8_all_b((bfloat256){a0, a1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, (minifloat256){0}, 1, 0));
            }
        }
    }
}
