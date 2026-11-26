// Fused Qwen-Image 2.1 prefix V + target V concatenation + 512-pad + per-head FP8 E4M3 quantization
// in paired 256-row all_lanes layout matching qwen_expmax_fp8 P_fp8 output.
// 24-core load-balanced across (32 heads * chunks) tiles, 0 B VLM, 0 spills.
//
// Inputs:
//   v_raw    : [4096, Q] BF16
//   pv       : [128, 32, P] BF16
//   v_scales : [64, 32] FP32 (element [0, h] = per-head V scale)
// Outputs:
//   vh_q_out : [256, k_pad/2, 32] FP8 E4M3 (zero-copy view [128, k_pad, 32])

void main(tensor v_raw, tensor pv, tensor v_scales, tensor vh_q_out, int Q, int P, int k_pad, int chunks) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total_tiles = 32 * chunks;
    const int t0 = begin[0];
    const int t1 = (end[0] < total_tiles) ? end[0] : total_tiles;
    if (t0 >= t1) return;

    const bfloat128 zero_bf = (bfloat128)0.0f;
    const minifloat256 zero_f8 = (minifloat256)0;
    const int kv_len = P + Q;
    const int total_blks = k_pad >> 8;
    const int blks_per_ch = (total_blks + chunks - 1) / chunks;

    for (int tid = t0; tid < t1; ++tid) {
        const int h = tid / chunks;
        const int ch = tid - h * chunks;
        const int b_start = ch * blks_per_ch;
        int b_end = b_start + blks_per_ch;
        if (b_end > total_blks) b_end = total_blks;
        if (b_start >= b_end) continue;
        const int col = h * 128;

        float vsc = s_f32_ld_g(gen_addr((int5){0, h, 0, 0, 0}, v_scales));
        float64 inv_vs = (float64)1.0f / (float64)vsc;
        bfloat128 inv_vsb = v_convert_f32_to_bf16_all_b((float128){inv_vs, inv_vs}, SW_RHNE, zero_bf, 1, 0);

        for (int blk = b_start; blk < b_end; ++blk) {
            const int src_base = blk << 8;
            const int dst_base = blk << 7;
            if (src_base >= P && src_base + 256 <= kv_len) {
                const int vb = src_base - P;
                for (int j = 0; j < 128; j += 2) {
                    bfloat128 lo0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, vb + j, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, vb + 128 + j, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 lo1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, vb + j + 1, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, vb + 129 + j, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    v_f8_st_tnsr((int5){0, dst_base + j, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo0, hi0}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                    v_f8_st_tnsr((int5){0, dst_base + j + 1, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo1, hi1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                }
            } else if (src_base + 256 <= P) {
                for (int j = 0; j < 128; j += 2) {
                    bfloat128 lo0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, src_base + j, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi0 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, src_base + 128 + j, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 lo1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, src_base + j + 1, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    bfloat128 hi1 = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, src_base + 129 + j, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    v_f8_st_tnsr((int5){0, dst_base + j, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo0, hi0}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                    v_f8_st_tnsr((int5){0, dst_base + j + 1, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo1, hi1}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                }
            } else {
                for (int j = 0; j < 128; ++j) {
                    const int sl = src_base + j;
                    const int sh = src_base + 128 + j;
                    bfloat128 lo = zero_bf, hi = zero_bf;
                    if (sl < P) {
                        lo = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, sl, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    } else if (sl < kv_len) {
                        lo = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, sl - P, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    }
                    if (sh < P) {
                        hi = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, sh, 0, 0}, pv), inv_vsb, 0, inv_vsb, 1, 0);
                    } else if (sh < kv_len) {
                        hi = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){col, sh - P, 0, 0, 0}, v_raw), inv_vsb, 0, inv_vsb, 1, 0);
                    }
                    v_f8_st_tnsr((int5){0, dst_base + j, h, 0, 0}, vh_q_out,
                        v_convert_bf16_to_f8_all_b((bfloat256){lo, hi}, SW_RHNE | SW_LINEAR | SW_CLIP_FP, zero_f8, 1, 0));
                }
            }
        }
    }
}
