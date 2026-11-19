// Fused Q/Gate split + Q-RMSNorm + K-RMSNorm + Partial-RoPE64 (split-half) + [1, H, rows, head_dim] transpose.
// Replaces ~30 serialized EDMA transposes/slices/norms/cats in every GatedAttention and MTP layer with 1 wave.
// Supports head_dim == 256 and rotary_dim == 64 (standard Qwen3.5 full-attention layout).
// cs: [128, rows] FP32 where cs[0..63, r] = [cos[0..31], cos[0..31]] and cs[64..127, r] = [-sin[0..31], +sin[0..31]].
static inline float64 ld_bf16_linear_f32(int5 pos, tensor t) {
    bfloat128 u = v_bf16_ld_tnsr_b(pos, t, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
    return v_convert_bf16_to_f32_b(u, 0, 0, 1, 0);
}

static inline float64 swap_halves_64(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    return v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, writes, 0);
}

void main(tensor qg, tensor xk, tensor qn_w, tensor kn_w, tensor cs,
          tensor xq_out, tensor xk_out, tensor gate_out,
          int num_heads, int num_kv_heads, int rows, int add_one, float eps) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total_heads = num_heads + num_kv_heads;
    const float inv_256 = 1.0f / 256.0f;

    for (int wid = begin[0]; wid < end[0] && wid < total_heads; ++wid) {
        const int is_q = (wid < num_heads);
        const int h = is_q ? wid : (wid - num_heads);

        bfloat128 wb23 = v_bf16_ld_tnsr_b((int5){128, 0, 0, 0, 0}, is_q ? qn_w : kn_w);
        if (add_one) {
            wb23 = v_bf16_add_b(wb23, (bfloat)1.0f, 0, wb23, 1, 0);
        }
        float64 wf0 = ld_bf16_linear_f32((int5){0, 0, 0, 0, 0}, is_q ? qn_w : kn_w);
        float64 wf1 = ld_bf16_linear_f32((int5){64, 0, 0, 0, 0}, is_q ? qn_w : kn_w);
        if (add_one) {
            wf0 = wf0 + 1.0f;
            wf1 = wf1 + 1.0f;
        }

        for (int r = 0; r < rows; ++r) {
            const int src_col = is_q ? (h * 512) : (h * 256);
            if (is_q) {
                bfloat128 g0 = v_bf16_ld_tnsr_b((int5){src_col + 256, r, 0, 0, 0}, qg);
                bfloat128 g1 = v_bf16_ld_tnsr_b((int5){src_col + 384, r, 0, 0, 0}, qg);
                v_bf16_st_tnsr((int5){h * 256, r, 0, 0, 0}, gate_out, g0);
                v_bf16_st_tnsr((int5){h * 256 + 128, r, 0, 0, 0}, gate_out, g1);
            }

            float64 x0 = ld_bf16_linear_f32((int5){src_col, r, 0, 0, 0}, is_q ? qg : xk);
            float64 x1 = ld_bf16_linear_f32((int5){src_col + 64, r, 0, 0, 0}, is_q ? qg : xk);
            bfloat128 xb23 = v_bf16_ld_tnsr_b((int5){src_col + 128, r, 0, 0, 0}, is_q ? qg : xk);
            float128 x23 = v_convert_bf16_to_f32_all_b(xb23, 0, (float128){0}, 1, 0);

            float64 ss = v_f32_reduce_nolut_add(x0 * x0 + x1 * x1 + x23.v1 * x23.v1 + x23.v2 * x23.v2);
            float64 rstd = rsqrt_f32(ss * inv_256 + eps);

            bfloat128 nb23 = v_convert_f32_to_bf16_all_b(
                (float128){x23.v1 * rstd, x23.v2 * rstd}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 out23 = v_bf16_mul_b(nb23, wb23, 0, nb23, 1, 0);

            float64 cos64 = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, cs);
            float64 sin64 = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, cs);
            float64 n0 = x0 * rstd * wf0;
            float64 n1 = x1 * rstd * wf1;
            float64 n0_rot = n0 * cos64 + swap_halves_64(n0) * sin64;
            bfloat128 out01 = v_convert_f32_to_bf16_all_b(
                (float128){n0_rot, n1}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0);

            if (is_q) {
                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, xq_out, out01);
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, xq_out, out23);
            } else {
                v_bf16_st_tnsr((int5){0, r, h, 0, 0}, xk_out, out01);
                v_bf16_st_tnsr((int5){128, r, h, 0, 0}, xk_out, out23);
            }
        }
    }
}
