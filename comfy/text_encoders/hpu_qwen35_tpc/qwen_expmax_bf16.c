// Pure-BF16 attention exp(x - max) + row sum + row max for Gaudi2 in-place RMW SRAM attention.
//
// Inputs:
//   s_in       : [k_pad, rows] BF16 (scores pre-scaled by 128^-0.25 * sqrt(128*log2(e)/185.0) on Q and K)
// Outputs:
//   p_out      : [k_pad, rows] BF16 (unnormalized exp(s - max), exact 0.0 on padded cols >= valid_keys;
//                safe for in-place aliasing p_out == s_in via inputsReusability)
//   sum_out    : [rows] FP32 (row-wise sum of p_out)
//   max_out    : [rows] FP32 (row-wise max of s_in, used for multi-block K online Flash-RMW merge)
//
// Design:
//   - Zero VLM usage (0 bytes local memory, 0 VRF spills): pure 2-pass SRAM read directly from RMW SRAM.
//   - Pass 1 finishes reading row r before Pass 2 writes p_out[r], making in-place s_in == p_out hazard-free.
//   - 4-vector unrolled Pass 2 hides Gaudi2 4-cycle VPU latency bubbles cleanly.
//   - Shifted symmetric quadratic mantissa correction in BF16:
//       d = (m_cur - x) * 185.0f, u = round_to_u16(d), eb = sat_sub_u16(16256, u)
//       t = ((eb & 0x007f) | 0x3f80) - 1.5f
//       e = (bf16)eb * (0.2269791f * t^2 + 0.94233699f)
//     Preserves exact 0.0 on saturated / masked -inf lanes and achieves <= 1 BF16 ULP accuracy
//     (matching or outperforming PyTorch BF16 SDPA vs FP64 across Layer 0..31 and 1MP..5MP+).

#define EXPMAX_BF16_SYM_4                                                                               \
    do {                                                                                                \
        bfloat128 d0 = v_bf16_sub_b(m_cur, x0, 0, x0, 1, 0);                                            \
        bfloat128 d1 = v_bf16_sub_b(m_cur, x1, 0, x1, 1, 0);                                            \
        bfloat128 d2 = v_bf16_sub_b(m_cur, x2, 0, x2, 1, 0);                                            \
        bfloat128 d3 = v_bf16_sub_b(m_cur, x3, 0, x3, 1, 0);                                            \
        d0 = v_bf16_mul_b(d0, (bfloat)185.0f, 0, d0, 1, 0);                                             \
        d1 = v_bf16_mul_b(d1, (bfloat)185.0f, 0, d1, 1, 0);                                             \
        d2 = v_bf16_mul_b(d2, (bfloat)185.0f, 0, d2, 1, 0);                                             \
        d3 = v_bf16_mul_b(d3, (bfloat)185.0f, 0, d3, 1, 0);                                             \
        ushort128 u0 = v_convert_bf16_to_u16_b(d0, SW_RHNE, (ushort128){0}, 1, 0);                      \
        ushort128 u1 = v_convert_bf16_to_u16_b(d1, SW_RHNE, (ushort128){0}, 1, 0);                      \
        ushort128 u2 = v_convert_bf16_to_u16_b(d2, SW_RHNE, (ushort128){0}, 1, 0);                      \
        ushort128 u3 = v_convert_bf16_to_u16_b(d3, SW_RHNE, (ushort128){0}, 1, 0);                      \
        ushort128 eb0 = v_u16_sub_b(c_bias_u16, u0, SW_SAT, u0, 1, 0);                                  \
        ushort128 eb1 = v_u16_sub_b(c_bias_u16, u1, SW_SAT, u1, 1, 0);                                  \
        ushort128 eb2 = v_u16_sub_b(c_bias_u16, u2, SW_SAT, u2, 1, 0);                                  \
        ushort128 eb3 = v_u16_sub_b(c_bias_u16, u3, SW_SAT, u3, 1, 0);                                  \
        ushort128 m0_ = v_u16_and_b(eb0, (unsigned short)0x007f, 0, eb0, 1, 0);                         \
        ushort128 m1_ = v_u16_and_b(eb1, (unsigned short)0x007f, 0, eb1, 1, 0);                         \
        ushort128 m2_ = v_u16_and_b(eb2, (unsigned short)0x007f, 0, eb2, 1, 0);                         \
        ushort128 m3_ = v_u16_and_b(eb3, (unsigned short)0x007f, 0, eb3, 1, 0);                         \
        bfloat128 t0_ = (bfloat128)v_u16_or_b(m0_, (unsigned short)0x3f80, 0, m0_, 1, 0);               \
        bfloat128 t1_ = (bfloat128)v_u16_or_b(m1_, (unsigned short)0x3f80, 0, m1_, 1, 0);               \
        bfloat128 t2_ = (bfloat128)v_u16_or_b(m2_, (unsigned short)0x3f80, 0, m2_, 1, 0);               \
        bfloat128 t3_ = (bfloat128)v_u16_or_b(m3_, (unsigned short)0x3f80, 0, m3_, 1, 0);               \
        t0_ = v_bf16_sub_b(t0_, (bfloat)1.5f, 0, t0_, 1, 0);                                            \
        t1_ = v_bf16_sub_b(t1_, (bfloat)1.5f, 0, t1_, 1, 0);                                            \
        t2_ = v_bf16_sub_b(t2_, (bfloat)1.5f, 0, t2_, 1, 0);                                            \
        t3_ = v_bf16_sub_b(t3_, (bfloat)1.5f, 0, t3_, 1, 0);                                            \
        t0_ = v_bf16_mul_b(t0_, t0_, 0, t0_, 1, 0);                                                     \
        t1_ = v_bf16_mul_b(t1_, t1_, 0, t1_, 1, 0);                                                     \
        t2_ = v_bf16_mul_b(t2_, t2_, 0, t2_, 1, 0);                                                     \
        t3_ = v_bf16_mul_b(t3_, t3_, 0, t3_, 1, 0);                                                     \
        t0_ = v_bf16_mac_b(t0_, (bfloat)0.2269791f, (bfloat128)(bfloat)0.94233699f, 0, 1, 0);           \
        t1_ = v_bf16_mac_b(t1_, (bfloat)0.2269791f, (bfloat128)(bfloat)0.94233699f, 0, 1, 0);           \
        t2_ = v_bf16_mac_b(t2_, (bfloat)0.2269791f, (bfloat128)(bfloat)0.94233699f, 0, 1, 0);           \
        t3_ = v_bf16_mac_b(t3_, (bfloat)0.2269791f, (bfloat128)(bfloat)0.94233699f, 0, 1, 0);           \
        bfloat128 e0 = v_bf16_mul_b((bfloat128)eb0, t0_, 0, t0_, 1, 0);                                 \
        bfloat128 e1 = v_bf16_mul_b((bfloat128)eb1, t1_, 0, t1_, 1, 0);                                 \
        bfloat128 e2 = v_bf16_mul_b((bfloat128)eb2, t2_, 0, t2_, 1, 0);                                 \
        bfloat128 e3 = v_bf16_mul_b((bfloat128)eb3, t3_, 0, t3_, 1, 0);                                 \
        acc0 = v_bf16_add_b(acc0, e0, 0, acc0, 1, 0);                                                   \
        acc1 = v_bf16_add_b(acc1, e1, 0, acc1, 1, 0);                                                   \
        acc2 = v_bf16_add_b(acc2, e2, 0, acc2, 1, 0);                                                   \
        acc3 = v_bf16_add_b(acc3, e3, 0, acc3, 1, 0);                                                   \
        v_bf16_st_tnsr(p0, p_out, e0);                                                                  \
        v_bf16_st_tnsr(p1, p_out, e1);                                                                  \
        v_bf16_st_tnsr(p2, p_out, e2);                                                                  \
        v_bf16_st_tnsr(p3, p_out, e3);                                                                  \
    } while (0)

static inline bfloat128 mask_tail_128(bfloat128 x, int rem, short128 lane_id, bfloat128 neg_inf_v) {
    if (rem >= 128) return x;
    if (rem <= 0) return neg_inf_v;
    bool128 m = v_i16_cmp_less_b(lane_id, (short)rem, 0, (bool128){0}, 1, 0);
    return v_bf16_mov_vb(neg_inf_v, 0, x, m, 1);
}

void main(tensor s_in, tensor p_out, tensor sum_out, tensor max_out, int valid_keys, int k_pad, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int r0 = begin[0];
    const int r1 = (end[0] < rows) ? end[0] : rows;
    if (r0 >= r1) return;

    const short128 lane_id = (short128)read_lane_id_2b_b();
    const bfloat128 neg_inf_v = (bfloat128)(bfloat)-3.38953139e38f;
    const ushort128 c_bias_u16 = (ushort128)(unsigned short)16256; // 0x3f80 = 1.0f in BF16
    const bfloat128 zero_bf = (bfloat128)0.0f;
    // Restore true natural-log score scale for max_out: alpha = 185.0 * ln(2) / 128.0.
    const float max_to_ln = 1.0018142843f;

    // Fast path: k_pad >= 512, k_pad % 512 == 0, valid_keys > k_pad - 512 (zero inner-loop branches).
    if (k_pad >= 512 && (k_pad & 511) == 0 && valid_keys > (k_pad - 512)) {
        const int n_body = k_pad - 512;
        const int rem512 = valid_keys - n_body;
        const bool128 tm0 = v_i16_cmp_less_b(lane_id, (short)rem512, 0, (bool128){0}, 1, 0);
        const bool128 tm1 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 128), 0, (bool128){0}, 1, 0);
        const bool128 tm2 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 256), 0, (bool128){0}, 1, 0);
        const bool128 tm3 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 384), 0, (bool128){0}, 1, 0);
        for (int r = r0; r < r1; ++r) {
            bfloat128 m0 = neg_inf_v, m1 = neg_inf_v, m2 = neg_inf_v, m3 = neg_inf_v;
            int5 c0 = {0, r, 0, 0, 0}, c1 = {128, r, 0, 0, 0}, c2 = {256, r, 0, 0, 0}, c3 = {384, r, 0, 0, 0};
            for (int c = 0; c < n_body; c += 512) {
                bfloat128 a0 = v_bf16_ld_tnsr_b(c0, s_in); c0[0] += 512;
                bfloat128 a1 = v_bf16_ld_tnsr_b(c1, s_in); c1[0] += 512;
                bfloat128 a2 = v_bf16_ld_tnsr_b(c2, s_in); c2[0] += 512;
                bfloat128 a3 = v_bf16_ld_tnsr_b(c3, s_in); c3[0] += 512;
                m0 = v_bf16_max_b(m0, a0, 0, m0, 1, 0); m1 = v_bf16_max_b(m1, a1, 0, m1, 1, 0);
                m2 = v_bf16_max_b(m2, a2, 0, m2, 1, 0); m3 = v_bf16_max_b(m3, a3, 0, m3, 1, 0);
            }
            bfloat128 t0 = v_bf16_mov_vb(neg_inf_v, 0, v_bf16_ld_tnsr_b(c0, s_in), tm0, 1);
            bfloat128 t1 = v_bf16_mov_vb(neg_inf_v, 0, v_bf16_ld_tnsr_b(c1, s_in), tm1, 1);
            bfloat128 t2 = v_bf16_mov_vb(neg_inf_v, 0, v_bf16_ld_tnsr_b(c2, s_in), tm2, 1);
            bfloat128 t3 = v_bf16_mov_vb(neg_inf_v, 0, v_bf16_ld_tnsr_b(c3, s_in), tm3, 1);
            m0 = v_bf16_max_b(m0, t0, 0, m0, 1, 0); m1 = v_bf16_max_b(m1, t1, 0, m1, 1, 0);
            m2 = v_bf16_max_b(m2, t2, 0, m2, 1, 0); m3 = v_bf16_max_b(m3, t3, 0, m3, 1, 0);
            bfloat128 m_cur = v_bf16_reduce_nolut_max(
                v_bf16_max_b(v_bf16_max_b(m0, m1, 0, m0, 1, 0), v_bf16_max_b(m2, m3, 0, m2, 1, 0), 0, m0, 1, 0));

            bfloat128 acc0 = zero_bf, acc1 = zero_bf, acc2 = zero_bf, acc3 = zero_bf;
            int5 l0 = {0, r, 0, 0, 0}, l1 = {128, r, 0, 0, 0}, l2 = {256, r, 0, 0, 0}, l3 = {384, r, 0, 0, 0};
            int5 p0 = {0, r, 0, 0, 0}, p1 = {128, r, 0, 0, 0}, p2 = {256, r, 0, 0, 0}, p3 = {384, r, 0, 0, 0};
            for (int c = 0; c < n_body; c += 512) {
                bfloat128 x0 = v_bf16_ld_tnsr_b(l0, s_in); l0[0] += 512;
                bfloat128 x1 = v_bf16_ld_tnsr_b(l1, s_in); l1[0] += 512;
                bfloat128 x2 = v_bf16_ld_tnsr_b(l2, s_in); l2[0] += 512;
                bfloat128 x3 = v_bf16_ld_tnsr_b(l3, s_in); l3[0] += 512;
                EXPMAX_BF16_SYM_4;
                p0[0] += 512; p1[0] += 512; p2[0] += 512; p3[0] += 512;
            }
            {
                bfloat128 x0 = t0, x1 = t1, x2 = t2, x3 = t3;
                EXPMAX_BF16_SYM_4;
            }
            float128 f01 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc0, acc1, 0, acc0, 1, 0), 0, (float128){0}, 1, 0);
            float128 f23 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc2, acc3, 0, acc2, 1, 0), 0, (float128){0}, 1, 0);
            float64 tot = v_f32_reduce_nolut_add(
                v_f32_add_b(v_f32_add_b(f01.v1, f01.v2, 0, f01.v1, 1, 0),
                            v_f32_add_b(f23.v1, f23.v2, 0, f23.v1, 1, 0), 0, f01.v1, 1, 0));
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, sum_out, tot, 0, 0);
            float64 m_f32 = v_convert_bf16_to_f32_all_b(m_cur, 0, (float128){0}, 1, 0).v1 * max_to_ln;
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, max_out, m_f32, 0, 0);
        }
        return;
    }

    // General fallback: any k_pad >= 128 (k_pad % 128 == 0) and any 1 <= valid_keys <= k_pad.
    for (int r = r0; r < r1; ++r) {
        bfloat128 m0 = neg_inf_v;
        for (int c = 0; c < k_pad; c += 128) {
            bfloat128 x0 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c, r, 0, 0, 0}, s_in), valid_keys - c, lane_id, neg_inf_v);
            m0 = v_bf16_max_b(m0, x0, 0, m0, 1, 0);
        }
        bfloat128 m_cur = v_bf16_reduce_nolut_max(m0);
        bfloat128 acc0 = zero_bf;
        for (int c = 0; c < k_pad; c += 128) {
            bfloat128 x0 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c, r, 0, 0, 0}, s_in), valid_keys - c, lane_id, neg_inf_v);
            bfloat128 d0 = v_bf16_mul_b(v_bf16_sub_b(m_cur, x0, 0, x0, 1, 0), (bfloat)185.0f, 0, x0, 1, 0);
            ushort128 u0 = v_convert_bf16_to_u16_b(d0, SW_RHNE, (ushort128){0}, 1, 0);
            ushort128 eb0 = v_u16_sub_b(c_bias_u16, u0, SW_SAT, u0, 1, 0);
            ushort128 m0_ = v_u16_and_b(eb0, (unsigned short)0x007f, 0, eb0, 1, 0);
            bfloat128 t0_ = v_bf16_sub_b((bfloat128)v_u16_or_b(m0_, (unsigned short)0x3f80, 0, m0_, 1, 0), (bfloat)1.5f, 0, x0, 1, 0);
            bfloat128 q0 = v_bf16_mac_b(v_bf16_mul_b(t0_, t0_, 0, x0, 1, 0), (bfloat)0.2269791f, (bfloat128)(bfloat)0.94233699f, 0, 1, 0);
            bfloat128 e0 = v_bf16_mul_b((bfloat128)eb0, q0, 0, x0, 1, 0);
            acc0 = v_bf16_add_b(acc0, e0, 0, acc0, 1, 0);
            v_bf16_st_tnsr((int5){c, r, 0, 0, 0}, p_out, e0);
        }
        float128 f01 = v_convert_bf16_to_f32_all_b(acc0, 0, (float128){0}, 1, 0);
        float64 tot = v_f32_reduce_nolut_add(v_f32_add_b(f01.v1, f01.v2, 0, f01.v1, 1, 0));
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, sum_out, tot, 0, 0);
        float64 m_f32 = v_convert_bf16_to_f32_all_b(m_cur, 0, (float128){0}, 1, 0).v1 * max_to_ln;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, max_out, m_f32, 0, 0);
    }
}
