// Fused Qwen-Image 2.1 FP8 attention exp(x - max) + FP8 E4M3 quantization + row sum + row max
// for multi-block K Flash-RMW SRAM attention on Gaudi2.
//
// Inputs:
//   s_in       : [k_pad, rows] BF16 (scores pre-scaled by 128 * log2(e) / sqrt(128) on Q, 1.0 on K)
// Outputs:
//   p_out      : [k_pad, rows] FP8 E4M3 (all_lanes 256-elem interleaved layout matching qwen_v_quant_fp8)
//   sum_out    : [rows] FP32 (row-wise sum of unnormalized quantized weights)
//   max_out    : [rows] FP32 (row-wise maximum in natural-log units: max_bf16 * ln(2) / 128)
//
// Fast path 1 (1536 <= k_pad <= 8704, k_pad % 512 == 0, valid_keys > k_pad - 512, r1 - r0 >= 2):
//   - Interleaved double-row single-read SRAM pipeline using 16 KiB small VLM (row_vlm[64], 0 VRF spills).
// Fast path 2 (k_pad >= 512, k_pad % 512 == 0, valid_keys > k_pad - 512):
//   - 0-VLM 2-pass 4-vector unrolled pipeline for 8704 < k_pad <= 10240 (e.g. 2MP pos 9728, 4MP pos 9216).
// General path:
//   - 2-pass fallback supporting any k_pad >= 256 (k_pad % 256 == 0), any 1 <= valid_keys <= k_pad, and rows >= 1.

__local__ bfloat128 row_vlm[64];

#define QWEN_EXPMAX_4(INC_P)                                                                     \
    do {                                                                                         \
        bfloat128 d0 = v_bf16_sub_b(m_cur, x0, 0, x0, 1, 0);                                     \
        bfloat128 d1 = v_bf16_sub_b(m_cur, x1, 0, x1, 1, 0);                                     \
        bfloat128 d2 = v_bf16_sub_b(m_cur, x2, 0, x2, 1, 0);                                     \
        bfloat128 d3 = v_bf16_sub_b(m_cur, x3, 0, x3, 1, 0);                                     \
        ushort128 u0 = v_convert_bf16_to_u16_b(d0, SW_RHNE, (ushort128){0}, 1, 0);               \
        ushort128 u1 = v_convert_bf16_to_u16_b(d1, SW_RHNE, (ushort128){0}, 1, 0);               \
        ushort128 u2 = v_convert_bf16_to_u16_b(d2, SW_RHNE, (ushort128){0}, 1, 0);               \
        ushort128 u3 = v_convert_bf16_to_u16_b(d3, SW_RHNE, (ushort128){0}, 1, 0);               \
        bfloat128 e0 = (bfloat128)v_u16_sub_b(c_bias_u16, u0, SW_SAT, u0, 1, 0);                 \
        bfloat128 e1 = (bfloat128)v_u16_sub_b(c_bias_u16, u1, SW_SAT, u1, 1, 0);                 \
        bfloat128 e2 = (bfloat128)v_u16_sub_b(c_bias_u16, u2, SW_SAT, u2, 1, 0);                 \
        bfloat128 e3 = (bfloat128)v_u16_sub_b(c_bias_u16, u3, SW_SAT, u3, 1, 0);                 \
        acc0 = v_bf16_add_b(acc0, e0, 0, acc0, 1, 0);                                            \
        acc1 = v_bf16_add_b(acc1, e1, 0, acc1, 1, 0);                                            \
        acc2 = v_bf16_add_b(acc2, e2, 0, acc2, 1, 0);                                            \
        acc3 = v_bf16_add_b(acc3, e3, 0, acc3, 1, 0);                                            \
        v_f8_st_tnsr(p0, p_out,                                                                  \
            v_convert_bf16_to_f8_all_b((bfloat256){e0, e1}, SW_RHNE | SW_CLIP_FP, zero_f8, 1, 0)); \
        v_f8_st_tnsr(p1, p_out,                                                                  \
            v_convert_bf16_to_f8_all_b((bfloat256){e2, e3}, SW_RHNE | SW_CLIP_FP, zero_f8, 1, 0)); \
        if (INC_P) {                                                                             \
            p0[0] += 512;                                                                        \
            p1[0] += 512;                                                                        \
        }                                                                                        \
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
    const ushort128 c_bias_u16 = (ushort128)(unsigned short)17262;
    const bfloat128 zero_bf = (bfloat128)0.0f;
    const minifloat256 zero_f8 = (minifloat256)0;
    const float max_to_ln = 0.00541521235f; // ln(2) / 128

    // Fast path 1: 16 KiB VLM double-row interleaved single-read (1536 <= k_pad <= 8704).
    if (k_pad >= 1536 && k_pad <= 8704 && (k_pad & 511) == 0 && valid_keys > (k_pad - 512) && (r1 - r0) >= 2) {
        const int n_body = k_pad - 512;
        const int body_v = n_body >> 7;
        const int loop_end = (body_v - 8) & ~7;
        const int odd4 = body_v & 4;
        const int rem512 = valid_keys - n_body;
        const bool128 tm0 = v_i16_cmp_less_b(lane_id, (short)rem512, 0, (bool128){0}, 1, 0);
        const bool128 tm1 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 128), 0, (bool128){0}, 1, 0);
        const bool128 tm2 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 256), 0, (bool128){0}, 1, 0);
        const bool128 tm3 = v_i16_cmp_less_b(lane_id, (short)(rem512 - 384), 0, (bool128){0}, 1, 0);

        bfloat128 m0 = neg_inf_v, m1 = neg_inf_v, m2 = neg_inf_v, m3 = neg_inf_v;
        int5 c0 = {0, r0, 0, 0, 0}, c1 = {128, r0, 0, 0, 0}, c2 = {256, r0, 0, 0, 0}, c3 = {384, r0, 0, 0, 0};
        for (int vi = 0; vi < body_v; vi += 4) {
            bfloat128 a0 = v_bf16_ld_tnsr_b(c0, s_in); c0[0] += 512;
            bfloat128 a1 = v_bf16_ld_tnsr_b(c1, s_in); c1[0] += 512;
            bfloat128 a2 = v_bf16_ld_tnsr_b(c2, s_in); c2[0] += 512;
            bfloat128 a3 = v_bf16_ld_tnsr_b(c3, s_in); c3[0] += 512;
            row_vlm[vi] = a0; row_vlm[vi + 1] = a1; row_vlm[vi + 2] = a2; row_vlm[vi + 3] = a3;
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

        for (int r = r0; r < r1 - 1; ++r) {
            const int rn = r + 1;
            bfloat128 nm0 = neg_inf_v, nm1 = neg_inf_v, nm2 = neg_inf_v, nm3 = neg_inf_v;
            bfloat128 acc0 = zero_bf, acc1 = zero_bf, acc2 = zero_bf, acc3 = zero_bf;
            int5 nc0 = {0, rn, 0, 0, 0}, nc1 = {128, rn, 0, 0, 0}, nc2 = {256, rn, 0, 0, 0}, nc3 = {384, rn, 0, 0, 0};
            int5 p0 = {0, r, 0, 0, 0}, p1 = {256, r, 0, 0, 0};

            bfloat128 a0 = v_bf16_ld_tnsr_b(nc0, s_in); nc0[0] += 512;
            bfloat128 a1 = v_bf16_ld_tnsr_b(nc1, s_in); nc1[0] += 512;
            bfloat128 a2 = v_bf16_ld_tnsr_b(nc2, s_in); nc2[0] += 512;
            bfloat128 a3 = v_bf16_ld_tnsr_b(nc3, s_in); nc3[0] += 512;
            bfloat128 b0 = v_bf16_ld_tnsr_b(nc0, s_in); nc0[0] += 512;
            bfloat128 b1 = v_bf16_ld_tnsr_b(nc1, s_in); nc1[0] += 512;
            bfloat128 b2 = v_bf16_ld_tnsr_b(nc2, s_in); nc2[0] += 512;
            bfloat128 b3 = v_bf16_ld_tnsr_b(nc3, s_in); nc3[0] += 512;

            for (int vi = 0; vi < loop_end; vi += 8) {
                {
                    bfloat128 x0 = row_vlm[vi], x1 = row_vlm[vi + 1], x2 = row_vlm[vi + 2], x3 = row_vlm[vi + 3];
                    row_vlm[vi] = a0; row_vlm[vi + 1] = a1; row_vlm[vi + 2] = a2; row_vlm[vi + 3] = a3;
                    nm0 = v_bf16_max_b(nm0, a0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, a1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, a2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, a3, 0, nm3, 1, 0);
                    a0 = v_bf16_ld_tnsr_b(nc0, s_in); nc0[0] += 512;
                    a1 = v_bf16_ld_tnsr_b(nc1, s_in); nc1[0] += 512;
                    a2 = v_bf16_ld_tnsr_b(nc2, s_in); nc2[0] += 512;
                    a3 = v_bf16_ld_tnsr_b(nc3, s_in); nc3[0] += 512;
                    QWEN_EXPMAX_4(1);
                }
                {
                    bfloat128 x0 = row_vlm[vi + 4], x1 = row_vlm[vi + 5], x2 = row_vlm[vi + 6], x3 = row_vlm[vi + 7];
                    row_vlm[vi + 4] = b0; row_vlm[vi + 5] = b1; row_vlm[vi + 6] = b2; row_vlm[vi + 7] = b3;
                    nm0 = v_bf16_max_b(nm0, b0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, b1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, b2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, b3, 0, nm3, 1, 0);
                    b0 = v_bf16_ld_tnsr_b(nc0, s_in); nc0[0] += 512;
                    b1 = v_bf16_ld_tnsr_b(nc1, s_in); nc1[0] += 512;
                    b2 = v_bf16_ld_tnsr_b(nc2, s_in); nc2[0] += 512;
                    b3 = v_bf16_ld_tnsr_b(nc3, s_in); nc3[0] += 512;
                    QWEN_EXPMAX_4(1);
                }
            }
            int vi = loop_end;
            if (odd4) {
                {
                    bfloat128 x0 = row_vlm[vi], x1 = row_vlm[vi + 1], x2 = row_vlm[vi + 2], x3 = row_vlm[vi + 3];
                    row_vlm[vi] = a0; row_vlm[vi + 1] = a1; row_vlm[vi + 2] = a2; row_vlm[vi + 3] = a3;
                    nm0 = v_bf16_max_b(nm0, a0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, a1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, a2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, a3, 0, nm3, 1, 0);
                    a0 = v_bf16_ld_tnsr_b(nc0, s_in); nc0[0] += 512;
                    a1 = v_bf16_ld_tnsr_b(nc1, s_in); nc1[0] += 512;
                    a2 = v_bf16_ld_tnsr_b(nc2, s_in); nc2[0] += 512;
                    a3 = v_bf16_ld_tnsr_b(nc3, s_in); nc3[0] += 512;
                    QWEN_EXPMAX_4(1);
                }
                {
                    bfloat128 x0 = row_vlm[vi + 4], x1 = row_vlm[vi + 5], x2 = row_vlm[vi + 6], x3 = row_vlm[vi + 7];
                    row_vlm[vi + 4] = b0; row_vlm[vi + 5] = b1; row_vlm[vi + 6] = b2; row_vlm[vi + 7] = b3;
                    nm0 = v_bf16_max_b(nm0, b0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, b1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, b2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, b3, 0, nm3, 1, 0);
                    b0 = v_bf16_ld_tnsr_b(nc0, s_in); b1 = v_bf16_ld_tnsr_b(nc1, s_in);
                    b2 = v_bf16_ld_tnsr_b(nc2, s_in); b3 = v_bf16_ld_tnsr_b(nc3, s_in);
                    QWEN_EXPMAX_4(1);
                }
                {
                    bfloat128 x0 = row_vlm[vi + 8], x1 = row_vlm[vi + 9], x2 = row_vlm[vi + 10], x3 = row_vlm[vi + 11];
                    row_vlm[vi + 8] = a0; row_vlm[vi + 9] = a1; row_vlm[vi + 10] = a2; row_vlm[vi + 11] = a3;
                    nm0 = v_bf16_max_b(nm0, a0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, a1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, a2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, a3, 0, nm3, 1, 0);
                    QWEN_EXPMAX_4(1);
                }
                { bfloat128 x0 = t0, x1 = t1, x2 = t2, x3 = t3; QWEN_EXPMAX_4(0); }
                t0 = v_bf16_mov_vb(neg_inf_v, 0, b0, tm0, 1); t1 = v_bf16_mov_vb(neg_inf_v, 0, b1, tm1, 1);
                t2 = v_bf16_mov_vb(neg_inf_v, 0, b2, tm2, 1); t3 = v_bf16_mov_vb(neg_inf_v, 0, b3, tm3, 1);
            } else {
                {
                    bfloat128 x0 = row_vlm[vi], x1 = row_vlm[vi + 1], x2 = row_vlm[vi + 2], x3 = row_vlm[vi + 3];
                    row_vlm[vi] = a0; row_vlm[vi + 1] = a1; row_vlm[vi + 2] = a2; row_vlm[vi + 3] = a3;
                    nm0 = v_bf16_max_b(nm0, a0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, a1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, a2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, a3, 0, nm3, 1, 0);
                    a0 = v_bf16_ld_tnsr_b(nc0, s_in); a1 = v_bf16_ld_tnsr_b(nc1, s_in);
                    a2 = v_bf16_ld_tnsr_b(nc2, s_in); a3 = v_bf16_ld_tnsr_b(nc3, s_in);
                    QWEN_EXPMAX_4(1);
                }
                {
                    bfloat128 x0 = row_vlm[vi + 4], x1 = row_vlm[vi + 5], x2 = row_vlm[vi + 6], x3 = row_vlm[vi + 7];
                    row_vlm[vi + 4] = b0; row_vlm[vi + 5] = b1; row_vlm[vi + 6] = b2; row_vlm[vi + 7] = b3;
                    nm0 = v_bf16_max_b(nm0, b0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, b1, 0, nm1, 1, 0);
                    nm2 = v_bf16_max_b(nm2, b2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, b3, 0, nm3, 1, 0);
                    QWEN_EXPMAX_4(1);
                }
                { bfloat128 x0 = t0, x1 = t1, x2 = t2, x3 = t3; QWEN_EXPMAX_4(0); }
                t0 = v_bf16_mov_vb(neg_inf_v, 0, a0, tm0, 1); t1 = v_bf16_mov_vb(neg_inf_v, 0, a1, tm1, 1);
                t2 = v_bf16_mov_vb(neg_inf_v, 0, a2, tm2, 1); t3 = v_bf16_mov_vb(neg_inf_v, 0, a3, tm3, 1);
            }
            nm0 = v_bf16_max_b(nm0, t0, 0, nm0, 1, 0); nm1 = v_bf16_max_b(nm1, t1, 0, nm1, 1, 0);
            nm2 = v_bf16_max_b(nm2, t2, 0, nm2, 1, 0); nm3 = v_bf16_max_b(nm3, t3, 0, nm3, 1, 0);
            float128 f01 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc0, acc1, 0, acc0, 1, 0), 0, (float128){0}, 1, 0);
            float128 f23 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc2, acc3, 0, acc2, 1, 0), 0, (float128){0}, 1, 0);
            float64 tot = v_f32_reduce_nolut_add(
                v_f32_add_b(v_f32_add_b(f01.v1, f01.v2, 0, f01.v1, 1, 0),
                            v_f32_add_b(f23.v1, f23.v2, 0, f23.v1, 1, 0), 0, f01.v1, 1, 0));
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, sum_out, tot, 0, 0);
            float64 m_f32 = v_convert_bf16_to_f32_all_b(m_cur, 0, (float128){0}, 1, 0).v1 * max_to_ln;
            v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, max_out, m_f32, 0, 0);
            m_cur = v_bf16_reduce_nolut_max(
                v_bf16_max_b(v_bf16_max_b(nm0, nm1, 0, nm0, 1, 0), v_bf16_max_b(nm2, nm3, 0, nm2, 1, 0), 0, nm0, 1, 0));
        }

        const int rl = r1 - 1;
        bfloat128 acc0 = zero_bf, acc1 = zero_bf, acc2 = zero_bf, acc3 = zero_bf;
        int5 p0 = {0, rl, 0, 0, 0}, p1 = {256, rl, 0, 0, 0};
        for (int vi = 0; vi < body_v; vi += 4) {
            bfloat128 x0 = row_vlm[vi], x1 = row_vlm[vi + 1], x2 = row_vlm[vi + 2], x3 = row_vlm[vi + 3];
            QWEN_EXPMAX_4(1);
        }
        { bfloat128 x0 = t0, x1 = t1, x2 = t2, x3 = t3; QWEN_EXPMAX_4(0); }
        float128 f01 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc0, acc1, 0, acc0, 1, 0), 0, (float128){0}, 1, 0);
        float128 f23 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc2, acc3, 0, acc2, 1, 0), 0, (float128){0}, 1, 0);
        float64 tot = v_f32_reduce_nolut_add(
            v_f32_add_b(v_f32_add_b(f01.v1, f01.v2, 0, f01.v1, 1, 0),
                        v_f32_add_b(f23.v1, f23.v2, 0, f23.v1, 1, 0), 0, f01.v1, 1, 0));
        v_f32_st_tnsr_partial((int5){rl, 0, 0, 0, 0}, sum_out, tot, 0, 0);
        float64 m_f32 = v_convert_bf16_to_f32_all_b(m_cur, 0, (float128){0}, 1, 0).v1 * max_to_ln;
        v_f32_st_tnsr_partial((int5){rl, 0, 0, 0, 0}, max_out, m_f32, 0, 0);
        return;
    }

    // Fast path 2: 0-VLM 2-pass 4-vector unrolled (k_pad >= 512, k_pad % 512 == 0, valid_keys > k_pad - 512).
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
            int5 p0 = {0, r, 0, 0, 0}, p1 = {256, r, 0, 0, 0};
            for (int c = 0; c < n_body; c += 512) {
                bfloat128 x0 = v_bf16_ld_tnsr_b(l0, s_in); l0[0] += 512;
                bfloat128 x1 = v_bf16_ld_tnsr_b(l1, s_in); l1[0] += 512;
                bfloat128 x2 = v_bf16_ld_tnsr_b(l2, s_in); l2[0] += 512;
                bfloat128 x3 = v_bf16_ld_tnsr_b(l3, s_in); l3[0] += 512;
                QWEN_EXPMAX_4(1);
            }
            { bfloat128 x0 = t0, x1 = t1, x2 = t2, x3 = t3; QWEN_EXPMAX_4(0); }
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

    // General fallback: any k_pad >= 256 (k_pad % 256 == 0).
    for (int r = r0; r < r1; ++r) {
        bfloat128 m0 = neg_inf_v, m1 = neg_inf_v;
        for (int c = 0; c < k_pad; c += 256) {
            bfloat128 x0 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c, r, 0, 0, 0}, s_in), valid_keys - c, lane_id, neg_inf_v);
            bfloat128 x1 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c + 128, r, 0, 0, 0}, s_in), valid_keys - (c + 128), lane_id, neg_inf_v);
            m0 = v_bf16_max_b(m0, x0, 0, m0, 1, 0);
            m1 = v_bf16_max_b(m1, x1, 0, m1, 1, 0);
        }
        bfloat128 m_cur = v_bf16_reduce_nolut_max(v_bf16_max_b(m0, m1, 0, m0, 1, 0));
        bfloat128 acc0 = zero_bf, acc1 = zero_bf;
        for (int c = 0; c < k_pad; c += 256) {
            bfloat128 x0 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c, r, 0, 0, 0}, s_in), valid_keys - c, lane_id, neg_inf_v);
            bfloat128 x1 = mask_tail_128(v_bf16_ld_tnsr_b((int5){c + 128, r, 0, 0, 0}, s_in), valid_keys - (c + 128), lane_id, neg_inf_v);
            bfloat128 d0 = v_bf16_sub_b(m_cur, x0, 0, x0, 1, 0);
            bfloat128 d1 = v_bf16_sub_b(m_cur, x1, 0, x1, 1, 0);
            ushort128 u0 = v_convert_bf16_to_u16_b(d0, SW_RHNE, (ushort128){0}, 1, 0);
            ushort128 u1 = v_convert_bf16_to_u16_b(d1, SW_RHNE, (ushort128){0}, 1, 0);
            bfloat128 e0 = (bfloat128)v_u16_sub_b(c_bias_u16, u0, SW_SAT, u0, 1, 0);
            bfloat128 e1 = (bfloat128)v_u16_sub_b(c_bias_u16, u1, SW_SAT, u1, 1, 0);
            acc0 = v_bf16_add_b(acc0, e0, 0, acc0, 1, 0);
            acc1 = v_bf16_add_b(acc1, e1, 0, acc1, 1, 0);
            v_f8_st_tnsr((int5){c, r, 0, 0, 0}, p_out,
                v_convert_bf16_to_f8_all_b((bfloat256){e0, e1}, SW_RHNE | SW_CLIP_FP, zero_f8, 1, 0));
        }
        float128 f01 = v_convert_bf16_to_f32_all_b(v_bf16_add_b(acc0, acc1, 0, acc0, 1, 0), 0, (float128){0}, 1, 0);
        float64 tot = v_f32_reduce_nolut_add(v_f32_add_b(f01.v1, f01.v2, 0, f01.v1, 1, 0));
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, sum_out, tot, 0, 0);
        float64 m_f32 = v_convert_bf16_to_f32_all_b(m_cur, 0, (float128){0}, 1, 0).v1 * max_to_ln;
        v_f32_st_tnsr_partial((int5){r, 0, 0, 0, 0}, max_out, m_f32, 0, 0);
    }
}
