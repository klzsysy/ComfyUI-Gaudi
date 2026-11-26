// Fused Qwen-Image 2.1 Q/K FP32 RMSNorm + interleaved RoPE + prefix K concat + 512-pad + HND layout.
// 24-core load-balanced across (32 heads * chunks) head-contiguous tiles (chunks=3 -> 96 tiles = 4 exact waves),
// 2-row interleaved (4 parallel rsqrt chains) for full VPU latency hiding, using linear even/odd split
// from v_convert_bf16_to_f32_all_b (v1=even lanes 0,2,...,126; v2=odd lanes 1,3,...,127).
//
// Inputs:
//   q_raw  : [4096, Q] BF16
//   k_raw  : [4096, Q] BF16
//   pk     : [128, 32, P] BF16
//   qn_w   : [128] BF16
//   kn_w   : [128] BF16
//   pe_cs  : [128, Q] FP32 (0..63 = cos64, 64..127 = sin64)
// Outputs:
//   qh_out : [128, Q, 32] BF16
//   kh_out : [128, k_pad, 32] BF16  (k_pad = ((P + Q + 511) / 512) * 512)

static inline float64 rsqrt_nolut_f32(float64 x) {
    float64 half_x = 0.5f * x;
    float64 y = rsqrt_quake_f32(x);
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return y;
}

void main(tensor q_raw, tensor k_raw, tensor pk, tensor qn_w, tensor kn_w, tensor pe_cs,
          tensor qh_out, tensor kh_out,
          int Q, int P, int k_pad, int chunks, float eps, float q_scale, float k_scale) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total_tiles = 32 * chunks;
    const int t0 = begin[0];
    const int t1 = (end[0] < total_tiles) ? end[0] : total_tiles;
    if (t0 >= t1) return;

    const float inv_128 = 1.0f / 128.0f;
    const bfloat128 zero_bf = (bfloat128)0.0f;
    const bfloat128 ksc_bf = (bfloat128)(bfloat)k_scale;
    const shuffle_mask_results_t m32 = gen_shuffle_map_32bit_nolut();
    float128 qwf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0, 0, 0, 0, 0}, qn_w), 0, (float128){0}, 1, 0);
    float128 kwf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){0, 0, 0, 0, 0}, kn_w), 0, (float128){0}, 1, 0);
    float64 qw0 = qwf.v1 * q_scale, qw1 = qwf.v2 * q_scale;
    float64 kw0 = kwf.v1 * k_scale, kw1 = kwf.v2 * k_scale;

    const int q_per_ch = ((Q + chunks - 1) / chunks + 1) & ~1;
    const int p_per_ch = (P + chunks - 1) / chunks;
    const int kv_len = P + Q;
    const int pad_len = k_pad - kv_len;
    const int pad_per_ch = (pad_len + chunks - 1) / chunks;

    for (int tid = t0; tid < t1; ++tid) {
        const int h = tid / chunks;
        const int ch = tid - h * chunks;
        const int col = h * 128;

        // 1. Copy prefix K slice [p0..p1)
        const int p0 = ch * p_per_ch;
        int p1 = p0 + p_per_ch;
        if (p1 > P) p1 = P;
        int p = p0;
        if (k_scale == 1.0f) {
            for (; p + 1 < p1; p += 2) {
                bfloat128 a = v_bf16_ld_tnsr_b((int5){0, h, p, 0, 0}, pk);
                bfloat128 b = v_bf16_ld_tnsr_b((int5){0, h, p + 1, 0, 0}, pk);
                v_bf16_st_tnsr((int5){0, p, h, 0, 0}, kh_out, a);
                v_bf16_st_tnsr((int5){0, p + 1, h, 0, 0}, kh_out, b);
            }
            if (p < p1) {
                v_bf16_st_tnsr((int5){0, p, h, 0, 0}, kh_out, v_bf16_ld_tnsr_b((int5){0, h, p, 0, 0}, pk));
            }
        } else {
            for (; p + 1 < p1; p += 2) {
                bfloat128 a = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, p, 0, 0}, pk), ksc_bf, 0, ksc_bf, 1, 0);
                bfloat128 b = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, p + 1, 0, 0}, pk), ksc_bf, 0, ksc_bf, 1, 0);
                v_bf16_st_tnsr((int5){0, p, h, 0, 0}, kh_out, a);
                v_bf16_st_tnsr((int5){0, p + 1, h, 0, 0}, kh_out, b);
            }
            if (p < p1) {
                bfloat128 a = v_bf16_mul_b(v_bf16_ld_tnsr_b((int5){0, h, p, 0, 0}, pk), ksc_bf, 0, ksc_bf, 1, 0);
                v_bf16_st_tnsr((int5){0, p, h, 0, 0}, kh_out, a);
            }
        }

        // 2. Target Q & K RMSNorm + interleaved RoPE (2 rows = 4 parallel rsqrt chains)
        const int r0 = ch * q_per_ch;
        int r1 = r0 + q_per_ch;
        if (r1 > Q) r1 = Q;
        int r = r0;
        for (; r + 1 < r1; r += 2) {
            float64 cA = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
            float64 sA = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
            float64 cB = v_f32_ld_tnsr_b((int5){0, r + 1, 0, 0, 0}, pe_cs);
            float64 sB = v_f32_ld_tnsr_b((int5){64, r + 1, 0, 0, 0}, pe_cs);

            float128 qA = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, q_raw), 0, (float128){0}, 1, 0);
            float128 qB = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r + 1, 0, 0, 0}, q_raw), 0, (float128){0}, 1, 0);
            float128 kA = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, k_raw), 0, (float128){0}, 1, 0);
            float128 kB = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r + 1, 0, 0, 0}, k_raw), 0, (float128){0}, 1, 0);

            float64 qrA = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(qA.v1 * qA.v1 + qA.v2 * qA.v2, m32) * inv_128 + eps);
            float64 qrB = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(qB.v1 * qB.v1 + qB.v2 * qB.v2, m32) * inv_128 + eps);
            float64 krA = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(kA.v1 * kA.v1 + kA.v2 * kA.v2, m32) * inv_128 + eps);
            float64 krB = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(kB.v1 * kB.v1 + kB.v2 * kB.v2, m32) * inv_128 + eps);

            float64 q0A = qA.v1 * qrA * qw0, q1A = qA.v2 * qrA * qw1;
            float64 q0B = qB.v1 * qrB * qw0, q1B = qB.v2 * qrB * qw1;
            float64 k0A = kA.v1 * krA * kw0, k1A = kA.v2 * krA * kw1;
            float64 k0B = kB.v1 * krB * kw0, k1B = kB.v2 * krB * kw1;

            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){q0A * cA - q1A * sA, q1A * cA + q0A * sA}, SW_RHNE, zero_bf, 1, 0));
            v_bf16_st_tnsr((int5){0, r + 1, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){q0B * cB - q1B * sB, q1B * cB + q0B * sB}, SW_RHNE, zero_bf, 1, 0));
            v_bf16_st_tnsr((int5){0, P + r, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){k0A * cA - k1A * sA, k1A * cA + k0A * sA}, SW_RHNE, zero_bf, 1, 0));
            v_bf16_st_tnsr((int5){0, P + r + 1, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){k0B * cB - k1B * sB, k1B * cB + k0B * sB}, SW_RHNE, zero_bf, 1, 0));
        }
        if (r < r1) {
            float64 cA = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, pe_cs);
            float64 sA = v_f32_ld_tnsr_b((int5){64, r, 0, 0, 0}, pe_cs);
            float128 qA = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, q_raw), 0, (float128){0}, 1, 0);
            float128 kA = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, k_raw), 0, (float128){0}, 1, 0);
            float64 qrA = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(qA.v1 * qA.v1 + qA.v2 * qA.v2, m32) * inv_128 + eps);
            float64 krA = rsqrt_nolut_f32(v_f32_reduce_nolut_noshuffle_add(kA.v1 * kA.v1 + kA.v2 * kA.v2, m32) * inv_128 + eps);
            float64 q0A = qA.v1 * qrA * qw0, q1A = qA.v2 * qrA * qw1;
            float64 k0A = kA.v1 * krA * kw0, k1A = kA.v2 * krA * kw1;
            v_bf16_st_tnsr((int5){0, r, h, 0, 0}, qh_out,
                v_convert_f32_to_bf16_all_b((float128){q0A * cA - q1A * sA, q1A * cA + q0A * sA}, SW_RHNE, zero_bf, 1, 0));
            v_bf16_st_tnsr((int5){0, P + r, h, 0, 0}, kh_out,
                v_convert_f32_to_bf16_all_b((float128){k0A * cA - k1A * sA, k1A * cA + k0A * sA}, SW_RHNE, zero_bf, 1, 0));
        }

        // 3. Zero-pad tail slice [kv_len..k_pad)
        if (pad_len > 0) {
            const int z0 = kv_len + ch * pad_per_ch;
            int z1 = z0 + pad_per_ch;
            if (z1 > k_pad) z1 = k_pad;
            for (int z = z0; z < z1; ++z) {
                v_bf16_st_tnsr((int5){0, z, h, 0, 0}, kh_out, zero_bf);
            }
        }
    }
}
