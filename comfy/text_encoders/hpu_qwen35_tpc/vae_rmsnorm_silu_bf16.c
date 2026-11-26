// Fused channel-first RMSNorm (+ optional SiLU) for Wan / Qwen-Image 2.1 VAE on Gaudi2.
// x:     [N_full, C, B] BF16 (zero-copy view of contiguous [B, C, 1, H_full, W])
// gamma: [C] BF16
// y:     [n_len, C, B] BF16 (contiguous strip output for [n_offset .. n_offset + n_len))
// Zero VLM, 0 VRF spills, 256 spatial pixels per work item, 1D indexSpace over B * ceil(n_len / 256).

static inline float load_gamma_f32(tensor gamma, int c) {
    unsigned short gu = s_u16_ld_g((__global__ unsigned short*)gen_addr((int5){c, 0, 0, 0, 0}, gamma));
    unsigned int bits = ((unsigned int)gu) << 16;
    return *(float*)&bits;
}

// 0-LUT full-FP32 rsqrt via Quake estimate + 2 Newton-Raphson steps (avoids LUT contention with tanh).
static inline float64 rsqrt_nolut_f32(float64 x) {
    float64 half_x = 0.5f * x;
    float64 y = rsqrt_quake_f32(x);
    y = y * (1.5f - half_x * y * y);
    y = y * (1.5f - half_x * y * y);
    return y;
}

static inline float64 silu_half_f32(float64 half_x) {
    return half_x * (1.0f + v_tanh_fast_f32(half_x));
}

void main(tensor x, tensor gamma, tensor y,
          int C, int B, int n_offset, int n_len, int n_blocks, int do_silu, float scale) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total = B * n_blocks;
    const float eff_scale = do_silu ? (0.5f * scale) : scale;

    for (int wid = begin[0]; wid < end[0] && wid < total; ++wid) {
        const int b = wid / n_blocks;
        const int nb = wid - b * n_blocks;
        const int n0 = nb * 256;
        const int n1 = n0 + 128;
        const int src0 = n_offset + n0;
        const int src1 = n_offset + n1;

        if (n1 < n_len) {
            float64 sq0 = 0.0f, sq1 = 0.0f, sq2 = 0.0f, sq3 = 0.0f;
            for (int c = 0; c < C; ++c) {
                bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                bfloat128 xb1 = v_bf16_ld_tnsr_b((int5){src1, c, b, 0, 0}, x);
                float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                float128 f1 = v_convert_bf16_to_f32_all_b(xb1, 0, (float128){0}, 1, 0);
                sq0 = v_f32_mac_b(f0.v1, f0.v1, sq0, 0, 1, 0);
                sq1 = v_f32_mac_b(f0.v2, f0.v2, sq1, 0, 1, 0);
                sq2 = v_f32_mac_b(f1.v1, f1.v1, sq2, 0, 1, 0);
                sq3 = v_f32_mac_b(f1.v2, f1.v2, sq3, 0, 1, 0);
            }
            float64 inv0 = rsqrt_nolut_f32(v_f32_max_b(sq0, 1e-24f, 0, sq0, 1, 0)) * eff_scale;
            float64 inv1 = rsqrt_nolut_f32(v_f32_max_b(sq1, 1e-24f, 0, sq1, 1, 0)) * eff_scale;
            float64 inv2 = rsqrt_nolut_f32(v_f32_max_b(sq2, 1e-24f, 0, sq2, 1, 0)) * eff_scale;
            float64 inv3 = rsqrt_nolut_f32(v_f32_max_b(sq3, 1e-24f, 0, sq3, 1, 0)) * eff_scale;

            if (do_silu) {
                for (int c = 0; c < C; ++c) {
                    float g = load_gamma_f32(gamma, c);
                    bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                    bfloat128 xb1 = v_bf16_ld_tnsr_b((int5){src1, c, b, 0, 0}, x);
                    float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                    float128 f1 = v_convert_bf16_to_f32_all_b(xb1, 0, (float128){0}, 1, 0);
                    float64 y0 = silu_half_f32((f0.v1 * inv0) * g);
                    float64 y1 = silu_half_f32((f0.v2 * inv1) * g);
                    float64 y2 = silu_half_f32((f1.v1 * inv2) * g);
                    float64 y3 = silu_half_f32((f1.v2 * inv3) * g);
                    bfloat128 ob0 = v_convert_f32_to_bf16_all_b((float128){y0, y1}, SW_RHNE, (bfloat128){0}, 1, 0);
                    bfloat128 ob1 = v_convert_f32_to_bf16_all_b((float128){y2, y3}, SW_RHNE, (bfloat128){0}, 1, 0);
                    v_bf16_st_tnsr((int5){n0, c, b, 0, 0}, y, ob0);
                    v_bf16_st_tnsr((int5){n1, c, b, 0, 0}, y, ob1);
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    float g = load_gamma_f32(gamma, c);
                    bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                    bfloat128 xb1 = v_bf16_ld_tnsr_b((int5){src1, c, b, 0, 0}, x);
                    float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                    float128 f1 = v_convert_bf16_to_f32_all_b(xb1, 0, (float128){0}, 1, 0);
                    bfloat128 ob0 = v_convert_f32_to_bf16_all_b(
                        (float128){(f0.v1 * inv0) * g, (f0.v2 * inv1) * g}, SW_RHNE, (bfloat128){0}, 1, 0);
                    bfloat128 ob1 = v_convert_f32_to_bf16_all_b(
                        (float128){(f1.v1 * inv2) * g, (f1.v2 * inv3) * g}, SW_RHNE, (bfloat128){0}, 1, 0);
                    v_bf16_st_tnsr((int5){n0, c, b, 0, 0}, y, ob0);
                    v_bf16_st_tnsr((int5){n1, c, b, 0, 0}, y, ob1);
                }
            }
        } else {
            float64 sq0 = 0.0f, sq1 = 0.0f;
            for (int c = 0; c < C; ++c) {
                bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                sq0 = v_f32_mac_b(f0.v1, f0.v1, sq0, 0, 1, 0);
                sq1 = v_f32_mac_b(f0.v2, f0.v2, sq1, 0, 1, 0);
            }
            float64 inv0 = rsqrt_nolut_f32(v_f32_max_b(sq0, 1e-24f, 0, sq0, 1, 0)) * eff_scale;
            float64 inv1 = rsqrt_nolut_f32(v_f32_max_b(sq1, 1e-24f, 0, sq1, 1, 0)) * eff_scale;

            if (do_silu) {
                for (int c = 0; c < C; ++c) {
                    float g = load_gamma_f32(gamma, c);
                    bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                    float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                    float64 y0 = silu_half_f32((f0.v1 * inv0) * g);
                    float64 y1 = silu_half_f32((f0.v2 * inv1) * g);
                    bfloat128 ob0 = v_convert_f32_to_bf16_all_b((float128){y0, y1}, SW_RHNE, (bfloat128){0}, 1, 0);
                    v_bf16_st_tnsr((int5){n0, c, b, 0, 0}, y, ob0);
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    float g = load_gamma_f32(gamma, c);
                    bfloat128 xb0 = v_bf16_ld_tnsr_b((int5){src0, c, b, 0, 0}, x);
                    float128 f0 = v_convert_bf16_to_f32_all_b(xb0, 0, (float128){0}, 1, 0);
                    bfloat128 ob0 = v_convert_f32_to_bf16_all_b(
                        (float128){(f0.v1 * inv0) * g, (f0.v2 * inv1) * g}, SW_RHNE, (bfloat128){0}, 1, 0);
                    v_bf16_st_tnsr((int5){n0, c, b, 0, 0}, y, ob0);
                }
            }
        }
    }
}
