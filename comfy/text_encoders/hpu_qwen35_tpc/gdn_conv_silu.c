// Fused deferred conv-state commit + 4-tap depthwise causal tree convolution + SiLU.
// conv_st, new_conv_st: [channels, 3, B] BF16
// prev_proj, cur_proj, conv_out: [channels, rows, B] BF16
// conv_w: [channels, 4] FP32 (tap 0 = current, taps 1..3 = lags 1..3)
// conv_bias: [channels] FP32 (dummy [1] when has_bias==0)
// conv_src: [rows*4] int32, ctl: [17] int32 ([prog[0..7], pending, slots[0..7]])
static inline float64 ld_bf16_linear_f32(int5 pos, tensor t) {
    bfloat128 u = v_bf16_ld_tnsr_b(pos, t, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
    return v_convert_bf16_to_f32_b(u, 0, 0, 1, 0);
}

static inline float64 silu_f32(float64 x) {
    return (0.5f * x) * (1.0f + tanh_f32(0.5f * x));
}

void main(tensor conv_st, tensor prev_proj, tensor cur_proj,
          tensor conv_w, tensor conv_bias, tensor conv_src, tensor ctl,
          tensor new_conv_st, tensor conv_out,
          int channels, int batches, int rows, int has_bias) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();

    int pending = s_i32_ld_g((__global__ int*)gen_addr((int5){8, 0, 0, 0, 0}, ctl));
    int slots[8];
    for (int i = 0; i < pending; ++i) {
        slots[i] = s_i32_ld_g((__global__ int*)gen_addr((int5){9 + i, 0, 0, 0, 0}, ctl));
    }
    int csrc[32];
    for (int i = 0; i < rows * 4; ++i) {
        csrc[i] = s_i32_ld_g((__global__ int*)gen_addr((int5){i, 0, 0, 0, 0}, conv_src));
    }

    for (int b = begin[1]; b < end[1] && b < batches; ++b) {
        for (int cg = begin[0]; cg < end[0]; ++cg) {
            const int ch = cg * 128;
            if (ch >= channels) continue;

            float64 wf0[11], wf1[11];
            bfloat128 old_st[3];
            for (int t = 0; t < 3; ++t) {
                old_st[t] = v_bf16_ld_tnsr_b((int5){ch, t, b, 0, 0}, conv_st);
            }
            for (int t = 0; t < 3; ++t) {
                int idx = pending + t;
                if (idx < 3) {
                    v_bf16_st_tnsr((int5){ch, t, b, 0, 0}, new_conv_st, old_st[idx]);
                    wf0[t] = ld_bf16_linear_f32((int5){ch, idx, b, 0, 0}, conv_st);
                    wf1[t] = ld_bf16_linear_f32((int5){ch + 64, idx, b, 0, 0}, conv_st);
                } else {
                    int sl = slots[idx - 3];
                    bfloat128 pb = v_bf16_ld_tnsr_b((int5){ch, sl, b, 0, 0}, prev_proj);
                    v_bf16_st_tnsr((int5){ch, t, b, 0, 0}, new_conv_st, pb);
                    wf0[t] = ld_bf16_linear_f32((int5){ch, sl, b, 0, 0}, prev_proj);
                    wf1[t] = ld_bf16_linear_f32((int5){ch + 64, sl, b, 0, 0}, prev_proj);
                }
            }
            for (int r = 0; r < rows; ++r) {
                wf0[3 + r] = ld_bf16_linear_f32((int5){ch, r, b, 0, 0}, cur_proj);
                wf1[3 + r] = ld_bf16_linear_f32((int5){ch + 64, r, b, 0, 0}, cur_proj);
            }

            float64 w0[4], w1[4];
            for (int t = 0; t < 4; ++t) {
                w0[t] = v_f32_ld_tnsr_b((int5){ch, t, 0, 0, 0}, conv_w);
                w1[t] = v_f32_ld_tnsr_b((int5){ch + 64, t, 0, 0, 0}, conv_w);
            }
            float64 b0 = has_bias ? v_f32_ld_tnsr_b((int5){ch, 0, 0, 0, 0}, conv_bias) : (float64)0.0f;
            float64 b1 = has_bias ? v_f32_ld_tnsr_b((int5){ch + 64, 0, 0, 0, 0}, conv_bias) : (float64)0.0f;

            for (int r = 0; r < rows; ++r) {
                float64 a0 = b0, a1 = b1;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int src = csrc[r * 4 + t];
                    a0 += wf0[src] * w0[t];
                    a1 += wf1[src] * w1[t];
                }
                bfloat128 y_bf = v_convert_f32_to_bf16_all_b((float128){a0, a1}, SW_RHNE | SW_LINEAR, (bfloat128){0}, 1, 0);
                float128 y = v_convert_bf16_to_f32_all_b(y_bf, 0, (float128){0}, 1, 0);
                bfloat128 out_bf = v_convert_f32_to_bf16_all_b((float128){silu_f32(y.v1), silu_f32(y.v2)}, SW_RHNE, (bfloat128){0}, 1, 0);
                v_bf16_st_tnsr((int5){ch, r, b, 0, 0}, conv_out, out_bf);
            }
        }
    }
}
