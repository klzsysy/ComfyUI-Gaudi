// Fused FP32-state deferred-commit + speculative-tree Gated DeltaNet recurrence (4x16 packed state).
// State layout: [64, 32, vh * v_blks, B] FP32 (PyTorch [B, vh * v_blks, 32, 64]) where v_blks = v_hd / 16.
// For (b, h, vb), wid = h * v_blks + vb, kg = khalf * 16 + j, lane = grp * 16 + vl:
//   S[kg][lane] = state[b, h, k = khalf * 64 + grp * 16 + j, v = vb * 16 + vl].
// This eliminates all horizontal v_f32_reduce_x2_add calls from the inner loop via native 1-cycle
// 16-lane group shuffles v_f32_shuffle_b(k, 128 + j) + 4-instruction group reduction.
// Also fuses softplus + exp + sigmoid gate computation from raw merged in_proj_ab [2*vh, rows, B] BF16.
// Local VLM footprint: 32 + 8 = 40 vectors = 10 KiB (well below 16 KiB limit).
__local__ float64 S_vlm[32];
__local__ float64 out_acc[8];

static inline float64 first_group(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    return v_f32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
}

static inline float64 bcast_lane0(float64 x) {
    return v_f32_shuffle_b(first_group(x), (unsigned char)128, 0, 0);
}

static inline float64 sum_4_groups(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    float64 s1 = x + v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, writes, 0);
    return s1 + v_f32_mov_dual_group_all_b(s1, 0xffffffff, 2, 2, 0, 0, writes, 0);
}

static inline float64 ld_bf16_linear_f32(int5 pos, tensor t) {
    bfloat128 u = v_bf16_ld_tnsr_b(pos, t, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
    return v_convert_bf16_to_f32_b(u, 0, 0, 1, 0);
}

void main(tensor state, tensor prev_conv, tensor cur_conv,
          tensor prev_ab, tensor cur_ab, tensor g_decay, tensor dt_bias, tensor ctl,
          tensor committed, tensor out,
          int v_hd, int vh, int kh, int batches, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int groups = vh / kh;
    const int key_dim = kh * 128;
    const int halves = v_hd >> 6;
    const int v_blks = v_hd >> 4;
    const float q_scale = 0.08838834764831845f; // 128 ** -0.5

    int instructions[8];
    int slots[8];
    for (int i = 0; i < rows; ++i) {
        instructions[i] = s_i32_ld_g((__global__ int*)gen_addr((int5){i, 0, 0, 0, 0}, ctl));
    }
    int pending = s_i32_ld_g((__global__ int*)gen_addr((int5){8, 0, 0, 0, 0}, ctl));
    for (int i = 0; i < pending; ++i) {
        slots[i] = s_i32_ld_g((__global__ int*)gen_addr((int5){9 + i, 0, 0, 0, 0}, ctl));
    }

    const int64 lane_i = (int64)read_lane_id_4b_b();
    const int64 grp_i = lane_i >> 4;

    for (int b = begin[1]; b < end[1] && b < batches; ++b) {
        int cached_h = -1;
        float64 p_dec_v = 1.0f, p_bet_v = 0.0f, p_invk_v = 1.0f;
        float64 c_dec_v = 1.0f, c_bet_v = 0.0f, c_invk_v = 1.0f, c_invq_v = 1.0f, qk_vec = 0.0f;
        int q_off = 0, k_off = 0, v_off = 0, out_h_off = 0;

        for (int wid = begin[0]; wid < end[0]; ++wid) {
            const int h = wid / halves;
            const int hf = wid - h * halves;
            if (h >= vh) break;
            const int v_base = hf * 64;

            if (h != cached_h) {
                cached_h = h;
                const int kh_i = h / groups;
                q_off = kh_i * 128;
                k_off = key_dim + kh_i * 128;
                v_off = 2 * key_dim + h * v_hd;
                out_h_off = h * v_hd;

                const float gd_s = s_f32_ld_g((__global__ float*)gen_addr((int5){h, 0, 0, 0, 0}, g_decay));
                const float dt_s = s_f32_ld_g((__global__ float*)gen_addr((int5){h, 0, 0, 0, 0}, dt_bias));

                float64 p_a = 0.0f, p_b = 0.0f;
                p_invk_v = 1.0f;
                for (int i = 0; i < pending; ++i) {
                    int sl = slots[i];
                    bool64 mi = v_i32_cmp_eq_b(lane_i, i, 0, (bool64){0}, 1, 0);
                    float64 av = bcast_lane0(ld_bf16_linear_f32((int5){h, sl, b, 0, 0}, prev_ab));
                    float64 bv = bcast_lane0(ld_bf16_linear_f32((int5){vh + h, sl, b, 0, 0}, prev_ab));
                    p_a = v_f32_mov_vb(av, 0, p_a, mi, 0);
                    p_b = v_f32_mov_vb(bv, 0, p_b, mi, 0);

                    float64 kf0 = ld_bf16_linear_f32((int5){k_off, sl, b, 0, 0}, prev_conv);
                    float64 kf1 = ld_bf16_linear_f32((int5){k_off + 64, sl, b, 0, 0}, prev_conv);
                    float64 ss = v_f32_reduce_nolut_add(kf0 * kf0 + kf1 * kf1);
                    float64 invn = rsqrt_f32(v_f32_max_b(ss, 1e-24f, 0, ss, 1, 0));
                    p_invk_v = v_f32_mov_vb(invn, 0, p_invk_v, mi, 0);
                }
                if (pending > 0) {
                    float64 px = p_a + dt_s;
                    float64 psp = v_f32_sel_grt_f32_b(px, 20.0f, px, log_f32(1.0f + exp_f32(px)), 0, px, 1, 0);
                    p_dec_v = first_group(exp_f32(gd_s * psp));
                    p_bet_v = first_group(0.5f * (1.0f + tanh_f32(0.5f * p_b)));
                    p_invk_v = first_group(p_invk_v);
                }

                float64 c_a = 0.0f, c_b = 0.0f;
                c_invq_v = 1.0f;
                c_invk_v = 1.0f;
                qk_vec = 0.0f;
                for (int r = 0; r < rows; ++r) {
                    bool64 mr = v_i32_cmp_eq_b(lane_i, r, 0, (bool64){0}, 1, 0);
                    float64 av = bcast_lane0(ld_bf16_linear_f32((int5){h, r, b, 0, 0}, cur_ab));
                    float64 bv = bcast_lane0(ld_bf16_linear_f32((int5){vh + h, r, b, 0, 0}, cur_ab));
                    c_a = v_f32_mov_vb(av, 0, c_a, mr, 0);
                    c_b = v_f32_mov_vb(bv, 0, c_b, mr, 0);

                    float64 qf0 = ld_bf16_linear_f32((int5){q_off, r, b, 0, 0}, cur_conv);
                    float64 qf1 = ld_bf16_linear_f32((int5){q_off + 64, r, b, 0, 0}, cur_conv);
                    float64 kf0 = ld_bf16_linear_f32((int5){k_off, r, b, 0, 0}, cur_conv);
                    float64 kf1 = ld_bf16_linear_f32((int5){k_off + 64, r, b, 0, 0}, cur_conv);
                    float128 ssqk = v_f32_reduce_x2_add(qf0 * qf0 + qf1 * qf1, kf0 * kf0 + kf1 * kf1);
                    float64 invq = rsqrt_f32(v_f32_max_b(ssqk.v1, 1e-24f, 0, ssqk.v1, 1, 0)) * q_scale;
                    float64 invk = rsqrt_f32(v_f32_max_b(ssqk.v2, 1e-24f, 0, ssqk.v2, 1, 0));
                    c_invq_v = v_f32_mov_vb(invq, 0, c_invq_v, mr, 0);
                    c_invk_v = v_f32_mov_vb(invk, 0, c_invk_v, mr, 0);

                    float64 qkd = v_f32_reduce_nolut_add((qf0 * kf0 + qf1 * kf1) * (invq * invk));
                    qk_vec = v_f32_mov_vb(qkd, 0, qk_vec, mr, 0);
                }
                float64 cx = c_a + dt_s;
                float64 csp = v_f32_sel_grt_f32_b(cx, 20.0f, cx, log_f32(1.0f + exp_f32(cx)), 0, cx, 1, 0);
                c_dec_v = first_group(exp_f32(gd_s * csp));
                c_bet_v = first_group(0.5f * (1.0f + tanh_f32(0.5f * c_b)));
                c_invq_v = first_group(c_invq_v);
                c_invk_v = first_group(c_invk_v);
                qk_vec = first_group(qk_vec);
            }

            for (int r = 0; r < rows; ++r) out_acc[r] = 0.0f;

            for (int sub = 0; sub < 4; ++sub) {
                const int vb = hf * 4 + sub;
                const int st_idx = h * v_blks + vb;
                const int vcol = v_base + sub * 16;
                const bool64 sub_mask = v_i32_cmp_eq_b(grp_i, sub, 0, (bool64){0}, 1, 0);

                // Load 32 packed 4x16 state vectors (8 KiB) into VLM.
                for (int kg = 0; kg < 32; ++kg) {
                    S_vlm[kg] = v_f32_ld_tnsr_b((int5){0, kg, st_idx, b, 0}, state);
                }

                // Deferred commit of accepted tokens from previous verify step.
                for (int i = 0; i < pending; ++i) {
                    int sl = slots[i];
                    float64 d = v_f32_shuffle_b(p_dec_v, (unsigned char)(128 + i), 0, 0);
                    float64 bt = v_f32_shuffle_b(p_bet_v, (unsigned char)(128 + i), 0, 0);
                    float64 invk = v_f32_shuffle_b(p_invk_v, (unsigned char)(128 + i), 0, 0);
                    float64 k0l = ld_bf16_linear_f32((int5){k_off, sl, b, 0, 0}, prev_conv) * invk;
                    float64 k0h = ld_bf16_linear_f32((int5){k_off + 8, sl, b, 0, 0}, prev_conv) * invk;
                    float64 k1l = ld_bf16_linear_f32((int5){k_off + 64, sl, b, 0, 0}, prev_conv) * invk;
                    float64 k1h = ld_bf16_linear_f32((int5){k_off + 72, sl, b, 0, 0}, prev_conv) * invk;
                    float64 v16 = first_group(ld_bf16_linear_f32((int5){v_off + vcol, sl, b, 0, 0}, prev_conv));

                    float64 p_acc = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < 8; ++j) {
                        unsigned char sh = (unsigned char)(128 + j);
                        p_acc = v_f32_mac_b(S_vlm[j],      v_f32_shuffle_b(k0l, sh, 0, 0), p_acc, 0, 1, 0);
                        p_acc = v_f32_mac_b(S_vlm[8 + j],  v_f32_shuffle_b(k0h, sh, 0, 0), p_acc, 0, 1, 0);
                        p_acc = v_f32_mac_b(S_vlm[16 + j], v_f32_shuffle_b(k1l, sh, 0, 0), p_acc, 0, 1, 0);
                        p_acc = v_f32_mac_b(S_vlm[24 + j], v_f32_shuffle_b(k1h, sh, 0, 0), p_acc, 0, 1, 0);
                    }
                    float64 pred = sum_4_groups(p_acc);
                    float64 upd = (v16 - pred * d) * bt;

                    #pragma unroll 4
                    for (int j = 0; j < 8; ++j) {
                        unsigned char sh = (unsigned char)(128 + j);
                        S_vlm[j]      = v_f32_mac_b(v_f32_shuffle_b(k0l, sh, 0, 0), upd, S_vlm[j] * d, 0, 1, 0);
                        S_vlm[8 + j]  = v_f32_mac_b(v_f32_shuffle_b(k0h, sh, 0, 0), upd, S_vlm[8 + j] * d, 0, 1, 0);
                        S_vlm[16 + j] = v_f32_mac_b(v_f32_shuffle_b(k1l, sh, 0, 0), upd, S_vlm[16 + j] * d, 0, 1, 0);
                        S_vlm[24 + j] = v_f32_mac_b(v_f32_shuffle_b(k1h, sh, 0, 0), upd, S_vlm[24 + j] * d, 0, 1, 0);
                    }
                }

                // Store committed state back to HBM.
                for (int kg = 0; kg < 32; ++kg) {
                    v_f32_st_tnsr((int5){0, kg, st_idx, b, 0}, committed, S_vlm[kg]);
                }

                // Speculative tree evaluation.
                for (int order = 0; order < rows; ++order) {
                    int instr = instructions[order];
                    int r = instr & 15;
                    float64 d = v_f32_shuffle_b(c_dec_v, (unsigned char)(128 + r), 0, 0);
                    float64 bt = v_f32_shuffle_b(c_bet_v, (unsigned char)(128 + r), 0, 0);
                    float64 invk = v_f32_shuffle_b(c_invk_v, (unsigned char)(128 + r), 0, 0);
                    float64 invq = v_f32_shuffle_b(c_invq_v, (unsigned char)(128 + r), 0, 0);
                    float64 qkd = v_f32_shuffle_b(qk_vec, (unsigned char)(128 + r), 0, 0);

                    float64 k0l = ld_bf16_linear_f32((int5){k_off, r, b, 0, 0}, cur_conv) * invk;
                    float64 k0h = ld_bf16_linear_f32((int5){k_off + 8, r, b, 0, 0}, cur_conv) * invk;
                    float64 k1l = ld_bf16_linear_f32((int5){k_off + 64, r, b, 0, 0}, cur_conv) * invk;
                    float64 k1h = ld_bf16_linear_f32((int5){k_off + 72, r, b, 0, 0}, cur_conv) * invk;
                    float64 q0l = ld_bf16_linear_f32((int5){q_off, r, b, 0, 0}, cur_conv) * invq;
                    float64 q0h = ld_bf16_linear_f32((int5){q_off + 8, r, b, 0, 0}, cur_conv) * invq;
                    float64 q1l = ld_bf16_linear_f32((int5){q_off + 64, r, b, 0, 0}, cur_conv) * invq;
                    float64 q1h = ld_bf16_linear_f32((int5){q_off + 72, r, b, 0, 0}, cur_conv) * invq;
                    float64 v16 = first_group(ld_bf16_linear_f32((int5){v_off + vcol, r, b, 0, 0}, cur_conv));

                    float64 p_acc = 0.0f, o_acc = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < 8; ++j) {
                        unsigned char sh = (unsigned char)(128 + j);
                        float64 s0 = S_vlm[j];
                        p_acc = v_f32_mac_b(s0, v_f32_shuffle_b(k0l, sh, 0, 0), p_acc, 0, 1, 0);
                        o_acc = v_f32_mac_b(s0, v_f32_shuffle_b(q0l, sh, 0, 0), o_acc, 0, 1, 0);
                        float64 s1 = S_vlm[8 + j];
                        p_acc = v_f32_mac_b(s1, v_f32_shuffle_b(k0h, sh, 0, 0), p_acc, 0, 1, 0);
                        o_acc = v_f32_mac_b(s1, v_f32_shuffle_b(q0h, sh, 0, 0), o_acc, 0, 1, 0);
                        float64 s2 = S_vlm[16 + j];
                        p_acc = v_f32_mac_b(s2, v_f32_shuffle_b(k1l, sh, 0, 0), p_acc, 0, 1, 0);
                        o_acc = v_f32_mac_b(s2, v_f32_shuffle_b(q1l, sh, 0, 0), o_acc, 0, 1, 0);
                        float64 s3 = S_vlm[24 + j];
                        p_acc = v_f32_mac_b(s3, v_f32_shuffle_b(k1h, sh, 0, 0), p_acc, 0, 1, 0);
                        o_acc = v_f32_mac_b(s3, v_f32_shuffle_b(q1h, sh, 0, 0), o_acc, 0, 1, 0);
                    }
                    float64 pred = sum_4_groups(p_acc);
                    float64 out_s = sum_4_groups(o_acc);
                    float64 upd = (v16 - pred * d) * bt;
                    float64 out16 = out_s * d + qkd * upd;

                    out_acc[r] = v_f32_mov_vb(out16, 0, out_acc[r], sub_mask, 0);

                    if (instr & 32) {
                        #pragma unroll 4
                        for (int j = 0; j < 8; ++j) {
                            unsigned char sh = (unsigned char)(128 + j);
                            S_vlm[j]      = v_f32_mac_b(v_f32_shuffle_b(k0l, sh, 0, 0), upd, S_vlm[j] * d, 0, 1, 0);
                            S_vlm[8 + j]  = v_f32_mac_b(v_f32_shuffle_b(k0h, sh, 0, 0), upd, S_vlm[8 + j] * d, 0, 1, 0);
                            S_vlm[16 + j] = v_f32_mac_b(v_f32_shuffle_b(k1l, sh, 0, 0), upd, S_vlm[16 + j] * d, 0, 1, 0);
                            S_vlm[24 + j] = v_f32_mac_b(v_f32_shuffle_b(k1h, sh, 0, 0), upd, S_vlm[24 + j] * d, 0, 1, 0);
                        }
                    }
                }
            }

            for (int r = 0; r < rows; ++r) {
                v_f32_st_tnsr((int5){out_h_off + v_base, r, b, 0, 0}, out, out_acc[r]);
            }
        }
    }
}
