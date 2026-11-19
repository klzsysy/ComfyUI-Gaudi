// Stage 1 of exact block-max Top-K: per-128-element block maxima with optional fused
// presence/repetition penalty masking (pen_mask + per-row tree ancestor tokens).
// index_space: [ceil(num_blocks / 64), rows]
// logits: [vocab, rows] BF16, pen_mask: [vocab] uint8, anc_toks: [anc_w, rows] int32
// out_bmax: [num_blocks, rows] FP32
void main(tensor logits, tensor pen_mask, tensor anc_toks, tensor out_bmax,
          int vocab, int rows, int anc_w, int num_blocks, int has_pen,
          float pres_pen, float rep_pen, float inv_rep_pen) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float neg_inf = -3.4028234e38f;
    const int use_rep = (rep_pen != 1.0f);
    const int use_pres = (pres_pen != 0.0f);
    const int64 lane_i = (int64)read_lane_id_4b_b();

    for (int r = begin[1]; r < end[1] && r < rows; ++r) {
        int anc[8];
        if (has_pen) {
            for (int a = 0; a < anc_w && a < 8; ++a) {
                anc[a] = s_i32_ld_g((__global__ int*)gen_addr((int5){a, r, 0, 0, 0}, anc_toks));
            }
        }
        for (int g = begin[0]; g < end[0]; ++g) {
            const int b_base = g * 64;
            const int g_col_lo = b_base * 128;
            const int g_col_hi = g_col_lo + 8192;
            int g_anc[8];
            int n_g_anc = 0;
            if (has_pen) {
                for (int a = 0; a < anc_w && a < 8; ++a) {
                    int tok = anc[a];
                    if (tok >= g_col_lo && tok < g_col_hi) {
                        g_anc[n_g_anc++] = tok;
                    }
                }
            }
            float64 out_vec = neg_inf;
            for (int bi = 0; bi < 64; ++bi) {
                const int b = b_base + bi;
                if (b >= num_blocks) break;
                const int col = b * 128;
                float64 v0, v1;
                if (!has_pen && col + 128 <= vocab) {
                    bfloat128 xb = v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, logits);
                    float128 xf = v_convert_bf16_to_f32_all_b(xb, 0, (float128){0}, 1, 0);
                    v0 = xf.v1;
                    v1 = xf.v2;
                } else {
                    bfloat128 u0 = v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, logits, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
                    bfloat128 u1 = v_bf16_ld_tnsr_b((int5){col + 64, r, 0, 0, 0}, logits, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
                    v0 = v_convert_bf16_to_f32_b(u0, 0, 0, 1, 0);
                    v1 = v_convert_bf16_to_f32_b(u1, 0, 0, 1, 0);
                    if (has_pen) {
                        uint64 pm0 = (uint64)v_u8_ld_tnsr_b((int5){col, 0, 0, 0, 0}, pen_mask, SW_UNPACK | SW_UNPCK_8_TO_32, (uchar256){0}, 1, 0);
                        bool64 m0 = v_u32_cmp_neq_b(pm0 & 0xffu, 0u, 0, (bool64){0}, 1, 0);
                        uint64 pm1 = (uint64)v_u8_ld_tnsr_b((int5){col + 64, 0, 0, 0, 0}, pen_mask, SW_UNPACK | SW_UNPCK_8_TO_32, (uchar256){0}, 1, 0);
                        bool64 m1 = v_u32_cmp_neq_b(pm1 & 0xffu, 0u, 0, (bool64){0}, 1, 0);
                        for (int a = 0; a < n_g_anc; ++a) {
                            int d0 = g_anc[a] - col;
                            if (d0 >= 0 && d0 < 64) {
                                bool64 am0 = v_i32_cmp_eq_b(lane_i, d0, 0, (bool64){0}, 1, 0);
                                m0 = to_bool64(v_i1_or_b(from_bool64(m0), from_bool64(am0), 0, (bool256){0}, 1, 0));
                            } else if (d0 >= 64 && d0 < 128) {
                                bool64 am1 = v_i32_cmp_eq_b(lane_i, d0 - 64, 0, (bool64){0}, 1, 0);
                                m1 = to_bool64(v_i1_or_b(from_bool64(m1), from_bool64(am1), 0, (bool256){0}, 1, 0));
                            }
                        }
                        if (use_rep) {
                            float64 r0 = v_f32_sel_less_f32_b(v0, 0.0f, v0 * rep_pen, v0 * inv_rep_pen, 0, v0, 1, 0);
                            v0 = v_f32_mov_vb(r0, 0, v0, m0, 0);
                            float64 r1 = v_f32_sel_less_f32_b(v1, 0.0f, v1 * rep_pen, v1 * inv_rep_pen, 0, v1, 1, 0);
                            v1 = v_f32_mov_vb(r1, 0, v1, m1, 0);
                        }
                        if (use_pres) {
                            v0 = v_f32_sub_vb(v0, pres_pen, 0, v0, m0, 0);
                            v1 = v_f32_sub_vb(v1, pres_pen, 0, v1, m1, 0);
                        }
                    }
                    if (col + 128 > vocab) {
                        bool64 oob0 = v_i32_cmp_geq_b(lane_i + col, vocab, 0, (bool64){0}, 1, 0);
                        v0 = v_f32_mov_vb(neg_inf, 0, v0, oob0, 0);
                        bool64 oob1 = v_i32_cmp_geq_b(lane_i + (col + 64), vocab, 0, (bool64){0}, 1, 0);
                        v1 = v_f32_mov_vb(neg_inf, 0, v1, oob1, 0);
                    }
                }
                float64 bmax = v_f32_reduce_nolut_max(v_f32_max_b(v0, v1, 0, v0, 1, 0));
                out_vec = v_f32_mov_vb(bmax, 0, out_vec, v_i32_cmp_eq_b(lane_i, bi, 0, (bool64){0}, 1, 0), 0);
            }
            v_f32_st_tnsr((int5){b_base, r, 0, 0, 0}, out_bmax, out_vec);
        }
    }
}
