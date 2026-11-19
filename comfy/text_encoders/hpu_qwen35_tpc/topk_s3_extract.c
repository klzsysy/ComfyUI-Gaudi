// Stage 3 of exact block-max Top-K: gather only the k winning 128-element blocks per row
// (2*k <= 48 vectors = 12 KiB VLM, supports k <= 24) and extract exact global top-k (vals, idx).
#define MAX_EXT_VECS 48
__local__ float64 ext_buf[MAX_EXT_VECS];

static inline int64 bcast_i32_0(int64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    int64 g = v_i32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
    return v_i32_shuffle_b(g, (unsigned char)128, 0, 0);
}

void main(tensor logits, tensor pen_mask, tensor anc_toks, tensor top_blocks,
          tensor out_vals, tensor out_idx,
          int vocab, int rows, int k, int anc_w, int has_pen,
          float pres_pen, float rep_pen, float inv_rep_pen) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float neg_inf = -3.4028234e38f;
    const int use_rep = (rep_pen != 1.0f);
    const int use_pres = (pres_pen != 0.0f);
    const uint64 lane_u = read_lane_id_4b_b();
    const int64 lane_i = (int64)lane_u;
    const int num_vecs = k * 2;

    for (int r = begin[0]; r < end[0] && r < rows; ++r) {
        int anc[8];
        if (has_pen) {
            for (int a = 0; a < anc_w && a < 8; ++a) {
                anc[a] = s_i32_ld_g((__global__ int*)gen_addr((int5){a, r, 0, 0, 0}, anc_toks));
            }
        }
        float64 cur_max = neg_inf;
        int64 cur_vidx = 0;
        int64 blk_bases = 0;

        for (int bi = 0; bi < k; ++bi) {
            int blk = s_i32_ld_g((__global__ int*)gen_addr((int5){bi, r, 0, 0, 0}, top_blocks));
            const int col = blk * 128;
            const int j0 = bi * 2;
            const int j1 = j0 + 1;

            blk_bases = v_i32_mov_vb(col, 0, blk_bases, v_i32_cmp_eq_b(lane_i, j0, 0, (bool64){0}, 1, 0), 0);
            blk_bases = v_i32_mov_vb(col + 64, 0, blk_bases, v_i32_cmp_eq_b(lane_i, j1, 0, (bool64){0}, 1, 0), 0);

            bfloat128 u0 = v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, logits, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
            bfloat128 u1 = v_bf16_ld_tnsr_b((int5){col + 64, r, 0, 0, 0}, logits, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
            float64 v0 = v_convert_bf16_to_f32_b(u0, 0, 0, 1, 0);
            float64 v1 = v_convert_bf16_to_f32_b(u1, 0, 0, 1, 0);

            if (has_pen) {
                uint64 pm0 = (uint64)v_u8_ld_tnsr_b((int5){col, 0, 0, 0, 0}, pen_mask, SW_UNPACK | SW_UNPCK_8_TO_32, (uchar256){0}, 1, 0);
                bool64 m0 = v_u32_cmp_neq_b(pm0 & 0xffu, 0u, 0, (bool64){0}, 1, 0);
                uint64 pm1 = (uint64)v_u8_ld_tnsr_b((int5){col + 64, 0, 0, 0, 0}, pen_mask, SW_UNPACK | SW_UNPCK_8_TO_32, (uchar256){0}, 1, 0);
                bool64 m1 = v_u32_cmp_neq_b(pm1 & 0xffu, 0u, 0, (bool64){0}, 1, 0);
                for (int a = 0; a < anc_w && a < 8; ++a) {
                    int d0 = anc[a] - col;
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
            ext_buf[j0] = v0;
            ext_buf[j1] = v1;

            bool64 gt0 = v_f32_cmp_grt_b(v0, cur_max, 0, (bool64){0}, 1, 0);
            cur_max = v_f32_mov_vb(v0, 0, cur_max, gt0, 0);
            cur_vidx = v_i32_mov_vb(j0, 0, cur_vidx, gt0, 0);

            bool64 gt1 = v_f32_cmp_grt_b(v1, cur_max, 0, (bool64){0}, 1, 0);
            cur_max = v_f32_mov_vb(v1, 0, cur_max, gt1, 0);
            cur_vidx = v_i32_mov_vb(j1, 0, cur_vidx, gt1, 0);
        }

        for (int t = 0; t < k; ++t) {
            uint64_float64_pair_t am = v_f32_reduce_argmax(cur_max);
            float64 win_val = am.v2;
            uint64 win_lane = am.v1;
            bool64 is_win = v_u32_cmp_eq_b(lane_u, win_lane, 0, (bool64){0}, 1, 0);
            int64 win_vidx = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(cur_vidx, 0, 0, is_win, 0)));

            bool64 is_v = v_i32_cmp_eq_b(lane_i, win_vidx, 0, (bool64){0}, 1, 0);
            int64 win_base = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(blk_bases, 0, 0, is_v, 0)));
            int64 gidx = win_base + (int64)win_lane;

            v_f32_st_tnsr_partial((int5){t, r, 0, 0, 0}, out_vals, win_val, 0, 0);
            v_i32_st_tnsr_partial((int5){t, r, 0, 0, 0}, out_idx, gidx, 0, 0);

            if (t + 1 < k) {
                float64 new_max = neg_inf;
                int64 new_vidx = 0;
                for (int j = 0; j < num_vecs; ++j) {
                    float64 vj = ext_buf[j];
                    bool64 is_hit = v_i32_cmp_eq_vb(win_vidx, j, 0, (bool64){0}, is_win, 0);
                    vj = v_f32_mov_vb(neg_inf, 0, vj, is_hit, 0);
                    ext_buf[j] = vj;
                    bool64 gt = v_f32_cmp_grt_b(vj, new_max, 0, (bool64){0}, 1, 0);
                    new_max = v_f32_mov_vb(vj, 0, new_max, gt, 0);
                    new_vidx = v_i32_mov_vb(j, 0, new_vidx, gt, 0);
                }
                cur_max = v_f32_mov_vb(new_max, 0, cur_max, is_win, 0);
                cur_vidx = v_i32_mov_vb(new_vidx, 0, cur_vidx, is_win, 0);
            }
        }
    }
}
