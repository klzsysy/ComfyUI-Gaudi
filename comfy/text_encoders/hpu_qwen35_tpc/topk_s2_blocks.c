// Stage 2 of exact block-max Top-K: select top-k block indices per row.
// Supports num_blocks <= 2560 (vocab <= 327,680) inside 16 KiB small VLM.
#define MAX_BLK_VECS 40
__local__ float64 bmax_buf[MAX_BLK_VECS];

static inline int64 bcast_i32_0(int64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    int64 g = v_i32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
    return v_i32_shuffle_b(g, (unsigned char)128, 0, 0);
}

void main(tensor in_bmax, tensor out_blocks, int num_blocks, int rows, int k) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const float neg_inf = -3.4028234e38f;
    const int num_vecs = (num_blocks + 63) >> 6;
    const uint64 lane_u = read_lane_id_4b_b();
    const int64 lane_i = (int64)lane_u;

    for (int r = begin[0]; r < end[0] && r < rows; ++r) {
        float64 cur_max = neg_inf;
        int64 cur_vidx = 0;
        for (int j = 0; j < num_vecs; ++j) {
            const int col = j * 64;
            float64 v = v_f32_ld_tnsr_b((int5){col, r, 0, 0, 0}, in_bmax);
            if (col + 64 > num_blocks) {
                bool64 oob = v_i32_cmp_geq_b(lane_i + col, num_blocks, 0, (bool64){0}, 1, 0);
                v = v_f32_mov_vb(neg_inf, 0, v, oob, 0);
            }
            bmax_buf[j] = v;
            bool64 gt = v_f32_cmp_grt_b(v, cur_max, 0, (bool64){0}, 1, 0);
            cur_max = v_f32_mov_vb(v, 0, cur_max, gt, 0);
            cur_vidx = v_i32_mov_vb(j, 0, cur_vidx, gt, 0);
        }

        for (int t = 0; t < k; ++t) {
            uint64_float64_pair_t am = v_f32_reduce_argmax(cur_max);
            uint64 win_lane = am.v1;
            bool64 is_win = v_u32_cmp_eq_b(lane_u, win_lane, 0, (bool64){0}, 1, 0);
            int64 win_vidx = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(cur_vidx, 0, 0, is_win, 0)));
            int64 blk_id = (win_vidx << 6) + (int64)win_lane;
            v_i32_st_tnsr_partial((int5){t, r, 0, 0, 0}, out_blocks, blk_id, 0, 0);

            if (t + 1 < k) {
                float64 new_max = neg_inf;
                int64 new_vidx = 0;
                for (int j = 0; j < num_vecs; ++j) {
                    float64 vj = bmax_buf[j];
                    bool64 is_hit = v_i32_cmp_eq_vb(win_vidx, j, 0, (bool64){0}, is_win, 0);
                    vj = v_f32_mov_vb(neg_inf, 0, vj, is_hit, 0);
                    bmax_buf[j] = vj;
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
