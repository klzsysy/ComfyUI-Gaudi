// Fused single-core post-TopK sampling (temperature, min_p, top_p, softmax, CDF) + speculative tree accept
// + on-device state/ring update. Replaces ~120 tiny serialized TPC ops (~210 us) in head_accept with ~4 us.
// All index tensors are int32; supports rows <= 8, k <= 24, depth <= 7, topk_k <= 7.
// state_in / state_out: [19] int32 = [prog[0..7], pending/acc_len, slots[0..7], pos, mtp_base]
// meta_out: [4] int32 = [cur, last_tok, old_pos, acc_len]
static inline float64 bcast_f32_0(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    float64 g = v_f32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
    return v_f32_shuffle_b(g, (unsigned char)128, 0, 0);
}

static inline int64 bcast_i32_0(int64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    int64 g = v_i32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
    return v_i32_shuffle_b(g, (unsigned char)128, 0, 0);
}

static inline int64 pick_lane_i32(int64 vec, int64 lane_i, int64 idx_bcast) {
    bool64 m = v_i32_cmp_eq_b(lane_i, idx_bcast, 0, (bool64){0}, 1, 0);
    return bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(vec, 0, 0, m, 0)));
}

void main(tensor top_vals, tensor top_idx, tensor verify_tokens, tensor tops,
          tensor rand_u, tensor state_in, tensor tree_parent,
          tensor state_out, tensor commit_toks_out, tensor safe_commit_out,
          tensor meta_out, tensor ranks_out,
          int rows, int k, int depth, int topk_k, int do_sample, int adaptive,
          float inv_temp, float top_p, float min_p) {
    const int5 begin = get_index_space_offset();
    if (begin[0] != 0) return;

    const int64 lane_i = (int64)read_lane_id_4b_b();
    const bool64 valid_k = v_i32_cmp_less_b(lane_i, k, 0, (bool64){0}, 1, 0);
    const bool64 lane_eq_0 = v_i32_cmp_eq_b(lane_i, 0, 0, (bool64){0}, 1, 0);

    // 1. Sample 1 token per tree row from descending top-k logits (or pick top_idx[r, 0] if greedy).
    int64 sampled_vec = 0;
    for (int r = 0; r < rows; ++r) {
        int64 tok_bcast;
        if (!do_sample || k <= 1) {
            int t0 = s_i32_ld_g((__global__ int*)gen_addr((int5){0, r, 0, 0, 0}, top_idx));
            tok_bcast = (int64)t0;
        } else {
            float64 v_vec = v_f32_ld_tnsr_b((int5){0, r, 0, 0, 0}, top_vals) * inv_temp;
            int64 idx_vec = v_i32_ld_tnsr_b((int5){0, r, 0, 0, 0}, top_idx);
            float64 v0 = bcast_f32_0(v_vec);
            bool64 valid = valid_k;
            float64 e_vec = v_f32_mov_vb(exp_f32(v_vec - v0), 0, 0.0f, valid, 0);
            float64 p_vec = e_vec * ((float64)1.0f / v_f32_reduce_nolut_add(e_vec));

            if (min_p > 0.0f) {
                float64 p0 = bcast_f32_0(p_vec);
                bool64 keep_m = v_f32_cmp_geq_b(p_vec, p0 * min_p, 0, (bool64){0}, 1, 0);
                valid = to_bool64(v_i1_and_b(from_bool64(valid), from_bool64(keep_m), 0, (bool256){0}, 1, 0));
                e_vec = v_f32_mov_vb(e_vec, 0, 0.0f, valid, 0);
                p_vec = e_vec * ((float64)1.0f / v_f32_reduce_nolut_add(e_vec));
            }
            if (top_p < 1.0f) {
                float64 prev_cum = 0.0f;
                for (int j = 1; j < k; ++j) {
                    bool64 lt_j = v_i32_cmp_less_b(lane_i, j, 0, (bool64){0}, 1, 0);
                    float64 s = v_f32_reduce_nolut_add(v_f32_mov_vb(p_vec, 0, 0.0f, lt_j, 0));
                    bool64 eq_j = v_i32_cmp_eq_b(lane_i, j, 0, (bool64){0}, 1, 0);
                    prev_cum = v_f32_mov_vb(s, 0, prev_cum, eq_j, 0);
                }
                bool64 keep_p = v_f32_cmp_less_b(prev_cum, top_p, 0, (bool64){0}, 1, 0);
                keep_p = to_bool64(v_i1_or_b(from_bool64(keep_p), from_bool64(lane_eq_0), 0, (bool256){0}, 1, 0));
                valid = to_bool64(v_i1_and_b(from_bool64(valid), from_bool64(keep_p), 0, (bool256){0}, 1, 0));
                e_vec = v_f32_mov_vb(e_vec, 0, 0.0f, valid, 0);
                p_vec = e_vec * ((float64)1.0f / v_f32_reduce_nolut_add(e_vec));
            }

            float u_s = s_f32_ld_g((__global__ float*)gen_addr((int5){0, r, 0, 0, 0}, rand_u));
            float64 cdf = (float64)2.0f;
            for (int j = 0; j < k; ++j) {
                bool64 le_j = v_i32_cmp_leq_b(lane_i, j, 0, (bool64){0}, 1, 0);
                float64 s = v_f32_reduce_nolut_add(v_f32_mov_vb(p_vec, 0, 0.0f, le_j, 0));
                bool64 eq_j = v_i32_cmp_eq_b(lane_i, j, 0, (bool64){0}, 1, 0);
                cdf = v_f32_mov_vb(s, 0, cdf, eq_j, 0);
            }
            bool64 le_u = v_f32_cmp_leq_b(cdf, (float64)u_s, 0, (bool64){0}, 1, 0);
            int64 cnt = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(1, 0, 0, le_u, 0)));
            cnt = v_i32_min_b(cnt, k - 1, 0, cnt, 1, 0);
            tok_bcast = pick_lane_i32(idx_vec, lane_i, cnt);
        }
        bool64 eq_r = v_i32_cmp_eq_b(lane_i, r, 0, (bool64){0}, 1, 0);
        sampled_vec = v_i32_mov_vb(tok_bcast, 0, sampled_vec, eq_r, 0);
    }

    // 2. Speculative tree walk (pure branchless vector over lanes 0..rows-1).
    int64 draft_vec = v_i32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, verify_tokens);
    int64 parent_vec = v_i32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, tree_parent);
    bool64 child_domain = to_bool64(v_i1_and_b(
        from_bool64(v_i32_cmp_grt_b(lane_i, 0, 0, (bool64){0}, 1, 0)),
        from_bool64(v_i32_cmp_less_b(lane_i, rows, 0, (bool64){0}, 1, 0)),
        0, (bool256){0}, 1, 0));

    int64 cur = 0;
    int64 acc_len = 1;
    int64 path_slots = lane_i;
    int64 commit_toks = 0;
    bool64 active = v_i32_cmp_eq_b(lane_i, lane_i, 0, (bool64){0}, 1, 0);

    for (int h = 0; h < depth; ++h) {
        int64 cur_tok = pick_lane_i32(sampled_vec, lane_i, cur);
        bool64 m_par = v_i32_cmp_eq_b(parent_vec, cur, 0, (bool64){0}, 1, 0);
        bool64 m_tok = v_i32_cmp_eq_b(draft_vec, cur_tok, 0, (bool64){0}, 1, 0);
        bool64 m_c = to_bool64(v_i1_and_b(from_bool64(m_par), from_bool64(m_tok), 0, (bool256){0}, 1, 0));
        m_c = to_bool64(v_i1_and_b(from_bool64(m_c), from_bool64(child_domain), 0, (bool256){0}, 1, 0));
        m_c = to_bool64(v_i1_and_b(from_bool64(m_c), from_bool64(active), 0, (bool256){0}, 1, 0));

        int64 next_node = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(lane_i, 0, 0, m_c, 0)));
        bool64 has_match = v_i32_cmp_grt_b(next_node, 0, 0, (bool64){0}, 1, 0);

        cur = v_i32_mov_vb(next_node, 0, cur, has_match, 0);
        acc_len = v_i32_add_vb(acc_len, 1, 0, acc_len, has_match, 0);

        bool64 set_p = to_bool64(v_i1_and_b(
            from_bool64(v_i32_cmp_eq_b(lane_i, h + 1, 0, (bool64){0}, 1, 0)),
            from_bool64(has_match), 0, (bool256){0}, 1, 0));
        path_slots = v_i32_mov_vb(next_node, 0, path_slots, set_p, 0);

        bool64 set_c = to_bool64(v_i1_and_b(
            from_bool64(v_i32_cmp_eq_b(lane_i, h, 0, (bool64){0}, 1, 0)),
            from_bool64(has_match), 0, (bool256){0}, 1, 0));
        commit_toks = v_i32_mov_vb(cur_tok, 0, commit_toks, set_c, 0);

        active = has_match;
    }

    int64 last_tok = pick_lane_i32(sampled_vec, lane_i, cur);
    bool64 eq_last = v_i32_cmp_eq_b(lane_i, acc_len - 1, 0, (bool64){0}, 1, 0);
    commit_toks = v_i32_mov_vb(last_tok, 0, commit_toks, eq_last, 0);
    bool64 lt_acc = v_i32_cmp_less_b(lane_i, acc_len, 0, (bool64){0}, 1, 0);
    int64 safe_commit = v_i32_mov_vb(commit_toks, 0, last_tok, lt_acc, 0);

    // 3. Build updated state_out [19] = [prog[0..7], acc_len, path_slots[0..7], new_pos, new_mtp_base].
    int64 st_in = v_i32_ld_tnsr_b((int5){0, 0, 0, 0, 0}, state_in);
    int64 old_pos = pick_lane_i32(st_in, lane_i, (int64)17);
    int64 old_mtp = pick_lane_i32(st_in, lane_i, (int64)18);
    int64 new_pos = old_pos + acc_len;
    bool64 on_chain = v_i32_cmp_eq_b(cur, acc_len - 1, 0, (bool64){0}, 1, 0);
    int64 kept = v_i32_mov_vb(acc_len, 0, acc_len - 1, on_chain, 0);
    kept = v_i32_min_b(kept, depth, 0, kept, 1, 0);
    int64 new_mtp = old_mtp + kept;

    int64 st_out = st_in;
    st_out = v_i32_mov_vb(acc_len, 0, st_out, v_i32_cmp_eq_b(lane_i, 8, 0, (bool64){0}, 1, 0), 0);
    for (int i = 0; i < 8; ++i) {
        int64 pi = pick_lane_i32(path_slots, lane_i, (int64)i);
        st_out = v_i32_mov_vb(pi, 0, st_out, v_i32_cmp_eq_b(lane_i, 9 + i, 0, (bool64){0}, 1, 0), 0);
    }
    st_out = v_i32_mov_vb(new_pos, 0, st_out, v_i32_cmp_eq_b(lane_i, 17, 0, (bool64){0}, 1, 0), 0);
    st_out = v_i32_mov_vb(new_mtp, 0, st_out, v_i32_cmp_eq_b(lane_i, 18, 0, (bool64){0}, 1, 0), 0);

    // 4. Build meta_out [4] = [cur, last_tok, old_pos, acc_len].
    int64 meta = cur;
    meta = v_i32_mov_vb(last_tok, 0, meta, v_i32_cmp_eq_b(lane_i, 1, 0, (bool64){0}, 1, 0), 0);
    meta = v_i32_mov_vb(old_pos, 0, meta, v_i32_cmp_eq_b(lane_i, 2, 0, (bool64){0}, 1, 0), 0);
    meta = v_i32_mov_vb(acc_len, 0, meta, v_i32_cmp_eq_b(lane_i, 3, 0, (bool64){0}, 1, 0), 0);

    // 5. Adaptive draft ranks [7] for nodes 0..depth-1.
    int64 ranks = (int64)7;
    if (adaptive) {
        bool64 lt_tk = v_i32_cmp_less_b(lane_i, topk_k, 0, (bool64){0}, 1, 0);
        for (int d = 0; d < depth; ++d) {
            int64 cand = v_i32_ld_tnsr_b((int5){0, 0, d, 0, 0}, tops);
            int64 tgt = pick_lane_i32(sampled_vec, lane_i, (int64)d);
            bool64 hit = to_bool64(v_i1_and_b(
                from_bool64(v_i32_cmp_eq_b(cand, tgt, 0, (bool64){0}, 1, 0)),
                from_bool64(lt_tk), 0, (bool256){0}, 1, 0));
            int64 cnt_hit = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(1, 0, 0, hit, 0)));
            int64 r_hit = bcast_i32_0(v_i32_reduce_nolut_add(v_i32_mov_vb(lane_i, 0, 0, hit, 0)));
            bool64 any_hit = v_i32_cmp_grt_b(cnt_hit, 0, 0, (bool64){0}, 1, 0);
            int64 r_val = v_i32_mov_vb(r_hit, 0, 7, any_hit, 0);
            ranks = v_i32_mov_vb(r_val, 0, ranks, v_i32_cmp_eq_b(lane_i, d, 0, (bool64){0}, 1, 0), 0);
        }
    }

    // Store all 5 tiny output vectors cleanly (hardware tensor bounds clip each store to its exact size).
    v_i32_st_tnsr((int5){0, 0, 0, 0, 0}, state_out, st_out);
    v_i32_st_tnsr((int5){0, 0, 0, 0, 0}, commit_toks_out, commit_toks);
    v_i32_st_tnsr((int5){0, 0, 0, 0, 0}, safe_commit_out, safe_commit);
    v_i32_st_tnsr((int5){0, 0, 0, 0, 0}, meta_out, meta);
    v_i32_st_tnsr((int5){0, 0, 0, 0, 0}, ranks_out, ranks);
}
