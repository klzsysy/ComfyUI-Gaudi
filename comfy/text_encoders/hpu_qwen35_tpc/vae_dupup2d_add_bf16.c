// Fused 2D spatial 2x nearest upsample (+ optional channel/row selection & residual add)
// for Wan 2.2 / Qwen-Image 2.1 VAE DupUp3D(T=1, first_chunk=True, factor_s=2) on Gaudi2.
// x:      [W, H, C_in, B] BF16
// main_t: [2*W, 2*H, C_out, B] BF16 (dummy [1, 1, 1, 1] when has_main == 0)
// out:    [2*W, 2*H, C_out, B] BF16 (out-of-place, avoids Synapse 1.24.2 conv-view alias bug)
// mode 0: C_in == C_out         -> row 2h & 2h+1 read x[b, c, h, w]
// mode 1: C_in == 2*C_out, ft=2 -> row 2h & 2h+1 read x[b, 2c+1, h, w]
// mode 2: C_in == 2*C_out, ft=1 -> row 2h reads x[b, 2c, h, w], row 2h+1 reads x[b, 2c+1, h, w]
// mode 3: C_in == 4*C_out, ft=2 -> row 2h reads x[b, 4c+2, h, w], row 2h+1 reads x[b, 4c+3, h, w]

static inline bfloat128 dup_64_bf16(int5 pos, tensor x) {
    bfloat128 u = v_bf16_ld_tnsr_b(pos, x, SW_UNPACK | SW_UNPCK_16_TO_32, (bfloat128){0}, 1, 0);
    uint64 w = (uint64)u;
    uint64 lo = w & 0xFFFFu;
    uint64 dup = lo | (lo << 16);
    return (bfloat128)dup;
}

void main(tensor x, tensor main_t, tensor out,
          int W, int H, int C_out, int B, int mode, int has_main) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total = B * C_out * H;

    for (int wid = begin[0]; wid < end[0] && wid < total; ++wid) {
        const int bc = wid / H;
        const int h = wid - bc * H;
        const int b = bc / C_out;
        const int c = bc - b * C_out;
        const int h0 = h * 2;
        const int h1 = h0 + 1;

        if (mode == 0 || mode == 1) {
            const int c_in = (mode == 0) ? c : (2 * c + 1);
            if (has_main) {
                int w = 0;
                for (; w + 64 < W; w += 128) {
                    const int ow0 = w * 2;
                    const int ow1 = ow0 + 128;
                    bfloat128 da = dup_64_bf16((int5){w, h, c_in, b, 0}, x);
                    bfloat128 db = dup_64_bf16((int5){w + 64, h, c_in, b, 0}, x);
                    bfloat128 m0a = v_bf16_ld_tnsr_b((int5){ow0, h0, c, b, 0}, main_t);
                    bfloat128 m0b = v_bf16_ld_tnsr_b((int5){ow1, h0, c, b, 0}, main_t);
                    bfloat128 m1a = v_bf16_ld_tnsr_b((int5){ow0, h1, c, b, 0}, main_t);
                    bfloat128 m1b = v_bf16_ld_tnsr_b((int5){ow1, h1, c, b, 0}, main_t);
                    v_bf16_st_tnsr((int5){ow0, h0, c, b, 0}, out, v_bf16_add_b(da, m0a, 0, da, 1, 0));
                    v_bf16_st_tnsr((int5){ow1, h0, c, b, 0}, out, v_bf16_add_b(db, m0b, 0, db, 1, 0));
                    v_bf16_st_tnsr((int5){ow0, h1, c, b, 0}, out, v_bf16_add_b(da, m1a, 0, da, 1, 0));
                    v_bf16_st_tnsr((int5){ow1, h1, c, b, 0}, out, v_bf16_add_b(db, m1b, 0, db, 1, 0));
                }
                if (w < W) {
                    const int ow = w * 2;
                    bfloat128 d = dup_64_bf16((int5){w, h, c_in, b, 0}, x);
                    bfloat128 m0 = v_bf16_ld_tnsr_b((int5){ow, h0, c, b, 0}, main_t);
                    bfloat128 m1 = v_bf16_ld_tnsr_b((int5){ow, h1, c, b, 0}, main_t);
                    v_bf16_st_tnsr((int5){ow, h0, c, b, 0}, out, v_bf16_add_b(d, m0, 0, d, 1, 0));
                    v_bf16_st_tnsr((int5){ow, h1, c, b, 0}, out, v_bf16_add_b(d, m1, 0, d, 1, 0));
                }
            } else {
                int w = 0;
                for (; w + 64 < W; w += 128) {
                    const int ow0 = w * 2;
                    const int ow1 = ow0 + 128;
                    bfloat128 da = dup_64_bf16((int5){w, h, c_in, b, 0}, x);
                    bfloat128 db = dup_64_bf16((int5){w + 64, h, c_in, b, 0}, x);
                    v_bf16_st_tnsr((int5){ow0, h0, c, b, 0}, out, da);
                    v_bf16_st_tnsr((int5){ow1, h0, c, b, 0}, out, db);
                    v_bf16_st_tnsr((int5){ow0, h1, c, b, 0}, out, da);
                    v_bf16_st_tnsr((int5){ow1, h1, c, b, 0}, out, db);
                }
                if (w < W) {
                    const int ow = w * 2;
                    bfloat128 d = dup_64_bf16((int5){w, h, c_in, b, 0}, x);
                    v_bf16_st_tnsr((int5){ow, h0, c, b, 0}, out, d);
                    v_bf16_st_tnsr((int5){ow, h1, c, b, 0}, out, d);
                }
            }
        } else {
            const int c_in0 = (mode == 2) ? (2 * c) : (4 * c + 2);
            const int c_in1 = c_in0 + 1;
            if (has_main) {
                int w = 0;
                for (; w + 64 < W; w += 128) {
                    const int ow0 = w * 2;
                    const int ow1 = ow0 + 128;
                    bfloat128 d0a = dup_64_bf16((int5){w, h, c_in0, b, 0}, x);
                    bfloat128 d0b = dup_64_bf16((int5){w + 64, h, c_in0, b, 0}, x);
                    bfloat128 d1a = dup_64_bf16((int5){w, h, c_in1, b, 0}, x);
                    bfloat128 d1b = dup_64_bf16((int5){w + 64, h, c_in1, b, 0}, x);
                    bfloat128 m0a = v_bf16_ld_tnsr_b((int5){ow0, h0, c, b, 0}, main_t);
                    bfloat128 m0b = v_bf16_ld_tnsr_b((int5){ow1, h0, c, b, 0}, main_t);
                    bfloat128 m1a = v_bf16_ld_tnsr_b((int5){ow0, h1, c, b, 0}, main_t);
                    bfloat128 m1b = v_bf16_ld_tnsr_b((int5){ow1, h1, c, b, 0}, main_t);
                    v_bf16_st_tnsr((int5){ow0, h0, c, b, 0}, out, v_bf16_add_b(d0a, m0a, 0, d0a, 1, 0));
                    v_bf16_st_tnsr((int5){ow1, h0, c, b, 0}, out, v_bf16_add_b(d0b, m0b, 0, d0b, 1, 0));
                    v_bf16_st_tnsr((int5){ow0, h1, c, b, 0}, out, v_bf16_add_b(d1a, m1a, 0, d1a, 1, 0));
                    v_bf16_st_tnsr((int5){ow1, h1, c, b, 0}, out, v_bf16_add_b(d1b, m1b, 0, d1b, 1, 0));
                }
                if (w < W) {
                    const int ow = w * 2;
                    bfloat128 d0 = dup_64_bf16((int5){w, h, c_in0, b, 0}, x);
                    bfloat128 d1 = dup_64_bf16((int5){w, h, c_in1, b, 0}, x);
                    bfloat128 m0 = v_bf16_ld_tnsr_b((int5){ow, h0, c, b, 0}, main_t);
                    bfloat128 m1 = v_bf16_ld_tnsr_b((int5){ow, h1, c, b, 0}, main_t);
                    v_bf16_st_tnsr((int5){ow, h0, c, b, 0}, out, v_bf16_add_b(d0, m0, 0, d0, 1, 0));
                    v_bf16_st_tnsr((int5){ow, h1, c, b, 0}, out, v_bf16_add_b(d1, m1, 0, d1, 1, 0));
                }
            } else {
                int w = 0;
                for (; w + 64 < W; w += 128) {
                    const int ow0 = w * 2;
                    const int ow1 = ow0 + 128;
                    bfloat128 d0a = dup_64_bf16((int5){w, h, c_in0, b, 0}, x);
                    bfloat128 d0b = dup_64_bf16((int5){w + 64, h, c_in0, b, 0}, x);
                    bfloat128 d1a = dup_64_bf16((int5){w, h, c_in1, b, 0}, x);
                    bfloat128 d1b = dup_64_bf16((int5){w + 64, h, c_in1, b, 0}, x);
                    v_bf16_st_tnsr((int5){ow0, h0, c, b, 0}, out, d0a);
                    v_bf16_st_tnsr((int5){ow1, h0, c, b, 0}, out, d0b);
                    v_bf16_st_tnsr((int5){ow0, h1, c, b, 0}, out, d1a);
                    v_bf16_st_tnsr((int5){ow1, h1, c, b, 0}, out, d1b);
                }
                if (w < W) {
                    const int ow = w * 2;
                    bfloat128 d0 = dup_64_bf16((int5){w, h, c_in0, b, 0}, x);
                    bfloat128 d1 = dup_64_bf16((int5){w, h, c_in1, b, 0}, x);
                    v_bf16_st_tnsr((int5){ow, h0, c, b, 0}, out, d0);
                    v_bf16_st_tnsr((int5){ow, h1, c, b, 0}, out, d1);
                }
            }
        }
    }
}
