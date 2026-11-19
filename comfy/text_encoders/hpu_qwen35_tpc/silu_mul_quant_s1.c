// Stage 1 of 24-core fused SiLU(gate) * up + FP8 E4M3 quantization for MLP down_proj.
// Uses 1D indexSpace = chunks * rows so Synapse distributes work evenly across all 24 TPC cores.
// cmax is [64, chunks, rows] FP32 (full 256-byte aligned stores, zero false sharing).
static inline float64 silu_f32(float64 x) {
    return (0.5f * x) * (1.0f + tanh_f32(0.5f * x));
}

void main(tensor gate, tensor up, tensor y_out, tensor cmax,
          int dim, int rows, int chunks, int chunk_sz) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total = rows * chunks;

    for (int wid = begin[0]; wid < end[0] && wid < total; ++wid) {
        const int r = wid / chunks;
        const int c = wid - r * chunks;
        const int col0 = c * chunk_sz;
        int col1 = col0 + chunk_sz;
        if (col1 > dim) col1 = dim;

        float64 amax0 = 0.0f, amax1 = 0.0f;
        for (int col = col0; col < col1; col += 128) {
            int5 p = {col, r, 0, 0, 0};
            bfloat128 gb = v_bf16_ld_tnsr_b(p, gate);
            bfloat128 ub = v_bf16_ld_tnsr_b(p, up);
            float128 gf = v_convert_bf16_to_f32_all_b(gb, 0, (float128){0}, 1, 0);
            bfloat128 sg = v_convert_f32_to_bf16_all_b(
                (float128){silu_f32(gf.v1), silu_f32(gf.v2)}, SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 yb = v_bf16_mul_b(sg, ub, 0, sg, 1, 0);
            v_bf16_st_tnsr(p, y_out, yb);
            float128 yf = v_convert_bf16_to_f32_all_b(yb, 0, (float128){0}, 1, 0);
            amax0 = v_f32_max_b(amax0, v_f32_abs_b(yf.v1, 0, yf.v1, 1, 0), 0, amax0, 1, 0);
            amax1 = v_f32_max_b(amax1, v_f32_abs_b(yf.v2, 0, yf.v2, 1, 0), 0, amax1, 1, 0);
        }
        float64 amax = v_f32_reduce_nolut_max(v_f32_max_b(amax0, amax1, 0, amax0, 1, 0));
        v_f32_st_tnsr((int5){0, c, r, 0, 0}, cmax, amax);
    }
}
