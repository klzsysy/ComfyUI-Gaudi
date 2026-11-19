// Stage 2 of 24-core fused SiLU(gate) * up + FP8 E4M3 quantization for MLP down_proj.
// Uses 1D indexSpace = chunks * rows so Synapse distributes work evenly across all 24 TPC cores.
void main(tensor y_in, tensor cmax, tensor out_q, tensor out_scale,
          int dim, int rows, int chunks, int chunk_sz) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int total = rows * chunks;
    const float inv_240 = 1.0f / 240.0f;

    for (int wid = begin[0]; wid < end[0] && wid < total; ++wid) {
        const int r = wid / chunks;
        const int c = wid - r * chunks;

        float row_max = 1e-12f;
        for (int cc = 0; cc < chunks; ++cc) {
            float v = s_f32_ld_g((__global__ float*)gen_addr((int5){0, cc, r, 0, 0}, cmax));
            if (v > row_max) row_max = v;
        }
        const float scale_s = row_max * inv_240;
        const float64 inv_s = (float64)(1.0f / scale_s);

        if (c == 0) {
            v_f32_st_tnsr_partial((int5){0, r, 0, 0, 0}, out_scale, (float64)scale_s, 0, 0);
        }
        const int col0 = c * chunk_sz;
        int col1 = col0 + chunk_sz;
        if (col1 > dim) col1 = dim;

        for (int col = col0; col < col1; col += 256) {
            bfloat128 b0 = v_bf16_ld_tnsr_b((int5){col, r, 0, 0, 0}, y_in);
            bfloat128 b1 = v_bf16_ld_tnsr_b((int5){col + 128, r, 0, 0, 0}, y_in);
            float128 y0 = v_convert_bf16_to_f32_all_b(b0, 0, (float128){0}, 1, 0);
            float128 y1 = v_convert_bf16_to_f32_all_b(b1, 0, (float128){0}, 1, 0);
            bfloat128 qbf0 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y0.v1 * inv_s, -240.0f, 0, y0.v1, 1, 0), 240.0f, 0, y0.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y0.v2 * inv_s, -240.0f, 0, y0.v2, 1, 0), 240.0f, 0, y0.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            bfloat128 qbf1 = v_convert_f32_to_bf16_all_b((float128){
                v_f32_min_b(v_f32_max_b(y1.v1 * inv_s, -240.0f, 0, y1.v1, 1, 0), 240.0f, 0, y1.v1, 1, 0),
                v_f32_min_b(v_f32_max_b(y1.v2 * inv_s, -240.0f, 0, y1.v2, 1, 0), 240.0f, 0, y1.v2, 1, 0)},
                SW_RHNE, (bfloat128){0}, 1, 0);
            minifloat256 q8 = v_convert_bf16_to_f8_all_b((bfloat256){qbf0, qbf1}, SW_RHNE | SW_LINEAR, (minifloat256){0}, 1, 0);
            v_f8_st_tnsr((int5){col, r, 0, 0, 0}, out_q, q8);
        }
    }
}
