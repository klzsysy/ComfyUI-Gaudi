#pragma once
#include <cstdint>

struct TopkS1Params {
    int32_t has_pen;
    float pres_pen;
    float rep_pen;
    float inv_rep_pen;
};

struct TopkS2Params {
    int32_t k;
};

struct TopkS3Params {
    int32_t k;
    int32_t has_pen;
    float pres_pen;
    float rep_pen;
    float inv_rep_pen;
};

struct GdnRecParams {
    int32_t kh;
    int32_t vh;
    int32_t v_hd;
};

struct GdnConvParams {
    int32_t has_bias;
};

struct RmsQuantParams {
    int32_t has_res;
    int32_t add_one;
    int32_t store_norm;
    float eps;
};

struct GatedRmsQuantParams {
    int32_t vh;
    float eps;
};

struct QkNormRopeParams {
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t add_one;
    float eps;
};

struct SampleTreeAcceptParams {
    int32_t depth;
    int32_t do_sample;
    int32_t adaptive;
    float inv_temp;
    float top_p;
    float min_p;
};

struct DualRmsCatQuantParams {
    int32_t add_one;
    float eps;
};

struct MossQkvPrepParams {
    int32_t aligned;
    float eps;
    float q_scale;
    float k_scale;
};

struct MossResAdalnParams {
    float eps;
};

struct MossExpMaxParams {
    int32_t valid_keys;
};

struct VaeRmsNormSiluParams {
    int32_t n_offset;
    int32_t n_len;
    int32_t do_silu;
    float scale;
};

struct VaeDupUp2dAddParams {
    int32_t c_out;
    int32_t mode;
    int32_t has_main;
};

struct QwenQkRopeParams {
    int32_t chunks;
    float eps;
    float q_scale;
    float k_scale;
};

struct QwenVQuantParams {
    int32_t chunks;
};

struct QwenResAdalnParams {
    int32_t has_res;
    float eps;
};
