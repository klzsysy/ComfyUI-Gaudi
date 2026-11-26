#include <torch/library.h>
#include <ATen/ATen.h>
#include <hpu_custom_op_pt2.h>
#include "params.h"

static inline int64_t silu_chunks(int64_t dim, int64_t rows) {
    return (dim >= 6144) ? ((rows <= 2) ? 12 : 3) : 1;
}

static habana::PartialOutputMetaDataVector meta_topk_s1(const at::Stack& s) {
    auto lg = s.at(0).toTensor();
    return {{at::kFloat, {lg.size(0), (lg.size(1) + 127) / 128}}};
}
static std::shared_ptr<void> fill_topk_s1(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(TopkS1Params);
    params->has_pen = static_cast<int32_t>(s.at(3).toInt());
    params->pres_pen = static_cast<float>(s.at(4).toDouble());
    double rep = s.at(5).toDouble();
    params->rep_pen = static_cast<float>(rep);
    params->inv_rep_pen = static_cast<float>(rep != 0.0 ? 1.0 / rep : 1.0);
    return params;
}
static at::Tensor exec_topk_s1(
    const at::Tensor& logits, const at::Tensor& pen_mask, const at::Tensor& anc_toks,
    int64_t has_pen, double pres_pen, double rep_pen) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_topk_s1");
    return op.execute({logits, pen_mask, anc_toks, has_pen, pres_pen, rep_pen})[0];
}

static habana::PartialOutputMetaDataVector meta_topk_s2(const at::Stack& s) {
    auto bm = s.at(0).toTensor();
    return {{at::kInt, {bm.size(0), s.at(1).toInt()}}};
}
static std::shared_ptr<void> fill_topk_s2(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(TopkS2Params);
    params->k = static_cast<int32_t>(s.at(1).toInt());
    return params;
}
static at::Tensor exec_topk_s2(const at::Tensor& bmax, int64_t k) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_topk_s2");
    return op.execute({bmax, k})[0];
}

static habana::PartialOutputMetaDataVector meta_topk_s3(const at::Stack& s) {
    auto lg = s.at(0).toTensor();
    int64_t k = s.at(4).toInt();
    return {{at::kFloat, {lg.size(0), k}}, {at::kInt, {lg.size(0), k}}};
}
static std::shared_ptr<void> fill_topk_s3(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(TopkS3Params);
    params->k = static_cast<int32_t>(s.at(4).toInt());
    params->has_pen = static_cast<int32_t>(s.at(5).toInt());
    params->pres_pen = static_cast<float>(s.at(6).toDouble());
    double rep = s.at(7).toDouble();
    params->rep_pen = static_cast<float>(rep);
    params->inv_rep_pen = static_cast<float>(rep != 0.0 ? 1.0 / rep : 1.0);
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_topk_s3(
    const at::Tensor& logits, const at::Tensor& pen_mask, const at::Tensor& anc_toks,
    const at::Tensor& top_blocks, int64_t k, int64_t has_pen, double pres_pen, double rep_pen) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_topk_s3");
    auto r = op.execute({logits, pen_mask, anc_toks, top_blocks, k, has_pen, pres_pen, rep_pen});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_gdn_rec(const at::Stack& s) {
    auto st = s.at(0).toTensor();
    auto cc = s.at(2).toTensor();
    int64_t vh = s.at(9).toInt();
    int64_t v_hd = s.at(10).toInt();
    return {{at::kFloat, st.sizes().vec()}, {at::kFloat, {cc.size(0), cc.size(1), vh * v_hd}}};
}
static std::shared_ptr<void> fill_gdn_rec(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(GdnRecParams);
    params->kh = static_cast<int32_t>(s.at(8).toInt());
    params->vh = static_cast<int32_t>(s.at(9).toInt());
    params->v_hd = static_cast<int32_t>(s.at(10).toInt());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_gdn_rec(
    const at::Tensor& state, const at::Tensor& prev_conv, const at::Tensor& cur_conv,
    const at::Tensor& prev_ab, const at::Tensor& cur_ab,
    const at::Tensor& g_decay, const at::Tensor& dt_bias, const at::Tensor& ctl,
    int64_t kh, int64_t vh, int64_t v_hd) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_gdn_rec");
    auto r = op.execute({state, prev_conv, cur_conv, prev_ab, cur_ab, g_decay, dt_bias, ctl, kh, vh, v_hd});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_gdn_conv(const at::Stack& s) {
    auto cst = s.at(0).toTensor();
    auto cp = s.at(2).toTensor();
    return {{at::kBFloat16, cst.sizes().vec()}, {at::kBFloat16, cp.sizes().vec()}};
}
static std::shared_ptr<void> fill_gdn_conv(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(GdnConvParams);
    params->has_bias = static_cast<int32_t>(s.at(7).toInt());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_gdn_conv(
    const at::Tensor& conv_st, const at::Tensor& prev_proj, const at::Tensor& cur_proj,
    const at::Tensor& conv_w, const at::Tensor& conv_bias, const at::Tensor& conv_src,
    const at::Tensor& ctl, int64_t has_bias) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_gdn_conv");
    auto r = op.execute({conv_st, prev_proj, cur_proj, conv_w, conv_bias, conv_src, ctl, has_bias});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_rmsnorm_quant(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    int64_t has_res = s.at(3).toInt();
    int64_t store_norm = s.at(5).toInt();
    std::vector<int64_t> ox_shape = has_res ? x.sizes().vec() : std::vector<int64_t>{1, 1};
    std::vector<int64_t> oy_shape = store_norm ? x.sizes().vec() : std::vector<int64_t>{1, 1};
    return {
        {at::kBFloat16, ox_shape},
        {at::kBFloat16, oy_shape},
        {at::kFloat8_e4m3fn, x.sizes().vec()},
        {at::kFloat, {x.size(0), 1}},
    };
}
static std::shared_ptr<void> fill_rmsnorm_quant(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(RmsQuantParams);
    params->has_res = static_cast<int32_t>(s.at(3).toInt());
    params->add_one = static_cast<int32_t>(s.at(4).toInt());
    params->store_norm = static_cast<int32_t>(s.at(5).toInt());
    params->eps = static_cast<float>(s.at(6).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> exec_rmsnorm_quant(
    const at::Tensor& x, const at::Tensor& res, const at::Tensor& weight,
    int64_t has_res, int64_t add_one, int64_t store_norm, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_rmsnorm_quant");
    auto r = op.execute({x, res, weight, has_res, add_one, store_norm, eps});
    return {r[0], r[1], r[2], r[3]};
}

static habana::PartialOutputMetaDataVector meta_gated_rms_quant(const at::Stack& s) {
    auto z = s.at(1).toTensor();
    return {{at::kFloat8_e4m3fn, z.sizes().vec()}, {at::kFloat, {z.size(0), 1}}};
}
static std::shared_ptr<void> fill_gated_rms_quant(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(GatedRmsQuantParams);
    params->vh = static_cast<int32_t>(s.at(3).toInt());
    params->eps = static_cast<float>(s.at(4).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_gated_rms_quant(
    const at::Tensor& out_f32, const at::Tensor& z, const at::Tensor& gdn_nw,
    int64_t vh, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_gated_rmsnorm_quant");
    auto r = op.execute({out_f32, z, gdn_nw, vh, eps});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_sig_mul_quant(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    return {{at::kFloat8_e4m3fn, x.sizes().vec()}, {at::kFloat, {x.size(0), 1}}};
}
static std::tuple<at::Tensor, at::Tensor> exec_sig_mul_quant(const at::Tensor& x, const at::Tensor& gate) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_sig_mul_quant");
    auto r = op.execute({x, gate});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_silu_mul_s1(const at::Stack& s) {
    auto g = s.at(0).toTensor();
    int64_t ch = silu_chunks(g.size(1), g.size(0));
    return {{at::kBFloat16, g.sizes().vec()}, {at::kFloat, {g.size(0), ch, 64}}};
}
static std::tuple<at::Tensor, at::Tensor> exec_silu_mul_s1(const at::Tensor& gate, const at::Tensor& up) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_silu_mul_quant_s1");
    auto r = op.execute({gate, up});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_silu_mul_s2(const at::Stack& s) {
    auto y = s.at(0).toTensor();
    return {{at::kFloat8_e4m3fn, y.sizes().vec()}, {at::kFloat, {y.size(0), 1}}};
}
static std::tuple<at::Tensor, at::Tensor> exec_silu_mul_s2(const at::Tensor& y_in, const at::Tensor& cmax) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_silu_mul_quant_s2");
    auto r = op.execute({y_in, cmax});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_qk_norm_rope(const at::Stack& s) {
    auto qg = s.at(0).toTensor();
    int64_t nh = s.at(5).toInt();
    int64_t nkh = s.at(6).toInt();
    int64_t rows = qg.size(0);
    return {
        {at::kBFloat16, {1, nh, rows, 256}},
        {at::kBFloat16, {1, nkh, rows, 256}},
        {at::kBFloat16, {rows, nh * 256}},
    };
}
static std::shared_ptr<void> fill_qk_norm_rope(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(QkNormRopeParams);
    params->num_heads = static_cast<int32_t>(s.at(5).toInt());
    params->num_kv_heads = static_cast<int32_t>(s.at(6).toInt());
    params->add_one = static_cast<int32_t>(s.at(7).toInt());
    params->eps = static_cast<float>(s.at(8).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> exec_qk_norm_rope(
    const at::Tensor& qg, const at::Tensor& xk, const at::Tensor& qn_w, const at::Tensor& kn_w, const at::Tensor& cs,
    int64_t num_heads, int64_t num_kv_heads, int64_t add_one, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_qk_norm_rope");
    auto r = op.execute({qg, xk, qn_w, kn_w, cs, num_heads, num_kv_heads, add_one, eps});
    return {r[0], r[1], r[2]};
}

static habana::PartialOutputMetaDataVector meta_sample_accept(const at::Stack&) {
    return {
        {at::kInt, {19}},
        {at::kInt, {8}},
        {at::kInt, {8}},
        {at::kInt, {4}},
        {at::kInt, {7}},
    };
}
static std::shared_ptr<void> fill_sample_accept(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(SampleTreeAcceptParams);
    params->depth = static_cast<int32_t>(s.at(7).toInt());
    params->do_sample = static_cast<int32_t>(s.at(8).toInt());
    params->adaptive = static_cast<int32_t>(s.at(9).toInt());
    double temp = s.at(10).toDouble();
    params->inv_temp = static_cast<float>(temp != 0.0 ? 1.0 / temp : 1.0);
    params->top_p = static_cast<float>(s.at(11).toDouble());
    params->min_p = static_cast<float>(s.at(12).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> exec_sample_accept(
    const at::Tensor& top_vals, const at::Tensor& top_idx, const at::Tensor& verify_tokens,
    const at::Tensor& tops, const at::Tensor& rand_u, const at::Tensor& state_in, const at::Tensor& tree_parent,
    int64_t depth, int64_t do_sample, int64_t adaptive, double temperature, double top_p, double min_p) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_sample_tree_accept");
    auto r = op.execute({top_vals, top_idx, verify_tokens, tops, rand_u, state_in, tree_parent,
                         depth, do_sample, adaptive, temperature, top_p, min_p});
    return {r[0], r[1], r[2], r[3], r[4]};
}

static habana::PartialOutputMetaDataVector meta_dual_rms_cat(const at::Stack& s) {
    auto e = s.at(0).toTensor();
    return {{at::kFloat8_e4m3fn, {e.size(0), e.size(1) * 2}}, {at::kFloat, {e.size(0), 1}}};
}
static std::shared_ptr<void> fill_dual_rms_cat(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(DualRmsCatQuantParams);
    params->add_one = static_cast<int32_t>(s.at(4).toInt());
    params->eps = static_cast<float>(s.at(5).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_dual_rms_cat(
    const at::Tensor& emb, const at::Tensor& hid, const at::Tensor& w_emb, const at::Tensor& w_hid,
    int64_t add_one, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen35_dual_rmsnorm_cat_quant");
    auto r = op.execute({emb, hid, w_emb, w_hid, add_one, eps});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_moss_situ_q(const at::Stack& s) {
    auto y = s.at(0).toTensor();
    return {{at::kFloat8_e4m3fn, {y.size(0), y.size(1) / 2}}, {at::kFloat, {y.size(0)}}};
}
static std::tuple<at::Tensor, at::Tensor> exec_moss_situ_q(const at::Tensor& y) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_situ_quant");
    auto r = op.execute({y});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_moss_qkv_prep(const at::Stack& s) {
    auto qkv = s.at(0).toTensor();
    int64_t rows = qkv.size(0);
    int64_t mode = s.at(4).toInt();
    if (mode == 2) {
        int64_t k_pad = ((rows + 511) / 512) * 512;
        return {
            {at::kBFloat16, {24, rows, 128}},
            {at::kBFloat16, {24, k_pad, 128}},
            {at::kFloat8_e4m3fn, {24, k_pad / 2, 256}},
            {at::kFloat, {24, 64}},
        };
    }
    int64_t k_pad = ((rows + 127) / 128) * 128;
    int64_t qk_dim = mode ? 128 : 192;
    return {
        {at::kBFloat16, {24, rows, qk_dim}},
        {at::kBFloat16, {24, k_pad, qk_dim}},
        {at::kFloat8_e4m3fn, {24, k_pad, 128}},
        {at::kFloat, {24, 64}},
    };
}
static std::shared_ptr<void> fill_moss_qkv_prep(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(MossQkvPrepParams);
    params->aligned = static_cast<int32_t>(s.at(4).toInt());
    params->eps = static_cast<float>(s.at(5).toDouble());
    params->q_scale = static_cast<float>(s.at(6).toDouble());
    params->k_scale = static_cast<float>(s.at(7).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> exec_moss_qkv_prep(
    const at::Tensor& qkv, const at::Tensor& qn_w, const at::Tensor& kn_w, const at::Tensor& pe_cs,
    int64_t aligned, double eps, double q_scale, double k_scale) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_qkv_prep");
    auto r = op.execute({qkv, qn_w, kn_w, pe_cs, aligned, eps, q_scale, k_scale});
    return {r[0], r[1], r[2], r[3]};
}

static habana::PartialOutputMetaDataVector meta_moss_res_adaln_q(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    return {
        {at::kBFloat16, {x.size(0), x.size(1)}},
        {at::kFloat8_e4m3fn, {x.size(0), x.size(1)}},
        {at::kFloat, {x.size(0)}},
    };
}
static std::shared_ptr<void> fill_moss_res_adaln_q(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(MossResAdalnParams);
    params->eps = static_cast<float>(s.at(5).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> exec_moss_res_adaln_q(
    const at::Tensor& x, const at::Tensor& res, const at::Tensor& gate,
    const at::Tensor& eff_scale, const at::Tensor& shift, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_res_adaln_quant");
    auto r = op.execute({x, res, gate, eff_scale, shift, eps});
    return {r[0], r[1], r[2]};
}

static habana::PartialOutputMetaDataVector meta_moss_hnd_scale_q(const at::Stack& s) {
    auto a = s.at(0).toTensor();
    return {
        {at::kFloat8_e4m3fn, {a.size(1), 3072}},
        {at::kFloat, {a.size(1)}},
    };
}
static std::tuple<at::Tensor, at::Tensor> exec_moss_hnd_scale_q(
    const at::Tensor& attn, const at::Tensor& scales) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_hnd_scale_quant");
    auto r = op.execute({attn, scales});
    return {r[0], r[1]};
}

static std::tuple<at::Tensor, at::Tensor> exec_moss_hnd_scale_sum_q(
    const at::Tensor& attn, const at::Tensor& scales, const at::Tensor& sum_in) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_hnd_scale_sum_quant");
    auto r = op.execute({attn, scales, sum_in});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_moss_expmax_fp8(const at::Stack& s) {
    auto scores = s.at(0).toTensor();
    return {
        {at::kFloat8_e4m3fn, {scores.size(0), scores.size(1)}},
        {at::kFloat, {scores.size(0)}},
    };
}
static std::shared_ptr<void> fill_moss_expmax_fp8(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(MossExpMaxParams);
    params->valid_keys = static_cast<int32_t>(s.at(1).toInt());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_moss_expmax_fp8(
    const at::Tensor& scores, int64_t valid_keys) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_moss_expmax_fp8");
    auto r = op.execute({scores, valid_keys});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_vae_rmsnorm_silu(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    int64_t n_len = s.at(3).toInt();
    return {{at::kBFloat16, {x.size(0), x.size(1), n_len}}};
}
static std::shared_ptr<void> fill_vae_rmsnorm_silu(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(VaeRmsNormSiluParams);
    params->n_offset = static_cast<int32_t>(s.at(2).toInt());
    params->n_len = static_cast<int32_t>(s.at(3).toInt());
    params->do_silu = static_cast<int32_t>(s.at(4).toInt());
    params->scale = static_cast<float>(s.at(5).toDouble());
    return params;
}
static at::Tensor exec_vae_rmsnorm_silu(
    const at::Tensor& x, const at::Tensor& gamma,
    int64_t n_offset, int64_t n_len, int64_t do_silu, double scale) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_vae_rmsnorm_silu_bf16");
    return op.execute({x, gamma, n_offset, n_len, do_silu, scale})[0];
}

static habana::PartialOutputMetaDataVector meta_vae_dupup2d_add(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    int64_t c_out = s.at(2).toInt();
    return {{at::kBFloat16, {x.size(0), c_out, x.size(2) * 2, x.size(3) * 2}}};
}
static std::shared_ptr<void> fill_vae_dupup2d_add(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(VaeDupUp2dAddParams);
    params->c_out = static_cast<int32_t>(s.at(2).toInt());
    params->mode = static_cast<int32_t>(s.at(3).toInt());
    params->has_main = static_cast<int32_t>(s.at(4).toInt());
    return params;
}
static at::Tensor exec_vae_dupup2d_add(
    const at::Tensor& x, const at::Tensor& main_t,
    int64_t c_out, int64_t mode, int64_t has_main) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_vae_dupup2d_add_bf16");
    return op.execute({x, main_t, c_out, mode, has_main})[0];
}

static habana::PartialOutputMetaDataVector meta_qwen_expmax_fp8(const at::Stack& s) {
    auto scores = s.at(0).toTensor();
    return {
        {at::kFloat8_e4m3fn, {scores.size(0), scores.size(1)}},
        {at::kFloat, {scores.size(0)}},
        {at::kFloat, {scores.size(0)}},
    };
}
static std::shared_ptr<void> fill_qwen_expmax_fp8(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(MossExpMaxParams);
    params->valid_keys = static_cast<int32_t>(s.at(1).toInt());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> exec_qwen_expmax_fp8(
    const at::Tensor& scores, int64_t valid_keys) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen_expmax_fp8");
    auto r = op.execute({scores, valid_keys});
    return {r[0], r[1], r[2]};
}

static habana::PartialOutputMetaDataVector meta_qwen_qk_rope_fp8(const at::Stack& s) {
    auto q = s.at(0).toTensor();
    auto pk = s.at(2).toTensor();
    int64_t Q = q.size(0);
    int64_t P = pk.size(0);
    int64_t k_pad = ((P + Q + 255) / 256) * 256;
    return {
        {at::kBFloat16, {32, Q, 128}},
        {at::kBFloat16, {32, k_pad, 128}},
    };
}
static std::shared_ptr<void> fill_qwen_qk_rope_fp8(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(QwenQkRopeParams);
    params->chunks = static_cast<int32_t>(s.at(6).toInt());
    params->eps = static_cast<float>(s.at(7).toDouble());
    params->q_scale = static_cast<float>(s.at(8).toDouble());
    params->k_scale = static_cast<float>(s.at(9).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor> exec_qwen_qk_rope_fp8(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& pk,
    const at::Tensor& qn_w, const at::Tensor& kn_w, const at::Tensor& pe_cs,
    int64_t chunks, double eps, double q_scale, double k_scale) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen_qk_rope_fp8");
    auto r = op.execute({q, k, pk, qn_w, kn_w, pe_cs, chunks, eps, q_scale, k_scale});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_qwen_v_quant_fp8(const at::Stack& s) {
    auto v = s.at(0).toTensor();
    auto pv = s.at(1).toTensor();
    int64_t k_pad = ((pv.size(0) + v.size(0) + 255) / 256) * 256;
    return {{at::kFloat8_e4m3fn, {32, k_pad / 2, 256}}};
}
static std::shared_ptr<void> fill_qwen_v_quant_fp8(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(QwenVQuantParams);
    params->chunks = static_cast<int32_t>(s.at(3).toInt());
    return params;
}
static at::Tensor exec_qwen_v_quant_fp8(
    const at::Tensor& v, const at::Tensor& pv, const at::Tensor& v_scales, int64_t chunks) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen_v_quant_fp8");
    return op.execute({v, pv, v_scales, chunks})[0];
}

static habana::PartialOutputMetaDataVector meta_qwen_hnd_scale_sum_q(const at::Stack& s) {
    auto a = s.at(0).toTensor();
    return {
        {at::kFloat8_e4m3fn, {a.size(1), 4096}},
        {at::kFloat, {a.size(1)}},
    };
}
static std::tuple<at::Tensor, at::Tensor> exec_qwen_hnd_scale_sum_q(
    const at::Tensor& attn, const at::Tensor& scales, const at::Tensor& sum_in) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen_hnd_scale_sum_quant");
    auto r = op.execute({attn, scales, sum_in});
    return {r[0], r[1]};
}

static habana::PartialOutputMetaDataVector meta_qwen_res_adaln_q(const at::Stack& s) {
    auto x = s.at(0).toTensor();
    int64_t has_res = s.at(4).toInt();
    return {
        {at::kBFloat16, has_res ? std::vector<int64_t>{x.size(0), x.size(1)} : std::vector<int64_t>{1, 1}},
        {at::kFloat8_e4m3fn, {x.size(0), x.size(1)}},
        {at::kFloat, {x.size(0)}},
    };
}
static std::shared_ptr<void> fill_qwen_res_adaln_q(const at::Stack& s, size_t& size) {
    HPU_PARAMS_STUB(QwenResAdalnParams);
    params->has_res = static_cast<int32_t>(s.at(4).toInt());
    params->eps = static_cast<float>(s.at(5).toDouble());
    return params;
}
static std::tuple<at::Tensor, at::Tensor, at::Tensor> exec_qwen_res_adaln_q(
    const at::Tensor& x, const at::Tensor& res, const at::Tensor& gate, const at::Tensor& scale_p1,
    int64_t has_res, double eps) {
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor("custom_op::comfy_qwen_res_adaln_quant");
    auto r = op.execute({x, res, gate, scale_p1, has_res, eps});
    return {r[0], r[1], r[2]};
}

TORCH_LIBRARY_FRAGMENT(custom_op, lib) {
    lib.def("comfy_qwen35_topk_s1(Tensor logits, Tensor pen_mask, Tensor anc_toks, int has_pen, float pres_pen, float rep_pen) -> Tensor");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_topk_s1", "comfy_topk_s1_bmax", meta_topk_s1, fill_topk_s1);

    lib.def("comfy_qwen35_topk_s2(Tensor bmax, int k) -> Tensor");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_topk_s2", "comfy_topk_s2_blocks", meta_topk_s2, fill_topk_s2);

    lib.def("comfy_qwen35_topk_s3(Tensor logits, Tensor pen_mask, Tensor anc_toks, Tensor top_blocks, int k, int has_pen, float pres_pen, float rep_pen) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_topk_s3", "comfy_topk_s3_extract", meta_topk_s3, fill_topk_s3);

    lib.def("comfy_qwen35_gdn_rec(Tensor state, Tensor prev_conv, Tensor cur_conv, Tensor prev_ab, Tensor cur_ab, Tensor g_decay, Tensor dt_bias, Tensor ctl, int kh, int vh, int v_hd) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_gdn_rec", "comfy_gdn_rec_f32", meta_gdn_rec, fill_gdn_rec);

    lib.def("comfy_qwen35_gdn_conv(Tensor conv_st, Tensor prev_proj, Tensor cur_proj, Tensor conv_w, Tensor conv_bias, Tensor conv_src, Tensor ctl, int has_bias) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_gdn_conv", "comfy_gdn_conv_bf16", meta_gdn_conv, fill_gdn_conv);

    lib.def("comfy_qwen35_rmsnorm_quant(Tensor x, Tensor res, Tensor weight, int has_res, int add_one, int store_norm, float eps) -> (Tensor, Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_rmsnorm_quant", "comfy_rmsnorm_quant_fp8", meta_rmsnorm_quant, fill_rmsnorm_quant);

    lib.def("comfy_qwen35_gated_rmsnorm_quant(Tensor out_f32, Tensor z, Tensor gdn_nw, int vh, float eps) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_gated_rmsnorm_quant", "comfy_gated_rmsnorm_quant_fp8", meta_gated_rms_quant, fill_gated_rms_quant);

    lib.def("comfy_qwen35_sig_mul_quant(Tensor x, Tensor gate) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_sig_mul_quant", "comfy_sig_mul_quant_fp8", meta_sig_mul_quant, nullptr);

    lib.def("comfy_qwen35_silu_mul_quant_s1(Tensor gate, Tensor up) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_silu_mul_quant_s1", "comfy_silu_mul_quant_s1", meta_silu_mul_s1, nullptr);

    lib.def("comfy_qwen35_silu_mul_quant_s2(Tensor y_in, Tensor cmax) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_silu_mul_quant_s2", "comfy_silu_mul_quant_s2", meta_silu_mul_s2, nullptr);

    lib.def("comfy_qwen35_qk_norm_rope(Tensor qg, Tensor xk, Tensor qn_w, Tensor kn_w, Tensor cs, int num_heads, int num_kv_heads, int add_one, float eps) -> (Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_qk_norm_rope", "comfy_qk_norm_rope_bf16", meta_qk_norm_rope, fill_qk_norm_rope);

    lib.def("comfy_qwen35_sample_tree_accept(Tensor top_vals, Tensor top_idx, Tensor verify_tokens, Tensor tops, Tensor rand_u, Tensor state_in, Tensor tree_parent, int depth, int do_sample, int adaptive, float temperature, float top_p, float min_p) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_sample_tree_accept", "comfy_sample_tree_accept_i32", meta_sample_accept, fill_sample_accept);

    lib.def("comfy_qwen35_dual_rmsnorm_cat_quant(Tensor emb, Tensor hid, Tensor w_emb, Tensor w_hid, int add_one, float eps) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen35_dual_rmsnorm_cat_quant", "comfy_dual_rmsnorm_cat_quant_fp8", meta_dual_rms_cat, fill_dual_rms_cat);

    lib.def("comfy_moss_situ_quant(Tensor y) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_situ_quant", "comfy_moss_situ_quant_fp8", meta_moss_situ_q, nullptr);

    lib.def("comfy_moss_qkv_prep(Tensor qkv, Tensor qn_w, Tensor kn_w, Tensor pe_cs, int aligned, float eps, float q_scale, float k_scale) -> (Tensor, Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_qkv_prep", "comfy_moss_qkv_prep_fp8", meta_moss_qkv_prep, fill_moss_qkv_prep);

    lib.def("comfy_moss_res_adaln_quant(Tensor x, Tensor res, Tensor gate, Tensor eff_scale, Tensor shift, float eps) -> (Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_res_adaln_quant", "comfy_moss_res_adaln_quant_fp8", meta_moss_res_adaln_q, fill_moss_res_adaln_q);

    lib.def("comfy_moss_hnd_scale_quant(Tensor attn, Tensor scales) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_hnd_scale_quant", "comfy_moss_hnd_scale_quant_fp8", meta_moss_hnd_scale_q, nullptr);

    lib.def("comfy_moss_hnd_scale_sum_quant(Tensor attn, Tensor scales, Tensor sum_in) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_hnd_scale_sum_quant", "comfy_moss_hnd_scale_sum_quant_fp8", meta_moss_hnd_scale_q, nullptr);

    lib.def("comfy_moss_expmax_fp8(Tensor scores, int valid_keys) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_moss_expmax_fp8", "comfy_moss_expmax_fp8", meta_moss_expmax_fp8, fill_moss_expmax_fp8);

    lib.def("comfy_vae_rmsnorm_silu_bf16(Tensor x, Tensor gamma, int n_offset, int n_len, int do_silu, float scale) -> Tensor");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_vae_rmsnorm_silu_bf16", "comfy_vae_rmsnorm_silu_bf16", meta_vae_rmsnorm_silu, fill_vae_rmsnorm_silu);

    lib.def("comfy_vae_dupup2d_add_bf16(Tensor x, Tensor main_t, int c_out, int mode, int has_main) -> Tensor");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_vae_dupup2d_add_bf16", "comfy_vae_dupup2d_add_bf16", meta_vae_dupup2d_add, fill_vae_dupup2d_add);

    lib.def("comfy_qwen_expmax_fp8(Tensor scores, int valid_keys) -> (Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen_expmax_fp8", "comfy_qwen_expmax_fp8", meta_qwen_expmax_fp8, fill_qwen_expmax_fp8);

    lib.def("comfy_qwen_qk_rope_fp8(Tensor q, Tensor k, Tensor pk, Tensor qn_w, Tensor kn_w, Tensor pe_cs, int chunks, float eps, float q_scale, float k_scale) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen_qk_rope_fp8", "comfy_qwen_qk_rope_fp8", meta_qwen_qk_rope_fp8, fill_qwen_qk_rope_fp8);

    lib.def("comfy_qwen_v_quant_fp8(Tensor v, Tensor pv, Tensor v_scales, int chunks) -> Tensor");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen_v_quant_fp8", "comfy_qwen_v_quant_fp8", meta_qwen_v_quant_fp8, fill_qwen_v_quant_fp8);

    lib.def("comfy_qwen_hnd_scale_sum_quant(Tensor attn, Tensor scales, Tensor sum_in) -> (Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen_hnd_scale_sum_quant", "comfy_qwen_hnd_scale_sum_quant_fp8", meta_qwen_hnd_scale_sum_q, nullptr);

    lib.def("comfy_qwen_res_adaln_quant(Tensor x, Tensor res, Tensor gate, Tensor scale_p1, int has_res, float eps) -> (Tensor, Tensor, Tensor)");
    habana::custom_op::registerUserCustomOp("custom_op::comfy_qwen_res_adaln_quant", "comfy_qwen_res_adaln_quant_fp8", meta_qwen_res_adaln_q, fill_qwen_res_adaln_q);
}

TORCH_LIBRARY_IMPL(custom_op, HPU, lib) {
    lib.impl("comfy_qwen35_topk_s1", exec_topk_s1);
    lib.impl("comfy_qwen35_topk_s2", exec_topk_s2);
    lib.impl("comfy_qwen35_topk_s3", exec_topk_s3);
    lib.impl("comfy_qwen35_gdn_rec", exec_gdn_rec);
    lib.impl("comfy_qwen35_gdn_conv", exec_gdn_conv);
    lib.impl("comfy_qwen35_rmsnorm_quant", exec_rmsnorm_quant);
    lib.impl("comfy_qwen35_gated_rmsnorm_quant", exec_gated_rms_quant);
    lib.impl("comfy_qwen35_sig_mul_quant", exec_sig_mul_quant);
    lib.impl("comfy_qwen35_silu_mul_quant_s1", exec_silu_mul_s1);
    lib.impl("comfy_qwen35_silu_mul_quant_s2", exec_silu_mul_s2);
    lib.impl("comfy_qwen35_qk_norm_rope", exec_qk_norm_rope);
    lib.impl("comfy_qwen35_sample_tree_accept", exec_sample_accept);
    lib.impl("comfy_qwen35_dual_rmsnorm_cat_quant", exec_dual_rms_cat);
    lib.impl("comfy_moss_situ_quant", exec_moss_situ_q);
    lib.impl("comfy_moss_qkv_prep", exec_moss_qkv_prep);
    lib.impl("comfy_moss_res_adaln_quant", exec_moss_res_adaln_q);
    lib.impl("comfy_moss_hnd_scale_quant", exec_moss_hnd_scale_q);
    lib.impl("comfy_moss_hnd_scale_sum_quant", exec_moss_hnd_scale_sum_q);
    lib.impl("comfy_moss_expmax_fp8", exec_moss_expmax_fp8);
    lib.impl("comfy_vae_rmsnorm_silu_bf16", exec_vae_rmsnorm_silu);
    lib.impl("comfy_vae_dupup2d_add_bf16", exec_vae_dupup2d_add);
    lib.impl("comfy_qwen_expmax_fp8", exec_qwen_expmax_fp8);
    lib.impl("comfy_qwen_qk_rope_fp8", exec_qwen_qk_rope_fp8);
    lib.impl("comfy_qwen_v_quant_fp8", exec_qwen_v_quant_fp8);
    lib.impl("comfy_qwen_hnd_scale_sum_quant", exec_qwen_hnd_scale_sum_q);
    lib.impl("comfy_qwen_res_adaln_quant", exec_qwen_res_adaln_q);
}
