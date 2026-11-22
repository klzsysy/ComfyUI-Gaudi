#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdint>
#include "params.h"

using namespace tpc_lib_api;

extern "C" const unsigned char topk_s1_start[], topk_s1_end[];
extern "C" const unsigned char topk_s2_start[], topk_s2_end[];
extern "C" const unsigned char topk_s3_start[], topk_s3_end[];
extern "C" const unsigned char gdn_rec_start[], gdn_rec_end[];
extern "C" const unsigned char gdn_conv_start[], gdn_conv_end[];
extern "C" const unsigned char rms_q_start[], rms_q_end[];
extern "C" const unsigned char gated_rms_q_start[], gated_rms_q_end[];
extern "C" const unsigned char sig_mul_q_start[], sig_mul_q_end[];
extern "C" const unsigned char silu_mul_s1_start[], silu_mul_s1_end[];
extern "C" const unsigned char silu_mul_s2_start[], silu_mul_s2_end[];
extern "C" const unsigned char qk_norm_rope_start[], qk_norm_rope_end[];
extern "C" const unsigned char sample_accept_start[], sample_accept_end[];
extern "C" const unsigned char dual_rms_cat_start[], dual_rms_cat_end[];
extern "C" const unsigned char moss_situ_q_start[], moss_situ_q_end[];
extern "C" const unsigned char moss_qkv_prep_start[], moss_qkv_prep_end[];

static const char* const kGuids[] = {
    "comfy_topk_s1_bmax",
    "comfy_topk_s2_blocks",
    "comfy_topk_s3_extract",
    "comfy_gdn_rec_f32",
    "comfy_gdn_conv_bf16",
    "comfy_rmsnorm_quant_fp8",
    "comfy_gated_rmsnorm_quant_fp8",
    "comfy_sig_mul_quant_fp8",
    "comfy_silu_mul_quant_s1",
    "comfy_silu_mul_quant_s2",
    "comfy_qk_norm_rope_bf16",
    "comfy_sample_tree_accept_i32",
    "comfy_dual_rmsnorm_cat_quant_fp8",
    "comfy_moss_situ_quant_fp8",
    "comfy_moss_qkv_prep_fp8",
};
static constexpr uint32_t kNumGuids = sizeof(kGuids) / sizeof(kGuids[0]);

static inline uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline int silu_chunks(int dim, int rows) {
    return (dim >= 6144) ? ((rows <= 2) ? 12 : 3) : 1;
}

static inline int silu_chunk_sz(int dim, int chunks) {
    return (((dim + chunks - 1) / chunks) + 255) & ~255;
}

static void set_all_required(HabanaKernelInstantiation* out, uint32_t nin, uint32_t nout) {
    out->auxiliaryTensorNr = 0;
    for (uint32_t i = 0; i < nin; ++i) {
        std::memset(&out->inputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->inputTensorAccessPattern[i].allRequired = 1;
    }
    for (uint32_t i = 0; i < nout; ++i) {
        std::memset(&out->outputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->outputTensorAccessPattern[i].allRequired = 1;
        out->outputTensorAccessPattern[i].noRmwAccess = 1;
    }
}

static GlueCodeReturn copy_elf(HabanaKernelInstantiation* out, const unsigned char* s, const unsigned char* e) {
    const uint32_t cap = out->kernel.elfSize;
    out->kernel.elfSize = static_cast<uint32_t>(e - s);
    if (!out->kernel.kernelElf || cap < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, s, out->kernel.elfSize);
    return GLUE_SUCCESS;
}

extern "C" uint64_t GetLibVersion() { return 1; }

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
    const uint32_t cap = *count;
    *count = (device == DEVICE_ID_GAUDI2) ? kNumGuids : 0;
    if (guids && cap >= kNumGuids && *count) {
        for (uint32_t i = 0; i < kNumGuids; ++i) {
            std::memset(&guids[i], 0, sizeof(GuidInfo));
            std::strcpy(guids[i].name, kGuids[i]);
        }
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* p, ShapeInferenceOutput* out) {
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    const char* name = (p->pGuid ? p->pGuid : &p->guid)->name;
    if (!std::strcmp(name, "comfy_topk_s1_bmax")) {
        const auto& lg = p->inputTensors[0]->geometry;
        auto& g = out->outputTensors[0]->geometry;
        g = lg;
        g.dataType = DATA_F32;
        g.maxSizes[0] = g.minSizes[0] = (lg.maxSizes[0] + 127) / 128;
        g.maxSizes[1] = g.minSizes[1] = lg.maxSizes[1];
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_topk_s2_blocks")) {
        const auto* prm = static_cast<const TopkS2Params*>(p->nodeParams.nodeParams);
        const auto& bm = p->inputTensors[0]->geometry;
        auto& g = out->outputTensors[0]->geometry;
        g = bm;
        g.dataType = DATA_I32;
        g.maxSizes[0] = g.minSizes[0] = static_cast<uint64_t>(prm->k);
        g.maxSizes[1] = g.minSizes[1] = bm.maxSizes[1];
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_topk_s3_extract")) {
        const auto* prm = static_cast<const TopkS3Params*>(p->nodeParams.nodeParams);
        const auto& lg = p->inputTensors[0]->geometry;
        for (int o = 0; o < 2; ++o) {
            auto& g = out->outputTensors[o]->geometry;
            g = lg;
            g.dataType = (o == 0) ? DATA_F32 : DATA_I32;
            g.maxSizes[0] = g.minSizes[0] = static_cast<uint64_t>(prm->k);
            g.maxSizes[1] = g.minSizes[1] = lg.maxSizes[1];
        }
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_gdn_rec_f32")) {
        const auto* prm = static_cast<const GdnRecParams*>(p->nodeParams.nodeParams);
        const auto& st = p->inputTensors[0]->geometry;
        const auto& cc = p->inputTensors[2]->geometry;
        out->outputTensors[0]->geometry = st;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = cc;
        o1.dataType = DATA_F32;
        o1.maxSizes[0] = o1.minSizes[0] = static_cast<uint64_t>(prm->vh) * static_cast<uint64_t>(prm->v_hd);
        o1.maxSizes[1] = o1.minSizes[1] = cc.maxSizes[1];
        o1.maxSizes[2] = o1.minSizes[2] = cc.maxSizes[2];
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_gdn_conv_bf16")) {
        out->outputTensors[0]->geometry = p->inputTensors[0]->geometry;
        out->outputTensors[1]->geometry = p->inputTensors[2]->geometry;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_rmsnorm_quant_fp8")) {
        const auto* prm = static_cast<const RmsQuantParams*>(p->nodeParams.nodeParams);
        const auto& xg = p->inputTensors[0]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = xg;
        if (!prm->has_res) { o0.maxSizes[0] = o0.minSizes[0] = 1; o0.maxSizes[1] = o0.minSizes[1] = 1; }
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = xg;
        if (!prm->store_norm) { o1.maxSizes[0] = o1.minSizes[0] = 1; o1.maxSizes[1] = o1.minSizes[1] = 1; }
        auto& o2 = out->outputTensors[2]->geometry;
        o2 = xg; o2.dataType = DATA_F8_143;
        auto& o3 = out->outputTensors[3]->geometry;
        o3 = xg; o3.dataType = DATA_F32; o3.maxSizes[0] = o3.minSizes[0] = 1;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_gated_rmsnorm_quant_fp8")) {
        const auto& zg = p->inputTensors[1]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = zg; o0.dataType = DATA_F8_143;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = zg; o1.dataType = DATA_F32; o1.maxSizes[0] = o1.minSizes[0] = 1;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_sig_mul_quant_fp8")) {
        const auto& xg = p->inputTensors[0]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = xg; o0.dataType = DATA_F8_143;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = xg; o1.dataType = DATA_F32; o1.maxSizes[0] = o1.minSizes[0] = 1;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_silu_mul_quant_s1")) {
        const auto& gg = p->inputTensors[0]->geometry;
        int dim = int(gg.maxSizes[0]);
        int rows = int(gg.maxSizes[1]);
        int ch = silu_chunks(dim, rows);
        out->outputTensors[0]->geometry = gg;
        auto& o1 = out->outputTensors[1]->geometry;
        std::memset(&o1, 0, sizeof(TensorGeometry));
        o1.dims = 3;
        o1.dataType = DATA_F32;
        o1.maxSizes[0] = o1.minSizes[0] = 64;
        o1.maxSizes[1] = o1.minSizes[1] = static_cast<uint64_t>(ch);
        o1.maxSizes[2] = o1.minSizes[2] = static_cast<uint64_t>(rows);
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_silu_mul_quant_s2")) {
        const auto& yg = p->inputTensors[0]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = yg; o0.dataType = DATA_F8_143;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = yg; o1.dataType = DATA_F32; o1.maxSizes[0] = o1.minSizes[0] = 1;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_qk_norm_rope_bf16")) {
        const auto* prm = static_cast<const QkNormRopeParams*>(p->nodeParams.nodeParams);
        const auto& qgg = p->inputTensors[0]->geometry;
        uint64_t rows = qgg.maxSizes[1];
        auto& o0 = out->outputTensors[0]->geometry;
        std::memset(&o0, 0, sizeof(TensorGeometry));
        o0.dims = 4; o0.dataType = DATA_BF16;
        o0.maxSizes[0] = o0.minSizes[0] = 256;
        o0.maxSizes[1] = o0.minSizes[1] = rows;
        o0.maxSizes[2] = o0.minSizes[2] = static_cast<uint64_t>(prm->num_heads);
        o0.maxSizes[3] = o0.minSizes[3] = 1;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = o0;
        o1.maxSizes[2] = o1.minSizes[2] = static_cast<uint64_t>(prm->num_kv_heads);
        auto& o2 = out->outputTensors[2]->geometry;
        o2 = qgg;
        o2.maxSizes[0] = o2.minSizes[0] = static_cast<uint64_t>(prm->num_heads) * 256;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_sample_tree_accept_i32")) {
        const uint64_t sizes[5] = {19, 8, 8, 4, 7};
        for (int i = 0; i < 5; ++i) {
            auto& g = out->outputTensors[i]->geometry;
            std::memset(&g, 0, sizeof(TensorGeometry));
            g.dims = 1;
            g.dataType = DATA_I32;
            g.maxSizes[0] = g.minSizes[0] = sizes[i];
        }
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_dual_rmsnorm_cat_quant_fp8")) {
        const auto& eg = p->inputTensors[0]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = eg; o0.dataType = DATA_F8_143;
        o0.maxSizes[0] = o0.minSizes[0] = eg.maxSizes[0] * 2;
        auto& o1 = out->outputTensors[1]->geometry;
        o1 = eg; o1.dataType = DATA_F32; o1.maxSizes[0] = o1.minSizes[0] = 1;
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_moss_situ_quant_fp8")) {
        const auto& yg = p->inputTensors[0]->geometry;
        auto& o0 = out->outputTensors[0]->geometry;
        o0 = yg; o0.dataType = DATA_F8_143;
        o0.maxSizes[0] = o0.minSizes[0] = yg.maxSizes[0] / 2;
        auto& o1 = out->outputTensors[1]->geometry;
        std::memset(&o1, 0, sizeof(TensorGeometry));
        o1.dims = 1; o1.dataType = DATA_F32;
        o1.maxSizes[0] = o1.minSizes[0] = yg.maxSizes[1];
        return GLUE_SUCCESS;
    }
    if (!std::strcmp(name, "comfy_moss_qkv_prep_fp8")) {
        const auto* prm = static_cast<const MossQkvPrepParams*>(p->nodeParams.nodeParams);
        const auto& qg = p->inputTensors[0]->geometry;
        uint64_t rows = qg.maxSizes[1];
        uint64_t k_pad = ((rows + 127) / 128) * 128;
        uint64_t qk_dim = prm->aligned ? 128 : 192;
        uint64_t d0s[3] = {qk_dim, qk_dim, 128};
        uint64_t d1s[3] = {rows, k_pad, k_pad};
        for (int i = 0; i < 3; ++i) {
            auto& o = out->outputTensors[i]->geometry;
            std::memset(&o, 0, sizeof(TensorGeometry));
            o.dims = 3;
            o.dataType = (i == 2) ? DATA_F8_143 : DATA_BF16;
            o.maxSizes[0] = o.minSizes[0] = d0s[i];
            o.maxSizes[1] = o.minSizes[1] = d1s[i];
            o.maxSizes[2] = o.minSizes[2] = 24;
        }
        auto& o3 = out->outputTensors[3]->geometry;
        std::memset(&o3, 0, sizeof(TensorGeometry));
        o3.dims = 2; o3.dataType = DATA_F32;
        o3.maxSizes[0] = o3.minSizes[0] = 64;
        o3.maxSizes[1] = o3.minSizes[1] = 24;
        return GLUE_SUCCESS;
    }
    return GLUE_NODE_NOT_FOUND;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (p->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    const char* name = p->guid.name;
    if (!std::strcmp(name, "comfy_topk_s1_bmax")) {
        if (p->inputTensorNr != 3 || p->outputTensorNr != 1) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const TopkS1Params*>(p->nodeParams.nodeParams);
        const auto& lg = p->inputTensors[0].geometry;
        const auto& an = p->inputTensors[2].geometry;
        int vocab = int(lg.maxSizes[0]);
        int rows = int(lg.maxSizes[1]);
        int anc_w = int(an.maxSizes[0]);
        int num_blocks = (vocab + 127) / 128;
        int num_groups = (num_blocks + 63) / 64;
        uint32_t sp[] = {uint32_t(vocab), uint32_t(rows), uint32_t(anc_w), uint32_t(num_blocks),
                         uint32_t(prm->has_pen), f2u(prm->pres_pen), f2u(prm->rep_pen), f2u(prm->inv_rep_pen)};
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = num_groups;
        out->indexSpaceGeometry[1] = rows;
        set_all_required(out, 3, 1);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 8;
        return copy_elf(out, topk_s1_start, topk_s1_end);
    }
    if (!std::strcmp(name, "comfy_topk_s2_blocks")) {
        if (p->inputTensorNr != 1 || p->outputTensorNr != 1) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const TopkS2Params*>(p->nodeParams.nodeParams);
        const auto& bm = p->inputTensors[0].geometry;
        uint32_t sp[] = {uint32_t(bm.maxSizes[0]), uint32_t(bm.maxSizes[1]), uint32_t(prm->k)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = bm.maxSizes[1];
        set_all_required(out, 1, 1);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 3;
        return copy_elf(out, topk_s2_start, topk_s2_end);
    }
    if (!std::strcmp(name, "comfy_topk_s3_extract")) {
        if (p->inputTensorNr != 4 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const TopkS3Params*>(p->nodeParams.nodeParams);
        const auto& lg = p->inputTensors[0].geometry;
        const auto& an = p->inputTensors[2].geometry;
        uint32_t sp[] = {uint32_t(lg.maxSizes[0]), uint32_t(lg.maxSizes[1]), uint32_t(prm->k),
                         uint32_t(an.maxSizes[0]), uint32_t(prm->has_pen),
                         f2u(prm->pres_pen), f2u(prm->rep_pen), f2u(prm->inv_rep_pen)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = lg.maxSizes[1];
        set_all_required(out, 4, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 8;
        return copy_elf(out, topk_s3_start, topk_s3_end);
    }
    if (!std::strcmp(name, "comfy_gdn_rec_f32")) {
        if (p->inputTensorNr != 8 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const GdnRecParams*>(p->nodeParams.nodeParams);
        const auto& st = p->inputTensors[0].geometry;
        const auto& cc = p->inputTensors[2].geometry;
        if (st.dataType != DATA_F32 || st.dims != 4 || st.maxSizes[0] != 64 || st.maxSizes[1] != 32
            || (prm->v_hd % 64) != 0)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        int batches = int(st.maxSizes[3]);
        int rows = int(cc.maxSizes[1]);
        int sp[] = {prm->v_hd, prm->vh, prm->kh, batches, rows};
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = static_cast<uint64_t>(prm->vh) * static_cast<uint64_t>(prm->v_hd / 64);
        out->indexSpaceGeometry[1] = static_cast<uint64_t>(batches);
        out->preferredSplitDim = 1;
        set_all_required(out, 8, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 5;
        return copy_elf(out, gdn_rec_start, gdn_rec_end);
    }
    if (!std::strcmp(name, "comfy_gdn_conv_bf16")) {
        if (p->inputTensorNr != 7 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const GdnConvParams*>(p->nodeParams.nodeParams);
        const auto& cp = p->inputTensors[2].geometry;
        int channels = int(cp.maxSizes[0]);
        int rows = int(cp.maxSizes[1]);
        int batches = int(cp.maxSizes[2]);
        int sp[] = {channels, batches, rows, prm->has_bias};
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = (channels + 127) / 128;
        out->indexSpaceGeometry[1] = batches;
        set_all_required(out, 7, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 4;
        return copy_elf(out, gdn_conv_start, gdn_conv_end);
    }
    if (!std::strcmp(name, "comfy_rmsnorm_quant_fp8")) {
        if (p->inputTensorNr != 3 || p->outputTensorNr != 4) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const RmsQuantParams*>(p->nodeParams.nodeParams);
        const auto& xg = p->inputTensors[0].geometry;
        uint32_t sp[] = {uint32_t(xg.maxSizes[0]), uint32_t(xg.maxSizes[1]),
                         uint32_t(prm->has_res), uint32_t(prm->add_one),
                         uint32_t(prm->store_norm), f2u(prm->eps)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = xg.maxSizes[1];
        set_all_required(out, 3, 4);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 6;
        return copy_elf(out, rms_q_start, rms_q_end);
    }
    if (!std::strcmp(name, "comfy_gated_rmsnorm_quant_fp8")) {
        if (p->inputTensorNr != 3 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const GatedRmsQuantParams*>(p->nodeParams.nodeParams);
        const auto& zg = p->inputTensors[1].geometry;
        uint32_t sp[] = {uint32_t(prm->vh), uint32_t(zg.maxSizes[1]), f2u(prm->eps)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = zg.maxSizes[1];
        set_all_required(out, 3, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 3;
        return copy_elf(out, gated_rms_q_start, gated_rms_q_end);
    }
    if (!std::strcmp(name, "comfy_sig_mul_quant_fp8")) {
        if (p->inputTensorNr != 2 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto& xg = p->inputTensors[0].geometry;
        uint32_t sp[] = {uint32_t(xg.maxSizes[0]), uint32_t(xg.maxSizes[1])};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = xg.maxSizes[1];
        set_all_required(out, 2, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 2;
        return copy_elf(out, sig_mul_q_start, sig_mul_q_end);
    }
    if (!std::strcmp(name, "comfy_silu_mul_quant_s1")) {
        if (p->inputTensorNr != 2 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto& gg = p->inputTensors[0].geometry;
        int dim = int(gg.maxSizes[0]);
        int rows = int(gg.maxSizes[1]);
        int ch = silu_chunks(dim, rows);
        int csz = silu_chunk_sz(dim, ch);
        uint32_t sp[] = {uint32_t(dim), uint32_t(rows), uint32_t(ch), uint32_t(csz)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = static_cast<uint64_t>(ch) * static_cast<uint64_t>(rows);
        set_all_required(out, 2, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 4;
        return copy_elf(out, silu_mul_s1_start, silu_mul_s1_end);
    }
    if (!std::strcmp(name, "comfy_silu_mul_quant_s2")) {
        if (p->inputTensorNr != 2 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto& yg = p->inputTensors[0].geometry;
        int dim = int(yg.maxSizes[0]);
        int rows = int(yg.maxSizes[1]);
        int ch = silu_chunks(dim, rows);
        int csz = silu_chunk_sz(dim, ch);
        uint32_t sp[] = {uint32_t(dim), uint32_t(rows), uint32_t(ch), uint32_t(csz)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = static_cast<uint64_t>(ch) * static_cast<uint64_t>(rows);
        set_all_required(out, 2, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 4;
        return copy_elf(out, silu_mul_s2_start, silu_mul_s2_end);
    }
    if (!std::strcmp(name, "comfy_qk_norm_rope_bf16")) {
        if (p->inputTensorNr != 5 || p->outputTensorNr != 3) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const QkNormRopeParams*>(p->nodeParams.nodeParams);
        const auto& qgg = p->inputTensors[0].geometry;
        uint32_t sp[] = {uint32_t(prm->num_heads), uint32_t(prm->num_kv_heads),
                         uint32_t(qgg.maxSizes[1]), uint32_t(prm->add_one), f2u(prm->eps)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = static_cast<uint64_t>(prm->num_heads + prm->num_kv_heads);
        set_all_required(out, 5, 3);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 5;
        return copy_elf(out, qk_norm_rope_start, qk_norm_rope_end);
    }
    if (!std::strcmp(name, "comfy_sample_tree_accept_i32")) {
        if (p->inputTensorNr != 7 || p->outputTensorNr != 5) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const SampleTreeAcceptParams*>(p->nodeParams.nodeParams);
        const auto& ti = p->inputTensors[1].geometry;
        const auto& tp = p->inputTensors[3].geometry;
        int k = int(ti.maxSizes[0]);
        int rows = int(ti.maxSizes[1]);
        int topk_k = int(tp.maxSizes[0]);
        uint32_t sp[] = {uint32_t(rows), uint32_t(k), uint32_t(prm->depth), uint32_t(topk_k),
                         uint32_t(prm->do_sample), uint32_t(prm->adaptive),
                         f2u(prm->inv_temp), f2u(prm->top_p), f2u(prm->min_p)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = 1;
        set_all_required(out, 7, 5);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 9;
        return copy_elf(out, sample_accept_start, sample_accept_end);
    }
    if (!std::strcmp(name, "comfy_dual_rmsnorm_cat_quant_fp8")) {
        if (p->inputTensorNr != 4 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const DualRmsCatQuantParams*>(p->nodeParams.nodeParams);
        const auto& eg = p->inputTensors[0].geometry;
        uint32_t sp[] = {uint32_t(eg.maxSizes[0]), uint32_t(eg.maxSizes[1]),
                         uint32_t(prm->add_one), f2u(prm->eps)};
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = eg.maxSizes[1];
        set_all_required(out, 4, 2);
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 4;
        return copy_elf(out, dual_rms_cat_start, dual_rms_cat_end);
    }
    if (!std::strcmp(name, "comfy_moss_situ_quant_fp8")) {
        if (p->inputTensorNr != 1 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto& yg = p->inputTensors[0].geometry;
        uint64_t width = yg.maxSizes[0];
        uint64_t dim = width / 2;
        uint64_t rows = yg.maxSizes[1];
        if (yg.dims != 2 || yg.dataType != DATA_BF16 || (width & 1) || dim == 0 || (dim % 256) != 0 || dim > 8192)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = (rows + 63) / 64;
        out->auxiliaryTensorNr = 0;
        out->preferredSplitDim = 0;
        auto& ia = out->inputTensorAccessPattern[0];
        std::memset(&ia, 0, sizeof(TensorAccessPattern));
        ia.allRequired = 0;
        ia.fullyAccessedOnce = 1;
        ia.mapping[0] = {0, 0.0f, 0.0f, float(width - 1), true};
        ia.mapping[1] = {0, 64.0f, 0.0f, 63.0f, false};
        auto& o0 = out->outputTensorAccessPattern[0];
        std::memset(&o0, 0, sizeof(TensorAccessPattern));
        o0.allRequired = 0;
        o0.noRmwAccess = 1;
        o0.fullyWritten = 1;
        o0.mapping[0] = {0, 0.0f, 0.0f, float(dim - 1), true};
        o0.mapping[1] = {0, 64.0f, 0.0f, 63.0f, false};
        auto& o1 = out->outputTensorAccessPattern[1];
        std::memset(&o1, 0, sizeof(TensorAccessPattern));
        o1.allRequired = 0;
        o1.noRmwAccess = 1;
        o1.fullyWritten = 1;
        o1.mapping[0] = {0, 64.0f, 0.0f, 63.0f, false};
        uint32_t sp[] = {uint32_t(dim), uint32_t(rows)};
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 2;
        return copy_elf(out, moss_situ_q_start, moss_situ_q_end);
    }
    if (!std::strcmp(name, "comfy_moss_qkv_prep_fp8")) {
        if (p->inputTensorNr != 4 || p->outputTensorNr != 4) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const auto* prm = static_cast<const MossQkvPrepParams*>(p->nodeParams.nodeParams);
        const auto& qg = p->inputTensors[0].geometry;
        uint64_t rows = qg.maxSizes[1];
        if (qg.dims != 2 || qg.dataType != DATA_BF16 || qg.maxSizes[0] != 9216 || rows == 0)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t k_pad = ((rows + 127) / 128) * 128;
        int is_aligned = (k_pad == rows) ? 1 : 0;
        if ((prm->aligned ? 1 : 0) != is_aligned)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        uint64_t qk_dim = is_aligned ? 128 : 192;
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = 24;
        out->auxiliaryTensorNr = 0;
        out->preferredSplitDim = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            std::memset(&out->inputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
            out->inputTensorAccessPattern[i].allRequired = 1;
        }
        uint64_t d0s[3] = {qk_dim, qk_dim, 128};
        uint64_t d1s[3] = {rows, k_pad, k_pad};
        for (uint32_t i = 0; i < 3; ++i) {
            auto& oa = out->outputTensorAccessPattern[i];
            std::memset(&oa, 0, sizeof(TensorAccessPattern));
            oa.allRequired = 0;
            oa.noRmwAccess = 1;
            oa.fullyWritten = 1;
            oa.mapping[0] = {0, 0.0f, 0.0f, float(d0s[i] - 1), true};
            oa.mapping[1] = {0, 0.0f, 0.0f, float(d1s[i] - 1), true};
            oa.mapping[2] = {0, 1.0f, 0.0f, 0.0f, false};
        }
        auto& os = out->outputTensorAccessPattern[3];
        std::memset(&os, 0, sizeof(TensorAccessPattern));
        os.allRequired = 0;
        os.noRmwAccess = 1;
        os.fullyWritten = 1;
        os.mapping[0] = {0, 0.0f, 0.0f, 63.0f, true};
        os.mapping[1] = {0, 1.0f, 0.0f, 0.0f, false};
        uint32_t sp[] = {uint32_t(rows), uint32_t(k_pad), uint32_t(qk_dim),
                         f2u(prm->eps), f2u(prm->q_scale), f2u(prm->k_scale)};
        std::memcpy(out->kernel.scalarParams, sp, sizeof(sp));
        out->kernel.paramsNr = 6;
        return copy_elf(out, moss_qkv_prep_start, moss_qkv_prep_end);
    }
    return GLUE_NODE_NOT_FOUND;
}
