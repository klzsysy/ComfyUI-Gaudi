"""On-chip RMW SRAM attention pipeline for Gaudi2 HPU."""
from collections import OrderedDict
import ctypes as C
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import torch


_RMW_CPP_SOURCE = r"""
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <perf_lib_layer_params.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Binding {
    std::string name;
    unsigned buffer;
    uint64_t offset;
    uint64_t id = 0;
    unsigned dims;
    uint64_t sizes[3];
};

struct RMWPlan {
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::vector<synSectionHandle> sections;
    std::vector<Binding> bindings;
    std::vector<synLaunchTensorInfo> launch_info;
    uint64_t workspace = 0;
    unsigned group_heads = 1;
    uint64_t head_strides[6] = {};

    ~RMWPlan() {
        if (recipe) synRecipeDestroy(recipe);
        if (graph) synGraphDestroy(graph);
        for (auto s : sections) synSectionDestroy(s);
    }
};

static void check(synStatus status, const char* op) {
    if (status != synSuccess) {
        throw std::runtime_error(std::string(op) + " returned " + std::to_string(status));
    }
}

static synSectionHandle make_section(RMWPlan& plan, bool persistent, bool rmw) {
    synSectionHandle s = nullptr;
    check(synSectionCreate(&s, 0, plan.graph), "section");
    plan.sections.push_back(s);
    check(synSectionSetPersistent(s, persistent), "persistent");
    if (rmw) check(synSectionSetRMW(s, true), "rmw");
    return s;
}

static synTensor make_tensor(RMWPlan& plan, const std::string& name, unsigned dims,
                             uint64_t s0, uint64_t s1, uint64_t s2,
                             synDataType dtype, synSectionHandle sec, uint64_t offset) {
    synTensor t = nullptr;
    check(synTensorHandleCreate(&t, plan.graph, DATA_TENSOR, name.c_str()), "tensor");
    synTensorGeometry g{};
    g.dims = dims;
    g.sizes[0] = s0;
    g.sizes[1] = s1;
    g.sizes[2] = s2;
    check(synTensorSetGeometry(t, &g, synGeometryMaxSizes), "geometry");
    check(synTensorSetDeviceDataType(t, dtype), "dtype");
    if (sec) check(synTensorAssignToSection(t, sec, offset), "assign");
    return t;
}

static synTensor make_persistent(RMWPlan& plan, const std::string& name, unsigned buffer, uint64_t offset,
                                 unsigned dims, uint64_t s0, uint64_t s1, uint64_t s2,
                                 synDataType dtype, synSectionHandle sec) {
    auto t = make_tensor(plan, name, dims, s0, s1, s2, dtype, sec, offset);
    Binding b{name, buffer, offset, 0, dims, {s0, s1, s2}};
    plan.bindings.push_back(b);
    return t;
}

extern "C" void* build_rmw_plan(unsigned queries, unsigned keys, unsigned group_heads,
                                unsigned tile, unsigned slots, int retained,
                                const char* recipe_name) {
    try {
        if (!queries || !keys || !group_heads || !tile || !slots) {
            throw std::runtime_error("invalid geometry");
        }
        const uint64_t score_bytes = ((uint64_t)keys * tile * 2 + 255ULL) & ~255ULL;
        if (slots * score_bytes > 16ULL * 1024ULL * 1024ULL) {
            throw std::runtime_error("RMW capacity exceeded");
        }

        auto plan = std::make_unique<RMWPlan>();
        plan->group_heads = group_heads;
        plan->head_strides[0] = (uint64_t)queries * 128 * 2;
        plan->head_strides[1] = (uint64_t)keys * 128 * 2;
        plan->head_strides[2] = (uint64_t)keys * 128 * 2;
        plan->head_strides[3] = (uint64_t)queries * 128 * 2;
        const uint64_t stat_head_stride = ((uint64_t)queries * 2 + 127ULL) & ~127ULL;
        plan->head_strides[4] = stat_head_stride;
        plan->head_strides[5] = stat_head_stride;

        check(synGraphCreate(&plan->graph, synDeviceGaudi2), "graph");
        std::vector<synSectionHandle> io_secs;
        for (unsigned i = 0; i < 6; ++i) io_secs.push_back(make_section(*plan, true, false));
        synSectionHandle rmw_sec = make_section(*plan, false, true);

        unsigned tiles_per_head = (queries + tile - 1) / tile;
        unsigned num_ops = group_heads * tiles_per_head;
        unsigned split_axis1 = 1;
        unsigned split_axis0 = 0;

        auto q_full = make_persistent(*plan, "query", 0, 0, 2, 128, queries * group_heads, 1, syn_type_bf16, io_secs[0]);
        auto k_full = make_persistent(*plan, "key", 1, 0, 2, 128, keys * group_heads, 1, syn_type_bf16, io_secs[1]);
        auto v_full = make_persistent(*plan, "val", 2, 0, 2, 128, keys * group_heads, 1, syn_type_bf16, io_secs[2]);
        auto o_full = make_persistent(*plan, "output", 3, 0, 2, 128, queries * group_heads, 1, syn_type_bf16, io_secs[3]);

        std::vector<synTensor> k_h(group_heads), v_h(group_heads);
        std::vector<synTensor> m_h(group_heads, nullptr), inv_h(group_heads, nullptr);
        if (retained) {
            for (unsigned h = 0; h < group_heads; ++h) {
                m_h[h] = make_persistent(*plan, "maximum_" + std::to_string(h), 4, h * stat_head_stride,
                                         2, queries, 1, 1, syn_type_bf16, io_secs[4]);
                inv_h[h] = make_persistent(*plan, "inverse_" + std::to_string(h), 5, h * stat_head_stride,
                                           2, queries, 1, 1, syn_type_bf16, io_secs[5]);
            }
        }

        if (group_heads == 1) {
            k_h[0] = k_full;
            v_h[0] = v_full;
        } else {
            for (unsigned h = 0; h < group_heads; ++h) {
                k_h[h] = make_tensor(*plan, "k_h" + std::to_string(h), 2, 128, keys, 1, syn_type_bf16, nullptr, 0);
                v_h[h] = make_tensor(*plan, "v_h" + std::to_string(h), 2, 128, keys, 1, syn_type_bf16, nullptr, 0);
            }
            check(synNodeCreate(plan->graph, &k_full, k_h.data(), 1, group_heads, &split_axis1, sizeof(split_axis1), "split", "split_k", nullptr, nullptr), "split_k");
            check(synNodeCreate(plan->graph, &v_full, v_h.data(), 1, group_heads, &split_axis1, sizeof(split_axis1), "split", "split_v", nullptr, nullptr), "split_v");
        }

        std::vector<synTensor> q_t(num_ops), o_t(num_ops), m_t(num_ops), inv_t(num_ops);
        std::vector<unsigned> op_head(num_ops), op_rows(num_ops);
        unsigned op = 0;
        for (unsigned h = 0; h < group_heads; ++h) {
            for (unsigned start = 0; start < queries; start += tile, ++op) {
                unsigned rows = std::min(tile, queries - start);
                std::string suf = std::to_string(op);
                q_t[op] = make_tensor(*plan, "q_" + suf, 2, 128, rows, 1, syn_type_bf16, nullptr, 0);
                o_t[op] = make_tensor(*plan, "o_" + suf, 2, 128, rows, 1, syn_type_bf16, nullptr, 0);
                if (retained) {
                    m_t[op] = make_tensor(*plan, "m_" + suf, 2, rows, 1, 1, syn_type_bf16, nullptr, 0);
                    inv_t[op] = make_tensor(*plan, "inv_" + suf, 2, rows, 1, 1, syn_type_bf16, nullptr, 0);
                }
                op_head[op] = h;
                op_rows[op] = rows;
            }
        }

        if (num_ops == 1) {
            q_t[0] = q_full;
            o_t[0] = o_full;
            if (retained) {
                m_t[0] = m_h[0];
                inv_t[0] = inv_h[0];
            }
        } else {
            check(synNodeCreate(plan->graph, &q_full, q_t.data(), 1, num_ops, &split_axis1, sizeof(split_axis1), "split", "split_q", nullptr, nullptr), "split_q");
            check(synNodeCreate(plan->graph, o_t.data(), &o_full, num_ops, 1, &split_axis1, sizeof(split_axis1), "concat", "concat_o", nullptr, nullptr), "concat_o");
            if (retained) {
                for (unsigned h = 0; h < group_heads; ++h) {
                    if (tiles_per_head == 1) {
                        m_t[h] = m_h[h];
                        inv_t[h] = inv_h[h];
                    } else {
                        check(synNodeCreate(plan->graph, &m_t[h * tiles_per_head], &m_h[h], tiles_per_head, 1, &split_axis0, sizeof(split_axis0), "concat", ("concat_m_" + std::to_string(h)).c_str(), nullptr, nullptr), "concat_m");
                        check(synNodeCreate(plan->graph, &inv_t[h * tiles_per_head], &inv_h[h], tiles_per_head, 1, &split_axis0, sizeof(split_axis0), "concat", ("concat_inv_" + std::to_string(h)).c_str(), nullptr, nullptr), "concat_inv");
                    }
                }
            }
        }

        std::vector<synTensor> s_t(num_ops), sv_t(num_ops), pv_t(num_ops), p_t(num_ops);
        for (unsigned i = 0; i < num_ops; ++i) {
            unsigned rows = op_rows[i];
            unsigned slot = i % slots;
            uint64_t base_off = slot * score_bytes;
            std::string suf = std::to_string(i);
            s_t[i] = make_tensor(*plan, "s_" + suf, 2, keys, rows, 1, syn_type_bf16, rmw_sec, base_off);
            sv_t[i] = make_tensor(*plan, "sv_" + suf, 3, keys, rows, 1, syn_type_bf16, nullptr, 0);
            pv_t[i] = make_tensor(*plan, "pv_" + suf, 3, keys, rows, 1, syn_type_bf16, nullptr, 0);
            p_t[i] = make_tensor(*plan, "p_" + suf, 2, keys, rows, 1, syn_type_bf16, rmw_sec, base_off);
        }

        synGEMMParams qk_p(false, true), pv_p(false, false);
        ns_Softmax::ParamsV8 accelerated{};
        accelerated.dim = 0;
        accelerated.dataTypeSize = DATA_32BIT;
        accelerated.validCountAxis = -1;
        accelerated.mode = static_cast<SoftmaxMode_t>(SOFTMAX_HF8_1C | (retained ? SQUEEZED_RETAINED : 0));

        std::vector<synNodeId> qk_id(num_ops), sm_id(num_ops), pv_id(num_ops);
        auto emit_qk = [&](unsigned i) {
            std::string suf = std::to_string(i);
            synTensor qk_in[] = {q_t[i], k_h[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, qk_in, &s_t[i], 2, 1, &qk_p, sizeof(qk_p), "gemm", ("sdpa_recomp_rmw_qk_" + suf).c_str(), &qk_id[i], nullptr, nullptr), "qk");
            check(synNodeCreate(plan->graph, &s_t[i], &sv_t[i], 1, 1, nullptr, 0, "reshape", ("r1_" + suf).c_str(), nullptr, nullptr), "r1");
        };
        auto emit_sm_pv = [&](unsigned i) {
            std::string suf = std::to_string(i);
            std::vector<synTensor> sm_out{pv_t[i]};
            if (retained) {
                sm_out.push_back(m_t[i]);
                sm_out.push_back(inv_t[i]);
            }
            check(synNodeCreateWithId(plan->graph, &sv_t[i], sm_out.data(), 1, sm_out.size(), &accelerated, sizeof(accelerated), "softmax_fwd_hf8", ("softmax_" + suf).c_str(), &sm_id[i], nullptr, nullptr), "softmax");
            check(synNodeCreate(plan->graph, &pv_t[i], &p_t[i], 1, 1, nullptr, 0, "reshape", ("r2_" + suf).c_str(), nullptr, nullptr), "r2");
            synTensor pv_in[] = {p_t[i], v_h[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, pv_in, &o_t[i], 2, 1, &pv_p, sizeof(pv_p), "gemm", ("sdpa_recomp_rmw_pv_" + suf).c_str(), &pv_id[i], nullptr, nullptr), "pv");
        };

        if (slots >= 2) {
            emit_qk(0);
            for (unsigned i = 1; i < num_ops; ++i) {
                emit_qk(i);
                emit_sm_pv(i - 1);
            }
            emit_sm_pv(num_ops - 1);
        } else {
            for (unsigned i = 0; i < num_ops; ++i) {
                emit_qk(i);
                emit_sm_pv(i);
            }
        }

        for (unsigned i = 0; i < num_ops; ++i) {
            if (i >= slots) {
                check(synNodeDependencySet(plan->graph, &pv_id[i - slots], &qk_id[i], 1, 1), "slot_reuse");
            }
            if (slots >= 2 && i >= 1) {
                check(synNodeDependencySet(plan->graph, &qk_id[i - 1], &qk_id[i], 1, 1), "qk_order");
                check(synNodeDependencySet(plan->graph, &qk_id[i], &pv_id[i - 1], 1, 1), "qk_before_prev_pv");
                check(synNodeDependencySet(plan->graph, &pv_id[i - 1], &pv_id[i], 1, 1), "pv_order");
            }
        }

        check(synGraphCompile(&plan->recipe, plan->graph, recipe_name, nullptr), "compile");
        check(synWorkspaceGetSize(&plan->workspace, plan->recipe), "workspace");

        std::vector<const char*> names;
        std::vector<uint64_t> ids(plan->bindings.size());
        for (const auto& b : plan->bindings) names.push_back(b.name.c_str());
        check(synTensorRetrieveIds(plan->recipe, names.data(), ids.data(), ids.size()), "tensor_ids");
        plan->launch_info.resize(plan->bindings.size());
        for (size_t i = 0; i < plan->bindings.size(); ++i) {
            plan->bindings[i].id = ids[i];
            synLaunchTensorInfo info{};
            info.tensorName = plan->bindings[i].name.c_str();
            info.tensorId = ids[i];
            info.tensorType = DATA_TENSOR;
            for (unsigned d = 0; d < plan->bindings[i].dims; ++d) {
                info.tensorSize[d] = plan->bindings[i].sizes[d];
            }
            plan->launch_info[i] = info;
        }
        return plan.release();
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" void* build_rmw_fp8_plan(unsigned queries, unsigned keys, unsigned qk_dim,
                                    unsigned group_heads, unsigned tile, unsigned slots,
                                    const char* recipe_name) {
    try {
        const uint64_t score_bytes = ((uint64_t)keys * tile * 2 + 255ULL) & ~255ULL;
        if (slots * score_bytes > 16ULL * 1024ULL * 1024ULL) return nullptr;

        auto plan = std::make_unique<RMWPlan>();
        plan->group_heads = group_heads;
        plan->head_strides[0] = (uint64_t)queries * qk_dim * 2;
        plan->head_strides[1] = (uint64_t)keys * qk_dim * 2;
        plan->head_strides[2] = (uint64_t)keys * 128 * 1;
        plan->head_strides[3] = (uint64_t)queries * 128 * 2;
        plan->head_strides[4] = 0;
        plan->head_strides[5] = 0;

        check(synGraphCreate(&plan->graph, synDeviceGaudi2), "graph_create");
        std::vector<synSectionHandle> io;
        for (int i = 0; i < 6; ++i) io.push_back(make_section(*plan, true, false));
        synSectionHandle rmw_sec = make_section(*plan, false, true);

        const unsigned tiles_per_head = (queries + tile - 1) / tile;
        const unsigned nops = group_heads * tiles_per_head;
        unsigned axis1 = 1;

        synTensor q_full = make_persistent(*plan, "q", 0, 0, 2, qk_dim, queries * group_heads, 1, syn_type_bf16, io[0]);
        synTensor k_full = make_persistent(*plan, "k", 1, 0, 2, qk_dim, keys * group_heads, 1, syn_type_bf16, io[1]);
        synTensor v_full = make_persistent(*plan, "v", 2, 0, 2, 128, keys * group_heads, 1, syn_type_fp8_143, io[2]);
        synTensor o_full = make_persistent(*plan, "o", 3, 0, 2, 128, queries * group_heads, 1, syn_type_bf16, io[3]);
        synTensor sin_t = make_persistent(*plan, "sin", 4, 0, 1, 1, 1, 1, syn_type_float, io[4]);
        synTensor sout_t = make_persistent(*plan, "sout", 5, 0, 1, 1, 1, 1, syn_type_float, io[5]);

        std::vector<synTensor> k_heads(group_heads), v_heads(group_heads);
        if (group_heads == 1) {
            k_heads[0] = k_full;
            v_heads[0] = v_full;
        } else {
            for (unsigned h = 0; h < group_heads; ++h) {
                k_heads[h] = make_tensor(*plan, "k" + std::to_string(h), 2, qk_dim, keys, 1, syn_type_bf16, nullptr, 0);
                v_heads[h] = make_tensor(*plan, "v" + std::to_string(h), 2, 128, keys, 1, syn_type_fp8_143, nullptr, 0);
            }
            check(synNodeCreate(plan->graph, &k_full, k_heads.data(), 1, group_heads, &axis1, sizeof(axis1),
                                "split", "split_k", nullptr, nullptr), "split_k");
            check(synNodeCreate(plan->graph, &v_full, v_heads.data(), 1, group_heads, &axis1, sizeof(axis1),
                                "split", "split_v", nullptr, nullptr), "split_v");
        }

        std::vector<synTensor> q_tiles(nops), o_tiles(nops);
        std::vector<unsigned> op_head(nops), op_rows(nops);
        unsigned op = 0;
        for (unsigned h = 0; h < group_heads; ++h) {
            for (unsigned start = 0; start < queries; start += tile, ++op) {
                unsigned rows = std::min(tile, queries - start);
                std::string suffix = std::to_string(op);
                q_tiles[op] = make_tensor(*plan, "q" + suffix, 2, qk_dim, rows, 1, syn_type_bf16, nullptr, 0);
                o_tiles[op] = make_tensor(*plan, "o" + suffix, 2, 128, rows, 1, syn_type_bf16, nullptr, 0);
                op_head[op] = h;
                op_rows[op] = rows;
            }
        }
        check(synNodeCreate(plan->graph, &q_full, q_tiles.data(), 1, nops, &axis1, sizeof(axis1),
                            "split", "split_q", nullptr, nullptr), "split_q");
        check(synNodeCreate(plan->graph, o_tiles.data(), &o_full, nops, 1, &axis1, sizeof(axis1),
                            "concat", "concat_o", nullptr, nullptr), "concat_o");

        std::vector<synTensor> s_t(nops), sv_t(nops), pv_t(nops), p_t(nops);
        for (unsigned i = 0; i < nops; ++i) {
            unsigned rows = op_rows[i];
            uint64_t off = (i % slots) * score_bytes;
            std::string suffix = std::to_string(i);
            s_t[i] = make_tensor(*plan, "s" + suffix, 2, keys, rows, 1, syn_type_bf16, rmw_sec, off);
            sv_t[i] = make_tensor(*plan, "sv" + suffix, 3, keys, rows, 1, syn_type_bf16, nullptr, 0);
            pv_t[i] = make_tensor(*plan, "pv" + suffix, 3, keys, rows, 1, syn_type_fp8_143, nullptr, 0);
            p_t[i] = make_tensor(*plan, "p" + suffix, 2, keys, rows, 1, syn_type_fp8_143, rmw_sec, off);
        }

        synGEMMParams qk_params(false, true), av_params(false, false);
        ns_Softmax::ParamsV8 sm_params{};
        sm_params.dim = 0;
        sm_params.dataTypeSize = DATA_32BIT;
        sm_params.validCountAxis = -1;
        sm_params.mode = SOFTMAX_HF8_1B;

        std::vector<synNodeId> qk_ids(nops), sm_ids(nops), av_ids(nops);
        auto emit_qk = [&](unsigned i) {
            std::string suffix = std::to_string(i);
            synTensor qk_in[] = {q_tiles[i], k_heads[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, qk_in, &s_t[i], 2, 1, &qk_params, sizeof(qk_params),
                                      "gemm", ("qk" + suffix).c_str(), &qk_ids[i], nullptr, nullptr), "qk");
            check(synNodeCreate(plan->graph, &s_t[i], &sv_t[i], 1, 1, nullptr, 0,
                                "reshape", ("r1_" + suffix).c_str(), nullptr, nullptr), "r1");
        };
        auto emit_sp = [&](unsigned i) {
            std::string suffix = std::to_string(i);
            synTensor sm_in[] = {sv_t[i], sin_t, sout_t};
            check(synNodeCreateWithId(plan->graph, sm_in, &pv_t[i], 3, 1, &sm_params, sizeof(sm_params),
                                      "softmax_fwd_hf8", ("sm" + suffix).c_str(), &sm_ids[i], nullptr, nullptr), "sm");
            check(synNodeCreate(plan->graph, &pv_t[i], &p_t[i], 1, 1, nullptr, 0,
                                "reshape", ("r2_" + suffix).c_str(), nullptr, nullptr), "r2");
            synTensor av_in[] = {p_t[i], v_heads[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, av_in, &o_tiles[i], 2, 1, &av_params, sizeof(av_params),
                                      "gemm", ("av" + suffix).c_str(), &av_ids[i], nullptr, nullptr), "av");
        };

        emit_qk(0);
        for (unsigned i = 1; i < nops; ++i) {
            emit_qk(i);
            emit_sp(i - 1);
        }
        emit_sp(nops - 1);

        for (unsigned i = 0; i < nops; ++i) {
            if (i >= slots) {
                check(synNodeDependencySet(plan->graph, &av_ids[i - slots], &qk_ids[i], 1, 1), "slot_reuse");
            }
            if (i >= 1) {
                check(synNodeDependencySet(plan->graph, &qk_ids[i - 1], &qk_ids[i], 1, 1), "qk_order");
                check(synNodeDependencySet(plan->graph, &qk_ids[i], &av_ids[i - 1], 1, 1), "qk_before_av");
                check(synNodeDependencySet(plan->graph, &av_ids[i - 1], &av_ids[i], 1, 1), "av_order");
            }
        }

        check(synGraphCompile(&plan->recipe, plan->graph, recipe_name, nullptr), "compile");
        check(synWorkspaceGetSize(&plan->workspace, plan->recipe), "workspace");

        std::vector<const char*> names;
        std::vector<uint64_t> ids(plan->bindings.size());
        for (const auto& b : plan->bindings) names.push_back(b.name.c_str());
        check(synTensorRetrieveIds(plan->recipe, names.data(), ids.data(), ids.size()), "tensor_ids");
        plan->launch_info.resize(plan->bindings.size());
        for (size_t i = 0; i < plan->bindings.size(); ++i) {
            plan->bindings[i].id = ids[i];
            synLaunchTensorInfo info{};
            info.tensorName = plan->bindings[i].name.c_str();
            info.tensorId = ids[i];
            info.tensorType = DATA_TENSOR;
            for (unsigned d = 0; d < plan->bindings[i].dims; ++d) {
                info.tensorSize[d] = plan->bindings[i].sizes[d];
            }
            plan->launch_info[i] = info;
        }
        return plan.release();
    } catch (const std::exception&) {
        return nullptr;
    }
}

extern "C" uint64_t rmw_plan_workspace_bytes(void* handle) {
    return static_cast<RMWPlan*>(handle)->workspace;
}

extern "C" int launch_rmw_plan(void* handle, void* stream, const uint64_t* base_addresses,
                               uint64_t workspace, unsigned total_heads) {
    auto& plan = *static_cast<RMWPlan*>(handle);
    const size_t n = plan.bindings.size();
    synLaunchTensorInfo* info = plan.launch_info.data();
    for (unsigned h = 0; h < total_heads; h += plan.group_heads) {
        for (size_t i = 0; i < n; ++i) {
            const auto& b = plan.bindings[i];
            info[i].pTensorAddress = base_addresses[b.buffer] + b.offset + (uint64_t)h * plan.head_strides[b.buffer];
        }
        synStatus st = synLaunch(static_cast<synStreamHandle>(stream), info, n, workspace, plan.recipe, 0);
        if (st != synSuccess) return static_cast<int>(st);
    }
    return 0;
}

extern "C" void destroy_rmw_plan(void* handle) {
    delete static_cast<RMWPlan*>(handle);
}
"""


def _prep_aligned(q, k, v):
    scale = torch.full((), q.shape[-1] ** -0.5, dtype=q.dtype, device=q.device)
    qh = (q * scale).contiguous()
    kh = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype).contiguous()
    vh = v.contiguous()
    return qh, kh, vh


def _prep_split_and_tail(q, k, v):
    # Compute 128-aligned split symbolically inside the compiled region so varying
    # caption lengths share one dynamic graph without hitting recompile_limit.
    split = (k.shape[-2] // 128) * 128
    scale = torch.full((), q.shape[-1] ** -0.5, dtype=q.dtype, device=q.device)
    qh = (q * scale).contiguous()
    kh = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    k1 = kh[..., :split, :].contiguous()
    v1 = v[..., :split, :].contiguous()
    k2 = kh[..., split:, :].contiguous()
    v2 = v[..., split:, :].contiguous()
    last, max2, inv2 = torch.ops.hpu.sdpa_recomp_fwd(
        qh, k2, v2, None, 0.0, 1.0, False, True, "fast", None, "right"
    )[:3]
    return qh, k1, v1, last, max2, inv2


def _merge_tail(first, max1, inv1, last, max2, inv2):
    m1 = max1.float()
    m2 = max2.float()
    maximum = torch.maximum(m1, m2)
    w1 = (m1 - maximum).exp() / inv1.float()
    w2 = (m2 - maximum).exp() / inv2.float()
    share = (w2 / (w1 + w2)).to(first.dtype)
    merged = torch.addcmul(first, last - first, share)
    return merged.transpose(1, 2).contiguous().transpose(1, 2)


def _transpose_nhd(out):
    return out.transpose(1, 2).contiguous().transpose(1, 2)


@lru_cache(maxsize=1)
def _compiled_helpers():
    options = {"force_static_compile": True}
    return (
        torch.compile(_prep_aligned, backend="hpu_backend", dynamic=None, options=options),
        torch.compile(_prep_split_and_tail, backend="hpu_backend", dynamic=None, options=options),
        torch.compile(_merge_tail, backend="hpu_backend", dynamic=None, options=options),
        torch.compile(_transpose_nhd, backend="hpu_backend", dynamic=None, options=options),
    )


def _load_rmw_library():
    digest = hashlib.sha256(_RMW_CPP_SOURCE.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir())
    so_path = cache_dir / f"comfy_hpu_rmw_attn_{digest}.so"
    if not so_path.exists():
        pid = os.getpid()
        cpp_tmp = cache_dir / f"comfy_hpu_rmw_attn_{digest}.{pid}.cpp"
        so_tmp = cache_dir / f"comfy_hpu_rmw_attn_{digest}.{pid}.so"
        try:
            cpp_tmp.write_text(_RMW_CPP_SOURCE, encoding="utf-8")
            subprocess.run(
                [
                    "g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                    "-I/usr/include/habanalabs",
                    str(cpp_tmp),
                    "-L/usr/lib/habanalabs", "-lSynapse",
                    "-o", str(so_tmp),
                ],
                check=True,
            )
            os.replace(so_tmp, so_path)
        finally:
            cpp_tmp.unlink(missing_ok=True)
            so_tmp.unlink(missing_ok=True)
    lib = C.CDLL(str(so_path))
    lib.build_rmw_plan.argtypes = [
        C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_int, C.c_char_p
    ]
    lib.build_rmw_plan.restype = C.c_void_p
    lib.build_rmw_fp8_plan.argtypes = [
        C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_char_p
    ]
    lib.build_rmw_fp8_plan.restype = C.c_void_p
    lib.rmw_plan_workspace_bytes.argtypes = [C.c_void_p]
    lib.rmw_plan_workspace_bytes.restype = C.c_uint64
    lib.launch_rmw_plan.argtypes = [
        C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.c_uint64, C.c_uint
    ]
    lib.launch_rmw_plan.restype = C.c_int
    lib.destroy_rmw_plan.argtypes = [C.c_void_p]
    lib.destroy_rmw_plan.restype = None
    return lib


class _RMWAttentionRuntime:
    def __init__(self, group_heads: int = 4, slots: int = 2, max_plans: int = 32):
        from habana_frameworks.torch.utils import _debug_eager_C
        from habana_frameworks.torch.utils.experimental import _compute_stream, _data_ptr

        self._debug_eager_C = _debug_eager_C
        self._compute_stream = _compute_stream
        self._data_ptr = _data_ptr
        self.lib = _load_rmw_library()
        self.group_heads = group_heads
        self.slots = slots
        self.max_plans = max_plans
        self.plans: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        self.ws: torch.Tensor | None = None
        self.last_workspace: int = 0

    def _address(self, tensor: torch.Tensor) -> int:
        return self._data_ptr(tensor) + tensor.storage_offset() * tensor.element_size()

    def get_plan(self, queries: int, aligned_keys: int, retained: int):
        key = (queries, aligned_keys, retained)
        plan = self.plans.get(key)
        if plan is not None:
            self.plans.move_to_end(key)
            return plan
        step = 64 if retained else 8
        max_tile = 448 if retained else 480
        tile = min(max_tile, (queries // step) * step or queries)
        while tile > 64 and self.slots * ((aligned_keys * tile * 2 + 255) & ~255) > 16 * 1024 * 1024:
            tile -= step
        if self.slots * ((aligned_keys * tile * 2 + 255) & ~255) > 16 * 1024 * 1024:
            return None
        name = f"comfy_rmw_q{queries}_k{aligned_keys}_r{retained}_g{self.group_heads}_s{self.slots}_t{tile}"
        plan = self.lib.build_rmw_plan(
            queries, aligned_keys, self.group_heads, tile, self.slots, retained, name.encode("utf-8")
        )
        if not plan:
            return None
        while len(self.plans) >= self.max_plans:
            _, evicted = self.plans.popitem(last=False)
            torch.hpu.synchronize()
            self.lib.destroy_rmw_plan(evicted)
        self.plans[key] = plan
        return plan

    def get_workspace(self, ws_bytes: int, device) -> torch.Tensor:
        needed = max(256, int(ws_bytes))
        if self.ws is None or self.ws.numel() < needed or self.ws.device != device:
            self.ws = torch.empty(needed, dtype=torch.uint8, device=device)
        return self.ws

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fallback_fn) -> torch.Tensor:
        queries = q.shape[-2]
        full_keys = k.shape[-2]
        aligned_keys = (full_keys // 128) * 128
        retained = 0 if aligned_keys == full_keys else 1
        plan = self.get_plan(queries, aligned_keys, retained)
        if plan is None:
            k_centered = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
            return fallback_fn(q.contiguous(), k_centered, v)

        prep_aligned, prep_split, merge_tail, transpose_nhd = _compiled_helpers()
        ws_bytes = int(self.lib.rmw_plan_workspace_bytes(plan))
        self.last_workspace = ws_bytes
        ws = self.get_workspace(ws_bytes, q.device)

        if not retained:
            qh, kh, vh = prep_aligned(q, k, v)
            out = torch.empty_like(qh)
            self._debug_eager_C.join_pending_pipeline_threads()
            ptrs = (C.c_uint64 * 6)(
                self._address(qh), self._address(kh), self._address(vh), self._address(out), 0, 0
            )
            st = self.lib.launch_rmw_plan(plan, self._compute_stream(), ptrs, self._address(ws), 32)
            if st != 0:
                raise RuntimeError(f"launch_rmw_plan aligned failed with synStatus={st}")
            return transpose_nhd(out)

        qh, k1, v1, last, max2, inv2 = prep_split(q, k, v)
        first = torch.empty_like(qh)
        stat_queries = ((queries + 63) // 64) * 64
        max1 = torch.empty(1, 32, stat_queries, 1, device=qh.device, dtype=qh.dtype)
        inv1 = torch.empty_like(max1)
        self._debug_eager_C.join_pending_pipeline_threads()
        ptrs = (C.c_uint64 * 6)(
            self._address(qh),
            self._address(k1),
            self._address(v1),
            self._address(first),
            self._address(max1),
            self._address(inv1),
        )
        st = self.lib.launch_rmw_plan(plan, self._compute_stream(), ptrs, self._address(ws), 32)
        if st != 0:
            raise RuntimeError(f"launch_rmw_plan split failed with synStatus={st}")
        return merge_tail(first, max1[:, :, :queries], inv1[:, :, :queries], last, max2, inv2)


@lru_cache(maxsize=1)
def _runtime() -> _RMWAttentionRuntime:
    return _RMWAttentionRuntime()


@torch._dynamo.disable
def rmw_fast_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fallback_fn) -> torch.Tensor:
    return _runtime().execute(q, k, v, fallback_fn)


def _rmw_fp8_tile(queries: int, keys: int, slots: int = 2) -> int | None:
    if queries <= 0 or keys <= 0 or (keys % 128) != 0:
        return None
    step = 8
    tile = min(480, (queries // step) * step or queries)
    while tile > 64 and slots * (((keys * tile * 2) + 255) & ~255) > 16 * 1024 * 1024:
        tile -= step
    if slots * (((keys * tile * 2) + 255) & ~255) > 16 * 1024 * 1024:
        return None
    return tile


def rmw_fp8_supported(queries: int, keys: int | None = None, heads: int = 24) -> bool:
    if keys is None:
        keys = queries
    if queries <= 0 or keys <= 0 or heads <= 0:
        return False
    k_pad = ((keys + 127) // 128) * 128
    return _rmw_fp8_tile(queries, k_pad) is not None


def _fp8_prep_aligned(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scale = torch.full((), q.shape[-1] ** -0.25, dtype=q.dtype, device=q.device)
    qh = (q[0] * scale).contiguous()
    kh = (k[0] * scale).contiguous()
    vh = v[0].contiguous()
    vmax = vh.abs().amax(dim=(-2, -1), keepdim=True).float().clamp_min(1e-12) / 240.0
    vq = torch.ops.hpu.cast_to_fp8_v2(vh, vmax.reciprocal(), False, False, torch.float8_e4m3fn)[0].contiguous()
    return qh, kh, vq, vmax / 128.0


def _fp8_prep_padded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # Algebraic 128-key alignment padding: column 128 holds +1.0 in Q and -8192.0 in padded K rows
    # (0.0 in real K rows), keeping scores bit-identical on real keys and forcing exp(-8192) = 0 on pad rows.
    heads, queries, _ = q.shape[1], q.shape[2], q.shape[3]
    keys = k.shape[2]
    k_pad = ((keys + 127) // 128) * 128
    pad_len = k_pad - keys
    scale = torch.full((), 128.0 ** -0.25, dtype=q.dtype, device=q.device)
    q_scaled = q[0] * scale
    k_scaled = k[0] * scale
    q_extra = torch.zeros((heads, queries, 64), dtype=q.dtype, device=q.device)
    q_extra[:, :, 0] = 1.0
    qh = torch.cat((q_scaled, q_extra), dim=-1).contiguous()
    k_extra_real = torch.zeros((heads, keys, 64), dtype=k.dtype, device=k.device)
    k_real = torch.cat((k_scaled, k_extra_real), dim=-1)
    k_pad_rows = torch.zeros((heads, pad_len, 192), dtype=k.dtype, device=k.device)
    k_pad_rows[:, :, 128] = -8192.0
    kh = torch.cat((k_real, k_pad_rows), dim=1).contiguous()
    vh = v[0].contiguous()
    vmax = vh.abs().amax(dim=(-2, -1), keepdim=True).float().clamp_min(1e-12) / 240.0
    vq_real = torch.ops.hpu.cast_to_fp8_v2(vh, vmax.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    vq_pad = torch.zeros((heads, pad_len, 128), dtype=torch.float8_e4m3fn, device=v.device)
    vq = torch.cat((vq_real, vq_pad), dim=1).contiguous()
    return qh, kh, vq, vmax / 128.0


def _fp8_scale_out(out: torch.Tensor, out_scales: torch.Tensor):
    return (out * out_scales.to(out.dtype)).unsqueeze(0)


@lru_cache(maxsize=1)
def _compiled_fp8_helpers():
    options = {"force_static_compile": True}
    return (
        torch.compile(_fp8_prep_aligned, backend="hpu_backend", dynamic=None, options=options),
        torch.compile(_fp8_prep_padded, backend="hpu_backend", dynamic=None, options=options),
        torch.compile(_fp8_scale_out, backend="hpu_backend", dynamic=None, options=options),
    )


class _RMWFP8AttentionRuntime:
    def __init__(self, slots: int = 2, max_plans: int = 32):
        from habana_frameworks.torch.utils import _debug_eager_C
        from habana_frameworks.torch.utils.experimental import _compute_stream, _data_ptr

        self._debug_eager_C = _debug_eager_C
        self._compute_stream = _compute_stream
        self._data_ptr = _data_ptr
        self.lib = _load_rmw_library()
        self.slots = slots
        self.max_plans = max_plans
        self.plans: OrderedDict[tuple[int, int, int, int], int] = OrderedDict()
        self.ws: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None
        self.sout: torch.Tensor | None = None
        self.last_workspace: int = 0

    def _address(self, tensor: torch.Tensor) -> int:
        return self._data_ptr(tensor) + tensor.storage_offset() * tensor.element_size()

    @staticmethod
    def _group_heads(heads: int) -> int:
        for gh in (4, 3, 2, 1):
            if heads % gh == 0:
                return gh
        return 1

    def get_plan(self, queries: int, keys: int, qk_dim: int, group_heads: int):
        key = (queries, keys, qk_dim, group_heads)
        plan = self.plans.get(key)
        if plan is not None:
            self.plans.move_to_end(key)
            return plan
        tile = _rmw_fp8_tile(queries, keys, self.slots)
        if tile is None:
            return None
        name = f"comfy_rmw_fp8_q{queries}_k{keys}_d{qk_dim}_g{group_heads}_s{self.slots}_t{tile}"
        plan = self.lib.build_rmw_fp8_plan(
            queries, keys, qk_dim, group_heads, tile, self.slots, name.encode("utf-8")
        )
        if not plan:
            return None
        while len(self.plans) >= self.max_plans:
            _, evicted = self.plans.popitem(last=False)
            torch.hpu.synchronize()
            self.lib.destroy_rmw_plan(evicted)
        self.plans[key] = plan
        return plan

    def _ensure_buffers(self, ws_bytes: int, device):
        needed = max(256, int(ws_bytes))
        if self.ws is None or self.ws.numel() < needed or self.ws.device != device:
            self.ws = torch.empty(needed, dtype=torch.uint8, device=device)
        if self.sin is None or self.sin.device != device:
            self.sin = torch.tensor([1.0], dtype=torch.float32, device=device)
            self.sout = torch.tensor([128.0], dtype=torch.float32, device=device)

    def execute_prepared(self, qh: torch.Tensor, kh: torch.Tensor, vh_fp8: torch.Tensor) -> torch.Tensor | None:
        heads, queries, qk_dim = qh.shape
        keys = kh.shape[1]
        gh = self._group_heads(heads)
        plan = self.get_plan(queries, keys, qk_dim, gh)
        if plan is None:
            return None
        ws_bytes = int(self.lib.rmw_plan_workspace_bytes(plan))
        self.last_workspace = ws_bytes
        self._ensure_buffers(ws_bytes, qh.device)
        out = torch.empty((heads, queries, 128), dtype=torch.bfloat16, device=qh.device)
        self._debug_eager_C.join_pending_pipeline_threads()
        ptrs = (C.c_uint64 * 6)(
            self._address(qh),
            self._address(kh),
            self._address(vh_fp8),
            self._address(out),
            self._address(self.sin),
            self._address(self.sout),
        )
        st = self.lib.launch_rmw_plan(plan, self._compute_stream(), ptrs, self._address(self.ws), heads)
        if st != 0:
            raise RuntimeError(f"launch_rmw_fp8_plan failed with synStatus={st}")
        return out

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fallback_fn) -> torch.Tensor:
        queries, keys = q.shape[-2], k.shape[-2]
        k_pad = ((keys + 127) // 128) * 128
        if _rmw_fp8_tile(queries, k_pad, self.slots) is None:
            return fallback_fn(q, k, v)
        prep_aligned, prep_padded, scale_out = _compiled_fp8_helpers()
        if k_pad == keys:
            qh, kh, vq, out_scales = prep_aligned(q, k, v)
        else:
            qh, kh, vq, out_scales = prep_padded(q, k, v)
        out = self.execute_prepared(qh, kh, vq)
        if out is None:
            return fallback_fn(q, k, v)
        return scale_out(out, out_scales)


@lru_cache(maxsize=1)
def _fp8_runtime() -> _RMWFP8AttentionRuntime:
    return _RMWFP8AttentionRuntime()


@torch._dynamo.disable
def rmw_fp8_attention_prepared(qh: torch.Tensor, kh: torch.Tensor, vh_fp8: torch.Tensor) -> torch.Tensor:
    out = _fp8_runtime().execute_prepared(qh, kh, vh_fp8)
    if out is None:
        raise RuntimeError(f"RMW FP8 attention plan unavailable for shapes q={tuple(qh.shape)} k={tuple(kh.shape)}")
    return out


@torch._dynamo.disable
def rmw_fp8_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, fallback_fn) -> torch.Tensor:
    return _fp8_runtime().execute(q, k, v, fallback_fn)
