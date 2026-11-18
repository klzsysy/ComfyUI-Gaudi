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
