// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU

#    include "gated_residual_onednn.hpp"

#    include <functional>
#    include <memory>
#    include <mutex>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <unordered_map>

#    include "intel_gpu/runtime/lru_cache.hpp"
#    include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::gr_onednn {

namespace {

struct DownKey {
    const void* engine;
    DownProblem problem;

    bool operator==(const DownKey& other) const {
        return engine == other.engine && problem.rows == other.problem.rows && problem.width == other.problem.width &&
               problem.lowrank == other.problem.lowrank && problem.branches == other.problem.branches && problem.weights == other.problem.weights &&
               problem.group_size == other.problem.group_size && problem.has_zp == other.problem.has_zp;
    }
};

struct DownKeyHash {
    size_t operator()(const DownKey& key) const {
        size_t h = std::hash<const void*>()(key.engine);
        const auto& p = key.problem;
        for (const size_t v : {p.rows, p.width, p.lowrank, p.branches, static_cast<size_t>(p.weights.hash()), p.group_size, static_cast<size_t>(p.has_zp)}) {
            h ^= std::hash<size_t>()(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

dnnl::memory::data_type weights_data_type(const ov::element::Type& t) {
    switch (t) {
    case ov::element::f16:
        return dnnl::memory::data_type::f16;
    case ov::element::u4:
        return dnnl::memory::data_type::u4;
    case ov::element::i4:
        return dnnl::memory::data_type::s4;
    case ov::element::u8:
        return dnnl::memory::data_type::u8;
    case ov::element::i8:
        return dnnl::memory::data_type::s8;
    default:
        OPENVINO_THROW("[GPU][OCL] gated_residual: unsupported weight type ", t, " of the oneDNN DOWN matmul");
    }
}

/// A created DOWN matmul. The engine is held so that the key (engine address) cannot be reused while the entry lives.
struct DownPrimitive {
    dnnl::engine engine;
    dnnl::matmul::primitive_desc pd;
    dnnl::matmul prim;
    dnnl::memory::desc scale_md;  // compressed weights only
    dnnl::memory::desc zp_md;     // with zero points only
};

/// Process-wide cache of the DOWN matmuls: every GatedResidual layer of a model has the same shapes, and a oneDNN primitive is
/// stateless, so layers (and infer requests) share one primitive per (engine, rows, C*D, L, C). Creation includes the oneDNN
/// kernel build, so it is done once under the lock. The least recently used primitives are dropped beyond `capacity`.
class DownPrimitiveCache {
public:
    static DownPrimitiveCache& instance() {
        // Intentionally leaked: the primitives reference OpenCL objects that must not be released during static destruction.
        static auto* cache = new DownPrimitiveCache();
        return *cache;
    }

    std::shared_ptr<const DownPrimitive> get(cldnn::engine& engine, const DownProblem& problem) {
        auto& dnnl_engine = engine.get_onednn_engine();
        const DownKey key{&dnnl_engine, problem};

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_cache.has(key)) {
            m_stats.hits++;
            return m_cache.get(key);
        }
        m_stats.misses++;
        auto entry = create(dnnl_engine, problem);
        m_cache.add(key, entry);
        return entry;
    }

    CacheStats stats() {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto stats = m_stats;
        stats.entries = m_cache.size();
        return stats;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
        m_stats = {};
    }

private:
    static constexpr size_t capacity = 128;

    DownPrimitiveCache() : m_cache(capacity) {}

    static std::shared_ptr<const DownPrimitive> create(dnnl::engine& engine, const DownProblem& problem) {
        using dnnl::memory;
        const auto rows = static_cast<memory::dim>(problem.rows);
        const auto width = static_cast<memory::dim>(problem.width);
        const auto lowrank = static_cast<memory::dim>(problem.lowrank);

        // W_down is stored [lowrank, width] with width contiguous: the K x N weights of the matmul with strides {1, width}.
        const bool compressed = problem.weights != ov::element::f16;
        const auto w_dt = weights_data_type(problem.weights);
        const memory::desc src_md({rows, width}, memory::data_type::f16, memory::dims{width, 1});
        const memory::desc weights_md({width, lowrank}, w_dt, memory::format_tag::ba);
        const memory::desc dst_md({rows, lowrank}, memory::data_type::f16, memory::dims{lowrank, 1});

        // T = SiLU(acc / C): both run on the f32 accumulator before the f16 store.
        dnnl::post_ops ops;
        ops.append_eltwise(dnnl::algorithm::eltwise_linear, 1.0f / static_cast<float>(problem.branches), 0.0f);
        ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        dnnl::primitive_attr attr;
        attr.set_post_ops(ops);
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        auto entry = std::make_shared<DownPrimitive>();
        if (compressed) {
            // decompression on the fly: f16 math on int weights, per-channel or per-K-group scales [G, lowrank]
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
            const auto groups = static_cast<memory::dim>(problem.group_size);
            const bool per_channel = problem.group_size == problem.width;
            const int mask = per_channel ? (1 << 1) : ((1 << 0) | (1 << 1));
            const memory::dims group_dims = per_channel ? memory::dims{} : memory::dims{groups, 1};
            attr.set_scales(DNNL_ARG_WEIGHTS, mask, group_dims, memory::data_type::f16);
            if (problem.has_zp) {
                attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, group_dims, w_dt);
            }
            if (per_channel) {
                entry->scale_md = memory::desc({lowrank}, memory::data_type::f16, memory::format_tag::a);
                entry->zp_md = memory::desc({lowrank}, w_dt, memory::format_tag::a);
            } else {
                const memory::dim num_groups = width / groups;
                entry->scale_md = memory::desc({num_groups, lowrank}, memory::data_type::f16, memory::format_tag::ab);
                entry->zp_md = memory::desc({num_groups, lowrank}, w_dt, memory::format_tag::ab);
            }
        }

        entry->engine = engine;
        try {
            entry->pd = dnnl::matmul::primitive_desc(engine, src_md, weights_md, dst_md, attr);
            entry->prim = dnnl::matmul(entry->pd);
        } catch (const dnnl::error& err) {
            OPENVINO_THROW("[GPU][OCL] gated_residual: can't create the oneDNN DOWN matmul for ", problem.rows, " rows: ", err.what());
        }
        return entry;
    }

    std::mutex m_mutex;
    cldnn::LruCache<DownKey, std::shared_ptr<const DownPrimitive>, DownKeyHash> m_cache;
    CacheStats m_stats;
};

}  // namespace

size_t down_scratchpad_bytes(cldnn::engine& engine, const DownProblem& problem) {
    return DownPrimitiveCache::instance().get(engine, problem)->pd.scratchpad_desc().get_size();
}

void execute_down(cldnn::stream& stream,
                  cldnn::engine& engine,
                  const DownProblem& problem,
                  const cldnn::memory::ptr& rbar,
                  const cldnn::memory::ptr& w_down,
                  const cldnn::memory::ptr& scale,
                  const cldnn::memory::ptr& zp,
                  const cldnn::memory::ptr& t,
                  const cldnn::memory::ptr& scratchpad) {
    const auto entry = DownPrimitiveCache::instance().get(engine, problem);

    std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, rbar->get_onednn_memory(entry->pd.src_desc())},
                                               {DNNL_ARG_WEIGHTS, w_down->get_onednn_memory(entry->pd.weights_desc())},
                                               {DNNL_ARG_DST, t->get_onednn_memory(entry->pd.dst_desc())}};
    if (problem.weights != ov::element::f16) {
        OPENVINO_ASSERT(scale != nullptr && (zp != nullptr) == problem.has_zp,
                        "[GPU][OCL] gated_residual: wrong scale / zero point inputs of the oneDNN DOWN matmul");
        args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale->get_onednn_memory(entry->scale_md));
        if (zp != nullptr) {
            args.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp->get_onednn_memory(entry->zp_md));
        }
    }
    if (entry->pd.scratchpad_desc().get_size() > 0) {
        OPENVINO_ASSERT(scratchpad != nullptr, "[GPU][OCL] gated_residual: the oneDNN DOWN matmul needs a scratchpad");
        args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad->get_onednn_memory(entry->pd.scratchpad_desc()));
    }

    try {
        entry->prim.execute(stream.get_onednn_stream(), args);
    } catch (const dnnl::error& err) {
        OPENVINO_THROW("[GPU][OCL] gated_residual: the oneDNN DOWN matmul failed: ", err.what());
    }
}

CacheStats cache_stats() {
    return DownPrimitiveCache::instance().stats();
}

void clear_cache() {
    DownPrimitiveCache::instance().clear();
}

}  // namespace ov::intel_gpu::ocl::gr_onednn

#endif  // ENABLE_ONEDNN_FOR_GPU
