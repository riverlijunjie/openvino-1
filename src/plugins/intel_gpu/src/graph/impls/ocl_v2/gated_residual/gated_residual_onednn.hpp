// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_ONEDNN_FOR_GPU

#    include <cstddef>

#    include "intel_gpu/runtime/engine.hpp"
#    include "intel_gpu/runtime/memory.hpp"
#    include "intel_gpu/runtime/stream.hpp"
#    include "openvino/core/type/element_type.hpp"

/// The oneDNN part of the GatedResidual READ: the DOWN projection T = SiLU(Rbar . W_down^T / C) as one dnnl::matmul with the
/// 1/C scale and the SiLU as post-ops. oneDNN types stay out of this header.
namespace ov::intel_gpu::ocl::gr_onednn {

/// The DOWN projection of `rows` token rows: Rbar [rows, width] x W_down [lowrank, width]^T -> T [rows, lowrank]. Rbar and T are
/// f16; W_down is f16 or i4 / u4 / i8 / u8 with f16 scales [width / group_size, lowrank] and optional zero points of the weight type.
struct DownProblem {
    size_t rows;
    size_t width;  ///< C * D
    size_t lowrank;
    size_t branches;  ///< C: T is scaled by 1 / C
    ov::element::Type weights = ov::element::f16;
    size_t group_size = 0;  ///< K per scale group of compressed weights (== width: one scale per output channel)
    bool has_zp = false;
};

/// Scratchpad bytes the matmul of `problem` needs (0 for most shapes). Creates the primitive on first use.
size_t down_scratchpad_bytes(cldnn::engine& engine, const DownProblem& problem);

/// Runs the DOWN projection on the in-order queue of `stream`. `scratchpad` may be null when down_scratchpad_bytes() is 0;
/// `scale` and `zp` are the compressed weights' scales and zero points (`zp` null without zero points).
void execute_down(cldnn::stream& stream,
                  cldnn::engine& engine,
                  const DownProblem& problem,
                  const cldnn::memory::ptr& rbar,
                  const cldnn::memory::ptr& w_down,
                  const cldnn::memory::ptr& scale,
                  const cldnn::memory::ptr& zp,
                  const cldnn::memory::ptr& t,
                  const cldnn::memory::ptr& scratchpad);

/// Counters of the process-wide primitive cache (tests and tuning).
struct CacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t entries = 0;
};

CacheStats cache_stats();
void clear_cache();

}  // namespace ov::intel_gpu::ocl::gr_onednn

#endif  // ENABLE_ONEDNN_FOR_GPU
