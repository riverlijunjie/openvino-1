// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <utility>

#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {
struct PagedAttentionManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::paged_attention")
    explicit PagedAttentionManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<paged_attention>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t idx = 0; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::bfyx;
        }
        out_fmts[0] = format::ybfx;
        for (size_t idx = 1; idx < node.get_outputs_count(); idx++) {
            out_fmts[idx] = format::bfyx;
        }

        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_q_types = {
            ov::element::f32,
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            return false;
        }

        return true;
    }
};
}  // namespace ov::intel_gpu::cm