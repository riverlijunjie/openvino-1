// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_cm.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "utils/jitter.hpp"
#include "utils/kernel_generator_cm.hpp"

#define CM_PA_ENABLE
#ifdef CM_PA_ENABLE
namespace ov::intel_gpu::cm {

using namespace ov;
using namespace ov::intel_gpu::ocl;
using namespace cldnn;
namespace {

// TODO: support MIXED mode
enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

constexpr auto get_pa_build_options() {
    return " -cmc -Qxcm_register_file_size=256 -mdump_asm -g2 ";
}

constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
constexpr size_t paged_attention_block_size = 16;
constexpr size_t seq_len_partition_size = 256;
constexpr size_t subgroup_size = 16;
constexpr size_t WG_SIZE = 16;
constexpr size_t kv_split_data_size = 16;

constexpr size_t attention_mask_idx = 3;
constexpr size_t split_output_idx = 3;
constexpr size_t lse_idx = 4;

}  // namespace

struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t paged_attention_aligned_seq_len;
    size_t sdpa_opt_max_seq_len;
    // size_t sdpa_opt_seq_len_partition_size;
};

inline size_t get_target_seq_len_block_size() {
    constexpr size_t block_size = 16;
    return block_size;
}

inline size_t get_generate_stage_block_size(size_t head_size) {
    auto preferred_block_size = {4, 2, 1};
    for (const auto& block_size : preferred_block_size) {
        if (head_size % (block_size * subgroup_size) == 0) {
            return block_size;
        }
    }
    return 1;
}

inline size_t get_element_size(ov::element::Type_t type) {
    switch (type) {
    case ov::element::Type_t::i64:
        return 8;
    case ov::element::Type_t::f32:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        return 4;
    case ov::element::Type_t::f16:
        return 2;
    case ov::element::Type_t::i8:
    case ov::element::Type_t::u8:
        return 1;
    default:
        OPENVINO_ASSERT(false, "Unsupported element type for get_element_size");
        return 0;  // Fallback case, should not be reached
    }
}

inline bool param_is_dynamic(const RuntimeParams& params) {
    if (params.is_dynamic()) {
        return true;
    }
    for (const auto& layout : params.input_layouts) {
        if (layout.data_padding.is_dynamic()) {
            return true;
        }
    }
    for (const auto& layout : params.output_layouts) {
        if (layout.data_padding.is_dynamic()) {
            return true;
        }
    }
    return false;
}

// This function returns the kv_step and kv_split_len based on the architecture.
// return {kv_step, kv_split_len}
inline std::pair<size_t, size_t> get_kv_split_size(size_t arch) {
    if (arch == 1) {
        return {8, 32};  // For Xe1
    } else if (arch == 2) {
        return {16, 32};  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for KV split size");
    return {0, 0};  // Fallback case, should not be reached
}

inline size_t get_q_step(size_t arch, bool is_single_token = false) {
    if (arch == 1) {
        return is_single_token ? 1 : 8;  // For Xe1
    } else if (arch == 2) {
        return is_single_token ? 1 : 16;  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for Q step");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_kv_len(const RuntimeParams& params, const PagedAttentionStage& stage) {
    if (stage == PagedAttentionStage::PREFILL) {
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        return kv_len;
    } else if (stage == PagedAttentionStage::GENERATE) {
        // TODO FIX: key_cache shape = [16, 128+4, 4, 2269]
        //  auto key_cache_shape = params.input_layouts[3].get_shape();
        //  const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        size_t i = 0;
        for (auto& l : params.input_layouts) {
            auto _shape = l.get_shape();
            std::cout << i++ << " shape: " << _shape.to_string() << std::endl;
        }
        std::cout << std::endl;
        return kv_len;
    }
    OPENVINO_ASSERT(false, "Unsupported PagedAttentionStage for get_kv_len");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_split_num(const RuntimeParams& params, const PagedAttentionStage& stage) {
    const size_t kv_len = get_kv_len(params, stage);
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    const size_t split_num = kv_len / get_kv_split_size(xe_arch).second;

    return split_num;
}

inline int64_t get_aligned_seq_len(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, int64_t target_seq_len_block_size = 16) {
    auto calculate_aligned_seq_len = [&]() {
        const auto& input_mem = impl_param.memory_deps;
        const auto subsequence_begins_input_idx = 6;
        const auto subsequence_begins_mem = input_mem.at(subsequence_begins_input_idx);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *impl_param.strm);

        auto aligned_seq_len = 0;
        for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
            auto prompt_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];
            aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
        }

        return aligned_seq_len;
    };

    int64_t aligned_seq_len = 0;
    if (stage == PagedAttentionStage::PREFILL) {
        const auto desc = impl_param.typed_desc<paged_attention>();
        if (static_cast<int64_t>(paged_attention::block_size) == target_seq_len_block_size) {
            const auto block_indices_input_idx = 7;
            const auto& block_indices_ps = impl_param.get_input_layout(block_indices_input_idx).get_partial_shape();

            aligned_seq_len = block_indices_ps[0].get_length() * target_seq_len_block_size;
        } else {
            aligned_seq_len = calculate_aligned_seq_len();
        }
    } else {
        aligned_seq_len = calculate_aligned_seq_len();
    }

    return aligned_seq_len;
}

inline std::pair<size_t, size_t> get_partitioning_params(const kernel_impl_params& params, size_t head_size, PagedAttentionStage stage) {
    const auto& input_mem = params.memory_deps;
    const auto max_context_len = input_mem.at(12);
    mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *params.strm);
    const auto paged_attention_max_len = max_context_len_mem_lock[0];

    size_t partition_size = 0;
    if (stage == PagedAttentionStage::PREFILL) {
        partition_size = head_size;
    } else {
        partition_size = seq_len_partition_size;
    }

    return {ceil_div(paged_attention_max_len, partition_size), partition_size};
}

static PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param) {
    const auto& query_shape = impl_param.get_input_layout(0).get_partial_shape();
    const auto& past_lens_shape = impl_param.get_input_layout(5).get_partial_shape();

    if (query_shape.is_static() && past_lens_shape.is_static()) {
        if (query_shape[0].get_length() == past_lens_shape[0].get_length()) {
            return PagedAttentionStage::GENERATE;
        }

        const auto past_lens_idx = 5;
        const auto& memory_deps = impl_param.memory_deps;
        const auto past_lens_mem = memory_deps.at(past_lens_idx);
        mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

        const auto past_lens_size = past_lens_mem_lock.size();
        for (size_t i = 0; i < past_lens_size; i++) {
            if (past_lens_mem_lock[i] != 0) {
                OPENVINO_ASSERT(false, "[GPU][CM] PagedAttentionGenerator: get_paged_attention_stage doesn't support MIXED stage. ");
                return PagedAttentionStage::MIXED;
            }
        }

        return PagedAttentionStage::PREFILL;
    }

    return PagedAttentionStage::UNKNOWN;
}

class PagedAttentionGeneratorBase : public KernelGenerator {
public:
    explicit PagedAttentionGeneratorBase(std::string_view kernel_name, std::string_view stage_suffix = "") : KernelGenerator(kernel_name, stage_suffix) {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_pa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        // std::cout << "PagedAttentionGeneratorBase::get_jit_constants: " << get_entry_point(params) << std::endl;

        auto desc = params.typed_desc<paged_attention>();
        jit.make("HEAD_SIZE", desc->k_head_size);
        jit.make("HEADS_NUM", desc->heads_num);
        jit.make("KV_HEADS_NUM", desc->kv_heads_num);

        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
        jit.make("SCALE_FACTOR", scale_factor);
        jit.make("CMFLA_SCALE_FACTOR", scale_factor);
        jit.make("CMFLA_NUM_HEADS", desc->heads_num);
        jit.make("CMFLA_HEAD_SIZE", desc->k_head_size);
        jit.make("CMFLA_NUM_KV_HEADS", desc->kv_heads_num);
        jit.make("WG_SIZE_HINT", WG_SIZE);

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        jit.make("XE_ARCH", xe_arch);

        auto split_size = get_kv_split_size(xe_arch);
        jit.make("KV_STEP", split_size.first);

        jit.make("WG_SIZE", WG_SIZE);
        return jit;
    }
};

class PagedAttentionGeneratorMultiToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorMultiToken() : PagedAttentionGeneratorBase("pa_sdpa_prefill") {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        const auto desc = params.typed_desc<paged_attention>();

        OPENVINO_ASSERT(!desc->has_scores_output(), "[GPU][CM] PagedAttentionGeneratorMultiToken with scores output is not supported yet");

        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // query
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // key
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // value

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_len

        return args;
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        const auto desc = params.typed_desc<paged_attention>();

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        jit.make("Q_STEP", get_q_step(xe_arch, false));

        // TODO: set causal mask only if needed
        auto causal_mask = 1;
        jit.make("CAUSAL_MASK", causal_mask);

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            auto desc = params.typed_desc<paged_attention>();
            // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
            const size_t heads_num = desc->heads_num;
            // const size_t head_size = desc->k_head_size;

            auto out_shape = params.output_layouts[0].get_shape();
            const size_t batch = out_shape.size() < 4 ? 1 : out_shape[0];
            const size_t q_len = out_shape[0];

            auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
            const size_t q_step = get_q_step(xe_arch, false);
            const size_t q_group_size = WG_SIZE * q_step;
            const size_t q_threads = align_to(q_len, q_group_size) / q_step;

            wgs.global = {batch, heads_num, q_threads};
            wgs.local = {1, 1, WG_SIZE};

            std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: "
                      << "out_shape: " << out_shape.to_string()
                      << ", batch: " << batch
                      << ", heads_num: " << heads_num
                      << ", q_threads: " << q_threads
                      << ", q_len: " << q_len
                      << ", q_step: " << q_step
                      << std::endl;

            // Prefill stage: kv_len == q_len
            auto kv_len = q_len;
            std::vector<size_t> scaler_value = {q_len, kv_len};
            scalars.resize(scaler_value.size());
            for (size_t i = 0; i < scaler_value.size(); ++i) {
                scalars[i].t = ScalarDescriptor::Types::INT32;
                scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
            }
        }};
    }
};

class PagedAttentionGeneratorSingleToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleToken() : PagedAttentionGeneratorBase("pa_sdpa_single_token") {}

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        // jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        jit.make("Q_STEP", get_q_step(xe_arch, true));
        auto kv_split_size = get_kv_split_size(xe_arch);
        jit.make("KV_STEP", kv_split_size.first);
        jit.make("KV_SPLIT_LEN", kv_split_size.second);

        const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
        jit.make("KV_LEN", kv_len);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        const auto desc = params.typed_desc<paged_attention>();
        // const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = params.output_layouts.size() > 1;

        OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleToken with scores output is not supported yet");

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // queries
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // keys cache
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // values cache

        // TODO: HAS_ATTN_MASK_INPUT
        // args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, attention_mask_idx}); // attention_mask
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split output
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // lse output

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len==1
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_len

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto desc = params.typed_desc<paged_attention>();

            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t split_num = get_split_num(params, rtp->stage);
            wgs.global = {batch, heads_num, split_num};
            wgs.local = {1, 1, WG_SIZE};

            // generate stage: q_len=1, kv_len=past_len + 1
            auto& scalars = kd.params.scalars;
            auto kv_len = rtp->paged_attention_aligned_seq_len;
            std::vector<size_t> scaler_value = {1, kv_len};
            scalars.resize(scaler_value.size());

            std::cout << "PagedAttentionGeneratorSingleToken::get_dispatch_data_func: "
                      << "batch: " << batch
                      << ", heads_num: " << heads_num
                      << ", split_num: " << split_num
                      << ", kv_len: " << kv_len
                      << std::endl;

            for (size_t i = 0; i < scaler_value.size(); ++i) {
                scalars[i].t = ScalarDescriptor::Types::INT32;
                scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
            }
        }};
    }
};

class PagedAttentionGeneratorSingleTokenFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleTokenFinalization() : PagedAttentionGeneratorBase("pa_sdpa_single_token_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);

        const auto desc = params.typed_desc<paged_attention>();
        jit.make("KV_SPLIT_DATA_SIZE", kv_split_data_size);
        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        jit.make("KV_SPLIT_LEN", get_kv_split_size(xe_arch).second);

        // auto key_cache_shape = params.input_layouts[3].get_shape();
        // const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
        const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
        jit.make("KV_LEN", kv_len);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        const auto has_scores_output = params.output_layouts.size() > 1;

        OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleTokenFinalization with scores output is not supported yet");

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split data
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                              // output
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // values cache

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.resize(1);

            const auto desc = params.typed_desc<paged_attention>();
            // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->k_head_size;

            wgs.global = {batch, heads_num, head_size / kv_split_data_size};
            wgs.local = {1, 1, 1};
        }};
    }
};

class KVCacheUpdateGenerator : public ov::intel_gpu::ocl::KernelGenerator {
public:
    KVCacheUpdateGenerator() : KernelGenerator("pa_kv_cache_update_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto& in_offsets_map = params.in_port_to_shape_info_offset;

        // std::cout << "params.is_dynamic() = " << params.is_dynamic() << ", pad_dyn = " << param_is_dynamic(params) << std::endl;
        // for (auto& item : jit) {
        //     if (item.name == "OPTIONAL_SHAPE_INFO_ARG" && item.value == "") {
        //         std::cout << "WARNING TO FIXED: OPTIONAL_SHAPE_INFO_ARG for param.is_dynamic()== true" << std::endl;
        //     }
        // }

        constexpr static std::array input_ids = {1, 2, 5, 7, 8, 6};
        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids.at(i);
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        constexpr size_t key_cache_id = 3;
        constexpr size_t value_cache_id = 4;

        jit.add(make_layout_jit_constants("OUTPUT", params.input_layouts[key_cache_id], in_offsets_map.at(key_cache_id)));
        jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(1), params.input_layouts[value_cache_id], in_offsets_map.at(value_cache_id)));

        const auto desc = params.typed_desc<paged_attention>();
        jit.make("K_HEAD_SIZE", desc->k_head_size);
        jit.make("V_HEAD_SIZE", desc->v_head_size);
        jit.make("HEADS_NUM", desc->heads_num);
        jit.make("KV_HEADS_NUM", desc->kv_heads_num);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        jit.make("GENERATE_STAGE_K_BLOCK_SIZE", get_generate_stage_block_size(desc->k_head_size));
        jit.make("GENERATE_STAGE_V_BLOCK_SIZE", get_generate_stage_block_size(desc->v_head_size));

        const bool is_kv_compressed = true;
        if (is_kv_compressed) {
            auto data_type = params.input_layouts[1].data_type; //key tensor data size
            auto scales_zp_size = get_element_size(data_type) * 2;  // scale + zp
            jit.make("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size);
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + scales_zp_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size + scales_zp_size);
        } else {
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size);
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        // if (params.is_dynamic()) {
        if (param_is_dynamic(params)) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        // Inputs
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // key
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // value
        args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // block_indices
        args.push_back({ArgumentDescriptor::Types::INPUT, 8});  // block_indices_begins
        args.push_back({ArgumentDescriptor::Types::INPUT, 6});  // subsequence_begins

        // Outputs
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // key_cache
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // value_cache

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.resize(1);

            const auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const auto is_prefill = rtp->stage == PagedAttentionStage::PREFILL;
            auto heads_number = desc->kv_heads_num;

            if (is_prefill) {
                const auto blocks_number = rtp->paged_attention_aligned_seq_len / paged_attention_block_size;

                wgs.global = {blocks_number, heads_number, subgroup_size};
                wgs.local = {1, 1, subgroup_size};
            } else {
                const auto& key_input = params.input_layouts[0];
                const auto sequences_number = key_input.get_partial_shape()[0].get_length();

                wgs.global = {static_cast<size_t>(sequences_number), heads_number, subgroup_size};
                wgs.local = {1, 1, subgroup_size};
            }

            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(is_prefill);
        }};
    }
};

class PagedAttentionImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::PagedAttentionImpl)

    Stage::Ptr kv_cache_update = make_stage<KVCacheUpdateGenerator>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_prefill = make_stage<PagedAttentionGeneratorMultiToken>();

    PagedAttentionImpl() : PrimitiveImplCM(PagedAttentionManager::get_type_info_static()) {}

    explicit PagedAttentionImpl(const kernel_impl_params& params) : PrimitiveImplCM(PagedAttentionManager::get_type_info_static()) {
        const auto desc = params.typed_desc<paged_attention>();
        const bool has_scores_output = params.output_layouts.size() > 1;
        const bool has_rotated_blocks = desc->has_rotated_blocks;

        OPENVINO_ASSERT(!has_scores_output && !has_rotated_blocks, "[GPU][CM] PagedAttentionImpl with scores output and rotated block is not supported yet");

        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_prefill, params);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        const auto& params = *instance.get_impl_params();
        const auto& desc = params.typed_desc<paged_attention>();
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
        }

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        rt_params->stage = get_paged_attention_stage(params);
        std::tie(rt_params->num_of_partitions, rt_params->partition_size) = get_partitioning_params(params, desc->k_head_size, rt_params->stage);
        rt_params->paged_attention_aligned_seq_len = static_cast<size_t>(get_aligned_seq_len(params, rt_params->stage));
        // rt_params->sdpa_opt_seq_len_partition_size = get_seq_len_partition_size(params.get_device_info(), desc->k_head_size, SDPAStage::MULTI_TOKENS);

        const auto& input_mem = params.memory_deps;
        const auto max_context_len = input_mem.at(12);
        mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *params.strm);
        rt_params->sdpa_opt_max_seq_len = static_cast<int64_t>(max_context_len_mem_lock[0]);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        /*
         * Internal buffers allocation owners and users:
         * +--------------------------------------+--------------------+--------------------+
         * | Stage                                | Allocates & uses   | Reuses             |
         * +--------------------------------------+--------------------+--------------------+
         * | KV_CACHE_UPDATE                      | [0, 1, 2]          |                    |
         * +--------------------------------------+--------------------+--------------------+
         * | SDPA (1st token)                     |                    | [0, 1, 2]          |
         * +--------------------------------------+--------------------+--------------------+
         * | PA_SDPA (2nd+ token)                 | [3, 4]          |                    |
         * +--------------------------------------+--------------------+--------------------+
         *
         * Description:
         * 0, 1, 2 - Buffers used for proper blocks distribution for kv_cache_update and
         *           sdpa_opt (1st token calculation) block configuration over target_seq_len dimension.
         *           Filled in paged_attention_inst::on_execute() call.
         * 3, 4 - Used for 2nd+ PA calculation (intermediate output and lse).
         *           Filled in PA/SDPA kernels.
         */

        std::vector<BufferDescriptor> internal_buffers;

        const auto desc = params.typed_desc<paged_attention>();

        const bool has_scores_output = params.output_layouts.size() > 1;
        OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] get_internal_buffer_descs with scores output is not supported yet");

        const auto indexes_dt = ov::element::i32;
        const int64_t target_seq_len_block_size = 16;
        auto stage = get_paged_attention_stage(params);
        int64_t paged_attention_aligned_seq_len = -1;
        if (stage == PagedAttentionStage::PREFILL && !params.is_dynamic()) {
            paged_attention_aligned_seq_len = get_aligned_seq_len(params, stage);
        }

        const auto target_seq_len = std::max<int64_t>(paged_attention_aligned_seq_len, 1);
        const auto indexes_buf_size = static_cast<int64_t>(ceil_div(target_seq_len, target_seq_len_block_size));

        const bool lockable = true;  // usm_host
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);

        const auto& input = params.input_layouts[0];
        const int64_t total_tokens = input.get_partial_shape()[0].get_length();

        const size_t split_num = get_split_num(params, stage);
        const size_t lse_size = total_tokens * desc->heads_num * split_num;
        const size_t split_output_size = lse_size * desc->k_head_size;
        internal_buffers.emplace_back(split_output_size, ov::element::f16);  // split output
        internal_buffers.emplace_back(lse_size, ov::element::f32);           // lse output

        return internal_buffers;
    }

    static size_t get_query_block_size(const PagedAttentionStage& stage) {
        const auto default_block_size = 16;
        return default_block_size;
    }

    static void prepare_internal_buffers(paged_attention_inst& instance, const PagedAttentionStage& stage) {
        const auto& desc = instance.get_impl_params()->typed_desc<paged_attention>();
        const bool has_scores_output = desc->has_scores_output();
        OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] prepare_internal_buffers with scores output and rotated block is not supported yet");

        if (stage == PagedAttentionStage::UNKNOWN) {
            return;
        }

        auto& stream = instance.get_network().get_stream();
        const auto past_lens_mem = instance.past_lens_memory_ptr();
        const auto subsequence_begins_mem = instance.subsequence_begins_memory_ptr();
        const auto& intermediates_memories = instance.get_intermediates_memories();
        mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, stream);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, stream);
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> subsequence_offsets_lock = nullptr;

        if (stage == PagedAttentionStage::GENERATE) {
            // For the generate stage it's not necessary to configure any other intermediate
            // buffers. Simply calculate the offsets and exit
            size_t subsequence_offsets_acc = 0;
            for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                const auto past_len = past_lens_mem_lock[i];
                const auto seq_start = subsequence_begins_mem_lock[i];
                const auto seq_end = subsequence_begins_mem_lock[i + 1];
                const auto seq_length = seq_end - seq_start;

                if (subsequence_offsets_lock) {
                    subsequence_offsets_lock->operator[](i) = static_cast<int32_t>(subsequence_offsets_acc);
                    subsequence_offsets_acc += seq_length + past_len;
                }
            }

            return;
        }

        OPENVINO_ASSERT(intermediates_memories.size() >= 3, "Unexpected number of intermediates buffers for Paged Attention at prefill stage");

        const auto blocks_indexes_start_idx = 0;
        const auto blocks_indexes_end_idx = 1;
        const auto blocked_gws_subseq_mapping_idx = 2;

        const auto& blocks_indexes_start_mem = intermediates_memories[blocks_indexes_start_idx];
        const auto& blocks_indexes_end_mem = intermediates_memories[blocks_indexes_end_idx];
        const auto& blocked_gws_subseq_mapping_mem = intermediates_memories[blocked_gws_subseq_mapping_idx];

        OPENVINO_ASSERT(subsequence_begins_mem->get_layout().data_type == data_types::i32);

        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_start_lock(blocks_indexes_start_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_end_lock(blocks_indexes_end_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocked_gws_subseq_mapping_mem_lock(blocked_gws_subseq_mapping_mem, stream);

        size_t index = 0;
        size_t subsequence_offsets_acc = 0;
        // size_t query_block_size = get_query_block_size(stage);
        const auto pa_block_size = static_cast<int>(paged_attention::block_size);
        for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
            const auto past_len = past_lens_mem_lock[i];
            const auto seq_start = subsequence_begins_mem_lock[i];
            const auto seq_end = subsequence_begins_mem_lock[i + 1];
            const auto seq_length = seq_end - seq_start;

            int32_t j = 0;
            if (past_len != 0) {
                auto block_start_pos = seq_start;
                auto empty_slots = pa_block_size - (past_len % pa_block_size);
                auto block_end_pos = seq_start + std::min(empty_slots, seq_length);

                blocks_indexes_start_lock[index] = block_start_pos;
                blocks_indexes_end_lock[index] = block_end_pos;
                blocked_gws_subseq_mapping_mem_lock[index] = static_cast<int32_t>(i);

                index++;

                auto added_slots = block_end_pos - block_start_pos;
                j += added_slots;
            }

            for (; j < seq_length; j += pa_block_size) {
                auto block_start_pos = subsequence_begins_mem_lock[i] + j;
                auto block_end_pos = std::min(block_start_pos + pa_block_size, seq_end);

                blocks_indexes_start_lock[index] = block_start_pos;
                blocks_indexes_end_lock[index] = block_end_pos;
                blocked_gws_subseq_mapping_mem_lock[index] = static_cast<int32_t>(i);

                index++;
            }

            if (subsequence_offsets_lock) {
                subsequence_offsets_lock->operator[](i) = static_cast<int32_t>(subsequence_offsets_acc);
                subsequence_offsets_acc += seq_length + past_len;
            }
        }
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();
        // const auto& head_size = desc->k_head_size;

        const bool has_scores_output = params.output_layouts.size() > 1;
        const bool has_rotated_blocks = desc->has_rotated_blocks;
        OPENVINO_ASSERT(!has_scores_output && !has_rotated_blocks, "[GPU][CM] Paged Attention with scores output and rotated blocks is not supported yet");

        std::cout << "ov::intel_gpu::cm::execute()..." << std::endl;
        update_rt_params(instance);

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        assert(rt_params != nullptr);
        prepare_internal_buffers(static_cast<paged_attention_inst&>(instance), rt_params->stage);

        std::cout << "ov::intel_gpu::cm::execute()...stage = " << static_cast<size_t>(rt_params->stage) << std::endl;

        std::vector<event::ptr> res_event = events;
        res_event = {execute_stage(res_event, instance, kv_cache_update)};
        for (auto& ev : res_event) {
            ev->wait();
        }
        std::cout << "ov::intel_gpu::cm::execute()...kv_cache_update done" << std::endl;

        if (rt_params->stage == PagedAttentionStage::PREFILL) {
            res_event = {execute_stage(res_event, instance, pa_prefill)};
            for (auto& ev : res_event) {
                ev->wait();
            }
            std::cout << "ov::intel_gpu::cm::execute()...pa_prefill done" << std::endl;
        } else if (rt_params->stage == PagedAttentionStage::GENERATE) {
            res_event = {execute_stage(res_event, instance, pa_single_token)};
            for (auto& ev : res_event) {
                ev->wait();
            }
            std::cout << "ov::intel_gpu::cm::execute()...pa_single_token done" << std::endl;
            res_event = {execute_stage(res_event, instance, pa_single_token_finalization)};
            for (auto& ev : res_event) {
                ev->wait();
            }
            std::cout << "ov::intel_gpu::cm::execute()...pa_single_token_finalization done" << std::endl;
        }

        std::cout << "ov::intel_gpu::cm::execute()...done" << std::endl << std::endl;
        return res_event.size() == 0 ? nullptr : res_event[0];
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedAttentionImpl>(this);
    }
};

std::unique_ptr<primitive_impl> PagedAttentionManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<paged_attention>());
    return std::make_unique<PagedAttentionImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::PagedAttentionImpl)

#endif