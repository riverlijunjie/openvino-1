// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_moe_3gemm_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

FuseMOE3GemmCompressed::FuseMOE3GemmCompressed() {
    auto hidden_state_m = any_input();
    auto routers_m = any_input();
    // Common Router Logits (MatMul)
    auto router_matmul_m = wrap_type<ov::op::v0::MatMul>({hidden_state_m, routers_m}, consumers_count(1));

    // =========================================================================
    // GEMM3 Branch (Qwen): MatMul -> Softmax -> TopK -> [Complex Scatter Logic]
    // =========================================================================
    auto softmax_g3_m = wrap_type<ov::op::v8::Softmax>({router_matmul_m}, consumers_count(1));
    auto topk_g3_m = wrap_type<ov::op::v11::TopK>({softmax_g3_m, any_input()});
    topk_g3_m->set_output_size(2);

    // GEMM3 Weight Normalization & Scatter Logic
    auto reduce_sum_g3_m = wrap_type<ov::op::v1::ReduceSum>({topk_g3_m->output(0), any_input()}, consumers_count(1));
    auto norm_g3_m = wrap_type<ov::op::v1::Divide>({topk_g3_m->output(0), reduce_sum_g3_m->output(0)}, consumers_count(1));

    auto shape_of_m = wrap_type<ov::op::v3::ShapeOf>({topk_g3_m->output(1)}, consumers_count(1));
    auto gather_m = wrap_type<ov::op::v8::Gather>({shape_of_m, any_input(), any_input()}, consumers_count(1));
    auto unsqueeze_m = wrap_type<ov::op::v0::Unsqueeze>({gather_m, any_input()});

    auto unsqueeze_const_m = wrap_type<ov::op::v0::Unsqueeze>({any_input(), any_input()});
    auto concat_m = wrap_type<ov::op::v0::Concat>({unsqueeze_m, unsqueeze_const_m}, consumers_count(1));
    auto concat1_m = wrap_type<ov::op::v0::Concat>({unsqueeze_const_m, unsqueeze_m, any_input()}, consumers_count(1));
    auto bc_m = wrap_type<ov::op::v3::Broadcast>({any_input(), concat_m}, consumers_count(1));

    auto scatter_m =
        wrap_type<ov::op::v12::ScatterElementsUpdate>({bc_m->output(0), topk_g3_m->output(1), norm_g3_m->output(0), any_input()}, consumers_count(1));
    auto transpose_m = wrap_type<ov::op::v1::Transpose>({scatter_m, any_input()}, consumers_count(1));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({transpose_m, concat1_m}, consumers_count(1));
    auto unsqueeze_moe_m = wrap_type<ov::op::v0::Unsqueeze>({reshape_m, any_input()}, consumers_count(1));

    // =========================================================================
    // GEMM2 Branch (GPT-OSS): logits -> TopK -> Softmax
    // =========================================================================
    // NOTE: ConvertMOEToMOECompressed(GEMM2) allows the TopK input to be an arbitrary subgraph
    // (not necessarily MatMul), so track it explicitly and avoid over-constraining consumer counts.
    auto g2_logits_m = any_input();
    auto topk_g2_m = wrap_type<ov::op::v11::TopK>({g2_logits_m, any_input()});
    topk_g2_m->set_output_size(2);
    auto norm_g2_m = wrap_type<ov::op::v8::Softmax>({topk_g2_m->output(0)});

    // Indices path in ConvertMOEToMOECompressed(GEMM2) can be wrapped with Convert.
    // The current convert pattern uses Convert({TopK}) (unspecified output),
    // therefore accept both Convert(TopK.output(1)) and Convert(TopK).
    auto topk_indices_convert_from_out1_m = wrap_type<ov::op::v0::Convert>({topk_g2_m->output(1)});
    auto topk_indices_convert_from_topk_m = wrap_type<ov::op::v0::Convert>({topk_g2_m});
    auto topk_indices_g2_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{topk_indices_convert_from_out1_m, topk_indices_convert_from_topk_m});

    // =========================================================================
    // Parameters Matching
    // =========================================================================
    auto gate_wei_m = wrap_type<ov::op::v0::Constant>();
    auto gate_scale_m = any_input();
    auto gate_zp_m = any_input();

    auto up_wei_m = wrap_type<ov::op::v0::Constant>();
    auto up_scale_m = any_input();
    auto up_zp_m = any_input();
    auto up_bias_m = any_input();  // GEMM2 specific

    auto down_wei_m = wrap_type<ov::op::v0::Constant>();
    auto down_scale_m = any_input();
    auto down_zp_m = any_input();
    auto down_bias_m = any_input();  // GEMM2 specific

    // 1. GEMM3 Pattern (12 inputs: Gate, Up, Down)
    auto moe_compressed_gemm3_m = wrap_type<ov::intel_gpu::op::MOECompressed>({hidden_state_m->output(0),
                                                                               unsqueeze_moe_m->output(0),  // Processed weights
                                                                               topk_g3_m->output(1),        // Indices
                                                                               gate_wei_m->output(0),
                                                                               gate_scale_m->output(0),
                                                                               gate_zp_m->output(0),
                                                                               up_wei_m->output(0),
                                                                               up_scale_m->output(0),
                                                                               up_zp_m->output(0),
                                                                               down_wei_m->output(0),
                                                                               down_scale_m->output(0),
                                                                               down_zp_m->output(0)});

    // 2. GEMM2 Pattern - No ZP (9 inputs: Up, Down, Bias)
    auto moe_compressed_gemm2_no_zp_m = wrap_type<ov::intel_gpu::op::MOECompressed>({hidden_state_m->output(0),
                                                                                     norm_g2_m->output(0),  // Softmax output
                                                                                     topk_indices_g2_m,
                                                                                     up_wei_m->output(0),
                                                                                     up_scale_m->output(0),
                                                                                     up_bias_m->output(0),
                                                                                     down_wei_m->output(0),
                                                                                     down_scale_m->output(0),
                                                                                     down_bias_m->output(0)});

    // 3. GEMM2 Pattern - With ZP (11 inputs: Up, Down, ZP, Bias)
    auto moe_compressed_gemm2_zp_m = wrap_type<ov::intel_gpu::op::MOECompressed>({hidden_state_m->output(0),
                                                                                  norm_g2_m->output(0),
                                                                                  topk_indices_g2_m,
                                                                                  up_wei_m->output(0),
                                                                                  up_scale_m->output(0),
                                                                                  up_zp_m->output(0),
                                                                                  up_bias_m->output(0),
                                                                                  down_wei_m->output(0),
                                                                                  down_scale_m->output(0),
                                                                                  down_zp_m->output(0),
                                                                                  down_bias_m->output(0)});

    auto moe_root_m =
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_compressed_gemm3_m, moe_compressed_gemm2_no_zp_m, moe_compressed_gemm2_zp_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root_node = m.get_match_root();
        auto moe_compressed = ov::as_type_ptr<ov::intel_gpu::op::MOECompressed>(root_node);
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }

        std::string support_gpt_oss = getenv("OV_FUSED_COMPRESSED_SUPPORT_GPT_OSS");
        if (support_gpt_oss == "0") {
            if (moe_compressed->get_config().fused_gate_up == true) {
                std::cout << "FuseMOE3GemmCompressed skipped for MOECompressed with GPT-OSS format." << std::endl;
                return false;
            }
        }

        // Router logits source:
        //   - GEMM3 (Qwen): MatMul(hidden_state, routers)
        //   - GEMM2 (GPT-OSS): input to TopK (arbitrary subgraph)
        ov::Output<ov::Node> router_logits;
        if (pattern_map.count(router_matmul_m) > 0) {
            router_logits = pattern_map.at(router_matmul_m);
        } else if (pattern_map.count(g2_logits_m) > 0) {
            router_logits = pattern_map.at(g2_logits_m);
        } else {
            return false;
        }

        // Construct arguments for Fused Op
        OutputVector args;
        args.reserve(moe_compressed->get_input_size());

        // Arg 0: Hidden States
        args.push_back(pattern_map.at(hidden_state_m));

        // Arg 1: Router Logits (Replacing original Weight + Index inputs with pure Logits)
        // Note: Fused kernel will re-compute TopK/Softmax internally
        args.push_back(router_logits);

        // Args 2...N: Parameters (Gate/Up/Down Weights, Scales, ZP, Bias)
        // Inputs 0, 1, 2 of MOECompressed are Hidden, Weights, Indices.
        // Parameters start at index 3. We simply copy them over.
        for (size_t i = 3; i < moe_compressed->get_input_size(); i++) {
            args.push_back(moe_compressed->input_value(i));
        }

        auto moe_3gemm_fused_compressed = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, moe_compressed->get_config());
        moe_3gemm_fused_compressed->set_friendly_name(moe_compressed->get_friendly_name());
        ov::copy_runtime_info(moe_compressed, moe_3gemm_fused_compressed);
        ov::replace_node(moe_compressed, moe_3gemm_fused_compressed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root_m, "FuseMOE3GemmCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
