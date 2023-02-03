// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise.hpp"

#include <dnnl_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"

#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "emitters/x64/jit_emitter.hpp"
#include "emitters/x64/jit_eltwise_emitters.hpp"
#include "emitters/x64/jit_dnnl_emitters.hpp"
#include "emitters/x64/jit_bf16_emitters.hpp"

namespace ov {
namespace intel_cpu {

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_eltwise_call_args_ptrs, field)

template<typename T>
struct SupportedPrecisions {
    void operator()(std::set<Precision> &precisions) {
        precisions = T::get_supported_precisions();
    }
};

struct EltwiseEmitterContext {
    std::shared_ptr<jit_emitter> emitter;
    jit_generator *host;
    cpu_isa_t host_isa;
    const EltwiseAttrs& opData;
    InferenceEngine::Precision exec_prc;
};

template<typename T>
struct EltwiseEmitter {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<T>(ctx.host, ctx.host_isa, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_dnnl_aux_emitter> {
    void operator()(EltwiseEmitterContext & ctx) {
        auto algKind = static_cast<dnnl_alg_kind_t>(DnnlExtensionUtils::convertToDnnlAlgorithm(ctx.opData.algorithm));
        ctx.emitter = std::make_shared<jit_dnnl_aux_emitter>(ctx.host, ctx.host_isa, algKind,
                                                               ctx.opData.alpha, ctx.opData.beta, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_power_static_emitter> {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<jit_power_static_emitter>(ctx.host, ctx.host_isa, ctx.opData.alpha,
                                                                 ctx.opData.beta, ctx.opData.gamma, ctx.exec_prc);
    }
};

template<>
struct EltwiseEmitter<jit_is_inf_emitter> {
    void operator()(EltwiseEmitterContext & ctx) {
        ctx.emitter = std::make_shared<jit_is_inf_emitter>(ctx.host, ctx.host_isa, ctx.exec_prc, ctx.opData.alpha, ctx.opData.beta);
    }
};

template <cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    explicit jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                                     const std::vector<EltwisePostOp>& eltwise_post_ops)
    : jit_uni_eltwise_kernel(jep), jit_generator(jit_name()), eltwise_post_ops_(eltwise_post_ops) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        Precision exec_prc = Precision::UNSPECIFIED;

        std::set<Precision> supported_precision_intersection = get_supported_precisions(jep_.attrs.algorithm);
        for (size_t i = 0; i < eltwise_post_ops_.size(); ++i) {
            if (eltwise_post_ops_[i].type == EltwisePostOpType::Eltwise) {
                std::set<Precision> prcs = get_supported_precisions(eltwise_post_ops_[i].eltwise.algorithm);
                std::set<Precision> prcs_intersect = {};

                std::set_intersection(supported_precision_intersection.begin(), supported_precision_intersection.end(),
                                    prcs.begin(), prcs.end(), std::inserter(prcs_intersect, prcs_intersect.begin()));

                supported_precision_intersection = prcs_intersect;
            }
        }

        static const Precision exec_precisions_priority[] = {
                Precision::U8,
                Precision::I8,
                Precision::U16,
                Precision::I16,
                Precision::BF16,
                Precision::I32,
                Precision::FP32
        };

        for (auto prc : exec_precisions_priority) {
            if (std::find(supported_precision_intersection.begin(), supported_precision_intersection.end(), prc) != supported_precision_intersection.end()) {
                exec_prc = prc;
                break;
            }
        }

        for (int i = 0; i < jep_.inputs_number; i++) {
            if (jep_.src_prc[i] != exec_prc) {
                exec_prc = Precision::FP32;
                break;
            }
        }

        if (exec_prc == Precision::UNSPECIFIED) {
            IE_THROW() << "Eltwise jitter failed to specify execution precision for Eltwise node";
        }

        eltwise_emitter = create_eltwise_emitter(jep_.attrs, exec_prc);
        for (int i = 0; i < eltwise_post_ops_.size(); ++i) {
            if (eltwise_post_ops_[i].type == EltwisePostOpType::Eltwise) {
                post_op_emitters.push_back(create_eltwise_emitter(eltwise_post_ops_[i].eltwise, exec_prc));
            } else if (eltwise_post_ops_[i].type == EltwisePostOpType::Dnnl) {
                const auto& p = eltwise_post_ops_[i].dnnlPostOps.get();
                if (p->len() != 1 || !p->entry_[0].is_quantization()) {
                    IE_THROW() << "Eltwise jitter error. Unsupported post op detected";
                }
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, p->entry_[0], vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        if (mayiuse(avx512_core))
            uni_vcvtneps2bf16.reset(new jit_uni_vcvtneps2bf16(this, isa));

        const auto &jep = jep_;

        this->preamble();

        const int offset_count = jep.input_size - 1;

        // ptrs initializing
        auto init_ptrs_with_offsets = [this, offset_count](Reg64 pointer, const std::vector<size_t>& offsets) {
            for (int j = 0; j < offset_count; j++) {
                if (jep_.dims[j] != 1 && offsets[j] != 0) {
                    mov(reg_tmp_64, offsets[j]);
                    imul(reg_tmp_64, ptr[reg_indexes + j * sizeof(size_t)]);
                    add(pointer, reg_tmp_64);
                }
            }
        };

        for (int i = 0; i < jep.inputs_number; i++) {
            mov(get_src_reg(i), ptr[reg_const_params + GET_OFF(src_ptr[0]) + i * sizeof(size_t)]);
            init_ptrs_with_offsets(get_src_reg(i), jep.src_offsets[i]);
        }

        mov(reg_dst, ptr[reg_const_params + GET_OFF(dst_ptr)]);
        init_ptrs_with_offsets(reg_dst, jep.dst_offsets);

        mov(reg_post_op_ptrs, ptr[reg_const_params + GET_OFF(post_op_data)]);

        xor_(reg_oc_off, reg_oc_off);
        init_ptrs_with_offsets(reg_oc_off, jep.oc_offsets);

        mov(reg_work_amount, jep.work_amount);

        Xbyak::Label unroll_loop_label;
        Xbyak::Label unroll_loop_end_label;
        Xbyak::Label main_loop_label;
        Xbyak::Label main_loop_end_label;
        Xbyak::Label tail_loop_label;
        Xbyak::Label tail_loop_end_label;

        if (isa == x64::avx512_core)
            vpxord(vmm_zero, vmm_zero, vmm_zero);

        for (int i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] == 1)
                load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, true);
        }

        size_t min_src_size = jep.dst_size;
        for (int i = 0; i < jep.inputs_number; i++) {
            if (jep.src_size[i] != 1)
                min_src_size = std::min(min_src_size, jep.src_size[i]);
        }
        if (jep_.oc_size > 1)
            min_src_size = std::min(min_src_size, jep_.oc_size);

        if (min_src_size != jep.dst_size) {
            bool is_valid_configuration = true;
            if (jep.dst_size % min_src_size != 0)
                is_valid_configuration = false;

            for (int i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1 && jep.src_size[i] != min_src_size && jep.src_size[i] != jep.dst_size)
                    is_valid_configuration = false;
            }

            if (jep_.oc_size > 1 && jep_.oc_size != min_src_size && jep_.oc_size != jep.dst_size)
                is_valid_configuration = false;

            if (!is_valid_configuration)
                IE_THROW() << "Eltwise jitter has invalid configuration for Eltwise node";

            L(unroll_loop_label);
            {
                size_t loop_step = min_src_size;
                size_t vec_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(unroll_loop_end_label, T_NEAR);

                for (int j = 0; j < min_src_size / vec_step; j++) {
                    for (int i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_vector(get_vmm_reg(i), ptr[get_src_reg(i) + j * vec_step * jep.src_prc[i].size()], jep.src_prc[i], exec_prc, false);
                    }

                    compute_eltwise_op();

                    apply_post_ops(false, jep_.oc_size > 1 ? j * vec_step * sizeof(float) : 0);

                    store_vector(ptr[reg_dst + j * vec_step * jep.dst_prc.size()], vmm_dst, exec_prc, jep.dst_prc);
                }

                int tail_start = min_src_size - min_src_size % vec_step;
                for (int j = tail_start; j < min_src_size; j++) {
                    for (int i = 0; i < jep.inputs_number; i++) {
                        if (jep.src_size[i] != 1)
                            load_scalar(get_xmm_reg(i), ptr[get_src_reg(i) + j * jep.src_prc[i].size()], jep.src_prc[i], exec_prc);
                    }

                    compute_eltwise_op();

                    apply_post_ops(true, jep_.oc_size > 1 ? j * sizeof(float) : 0);

                    store_scalar(ptr[reg_dst + j * jep.dst_prc.size()], xmm_dst, exec_prc, jep.dst_prc);
                }

                for (int i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] == jep.dst_size)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1 && jep_.oc_size != min_src_size)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(unroll_loop_label, T_NEAR);
            }

            L(unroll_loop_end_label);
        }

        if (min_src_size == jep.dst_size) {
            L(main_loop_label);
            {
                size_t loop_step = cpu_isa_traits<isa>::vlen / exec_prc.size();

                cmp(reg_work_amount, loop_step);
                jl(main_loop_end_label, T_NEAR);

                for (int i = 0; i < jep.inputs_number; i++) {
                    if (jep.src_size[i] != 1)
                        load_vector(get_vmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc, false);
                }

                compute_eltwise_op();

                apply_post_ops(false);

                store_vector(ptr[reg_dst], vmm_dst, exec_prc, jep.dst_prc);

                for (int i = 0; i < jep.inputs_number; i++)
                    if (jep.src_size[i] != 1)
                        add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

                add(reg_dst, jep.dst_prc.size() * loop_step);
                sub(reg_work_amount, loop_step);
                if (jep_.oc_size > 1)
                    add(reg_oc_off, loop_step * sizeof(float));

                jmp(main_loop_label, T_NEAR);
            }

            L(main_loop_end_label);
        }

        L(tail_loop_label);
        {
            size_t loop_step = 1;

            cmp(reg_work_amount, loop_step);
            jl(tail_loop_end_label, T_NEAR);

            for (int i = 0; i < jep.inputs_number; i++) {
                if (jep.src_size[i] != 1)
                    load_scalar(get_xmm_reg(i), ptr[get_src_reg(i)], jep.src_prc[i], exec_prc);
            }

            compute_eltwise_op();

            apply_post_ops(true);

            store_scalar(ptr[reg_dst], xmm_dst, exec_prc, jep.dst_prc);

            for (int i = 0; i < jep.inputs_number; i++)
                if (jep.src_size[i] != 1)
                    add(get_src_reg(i), jep.src_prc[i].size() * loop_step);

            add(reg_dst, jep.dst_prc.size() * loop_step);
            sub(reg_work_amount, loop_step);
            if (jep_.oc_size > 1)
                add(reg_oc_off, loop_step * sizeof(float));

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_end_label);

        this->postamble();

        if (uni_vcvtneps2bf16)
            uni_vcvtneps2bf16->emit_data();

        eltwise_emitter->emit_data();
        for (int i = 0; i < post_op_emitters.size(); i++) {
            post_op_emitters[i]->emit_data();
        }
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xmm, isa == x64::avx2, Ymm, Zmm>::type;

    Reg64 get_src_reg(int idx) {
        return Reg64(r8.getIdx() + idx);
    }

    Vmm get_vmm_reg(int idx) {
        return Vmm(1 + idx);
    }

    Vmm get_aux_vmm(int idx) {
        return Vmm(10 + idx);
    }

    Xmm get_xmm_reg(int idx) {
        return Xmm(get_vmm_reg(idx).getIdx());
    }

    Reg64 reg_post_op_ptrs = rax;
    Reg64 reg_dst = rbx;
    Reg64 reg_work_amount = rdx;

    Reg64 reg_oc_off = abi_not_param1;
    Reg64 reg_const_params = abi_param1;
    Reg64 reg_indexes = abi_param2;  // reg_d_bias

    Reg8 reg_tmp_8 = Reg8(r15.getIdx());
    Reg32 reg_tmp_32 = Reg32(r15.getIdx());
    Reg64 reg_tmp_64 = Reg64(r15.getIdx());

    Reg64 reg_d_weights = rbp;
    Reg64 reg_d_bias = rsi;

    Vmm vmm_dst = Vmm(9);
    Xmm xmm_dst = Xmm(9);

    Vmm vmm_d_weights = Vmm(12);
    Vmm vmm_d_bias = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    std::shared_ptr<jit_uni_vcvtneps2bf16> uni_vcvtneps2bf16;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters = {};

    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors = {};

    const std::vector<EltwisePostOp>& eltwise_post_ops_;

    std::set<Precision> get_supported_precisions(Algorithm algo) {
        std::set<Precision> precisions;

        OV_SWITCH(intel_cpu, SupportedPrecisions, precisions, algo,
        OV_CASE(Algorithm::EltwiseRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluErf, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseElu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAbs, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSqrt, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSoftRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseExp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseClamp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSwish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHswish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseMish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHsigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfToEven, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfAwayFromZero, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
        OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
        OV_CASE(Algorithm::EltwiseFloorMod, jit_floor_mod_emitter),
        OV_CASE(Algorithm::EltwiseMod, jit_mod_emitter),
        OV_CASE(Algorithm::EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(Algorithm::EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(Algorithm::EltwiseSquaredDifference, jit_squared_difference_emitter),
        OV_CASE(Algorithm::EltwisePowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Algorithm::EltwiseEqual, jit_equal_emitter),
        OV_CASE(Algorithm::EltwiseNotEqual, jit_not_equal_emitter),
        OV_CASE(Algorithm::EltwiseGreater, jit_greater_emitter),
        OV_CASE(Algorithm::EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Algorithm::EltwiseLess, jit_less_emitter),
        OV_CASE(Algorithm::EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(Algorithm::EltwiseLogicalAnd, jit_logical_and_emitter),
        OV_CASE(Algorithm::EltwiseLogicalOr, jit_logical_or_emitter),
        OV_CASE(Algorithm::EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(Algorithm::EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
        OV_CASE(Algorithm::EltwiseErf, jit_erf_emitter),
        OV_CASE(Algorithm::EltwiseSoftSign, jit_soft_sign_emitter),
        OV_CASE(Algorithm::EltwiseIsFinite, jit_is_finite_emitter),
        OV_CASE(Algorithm::EltwiseIsInf, jit_is_inf_emitter),
        OV_CASE(Algorithm::EltwiseIsNaN, jit_is_nan_emitter));

        if (precisions.empty())
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return precisions;
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseAttrs& attrs, Precision exec_prec) {
        EltwiseEmitterContext ctx = {
            nullptr,
            this,
            isa,
            attrs,
            exec_prec
        };

        OV_SWITCH(intel_cpu, EltwiseEmitter, ctx, attrs.algorithm,
        OV_CASE(Algorithm::EltwiseRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluErf, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseGeluTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseElu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseTanh, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAbs, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSqrt, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSoftRelu, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseExp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseClamp, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseSwish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHswish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseMish, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseHsigmoid, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfToEven, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseRoundHalfAwayFromZero, jit_dnnl_aux_emitter),
        OV_CASE(Algorithm::EltwiseAdd, jit_add_emitter),
        OV_CASE(Algorithm::EltwiseMulAdd, jit_mul_add_emitter),
        OV_CASE(Algorithm::EltwiseSubtract, jit_subtract_emitter),
        OV_CASE(Algorithm::EltwiseMultiply, jit_multiply_emitter),
        OV_CASE(Algorithm::EltwiseDivide, jit_divide_emitter),
        OV_CASE(Algorithm::EltwiseFloorMod, jit_floor_mod_emitter),
        OV_CASE(Algorithm::EltwiseMod, jit_mod_emitter),
        OV_CASE(Algorithm::EltwiseMaximum, jit_maximum_emitter),
        OV_CASE(Algorithm::EltwiseMinimum, jit_minimum_emitter),
        OV_CASE(Algorithm::EltwiseSquaredDifference, jit_squared_difference_emitter),
        OV_CASE(Algorithm::EltwisePowerDynamic, jit_power_dynamic_emitter),
        OV_CASE(Algorithm::EltwiseEqual, jit_equal_emitter),
        OV_CASE(Algorithm::EltwiseNotEqual, jit_not_equal_emitter),
        OV_CASE(Algorithm::EltwiseGreater, jit_greater_emitter),
        OV_CASE(Algorithm::EltwiseGreaterEqual, jit_greater_equal_emitter),
        OV_CASE(Algorithm::EltwiseLess, jit_less_emitter),
        OV_CASE(Algorithm::EltwiseLessEqual, jit_less_equal_emitter),
        OV_CASE(Algorithm::EltwiseLogicalAnd, jit_logical_and_emitter),
        OV_CASE(Algorithm::EltwiseLogicalOr, jit_logical_or_emitter),
        OV_CASE(Algorithm::EltwiseLogicalXor, jit_logical_xor_emitter),
        OV_CASE(Algorithm::EltwiseLogicalNot, jit_logical_not_emitter),
        OV_CASE(Algorithm::EltwisePowerStatic, jit_power_static_emitter),
        OV_CASE(Algorithm::EltwisePrelu, jit_prelu_emitter),
        OV_CASE(Algorithm::EltwiseErf, jit_erf_emitter),
        OV_CASE(Algorithm::EltwiseSoftSign, jit_soft_sign_emitter),
        OV_CASE(Algorithm::EltwiseIsFinite, jit_is_finite_emitter),
        OV_CASE(Algorithm::EltwiseIsInf, jit_is_inf_emitter),
        OV_CASE(Algorithm::EltwiseIsNaN, jit_is_nan_emitter));

        if (!ctx.emitter)
            IE_THROW() << "Unsupported operation type for Eltwise emitter";

        return ctx.emitter;
    }

    inline void compute_eltwise_op() {
        std::vector<size_t> in_idxs;
        std::vector<size_t> aux_idxs;
        for (int i = 0; i < eltwise_emitter->get_inputs_num(); i++)
            in_idxs.push_back(get_vmm_reg(i).getIdx());
        for (int i = 0; i < eltwise_emitter->aux_vecs_count(); i++)
            aux_idxs.push_back(get_aux_vmm(i).getIdx());

        std::vector<size_t> out_idxs;
        out_idxs.push_back(vmm_dst.getIdx());

        eltwise_emitter->emit_code(in_idxs, out_idxs, aux_idxs);
    }

    inline void apply_post_ops(bool is_scalar, int offset = 0) {
        int input_idx = eltwise_emitter->get_inputs_num();
        int eltwise_post_op_idx = 0;
        int quantization_post_op_idx = 0;
        for (int i = 0; i < eltwise_post_ops_.size(); i++) {
            if (eltwise_post_ops_[i].type == EltwisePostOpType::Eltwise) {
                std::vector<size_t> in_idxs;
                std::vector<size_t> aux_idxs;
                in_idxs.push_back(vmm_dst.getIdx());
                for (int j = 1; j < post_op_emitters[eltwise_post_op_idx]->get_inputs_num(); j++)
                    in_idxs.push_back(get_vmm_reg(input_idx++).getIdx());
                for (int j = 0; j < post_op_emitters[eltwise_post_op_idx]->aux_vecs_count(); j++)
                    aux_idxs.push_back(get_aux_vmm(j).getIdx());

                std::vector<size_t> out_idxs;
                out_idxs.push_back(vmm_dst.getIdx());

                post_op_emitters[eltwise_post_op_idx]->emit_code(in_idxs, out_idxs, aux_idxs);

                eltwise_post_op_idx++;
            } else if (eltwise_post_ops_[i].type == EltwisePostOpType::Dnnl) {
                auto& p = eltwise_post_ops_[i].dnnlPostOps.get()->entry_[0];
                bool do_dequantization = p.quantization.alg == dnnl::impl::alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || jep_.dst_prc == Precision::FP32 || i != eltwise_post_ops_.size() - 1;
                int s_idx = vmm_dst.getIdx();

                size_t ptrs_table_off = quantization_post_op_idx * quantization_injectors[quantization_post_op_idx]->memoryStep();

                quantization_injectors[quantization_post_op_idx]->init_crop_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_crop(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_input_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_input_scale_shift(s_idx, s_idx + 1, offset, do_rounding,
                                                                                            is_scalar, jep_.oc_size == 1);

                quantization_injectors[quantization_post_op_idx]->init_output_scale_shift_ptrs(reg_post_op_ptrs + ptrs_table_off, reg_oc_off);
                quantization_injectors[quantization_post_op_idx]->compute_output_scale_shift(s_idx, s_idx + 1, offset, is_scalar, jep_.oc_size == 1);

                quantization_post_op_idx++;
            } else {
                IE_THROW(Unexpected) << "Eltwise jit kernel: unexpected operation type";
            }
        }
    }

    inline void load_vector(Vmm vmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc, bool broadcast) {
        Xmm xmm_src = Xmm(vmm_src.getIdx());

        if (broadcast) {
            load_scalar(xmm_src, op, src_prc, dst_prc);
            uni_vbroadcastss(vmm_src, xmm_src);
        } else {
            switch (src_prc) {
                case Precision::FP32:
                case Precision::I32:
                    uni_vmovups(vmm_src, op);
                    break;
                case Precision::BF16:
                    vpmovzxwd(vmm_src, op);
                    uni_vpslld(vmm_src, vmm_src, 16);
                    break;
                case Precision::U16:
                    uni_vpmovzxwd(vmm_src, op);
                    break;
                case Precision::I16:
                    uni_vpmovsxwd(vmm_src, op);
                    break;
                case Precision::I8:
                    uni_vpmovsxbd(vmm_src, op);
                    break;
                case Precision::U8:
                    uni_vpmovzxbd(vmm_src, op);
                    break;
                default:
                    assert(!"unknown src_prc");
            }

            switch (dst_prc) {
                case Precision::FP32:
                    if (src_prc != Precision::FP32 && src_prc != Precision::BF16)
                        uni_vcvtdq2ps(vmm_src, vmm_src);
                    break;
                case Precision::I32:
                    if (src_prc == Precision::FP32 || src_prc == Precision::BF16)
                        uni_vcvtps2dq(vmm_src, vmm_src);
                    break;
                default:
                    assert(!"unknown dst_prc");
            }
        }
    }

    inline void load_scalar(Xmm xmm_src, const Xbyak::Address &op, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(xmm_src, op);
                break;
            case Precision::BF16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpslld(xmm_src, xmm_src, 16);
                break;
            case Precision::I16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovsxwd(xmm_src, op);
                break;
            case Precision::U16:
                uni_vpinsrw(xmm_src, xmm_src, op, 0);
                uni_vpmovzxwd(xmm_src, op);
                break;
            case Precision::I8:
                movsx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            case Precision::U8:
                movzx(reg_tmp_32, op);
                uni_vmovq(xmm_src, reg_tmp_64);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
                if (src_prc != Precision::FP32 && src_prc != Precision::BF16)
                    uni_vcvtdq2ps(xmm_src, xmm_src);
                break;
            case Precision::I32:
                if (src_prc == Precision::FP32 || src_prc == Precision::BF16)
                    uni_vcvtps2dq(xmm_src, xmm_src);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_vector(const Xbyak::Address &op, Vmm vmm_dst, Precision src_prc, Precision dst_prc) {
        Xmm xmm_dst = Xmm(vmm_dst.getIdx());
        Ymm ymm_dst = Ymm(vmm_dst.getIdx());

        switch (src_prc) {
            case Precision::FP32:
                if (dst_prc != Precision::FP32 && dst_prc != Precision::BF16)
                    uni_vcvtps2dq(vmm_dst, vmm_dst);
                break;
            case Precision::I32:
                if (dst_prc == Precision::FP32 || dst_prc == Precision::BF16)
                    uni_vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovups(op, vmm_dst);
                break;
            case Precision::BF16:
                uni_vcvtneps2bf16->emit_code({static_cast<size_t>(vmm_dst.getIdx())}, {static_cast<size_t>(ymm_dst.getIdx())});
                vmovdqu16(op, ymm_dst);
                break;
            case Precision::I16:
                if (isa == x64::avx512_core) {
                    vpmovsdw(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::U16:
                if (isa == x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdw(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41) {
                        vpermq(ymm_dst, ymm_dst, 0x08);
                        uni_vmovdqu(op, xmm_dst);
                    } else {
                        movq(op, xmm_dst);
                    }
                }
                break;
            case Precision::I8:
                if (isa == x64::avx512_core) {
                    vpmovsdb(op, vmm_dst);
                } else {
                    uni_vpackssdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpacksswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            case Precision::U8:
                if (isa == x64::avx512_core) {
                    vpmaxsd(vmm_dst, vmm_zero, vmm_dst);
                    vpmovusdb(op, vmm_dst);
                } else {
                    uni_vpackusdw(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vpermq(ymm_dst, ymm_dst, 0x08);
                    uni_vpackuswb(vmm_dst, vmm_dst, vmm_dst);
                    if (isa != x64::sse41)
                        vmovq(op, xmm_dst);
                    else
                        movd(op, xmm_dst);
                }
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }

    inline void store_scalar(const Xbyak::Address &op, Xmm xmm_dst, Precision src_prc, Precision dst_prc) {
        switch (src_prc) {
            case Precision::FP32:
                if (dst_prc != Precision::FP32 && dst_prc != Precision::BF16)
                    uni_vcvtps2dq(xmm_dst, xmm_dst);
                break;
            case Precision::I32:
                if (dst_prc == Precision::FP32 || dst_prc == Precision::BF16)
                    uni_vcvtdq2ps(xmm_dst, xmm_dst);
                break;
            default:
                assert(!"unknown src_prc");
        }

        switch (dst_prc) {
            case Precision::FP32:
            case Precision::I32:
                uni_vmovss(op, xmm_dst);
                break;
            case Precision::BF16:
                uni_vpsrld(xmm_dst, xmm_dst, 16);
                uni_vpextrw(op, xmm_dst, 0x0);
                break;
            case Precision::I16:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U16:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::I8:
                uni_vpackssdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpacksswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            case Precision::U8:
                uni_vpackusdw(xmm_dst, xmm_dst, xmm_dst);
                uni_vpackuswb(xmm_dst, xmm_dst, xmm_dst);
                movq(reg_tmp_64, xmm_dst);
                mov(op, reg_tmp_8);
                break;
            default:
                assert(!"unknown dst_prc");
        }
    }
};

static void offset_out_calc(VectorDims& offset, const VectorDims& dims) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

static void offset_in_calc(VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

JitEltwiseExecutor::JitEltwiseExecutor(const ExecutorContext::CPtr context) : EltwiseExecutor(context) {}

bool JitEltwiseExecutor::init(const EltwiseAttrs& eltwiseAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              const std::vector<EltwisePostOp>& postOps) {
    // TODO: add support
    bool useDynBatch = false;

    auto outBlockingDesc = MemoryDescUtils::convertToBlockedMemoryDesc(dstDescs[0]);
    const auto &outOrder = outBlockingDesc->getOrder();
    const auto &currentOutBlkDims = outBlockingDesc->getBlockDims();

    size_t input_size = std::max(static_cast<size_t>(JitEltwiseExecutor::optimalTensorRank), currentOutBlkDims.size());
    auto inputNum = srcDescs.size();
    // init dims
    std::vector<VectorDims> inpDims(inputNum);
    for (int i = 0; i < inputNum; i++) {
        inpDims[i].resize(input_size, 1);
    }

    size_t outRank = currentOutBlkDims.size();
    for (int i = 0; i < inputNum; i++) {
        std::vector<VectorDims> currentInBlkDims(inputNum);
        auto inBlockingDesc = MemoryDescUtils::convertToBlockedMemoryDesc(srcDescs[i]);
        currentInBlkDims[i] = inBlockingDesc->getBlockDims();
        size_t inRank = currentInBlkDims[i].size();

        // WA to normalize blocked and planar layouts
        const auto &inOrder = inBlockingDesc->getOrder();
        size_t startOff = outOrder.size() != outBlockingDesc->getShape().getRank() &&
                          outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

        // WA to handle nspc layout with 1D tensors
        if (1 == inRank) {
            if (outRank > 2 && 1 == outOrder.back()) startOff = 1;
        }

        for (int j = 0; j < inRank; j++) {
            inpDims[i][inpDims[i].size() - 1 - j - startOff] = currentInBlkDims[i][inRank - 1 - j];
        }
    }

    auto collapseLastDims = [](std::vector<size_t>& dims, int dimsToCollapse) {
        for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
            dims[dims.size() - 1] *= dims[i];
        }

        for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
            dims[i] = dims[i - dimsToCollapse];
        }

        for (int i = dimsToCollapse - 1; i >= 0; i--) {
            dims[i] = 1;
        }
    };

    auto collapseLastOffsets = [](std::vector<size_t>& dims, int dimsToCollapse) {
        for (int i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
            if (dims[dims.size() - 1] > 0 || dims[i] > 0)
                dims[dims.size() - 1] = std::max(dims[dims.size() - 1], static_cast<size_t>(1)) * std::max(dims[i], static_cast<size_t>(1));
            else
                dims[dims.size() - 1] *= dims[i];
        }

        for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
            dims[i] = dims[i - dimsToCollapse];
        }

        for (int i = dimsToCollapse - 1; i >= 0; i--) {
            dims[i] = 0;
        }
    };

    auto isFusedWith = [&](EltwisePostOpType type_) {
        auto start_itr = postOps.begin();
        for (const auto& postOp : postOps) {
            if (postOp.type == type_)
                return true;
        }
        return false;
    };

    if (inpDims.empty()) {
        IE_THROW() << "Can not make Eltwise executor from empty input dims array";
    } else if (inpDims.front().empty()) {
        IE_THROW() << "Can not make Eltwise executor from empty input dims members";
    }

    size_t inputsNumber = inpDims.size();

    jep.attrs = eltwiseAttrs;
    jep.input_size = inpDims.front().size();

    jep.dims.resize(jep.input_size, 1);

    if (currentOutBlkDims.empty()) {
        IE_THROW() << "Can not make Eltwise executor from empty block dims vector";
    }

    for (int i = 0; i < outRank; i++) {
        jep.dims[jep.dims.size() - 1 - i] = currentOutBlkDims[outRank - 1 - i];
    }

    for (int i = 0; i < inpDims.size(); i++) {
        for (int j = 0; j < inpDims[i].size(); j++) {
            if (inpDims[i][j] != jep.dims[j] && inpDims[i][j] != 1)
                IE_THROW() << "Eltwise executor got invalid input/output dims configuration.";
        }
    }

    if (currentOutBlkDims.size() != outOrder.size()) {
        IE_THROW() << "Can not make Elwtise executor due to out blocked dims and out order vectors size mismatch.";
    }

    int lastUnchangedAxis = 0;
    size_t oc_size = 0;
    jep.oc_offsets.resize(jep.input_size, 0);
    std::fill(jep.oc_offsets.begin(), jep.oc_offsets.end(), 0);
    if (isFusedWith(EltwisePostOpType::Dnnl)) {
        size_t offset_oc = 1;
        for (int i = outOrder.size() - 1; i >= 0; i--) {
            if (outOrder[i] == 1) {
                int oc_dim_idx = i + (jep.input_size - outOrder.size());
                jep.oc_offsets[oc_dim_idx] = offset_oc;
                offset_oc *= jep.dims[oc_dim_idx];
                if (oc_dim_idx + 1 != jep.input_size) { // since in nspc case we can safely collapse the last axis
                    lastUnchangedAxis = oc_dim_idx;
                }
            }
        }
        oc_size = jep.oc_offsets[jep.dims.size() - 1] != 0 ? jep.dims[jep.dims.size() - 1] : 1;
    }

    int maxCollapsedDims = static_cast<int>(jep.dims.size()) - lastUnchangedAxis - 2;

    size_t fullWorkAmount = 1;
    for (int i = 0; i < jep.dims.size(); i++) {
        fullWorkAmount *= jep.dims[i];
    }

    size_t minimalConcurrency = parallel_get_max_threads();
    size_t minimalJitWorkAmount = 256;
    size_t currentJitWorkAmount = jep.dims[jep.dims.size() - 1];
    int collapsedDims = 0;

    bool hasDifferentDims = false;
    while (currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount &&
            // we shouldn't collapse batch dimension in case dynamic batch is enabled
            (!useDynBatch || (currentOutBlkDims.size() - collapsedDims > 2))) {
        if (collapsedDims >= maxCollapsedDims)
            break;

        for (int j = 1; j < inpDims.size(); j++) {
            if (inpDims[j].back() != inpDims[0].back()) {
                hasDifferentDims = true;
            }
        }

        if (oc_size > 1 && oc_size != inpDims[0][inpDims[0].size() - 1]) {
            hasDifferentDims = true;
        }

        bool canCollapse = true;
        for (int i = 0; i < inpDims.size(); i++) {
            if (inpDims[i][inpDims[i].size() - 2] != 1) {
                if (hasDifferentDims) {
                    canCollapse = false;
                    break;
                }
            }
        }

        if (!canCollapse) {
            break;
        }

        size_t nextJitWorkAmount = currentJitWorkAmount * jep.dims[jep.dims.size() - 2];
        if (fullWorkAmount / nextJitWorkAmount >= minimalConcurrency) {
            currentJitWorkAmount = nextJitWorkAmount;
            collapsedDims++;

            for (int i = 0; i < inpDims.size(); i++) {
                collapseLastDims(inpDims[i], 1);
            }
            collapseLastDims(jep.dims, 1);

            if (isFusedWith(EltwisePostOpType::Dnnl)) {
                collapseLastOffsets(jep.oc_offsets, 1);
            }
        } else {
            break;
        }
    }

    _batchDimIdx = jep.input_size - currentOutBlkDims.size() + collapsedDims;
    _schedulerWorkAmount = fullWorkAmount / jep.dims[jep.dims.size() - 1];

    // init offset
    jep.dst_offsets.resize(jep.input_size, 1);
    offset_out_calc(jep.dst_offsets, jep.dims);
    for (int j = 0; j < jep.input_size; j++) {
        jep.dst_offsets[j] *= dstDescs[0]->getPrecision().size();
    }

    for (int i = 0; i < inputsNumber; i++) {
        jep.src_offsets[i].resize(jep.input_size, 1);
        offset_in_calc(jep.src_offsets[i], inpDims[i], jep.dims);
        for (int j = 0; j < jep.input_size; j++) {
            jep.src_offsets[i][j] *= srcDescs[i]->getPrecision().size();
        }
    }

    jep.inputs_number = inputsNumber;
    for (int i = 0; i < inputsNumber; i++) {
        jep.src_prc[i] = srcDescs[i]->getPrecision();
        jep.src_size[i] = inpDims[i][inpDims[i].size() - 1];
    }
    jep.dst_prc = dstDescs[0]->getPrecision();
    jep.work_amount = jep.dst_size = jep.dims.back();
    jep.oc_size = oc_size;

    std::transform(jep.oc_offsets.begin(), jep.oc_offsets.end(), jep.oc_offsets.begin(),
                    [](size_t& offset) { return offset * sizeof(float);});

    if (mayiuse(x64::avx512_core)) {
        _pKernel.reset(new jit_uni_eltwise_generic<x64::avx512_core>(jep, postOps));
        implType = impl_desc_type::jit_avx512;
    } else if (mayiuse(x64::avx2)) {
        _pKernel.reset(new jit_uni_eltwise_generic<x64::avx2>(jep, postOps));
        implType = impl_desc_type::jit_avx2;
    } else if (mayiuse(x64::sse41)) {
        _pKernel.reset(new jit_uni_eltwise_generic<x64::sse41>(jep, postOps));
        implType = impl_desc_type::jit_sse42;
    } else {
        IE_THROW() << "Can't create jit eltwise kernel";
    }

    if (_pKernel)
        _pKernel->create_ker();

    return true;
}

void JitEltwiseExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    if (!_pKernel)
        IE_THROW() << "Can't execute, kernel for eltwise node is not compiled";

    jit_eltwise_call_args_ptrs args_ptrs = {};
    VectorDims dims_out = jep.dims;
    for (int i = 0; i < src.size(); i++)
        args_ptrs.src_ptr[i] = reinterpret_cast<const uint8_t*>(src[i]->GetPtr());
    args_ptrs.dst_ptr = reinterpret_cast<uint8_t*>(dst[0]->GetPtr());

    // TODO: add support
    // In general case we need to recompute offsets as well but currently all supported layout assumes batch to be outermost dimension
    // if (isDynBatchEnabled) {
        // auto batchDimIdx = execPtr->getBatchDimIdx();
        // if (dims_out.size() <= batchDimIdx)
        //     IE_THROW() << "Can't set batch dims for eltwise node with rank: " << dims_out.size() << " and batch idx: " << batchDimIdx;
        // dims_out[batchDimIdx] = static_cast<size_t>(batchToProcess());
    // }

    args_ptrs.post_op_data = post_ops_data_;

    if (_pKernel->jep_.input_size == optimalTensorRank) {
        // execute Optimized 6D
        parallel_for5d(dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4],
                        [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                            auto args = jit_eltwise_call_args_indexes();
                            args.indexes[0] = i0;
                            args.indexes[1] = i1;
                            args.indexes[2] = i2;
                            args.indexes[3] = i3;
                            args.indexes[4] = i4;

                            (*_pKernel)(&args_ptrs, &args);
                        });
    } else {
        // execute Optimized Generic
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0;
            splitter(_schedulerWorkAmount, nthr, ithr, start, end);

            std::vector<size_t> counters(dims_out.size() - 1, 0);
            auto args = jit_eltwise_call_args_indexes();
            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t tmp = iwork;
                for (ptrdiff_t j = dims_out.size() - 2; j >= 0; j--) {
                    counters[j] = tmp % dims_out[j];
                    tmp /= dims_out[j];
                }

                for (size_t j = 0; j < counters.size(); j++)
                    args.indexes[j] = counters[j];

                (*_pKernel)(&args_ptrs, &args);
            }
        });
    }
}

}   // namespace intel_cpu
}   // namespace ov