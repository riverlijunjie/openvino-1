// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"

#include <dnnl_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "emitters/jit_bf16_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>

namespace ov {
namespace intel_cpu {

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

using namespace InferenceEngine;
using namespace dnnl;
using namespace dnnl::impl::utils;
using namespace Xbyak;

// some utility functions
static inline bool isFloatCompatible(Precision prc) {
    return Precision::FP32 == prc || Precision::BF16 == prc;
}

static inline bool isFloatCompatible(memory::data_type type) {
    return memory::data_type::f32 == type || memory::data_type::bf16 == type;
}

// normalize_variance = false : src->mean
// normalize_variance = true : src+mean->variance:sqr(x-mean)
template <cpu_isa_t isa>
struct jit_uni_mvn_mean_variance_kernel_f32 : public jit_uni_mvn_mean_variance_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_mean_kernel_f32)

    explicit jit_uni_mvn_mean_variance_kernel_f32(jit_mvn_config_params jcp) : jit_uni_mvn_mean_variance_kernel(jcp), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        tail_step = jcp_.layout == MVNLayoutType::mvn_planar ? (jcp_.D * jcp_.H * jcp_.W) - ((jcp_.D * jcp_.H * jcp_.W) / vector_step) * vector_step :
                   jcp_.C - (jcp_.C / vector_step) * vector_step;

        Precision dst_prc = isFloatCompatible(jcp_.src_prc) ? Precision::FP32 : Precision::I32;
        load_vector_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, vector_step));
        load_tail_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, tail_step));
        load_tail_with_fill_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, tail_step, Precision::FP32, true, "zero"));
        load_scalar_with_fill_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, dst_prc, scalar_step, Precision::FP32, true, "zero"));

        this->preamble();
        mov(reg_table, l_table);
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        if (jcp_.normalize_variance) {
            mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
            mov(reg_variance, ptr[reg_params + GET_OFF(variance)]);
            uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
        } else {
            mov(reg_sum, ptr[reg_params + GET_OFF(sum)]);
            uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
        }
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (jcp_.normalize_variance) {
            if (jcp_.layout == MVNLayoutType::mvn_planar || jcp_.across_channels) {
                uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            } else {
                uni_vmovups(vmm_mean, ptr[reg_mean]);
            }
        }

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};

        if (jcp_.layout == MVNLayoutType::mvn_planar) {
            worker_unroll();
            if (tail_step != 0) {
                worker_partial(false, true);
            }

            // hsum+store
            if (!jcp_.normalize_variance && !isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_sum, vmm_sum);
            Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
            reduce_sum_store_vmm(vmm_dst.getIdx());
        } else if (jcp_.layout == MVNLayoutType::mvn_by_channel) {
            if (jcp_.across_channels)
                nspc_ac_ker();
            else
                nspc_pc_ker();
        } else {
            // blk
            int repeats = (isa == cpu::x64::sse41) ? 2 : 1; // block size is also 8 on cpu::x64::sse41 with two step process
            int sse42_step = 4;
            for (int i = 0; i < repeats; i++) {
                int offset_sse42 = i * sse42_step;
                if (i > 0) {
                    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                    add(reg_src, offset_sse42 * jcp_.src_data_size);

                    if (jcp_.normalize_variance) {
                        // mean and vaiance for variance kernel
                        if (!jcp_.across_channels) {
                            // mean is bc when across_channel, no need shift
                            add(reg_mean, offset_sse42 * sizeof(float));
                            uni_vmovups(vmm_mean, ptr[reg_mean]);
                        }
                        add(reg_variance, offset_sse42 * sizeof(float));
                        uni_vpxor(vmm_variance, vmm_variance, vmm_variance);
                    } else {
                        // sum for mean kernel
                        add(reg_sum, offset_sse42 * sizeof(float));
                        uni_vpxor(vmm_sum, vmm_sum, vmm_sum);
                    }
                    add(reg_oc_off, offset_sse42 * sizeof(float));
                }

                Xbyak::Label label_empty_2half_sse42;
                if (tail_step == 0) {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    worker_unroll();
                } else {
                    // maybe tail blk
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    Xbyak::Label label_full_size;
                    Xbyak::Label label_size_end;
                    cmp(reg_oc_off, static_cast<int>((jcp_.C - vector_step) * sizeof(float)));
                    jle(label_full_size, T_NEAR);

                    // no need care and fill rest
                    // for per_channel, do not use tail mean(variance), do not store computed tail values.
                    // for across_channel, partial sum for tail one time out of kernel from perf.
                    worker_unroll(true);

                    jmp(label_size_end, T_NEAR);
                    L(label_full_size);
                    {
                        worker_unroll();
                    }
                    L(label_size_end);
                }

                // add input_base value and store for per_channel
                // store for across_channels
                if (jcp_.normalize_variance) {
                    if (!jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_variance]);
                        uni_vaddps(vmm_variance, vmm_variance, vmm_val);
                    }
                    uni_vmovups(ptr[reg_variance], vmm_variance);
                } else {
                    if (!isFloatCompatible(jcp_.src_prc))  // add with int for int-family data type, other compute go with float
                        uni_vcvtdq2ps(vmm_sum, vmm_sum);

                    if (!jcp_.across_channels) {
                        uni_vmovups(vmm_val, ptr[reg_sum]);
                        uni_vaddps(vmm_sum, vmm_sum, vmm_val);
                    }
                    uni_vmovups(ptr[reg_sum], vmm_sum);
                }

                L(label_empty_2half_sse42);
            }
        }

        this->postamble();

        load_vector_emitter->emit_data();
        load_tail_emitter->emit_data();
        load_tail_with_fill_emitter->emit_data();
        load_scalar_with_fill_emitter->emit_data();

        prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    int tail_step = 0;
    int scalar_step = 1;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    Xbyak::Reg64 reg_stride = r12;
    Xbyak::Reg64 reg_sum = reg_mean;
    Xbyak::Reg64 reg_params = abi_param1;
    Xbyak::Reg64 reg_load_table = r13;
    Xbyak::Reg64 reg_load_store_mask = r14;
    Xbyak::Reg64 reg_aux = r15;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_table = rdx;
    Xbyak::Label l_table;

    Vmm vmm_val = Vmm(1);
    Vmm vmm_mean = Vmm(2);
    Vmm vmm_variance = Vmm(3);
    Vmm vmm_sum = vmm_mean;
    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(4);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(5);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(6);
    Vmm vmm_zero = Vmm(0);
    // 8-15 for unloop

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);

    std::unique_ptr<jit_load_emitter> load_vector_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_tail_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_tail_with_fill_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_scalar_with_fill_emitter = nullptr;

    std::vector<size_t> load_pool_gpr_idxs;

    // nspc across channel
    inline void nspc_ac_ker() {
        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        Xbyak::Label scalar_loop_label;
        Xbyak::Label scalar_loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, vector_step);
            jl(loop_end_label, T_NEAR);

            worker_full_size();
            add(reg_src, vector_step * jcp_.src_data_size);

            sub(reg_work_amount, vector_step);
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);

        L(scalar_loop_label);
        {
            cmp(reg_work_amount, 1);
            jl(scalar_loop_end_label, T_NEAR);

            worker_partial(true, true);
            add(reg_src, scalar_step * jcp_.src_data_size);

            sub(reg_work_amount, scalar_step);
            jmp(scalar_loop_label, T_NEAR);
        }
        L(scalar_loop_end_label);

        if (!jcp_.normalize_variance && !isFloatCompatible(jcp_.src_prc))
            uni_vcvtdq2ps(vmm_sum, vmm_sum);
        Vmm vmm_dst = jcp_.normalize_variance ? vmm_variance : vmm_sum;
        reduce_sum_store_vmm(vmm_dst.getIdx());
    }

    // nspc per channel with unroll
    inline void nspc_pc_ker() {
        // 4 unroll vector
        size_t unroll_size = 4;
        size_t vec_num = div_up(jcp_.C, vector_step);
        unroll_size = vec_num >= unroll_size ? unroll_size : vec_num;
        size_t unroll_number = div_up(vec_num, unroll_size);

        int ur_base = 4;
        Xbyak::Reg64 reg_src_aux = reg_stride;
        Xbyak::Reg64 reg_work_amount_bk = rbx;
        mov(reg_work_amount_bk, reg_work_amount);
        for (int ur_num = 0; ur_num < unroll_number; ur_num++) {
            // 4-15 for unroll. 4-7 for src, 8-11 for m/v sum, 12-15 for mean
            int ur_offset_elt = ur_num * unroll_size * vector_step;
            int ur_offset = ur_offset_elt * sizeof(float);
            size_t unroll_size_rt = std::min(vec_num - ur_num * unroll_size, unroll_size);
            size_t elt_num = std::min(jcp_.C - ur_num * unroll_size * vector_step, unroll_size * vector_step);
            for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                uni_vpxor(Vmm(ur_base + 4 + ur_size), Vmm(ur_base + 4 + ur_size), Vmm(ur_base + 4 + ur_size));
            }
            if (jcp_.normalize_variance) {
                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    uni_vmovups(Vmm(ur_base + 8 + ur_size), ptr[reg_mean + ur_offset + ur_size * vector_step * sizeof(float)]);
                }
            }

            mov(reg_src_aux, reg_src);
            mov(reg_work_amount, reg_work_amount_bk);

            Xbyak::Label loop_label;
            Xbyak::Label loop_end_label;
            L(loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(loop_end_label, T_NEAR);

                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    bool is_tails = ur_offset_elt + ur_size * vector_step + vector_step > jcp_.C;
                    if (is_tails) {
                        load_tail_emitter->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                            {static_cast<size_t>(ur_base + ur_size)}, {}, {load_pool_gpr_idxs});
                        add(reg_src_aux, tail_step * jcp_.src_data_size);
                    } else {
                        load_vector_emitter->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                            {static_cast<size_t>(ur_base + ur_size)}, {}, {load_pool_gpr_idxs});
                        add(reg_src_aux, vector_step * jcp_.src_data_size);
                    }
                }
                add(reg_src_aux, (jcp_.C - elt_num) * jcp_.src_data_size);
                prefetcht0(ptr[reg_src_aux]);

                if (jcp_.normalize_variance) {
                    if (!isFloatCompatible(jcp_.src_prc)) {
                        for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                            uni_vcvtdq2ps(Vmm(ur_base + ur_size), Vmm(ur_base + ur_size));
                        }
                    }
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        uni_vsubps(Vmm(ur_base + ur_size), Vmm(ur_base + ur_size), Vmm(ur_base + 8 + ur_size));
                    }
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        uni_vfmadd231ps(Vmm(ur_base + 4 + ur_size), Vmm(ur_base + ur_size), Vmm(ur_base + ur_size));
                    }
                } else {
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        if (!isFloatCompatible(jcp_.src_prc))
                            uni_vpaddd(Vmm(ur_base + 4 + ur_size), Vmm(ur_base + 4 + ur_size), Vmm(ur_base + ur_size));
                        else
                            uni_vaddps(Vmm(ur_base + 4 + ur_size), Vmm(ur_base + 4 + ur_size), Vmm(ur_base + ur_size));
                    }
                }

                sub(reg_work_amount, 1);
                jmp(loop_label, T_NEAR);
            }
            L(loop_end_label);

            // store sum/variance
            for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                if (jcp_.normalize_variance) {
                    uni_vmovups(ptr[reg_variance + ur_offset + ur_size * vector_step * sizeof(float)], Vmm(ur_base + 4 + ur_size));
                } else {
                    if (!isFloatCompatible(jcp_.src_prc))
                        uni_vcvtdq2ps(Vmm(ur_base + 4 + ur_size), Vmm(ur_base + 4 + ur_size));
                    uni_vmovups(ptr[reg_sum + ur_offset + ur_size * vector_step * sizeof(float)], Vmm(ur_base + 4 + ur_size));
                }
            }

            add(reg_src, unroll_size_rt * vector_step * jcp_.src_data_size);
        }
    }

    inline void worker_unroll(bool is_tail = false) {
        // if mean(sum) for continous data, then fast pass for major part
        if (!jcp_.normalize_variance && jcp_.layout == MVNLayoutType::mvn_planar) {
            Vmm vmm_one = Vmm(15);
            // i8/u8 fast path
            if (mayiuse(avx512_core_vnni) && jcp_.src_data_size == 1) {
                uni_vmovups(vmm_one, ptr[reg_table]);
                Xbyak::Label loop_8bit_label;
                Xbyak::Label loop_8bit_end_label;
                L(loop_8bit_label);
                {
                    cmp(reg_work_amount, 4);
                    jl(loop_8bit_end_label, T_NEAR);

                    if (jcp_.src_prc == Precision::I8) {
                        vpdpbusd(vmm_sum, vmm_one, ptr[reg_src]);
                    } else {
                        uni_vmovdqu(vmm_val, ptr[reg_src]);
                        vpdpbusd(vmm_sum, vmm_val, vmm_one);
                    }

                    add(reg_src, vlen);
                    sub(reg_work_amount, 4);

                    jmp(loop_8bit_label, T_NEAR);
                }
                L(loop_8bit_end_label);
            }
            // bf16 fast path
            if (mayiuse(avx512_core_bf16) && jcp_.src_prc == Precision::BF16) {
                uni_vmovups(vmm_one, ptr[reg_table]);
                Xbyak::Label loop_bf16_label;
                Xbyak::Label loop_bf16_end_label;
                L(loop_bf16_label);
                {
                    cmp(reg_work_amount, 2);
                    jl(loop_bf16_end_label, T_NEAR);

                    vdpbf16ps(vmm_sum, vmm_one, ptr[reg_src]);

                    add(reg_src, vlen);
                    sub(reg_work_amount, 2);

                    jmp(loop_bf16_label, T_NEAR);
                }
                L(loop_bf16_end_label);
            }
        }

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end_label, T_NEAR);

            if (jcp_.layout != MVNLayoutType::mvn_planar && is_tail) {
                worker_partial(false, false);
            } else {
                worker_full_size();
            }

            add(reg_src, reg_stride);
            sub(reg_work_amount, 1);

            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    inline void worker_full_size() {
        load_vector_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                       {}, {load_pool_gpr_idxs});

        if (jcp_.normalize_variance) {
            // all with float
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);

            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            // for sum, int execute prc for int-family data type
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    // needed and supported case: 1. scalar with zero pad. 2. tails w/ or w/o zero pad
    inline void worker_partial(bool is_scalar, bool is_zero_pad) {
        if (is_scalar) {
            load_scalar_with_fill_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                               {}, {load_pool_gpr_idxs});
        } else {
            if (is_zero_pad)
                load_tail_with_fill_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                               {}, {load_pool_gpr_idxs});
            else
                load_tail_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
                                               {}, {load_pool_gpr_idxs});
        }
        if (jcp_.normalize_variance) {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vcvtdq2ps(vmm_val, vmm_val);
            uni_vsubps(vmm_val, vmm_val, vmm_mean);
            if (is_zero_pad) {
                int elt_num = is_scalar ? 1 : tail_step;
                uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
                if (isa == cpu::x64::sse41) {
                    uint8 imm = 1;
                    imm = ~((imm << elt_num) - imm);
                    blendps(vmm_val, vmm_zero, imm);
                } else if (isa == cpu::x64::avx2) {
                    uint8 imm = 1;
                    imm = ~((imm << elt_num) - imm);
                    vblendps(vmm_val, vmm_val, vmm_zero, imm);
                } else if (isa == cpu::x64::avx512_core) {
                    uint64_t tail_mask = 1;
                    tail_mask = ~((tail_mask << elt_num) - tail_mask);
                    mov(reg_aux, tail_mask);
                    kmovq(k_mask, reg_aux);
                    vblendmps(vmm_val | k_mask, vmm_val, vmm_zero);
                }
            }
            uni_vfmadd231ps(vmm_variance, vmm_val, vmm_val);
        } else {
            if (!isFloatCompatible(jcp_.src_prc))
                uni_vpaddd(vmm_sum, vmm_sum, vmm_val);
            else
                uni_vaddps(vmm_sum, vmm_sum, vmm_val);
        }
    }

    inline void reduce_sum_store_xmm(Xbyak::Xmm xmm_sum) {
        uni_vmovshdup(xmm_aux3, xmm_sum);            //  sum:1,2,3,4; aux3:2,2,4,4
        uni_vaddps(xmm_sum, xmm_sum, xmm_aux3);      //  sum:1+2,2+2,3+4,4+4
        uni_vmovhlps(xmm_aux3, xmm_aux3, xmm_sum);   //  aux3:3+4,4+4,4,4
        uni_vaddps(xmm_sum, xmm_sum,  xmm_aux3);     //  sum:1+2+3+4,...
        if (jcp_.normalize_variance) {
            uni_vmovss(ptr[reg_variance], xmm_sum);
        } else {
            uni_vmovss(ptr[reg_sum], xmm_sum);
        }
    }

    inline void reduce_sum_store_vmm(int vmm_idx) {
        if (isa == cpu::x64::sse41) {
            reduce_sum_store_xmm(Xmm(vmm_idx));
        } else if (isa == cpu::x64::avx2) {
            Xbyak::Ymm ymm_sum = Xbyak::Ymm(vmm_idx);
            vextractf128(xmm_aux1, ymm_sum, 0);
            vextractf128(xmm_aux2, ymm_sum, 1);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            reduce_sum_store_xmm(xmm_aux1);
        } else {
            Xbyak::Zmm zmm_sum = Xbyak::Zmm(vmm_idx);
            vextractf32x4(xmm_aux1, zmm_sum, 0);
            vextractf32x4(xmm_aux2, zmm_sum, 1);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            vextractf32x4(xmm_aux2, zmm_sum, 2);
            vextractf32x4(xmm_aux3, zmm_sum, 3);
            uni_vaddps(xmm_aux2, xmm_aux2, xmm_aux3);
            uni_vaddps(xmm_aux1, xmm_aux1, xmm_aux2);
            reduce_sum_store_xmm(xmm_aux1);
        }
    }

    void prepare_table() {
        const unsigned int cvals[] = {
            // 4 * 8 = 32 bit
            0x01010101,  // one byte
            0x3f803f80   // one bf16
        };

        align(64);
        L(l_table);

        if (mayiuse(avx512_core_vnni) && (jcp_.src_prc == Precision::U8 || jcp_.src_prc == Precision::I8)) {
            for (size_t d = 0; d < vector_step; ++d) {
                dd(cvals[0]);
            }
        }
        if (mayiuse(avx512_core_bf16) && jcp_.src_prc == Precision::BF16) {
            for (size_t d = 0; d < vector_step; ++d) {
                dd(cvals[1]);
            }
        }
    }
};

// mean,variance->mvn
template <cpu_isa_t isa>
struct jit_uni_mvn_kernel_f32 : public jit_uni_mvn_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_mvn_kernel_f32)

    explicit jit_uni_mvn_kernel_f32(jit_mvn_config_params jcp, const dnnl::primitive_attr &attr) : jit_uni_mvn_kernel(jcp, attr), jit_generator(jit_name()) {
        const auto &p = attr_.get_post_ops();
        bool opt_scaleshift_applicable = jcp_.layout == MVNLayoutType::mvn_by_channel && isa == cpu::x64::avx512_core && !jcp_.across_channels;
        if (opt_scaleshift_applicable) {
            for (int i = 0; i < p.len(); i++) {
                auto &post_op = p.get()->entry_[i];
                if (post_op.is_depthwise()) {
                    if (0 == i && post_op.depthwise.alg == alg_kind::depthwise_scale_shift) {
                        optimized_scaleshift_num = 1;
                    } else if (1 == i && optimized_scaleshift_num == 1 && post_op.depthwise.alg == alg_kind::depthwise_scale_shift) {
                        optimized_scaleshift_num = 2;
                    }
                }
            }
        }
    }

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override {
        const auto &p = attr_.get_post_ops();
        for (int i = 0; i < p.len(); i++) {
            auto &post_op = p.get()->entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(std::make_shared<jit_uni_eltwise_injector_f32<isa>>(
                        this, post_op.eltwise.alg, post_op.eltwise.alpha, post_op.eltwise.beta, post_op.eltwise.scale));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(std::make_shared<jit_uni_depthwise_injector_f32<isa>>(
                        this, post_op));
            } else if (post_op.is_quantization()) {
                quantization_injectors.push_back(std::make_shared<jit_uni_quantization_injector_f32<isa>>(
                        this, post_op, vmm_d_weights, vmm_d_bias, reg_d_weights, reg_d_bias));
            }
        }

        tail_step = jcp_.layout == MVNLayoutType::mvn_planar ? (jcp_.D * jcp_.H * jcp_.W) - ((jcp_.D * jcp_.H * jcp_.W) / vector_step) * vector_step :
                                jcp_.C - (jcp_.C / vector_step) * vector_step;

        load_vector_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, Precision::FP32, vector_step));
        load_tail_emitter.reset(new jit_load_emitter(this, isa, jcp_.src_prc, Precision::FP32, tail_step));
        store_vector_emitter.reset(new jit_store_emitter(this, isa, Precision::FP32, jcp_.dst_prc, vector_step));
        store_tail_emitter.reset(new jit_store_emitter(this, isa, Precision::FP32, jcp_.dst_prc, tail_step));

        this->preamble();

        mov(reg_post_ops_data, ptr[reg_params + GET_OFF(post_op_data)]);
        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_mean, ptr[reg_params + GET_OFF(mean)]);
        if (jcp_.normalize_variance)
            mov(reg_variance_inv, ptr[reg_params + GET_OFF(variance)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);
        mov(reg_src_stride, ptr[reg_params + GET_OFF(src_stride)]);
        mov(reg_dst_stride, ptr[reg_params + GET_OFF(dst_stride)]);
        mov(reg_oc_off, ptr[reg_params + GET_OFF(oc_off)]);

        if (jcp_.layout == MVNLayoutType::mvn_planar || jcp_.across_channels) {
            uni_vbroadcastss(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vbroadcastss(vmm_variance_inv, ptr[reg_variance_inv]);
        } else {
            uni_vmovups(vmm_mean, ptr[reg_mean]);
            if (jcp_.normalize_variance)
                uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
        }

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx()), static_cast<size_t>(vmm_val.getIdx())};

        if (jcp_.layout == MVNLayoutType::mvn_planar) {
            worker_mvn_unroll();
            if (tail_step != 0) {
                worker_mvn(true);
            }
        } else if (jcp_.layout == MVNLayoutType::mvn_by_channel) {
            if (jcp_.across_channels)
                norm_nspc_ac_ker();
            else
                norm_nspc_pc_ker();
        } else {
            // blk
            int repeats = (isa == cpu::x64::sse41) ? 2 : 1;  // block size is also 8 on cpu::x64::sse41
            for (int i = 0; i < repeats; i++) {
                int offset_sse42 = i * 4;
                if (i > 0) {
                    // reset modified input
                    mov(reg_src, ptr[reg_params + GET_OFF(src)]);
                    mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);
                    mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

                    add(reg_src, offset_sse42 * jcp_.src_data_size);
                    add(reg_dst, offset_sse42 * jcp_.dst_data_size);
                    add(reg_oc_off, offset_sse42 * sizeof(float));

                    if (!jcp_.across_channels) {
                        add(reg_mean, offset_sse42 * sizeof(float));
                        uni_vmovups(vmm_mean, ptr[reg_mean]);
                        if (jcp_.normalize_variance) {
                            add(reg_variance_inv, offset_sse42 * sizeof(float));
                            uni_vmovups(vmm_variance_inv, ptr[reg_variance_inv]);
                        }
                    }
                }

                Xbyak::Label label_empty_2half_sse42;
                if (tail_step == 0) {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);
                    worker_mvn_unroll();
                } else {
                    cmp(reg_oc_off, static_cast<int>(jcp_.C * sizeof(float)));
                    jae(label_empty_2half_sse42, T_NEAR);

                    Xbyak::Label label_full_size_block;
                    Xbyak::Label label_size_end;

                    cmp(reg_oc_off, static_cast<int>((jcp_.C - vector_step) * sizeof(float)));
                    jle(label_full_size_block, T_NEAR);

                    worker_mvn_unroll(true);
                    jmp(label_size_end, T_NEAR);

                    L(label_full_size_block);
                    {
                        worker_mvn_unroll();
                    }
                    L(label_size_end);
                }
                L(label_empty_2half_sse42);
            }
        }

        this->postamble();

        load_vector_emitter->emit_data();
        load_tail_emitter->emit_data();
        store_vector_emitter->emit_data();
        store_tail_emitter->emit_data();

        for (auto& inj : eltwise_injectors)
            inj->prepare_table();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    int tail_step = 0;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_mean = r9;
    Xbyak::Reg64 reg_variance_inv = r10;
    Xbyak::Reg64 reg_dst = r11;
    Xbyak::Reg64 reg_work_amount = r12;
    Xbyak::Reg64 reg_src_stride = r13;
    Xbyak::Reg64 reg_dst_stride = r14;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_oc_off = rax;
    Xbyak::Reg64 reg_d_weights = rbx;
    Xbyak::Reg64 reg_d_bias = rdx;
    Xbyak::Reg64 reg_post_ops_data = rsi;

    Xbyak::Reg64 reg_load_table = r15;
    Xbyak::Reg64 reg_load_store_mask = rbp;

    Vmm vmm_val = Vmm(3);
    Vmm vmm_mean = Vmm(4);
    Vmm vmm_variance_inv = Vmm(5);
    Vmm vmm_zero = Vmm(2);

    Vmm vmm_d_weights = Vmm(0);
    Vmm vmm_d_bias = Vmm(1);

    std::unique_ptr<jit_load_emitter> load_vector_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_tail_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_vector_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_tail_emitter = nullptr;

    std::vector<std::shared_ptr<jit_uni_eltwise_injector_f32<isa>>> eltwise_injectors;
    std::vector<std::shared_ptr<jit_uni_depthwise_injector_f32<isa>>> depthwise_injectors;
    std::vector<std::shared_ptr<jit_uni_quantization_injector_f32<isa>>> quantization_injectors;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    // nspc norm per channel with unroll
    inline void norm_nspc_pc_ker() {
        // 4 unroll vector
        size_t unroll_size = 4;
        size_t vec_num = div_up(jcp_.C, vector_step);
        unroll_size = vec_num >= unroll_size ? unroll_size : vec_num;
        size_t unroll_number = div_up(vec_num, unroll_size);

        int ur_base = 4;
        Xbyak::Reg64 reg_src_aux = reg_src_stride;
        Xbyak::Reg64 reg_dst_aux = reg_dst_stride;
        // 2 abi
        Xbyak::Reg64 reg_work_amount_bk = rcx;
        Xbyak::Reg64 reg_oc_off_bk = rdi;
        mov(reg_oc_off_bk, reg_oc_off);
        mov(reg_work_amount_bk, reg_work_amount);
        for (int ur_num = 0; ur_num < unroll_number; ur_num++) {
            // 4-15 for unroll. 4-7 for src, 8-11 for m, 12-15 for v
            int ur_offset_elt = ur_num * unroll_size * vector_step;
            int ur_offset = ur_offset_elt * sizeof(float);
            size_t unroll_size_rt = std::min(vec_num - ur_num * unroll_size, unroll_size);
            size_t elt_num = std::min(jcp_.C - ur_num * unroll_size * vector_step, unroll_size * vector_step);
            for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                uni_vmovups(Vmm(ur_base + 4 + ur_size), ptr[reg_mean + ur_offset + ur_size * vector_step * sizeof(float)]);
            }
            if (jcp_.normalize_variance) {
                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    uni_vmovups(Vmm(ur_base + 8 + ur_size), ptr[reg_variance_inv + ur_offset + ur_size * vector_step * sizeof(float)]);
                }
            }
            // optimized scaleshift
            size_t post_ops_data_offset = 0;
            for (int i = 0; i < optimized_scaleshift_num; i++) {
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, ur_offset);
                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    uni_vmovups(Vmm(16 + i * 4 + ur_size), ptr[reg_d_weights]);
                    add(reg_d_weights, vector_step * sizeof(float));
                }
                mov(reg_d_bias, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_bias, ur_offset + jcp_.C * sizeof(float));
                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    uni_vmovups(Vmm(24 + i * 4 + ur_size), ptr[reg_d_bias]);
                    add(reg_d_bias, vector_step * sizeof(float));
                }
                post_ops_data_offset += sizeof(float*);
            }

            mov(reg_src_aux, reg_src);
            mov(reg_dst_aux, reg_dst);
            mov(reg_work_amount, reg_work_amount_bk);
            mov(reg_oc_off, reg_oc_off_bk);

            Xbyak::Label loop_label;
            Xbyak::Label loop_end_label;
            L(loop_label);
            {
                cmp(reg_work_amount, 0);
                jle(loop_end_label, T_NEAR);

                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    bool is_tails = ur_offset_elt + ur_size * vector_step + vector_step > jcp_.C;
                    if (is_tails) {
                        load_tail_emitter->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                            {static_cast<size_t>(ur_base + ur_size)}, {}, {load_pool_gpr_idxs});
                        add(reg_src_aux, tail_step * jcp_.src_data_size);
                    } else {
                        load_vector_emitter->emit_code({static_cast<size_t>(reg_src_aux.getIdx())},
                            {static_cast<size_t>(ur_base + ur_size)}, {}, {load_pool_gpr_idxs});
                        add(reg_src_aux, vector_step * jcp_.src_data_size);
                    }
                }
                add(reg_src_aux, (jcp_.C - elt_num) * jcp_.src_data_size);
                prefetcht0(ptr[reg_src_aux]);

                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    uni_vsubps(Vmm(ur_base + ur_size), Vmm(ur_base + ur_size), Vmm(ur_base + 4 + ur_size));
                }
                if (jcp_.normalize_variance) {
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        uni_vmulps(Vmm(ur_base + ur_size), Vmm(ur_base + ur_size), Vmm(ur_base + 8 + ur_size));
                    }
                }

                for (int i = 0; i < optimized_scaleshift_num; i++) {
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        uni_vfmadd132ps(Vmm(ur_base + ur_size), Vmm(24 + i * 4 + ur_size), Vmm(16 + i * 4 + ur_size));
                    }
                }

                if (attr_.get_post_ops().len() != 0) {
                    for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                        apply_post_ops(jcp_.dst_prc, ur_base + ur_size, false);
                        bool is_tails = ur_offset_elt + ur_size * vector_step + vector_step > jcp_.C;
                        if (is_tails)
                            add(reg_oc_off, tail_step * sizeof(float));
                        else
                            add(reg_oc_off, vector_step * sizeof(float));
                    }
                }

                for (int ur_size = 0; ur_size < unroll_size_rt; ur_size++) {
                    bool is_tails = ur_offset_elt + ur_size * vector_step + vector_step > jcp_.C;
                    if (is_tails) {
                        store_tail_emitter->emit_code({static_cast<size_t>(ur_base + ur_size)}, {static_cast<size_t>(reg_dst_aux.getIdx())},
                            {store_pool_vec_idxs}, {store_pool_gpr_idxs});
                        add(reg_dst_aux, tail_step * jcp_.dst_data_size);
                    } else {
                        store_vector_emitter->emit_code({static_cast<size_t>(ur_base + ur_size)}, {static_cast<size_t>(reg_dst_aux.getIdx())},
                            {store_pool_vec_idxs}, {store_pool_gpr_idxs});
                        add(reg_dst_aux, vector_step * jcp_.dst_data_size);
                    }
                }

                add(reg_dst_aux, (jcp_.C - elt_num) * jcp_.dst_data_size);
                sub(reg_oc_off, elt_num * sizeof(float));
                sub(reg_work_amount, 1);
                jmp(loop_label, T_NEAR);
            }
            L(loop_end_label);

            add(reg_src, unroll_size_rt * vector_step * jcp_.src_data_size);
            add(reg_dst, unroll_size_rt * vector_step * jcp_.dst_data_size);
            add(reg_oc_off_bk, unroll_size_rt * vector_step * sizeof(float));
        }
    }

    inline void norm_nspc_ac_ker() {
        Xbyak::Reg64 reg_oc_off_bk = reg_src_stride;
        if (attr_.get_post_ops().len() != 0) {
            mov(reg_oc_off_bk, reg_oc_off);
        }

        size_t vec_num = div_up(jcp_.C, vector_step);

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(loop_end_label, T_NEAR);

            if (attr_.get_post_ops().len() != 0) {
                mov(reg_oc_off, reg_oc_off_bk);
            }

            for (int v_num = 0; v_num < vec_num; v_num++) {
                bool is_tail = (v_num * vector_step + vector_step > jcp_.C) ? true : false;
                worker_mvn(is_tail);
                if (is_tail) {
                    add(reg_src, tail_step * jcp_.src_data_size);
                    add(reg_dst, tail_step * jcp_.dst_data_size);
                    if (attr_.get_post_ops().len() != 0)
                        add(reg_oc_off, tail_step * sizeof(float));
                } else {
                    add(reg_src, vector_step * jcp_.src_data_size);
                    add(reg_dst, vector_step * jcp_.dst_data_size);
                    if (attr_.get_post_ops().len() != 0)
                        add(reg_oc_off, vector_step * sizeof(float));
                }
            }

            sub(reg_work_amount, 1);
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);
    }

    inline void worker_mvn(bool is_tail) {
        const auto& load_emitter = is_tail ? load_tail_emitter : load_vector_emitter;
        const auto& store_emitter = is_tail ? store_tail_emitter : store_vector_emitter;

        load_emitter->emit_code({static_cast<size_t>(reg_src.getIdx())}, {static_cast<size_t>(vmm_val.getIdx())},
            {}, {load_pool_gpr_idxs});

        uni_vsubps(vmm_val, vmm_val, vmm_mean);
        if (jcp_.normalize_variance)
            uni_vmulps(vmm_val, vmm_val, vmm_variance_inv);

        apply_post_ops(jcp_.dst_prc, vmm_val.getIdx(), jcp_.layout == MVNLayoutType::mvn_planar);

        store_emitter->emit_code({static_cast<size_t>(vmm_val.getIdx())}, {static_cast<size_t>(reg_dst.getIdx())},
            {store_pool_vec_idxs}, {store_pool_gpr_idxs});
    }

    inline void worker_mvn_unroll(bool is_tail = false) {
        Xbyak::Label mvn_loop_label;
        Xbyak::Label mvn_loop_end_label;

        L(mvn_loop_label);
        {
            cmp(reg_work_amount, 0);
            jle(mvn_loop_end_label, T_NEAR);

            worker_mvn(is_tail);

            add(reg_src, reg_src_stride);
            add(reg_dst, reg_dst_stride);
            sub(reg_work_amount, 1);

            jmp(mvn_loop_label, T_NEAR);
        }
        L(mvn_loop_end_label);
    }

    void apply_post_ops(InferenceEngine::Precision dst_prc, size_t vmm_idx, bool is_broadcast) {
        const auto &p = attr_.get_post_ops();
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        int quantization_inj_idx = 0;
        int post_ops_data_offset = 0;
        for (int i = 0; i < p.len(); i++) {
            auto& post_op = p.get()->entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors[eltwise_inj_idx]->compute_vector_range(vmm_idx, vmm_idx + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                if (post_op.depthwise.alg == alg_kind::depthwise_scale_shift && i < optimized_scaleshift_num) {
                    post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                    depthwise_inj_idx++;
                    continue;
                }
                mov(reg_d_weights, ptr[reg_post_ops_data + post_ops_data_offset]);
                add(reg_d_weights, reg_oc_off);

                depthwise_injectors[depthwise_inj_idx]->compute_vector_range(
                        vmm_idx, vmm_idx + 1, reg_d_weights, reg_d_weights, is_broadcast);

                post_ops_data_offset += depthwise_injectors[depthwise_inj_idx]->memoryStep();
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || isFloatCompatible(dst_prc) || i != p.len() - 1;

                quantization_injectors[quantization_inj_idx]->init_crop_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_crop(vmm_idx, vmm_idx + 1, 0, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_input_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_input_scale_shift(vmm_idx, vmm_idx + 1, 0, do_rounding, 0, is_broadcast);

                quantization_injectors[quantization_inj_idx]->init_output_scale_shift_ptrs(reg_post_ops_data + post_ops_data_offset, reg_oc_off);
                quantization_injectors[quantization_inj_idx]->compute_output_scale_shift(vmm_idx, vmm_idx + 1, 0, 0, is_broadcast);

                post_ops_data_offset += quantization_injectors[quantization_inj_idx]->memoryStep();
                quantization_inj_idx++;
            }
        }
    }
};
//////////////////////////////////////////////////////////////////////////////////

MVNJitExecutor::MVNJitExecutor() : MVNExecutor() {}

bool MVNJitExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescCPtr>& srcDescs,
                          const std::vector<MemoryDescCPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    // TODO: parameters validation

    this->mvnAttrs = mvnAttrs;

    jcp.src_prc = srcDescs[0]->getPrecision();
    jcp.dst_prc = dstDescs[0]->getPrecision();
    jcp.src_data_size = jcp.src_prc.size();
    jcp.dst_data_size = jcp.dst_prc.size();

    if (srcDescs[0]->hasLayoutType(LayoutType::ncsp)) {
        jcp.layout = MVNLayoutType::mvn_planar;
    } else if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        jcp.layout = MVNLayoutType::mvn_by_channel;
    } else {
        jcp.layout = MVNLayoutType::mvn_block;
    }

    auto shape5D = transformTo5DCase(srcDescs[0]->getShape().getStaticDims(), mvnAttrs.initAcrossChannels_);
    jcp.N = shape5D[0];
    jcp.C = shape5D[1];
    jcp.D = shape5D[2];
    jcp.H = shape5D[3];
    jcp.W = shape5D[4];

    jcp.normalize_variance = mvnAttrs.normalizeVariance_;
    jcp.across_channels = mvnAttrs.initAcrossChannels_;
    if (one_of(srcDescs[0]->getShape().getRank(), 1, 2) && mvnAttrs.initAcrossChannels_) {
        jcp.across_channels = false;
    }

    if (mayiuse(cpu::x64::avx512_core)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx512_core>(jcp, attr));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_core>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx512_core>(jcp));
        }
        implType = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::avx2>(jcp, attr));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::avx2>(jcp));
        }
        implType = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        mvn_kernel.reset(new jit_uni_mvn_kernel_f32<cpu::x64::sse41>(jcp, attr));
        jcp.normalize_variance = false;
        mvn_mean_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        if (mvnAttrs.normalizeVariance_) {
            jcp.normalize_variance = true;
            mvn_variance_kernel.reset(new jit_uni_mvn_mean_variance_kernel_f32<cpu::x64::sse41>(jcp));
        }
        implType = impl_desc_type::jit_sse42;
    } else {
        return false;
    }

    if (mvn_kernel)
        mvn_kernel->create_ker();
    if (mvn_mean_kernel)
        mvn_mean_kernel->create_ker();
    if (mvn_variance_kernel)
        mvn_variance_kernel->create_ker();

    return true;
}

void MVNJitExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    if (!mvn_mean_kernel || (mvnAttrs.normalizeVariance_ && !mvn_variance_kernel) || !mvn_kernel) {
        IE_THROW() << "MVN layer doesn't create kernel to execute on sse41 above platform.";
    }

    uint8_t *src_data = reinterpret_cast<uint8_t*>(src[0]->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dst[0]->GetPtr());
    if (jcp.layout == MVNLayoutType::mvn_planar) {
        mvn_pln(src_data, dst_data, post_ops_data_);
    } else if (jcp.layout == MVNLayoutType::mvn_by_channel) {
        mvn_nspc(src_data, dst_data, post_ops_data_);
    } else {
        mvn_blk(src_data, dst_data, post_ops_data_);
    }
}

void MVNJitExecutor::mvn_pln(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_) {
    size_t blk_size = 1;  // blk size in vmm
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else if (mayiuse(cpu::x64::sse41)) {
        blk_size = 4;
    }

    size_t C1 = jcp.H * jcp.W;
    size_t C2 = C1 * jcp.D;
    size_t C3 = C2 * jcp.C;

    size_t src_stride_size = static_cast<size_t>(blk_size * jcp.src_data_size);
    size_t dst_stride_size = static_cast<size_t>(blk_size * jcp.dst_data_size);

    if (jcp.across_channels) {
        parallel_for(jcp.N, [&](int b) {
            size_t cb = b * C3;
            // Calculate mean value for one instance in batch
            // Parallel sum for each channel
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum(jcp.C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                auto arg = jit_mvn_call_args();
                arg.src = src_data + cc * jcp.src_data_size;
                arg.sum = static_cast<float*>(&mean_internal);
                arg.src_stride = src_stride_size;
                arg.work_amount = static_cast<size_t>(C2 / blk_size); // for vector part
                arg.post_op_data = post_ops_data_;
                (*mvn_mean_kernel)(&arg);
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            // calculate variance value for one instance in batch
            // parallel sum for each channel
            if (mvnAttrs.normalizeVariance_) {
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(jcp.C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * jcp.src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance_internal);
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // vector part
                    arg.post_op_data = post_ops_data_;
                    (*mvn_variance_kernel)(&arg);
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C3inv) + mvnAttrs.epsValue_;

                // mvn for one instance in batch
                parallel_for(jcp.C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * jcp.src_data_size;
                    arg.dst = dst_data + cc * jcp.dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);  // work amount for vector part
                    arg.oc_off = sizeof(float) * c;
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for(jcp.C, [&](int c) {
                    size_t cc = cb + c * C2;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + cc * jcp.src_data_size;
                    arg.dst = dst_data + cc * jcp.dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(C2 / blk_size);
                    arg.oc_off = sizeof(float) * c;
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        });
    } else {
        parallel_for2d(jcp.N, jcp.C, [&](size_t b, size_t c) {
            size_t cb = b * C3;
            size_t cc = cb + c * C2;
            float C2inv = 1.f / static_cast<float>(C2);

            // mean for this channel
            float mean = 0.f;
            // the same arg for three kernels
            auto arg = jit_mvn_call_args();
            arg.src = src_data + cc * jcp.src_data_size;
            arg.dst = dst_data + cc * jcp.dst_data_size;
            arg.sum = static_cast<float*>(&mean);
            arg.src_stride = src_stride_size;
            arg.dst_stride = dst_stride_size;
            arg.work_amount = static_cast<size_t>(C2 / blk_size);
            arg.oc_off = static_cast<size_t>(c * sizeof(float));
            arg.post_op_data = post_ops_data_;
            (*mvn_mean_kernel)(&arg);

            mean *= C2inv;

            if (mvnAttrs.normalizeVariance_) {
                // variance for this channel
                float variance = 0.f;
                arg.mean = static_cast<float*>(&mean);
                arg.variance = static_cast<float*>(&variance);
                (*mvn_variance_kernel)(&arg);

                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance = 1.f / sqrtf(variance * C2inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance = 1.f / (sqrtf(variance * C2inv) + mvnAttrs.epsValue_);

                // mvn for this channel
                (*mvn_kernel)(&arg);
            } else {
                // mvn for this channel
                arg.mean = static_cast<float*>(&mean);
                (*mvn_kernel)(&arg);
            }
        });
    }
}

void MVNJitExecutor::mvn_nspc(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else if (mayiuse(cpu::x64::avx2)) {
        blk_size = 8;
    } else {
        blk_size = 4;
    }

    size_t threads_num = parallel_get_num_threads();
    size_t aux_buffer_size = jcp.across_channels ? 1 : rnd_up(jcp.C, blk_size);
    parallel_for(jcp.N, [&](size_t b) {
        std::vector<float> mean_buffer(aux_buffer_size * threads_num);
        std::vector<float> variance_buffer;
        if (mvnAttrs.normalizeVariance_) {
            variance_buffer.resize(aux_buffer_size * threads_num);
        }
        size_t b_offset = b * jcp.C * jcp.D * jcp.H * jcp.W;

        // kernel_type: 0 for mean, 1 for variance, 2 for normalization
        auto worker = [&](const bool across_channel, const int kernel_type) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                size_t start = 0, end = 0;
                splitter(static_cast<size_t>(jcp.D * jcp.H * jcp.W), nthr, ithr, start, end);

                auto arg = jit_mvn_call_args();
                arg.src = src_data + (b_offset + (start * jcp.C)) * jcp.src_data_size;
                if (0 == kernel_type) {
                    arg.sum = &mean_buffer[aux_buffer_size * ithr];
                } else if (1 == kernel_type) {
                    arg.mean = &mean_buffer[0];
                    arg.variance = &variance_buffer[aux_buffer_size * ithr];
                } else if (2 == kernel_type) {
                    arg.dst = dst_data + (b_offset + (start * jcp.C)) * jcp.dst_data_size;
                    arg.mean = &mean_buffer[0];
                    if (mvnAttrs.normalizeVariance_)
                        arg.variance = &variance_buffer[0];
                    arg.oc_off = 0;
                    arg.post_op_data = post_ops_data_;
                }
                arg.work_amount = (across_channel && kernel_type != 2) ? (end - start) * jcp.C : (end - start);

                if (0 == kernel_type) {
                    (*mvn_mean_kernel)(&arg);
                } else if (1 == kernel_type) {
                    (*mvn_variance_kernel)(&arg);
                } else if (2 == kernel_type) {
                    (*mvn_kernel)(&arg);
                }
            });
        };

        if (jcp.across_channels) {
            float size_inv = 1.f / static_cast<float>(jcp.C * jcp.D * jcp.H * jcp.W);
            worker(true, 0);
            for (size_t i = 1; i < threads_num; i++) {
                mean_buffer[0] += mean_buffer[i];
            }
            mean_buffer[0] *= size_inv;
            if (mvnAttrs.normalizeVariance_) {
                worker(true, 1);
                for (size_t i = 1; i < threads_num; i++) {
                    variance_buffer[0] += variance_buffer[i];
                }
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance_buffer[0] = 1.f / sqrtf(variance_buffer[0] * size_inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance_buffer[0] = 1.f / (sqrtf(variance_buffer[0] * size_inv) + mvnAttrs.epsValue_);
            }
            worker(true, 2);
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(jcp.D * jcp.H * jcp.W);
            worker(false, 0);
            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < jcp.C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < jcp.C; c++)
                mean_buffer[c] *= size_inv;
            if (mvnAttrs.normalizeVariance_) {
                worker(false, 1);
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < jcp.C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < jcp.C; c++) {
                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + mvnAttrs.epsValue_);
                }
            }
            worker(false, 2);
        }
    });
}

void MVNJitExecutor::mvn_blk(const uint8_t* src_data, uint8_t* dst_data, const void *post_ops_data_) {
    size_t blk_size = 1;  // channel blk for memory layout
    if (mayiuse(cpu::x64::avx512_core)) {
        blk_size = 16;
    } else {
        blk_size = 8;
    }

    size_t CB = div_up(jcp.C, blk_size);

    size_t C0 = jcp.W * blk_size;
    size_t C1 = C0 * jcp.H;
    size_t C2 = C1 * jcp.D;
    size_t C3 = C2 * CB;
    size_t C5 = jcp.C * jcp.D * jcp.H * jcp.W;

    size_t threads_num = parallel_get_num_threads();
    size_t aux_buffer_size = jcp.across_channels ? blk_size : rnd_up(jcp.C, blk_size);
    std::vector<float> mean_buffer(aux_buffer_size * threads_num);
    std::vector<float> variance_buffer(aux_buffer_size * threads_num);

    size_t src_stride_size = static_cast<size_t>(blk_size * jcp.src_data_size);
    size_t dst_stride_size = static_cast<size_t>(blk_size * jcp.dst_data_size);

    for (size_t b = 0lu; b < jcp.N; b++) {
        size_t b_offset = b * C3;
        if (jcp.across_channels) {
            // mean for this instance in batch
            float C5inv = 1.f / static_cast<float>(C5);
            float mean_temp = 0.0f;
            mean_temp = parallel_sum3d(CB, jcp.D, jcp.H, mean_temp, [&](size_t cb, size_t d, size_t h)->float {
                size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                float mean_internal = 0.0f;
                /////////////////////////////////
                //          W           //  |
                //                      //  |
                //                      //  |
                //blk +  +  +  +  +  +  //  |  +
                //                      //  |
                //                      //  |
                //                      // \|/
                /////////////////////////////////
                auto mean_buffer_ptr = &mean_buffer[blk_size * parallel_get_thread_num()];
                for (int i = 0; i < blk_size; i++)
                    mean_buffer_ptr[i] = 0.f;

                auto arg = jit_mvn_call_args();
                arg.src = src_data + src_offset * jcp.src_data_size;
                arg.sum = mean_buffer_ptr;
                arg.src_stride = src_stride_size;
                arg.work_amount = static_cast<size_t>(jcp.W);
                arg.oc_off = static_cast<size_t>(cb * blk_size * sizeof(float));  // for tail process
                (*mvn_mean_kernel)(&arg); // for W * blk

                size_t min_cb = (std::min)(blk_size, jcp.C - cb * blk_size);
                for (int i = 0; i < min_cb; i++)
                    mean_internal += mean_buffer_ptr[i];
                return mean_internal;
            });
            float mean = mean_temp * C5inv;

            if (mvnAttrs.normalizeVariance_) {
                // variance: sum((x-mean)*(x-mean)) for one instance in batch
                float variance_temp = 0.0f;
                variance_temp = parallel_sum3d(CB, jcp.D, jcp.H, variance_temp, [&](size_t cb, size_t d, size_t h)->float {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;

                    float variance_internal = 0.0f;
                    auto variance_buffer_ptr = &variance_buffer[blk_size * parallel_get_thread_num()];
                    for (int i = 0; i < blk_size; i++)
                        variance_buffer_ptr[i] = 0.f;

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * jcp.src_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = variance_buffer_ptr;
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(jcp.W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_variance_kernel)(&arg);

                    size_t min_cb = (std::min)(blk_size, C - cb * blk_size);
                    for (int i = 0; i < min_cb; i++)
                        variance_internal += variance_buffer_ptr[i];
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance /= sqrtf(variance_temp * C5inv) + mvnAttrs.epsValue_;

                // mvn for one instance in batch
                parallel_for3d(CB, jcp.D, jcp.H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * jcp.src_data_size;
                    arg.dst = dst_data + src_offset * jcp.dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.variance = static_cast<float*>(&variance);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(jcp.W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            } else {
                // mvn for one instance in batch
                parallel_for3d(CB, jcp.D, jcp.H, [&](size_t cb, size_t d, size_t h) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * jcp.src_data_size;
                    arg.dst = dst_data + src_offset * jcp.dst_data_size;
                    arg.mean = static_cast<float*>(&mean);
                    arg.src_stride = src_stride_size;
                    arg.dst_stride = dst_stride_size;
                    arg.work_amount = static_cast<size_t>(jcp.W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_kernel)(&arg);
                });
            }
        } else {  // for per_channel
            float size_inv = 1.f / static_cast<float>(jcp.D * jcp.H * jcp.W);
            for (int i = 0; i < mean_buffer.size(); i++)
                mean_buffer[i] = 0.f;

            // one thread for one C*W size(the same H) to get C size result for the same H, added to last group result
            // keep the compute order the same as planar
            parallel_for2d(jcp.D, jcp.H, [&](size_t thr_idx, size_t d, size_t h) {
                for (size_t cb = 0; cb < CB; cb++) {
                    size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                    auto mean_buffer_ptr = &mean_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                    auto arg = jit_mvn_call_args();
                    arg.src = src_data + src_offset * jcp.src_data_size;
                    arg.sum = mean_buffer_ptr;
                    arg.src_stride = src_stride_size;
                    arg.work_amount = static_cast<size_t>(jcp.W);
                    arg.oc_off = cb * blk_size * sizeof(float);
                    arg.post_op_data = post_ops_data_;
                    (*mvn_mean_kernel)(&arg);
                }
            });

            for (size_t i = 1; i < threads_num; i++) {
                for (size_t c = 0; c < jcp.C; c++)
                    mean_buffer[c] += mean_buffer[c + aux_buffer_size * i];
            }
            for (size_t c = 0; c < jcp.C; c++)
                mean_buffer[c] *= size_inv;

            if (mvnAttrs.normalizeVariance_) {
                for (int i = 0; i < variance_buffer.size(); i++)
                    variance_buffer[i] = 0.f;

                parallel_for2d(jcp.D, jcp.H, [&](size_t thr_idx, size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb + aux_buffer_size * thr_idx];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * jcp.src_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.work_amount = static_cast<size_t>(jcp.W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_variance_kernel)(&arg);
                    }
                });
                for (size_t i = 1; i < threads_num; i++) {
                    for (size_t c = 0; c < jcp.C; c++)
                        variance_buffer[c] += variance_buffer[c + aux_buffer_size * i];
                }
                for (size_t c = 0; c < jcp.C; c++) {
                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance_buffer[c] = 1.f / sqrtf(variance_buffer[c] * size_inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance_buffer[c] = 1.f / (sqrtf(variance_buffer[c] * size_inv) + mvnAttrs.epsValue_);
                }

                parallel_for2d(jcp.D, jcp.H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];
                        auto variance_buffer_ptr = &variance_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * jcp.src_data_size;
                        arg.dst = dst_data + src_offset * jcp.dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.variance = variance_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.dst_stride = dst_stride_size;
                        arg.work_amount = static_cast<size_t>(jcp.W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            } else {
                // normalizeVariance_ == false
                parallel_for2d(jcp.D, jcp.H, [&](size_t d, size_t h) {
                    for (size_t cb = 0; cb < CB; cb++) {
                        size_t src_offset = b_offset + cb * C2 + d * C1 + h * C0;
                        auto mean_buffer_ptr = &mean_buffer[blk_size * cb];

                        auto arg = jit_mvn_call_args();
                        arg.src = src_data + src_offset * jcp.src_data_size;
                        arg.dst = dst_data + src_offset * jcp.dst_data_size;
                        arg.mean = mean_buffer_ptr;
                        arg.src_stride = src_stride_size;
                        arg.dst_stride = dst_stride_size;
                        arg.work_amount = static_cast<size_t>(jcp.W);
                        arg.oc_off = cb * blk_size * sizeof(float);
                        arg.post_op_data = post_ops_data_;
                        (*mvn_kernel)(&arg);
                    }
                });
            }
        }
    }
}

MVNJitExecutor::Key::Key(const MVNAttrs& mvnAttrs,
                    const std::vector<MemoryDescCPtr>& srcDescs,
                    const std::vector<MemoryDescCPtr>& dstDescs,
                    const dnnl::primitive_attr &attr) {
    auto blockedDesc = srcDescs[0]->as<BlockedMemoryDesc>();
    this->mvnAttrs = mvnAttrs;
    this->srcDims = blockedDesc->getShape().getStaticDims();
    this->srcOrder = blockedDesc->getOrder();
    this->srcPrc = srcDescs[0]->getPrecision();
    this->dstPrc = dstDescs[0]->getPrecision();
}

size_t MVNJitExecutor::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = get_vector_hash(seed, srcDims);
    seed = get_vector_hash(seed, srcOrder);
    seed = hash_combine(seed, srcPrc.getPrecVal());
    seed = hash_combine(seed, dstPrc.getPrecVal());
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNJitExecutor::Key::operator==(const Key& rhs) const {
    bool retVal = true;
    retVal = retVal &&
             mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ &&
             mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             srcDims == rhs.srcDims &&
             srcOrder == rhs.srcOrder &&
             srcPrc == rhs.srcPrc &&
             dstPrc == rhs.dstPrc;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}

}   // namespace intel_cpu
}   // namespace ov