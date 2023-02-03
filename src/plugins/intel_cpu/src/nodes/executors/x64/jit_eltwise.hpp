// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../eltwise.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

#define MAX_ELTWISE_INPUTS 7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_params {
    EltwiseAttrs attrs;

    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
};

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
    //ptr to array of post op inputs pointers (flat list)
    const void* post_op_data;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

class JitEltwiseExecutor : public EltwiseExecutor {
public:
    JitEltwiseExecutor(const ExecutorContext::CPtr context);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    // struct Key {
    //     EltwiseAttrs mvnAttrs;
    //     VectorDims srcDims;
    //     VectorDims srcOrder;
    //     InferenceEngine::Precision srcPrc;
    //     InferenceEngine::Precision dstPrc;
    //     dnnl::primitive_attr attr;

    //     Key(const EltwiseAttrs& mvnAttrs,
    //         const std::vector<MemoryDescCPtr>& srcDescs,
    //         const std::vector<MemoryDescCPtr>& dstDescs,
    //         const dnnl::primitive_attr &attr);
    //     size_t hash() const;
    //     bool operator==(const Key& rhs) const;
    // };

    const VectorDims& getOutDims() const {
        if (!_pKernel)
            IE_THROW() << "Can't get jit eltwise params, kernel for Eltwise executor is not compiled";
        return _pKernel->jep_.dims;
    }
    size_t getBatchDimIdx() const {
        return _batchDimIdx;
    }

private:
    impl_desc_type implType = impl_desc_type::jit_uni;

    jit_eltwise_params jep = {};
    std::unique_ptr<jit_uni_eltwise_kernel> _pKernel;
    size_t _schedulerWorkAmount = 0;
    size_t _batchDimIdx = 0;
    static const int optimalTensorRank = 6;
};

class JitEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& mvnAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const override {
        return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
    }

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<JitEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov