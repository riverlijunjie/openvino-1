// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/reduce.hpp"

namespace ov {
namespace intel_cpu {

class RefReduceExecutor : public ReduceExecutor {
public:
    RefReduceExecutor(const ExecutorContext::CPtr context);

    bool init(const ReduceAttrs& reduceAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    void reduce_ref_process(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, float init_value, std::function<float(float, float)> func);
    inline void reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);
    inline void calc_process_dst_dims(const InferenceEngine::SizeVector &dst_dim);

    impl_desc_type implType = impl_desc_type::ref;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;
};

class RefReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefReduceExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov