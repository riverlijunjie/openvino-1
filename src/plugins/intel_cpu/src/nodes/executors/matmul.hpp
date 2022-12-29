// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "dnnl_scratch_pad.h"

namespace ov {
namespace intel_cpu {

struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
    bool withBias;
};

class MatMulExecutor {
public:
    MatMulExecutor();
    virtual bool init(const MatMulAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~MatMulExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

    void setEngine(const dnnl::engine& engine) {
        this->engine = engine;
    }

    void setScratchPad(const DnnlScratchPadPtr& scratchPad) {
        this->scratchPad = scratchPad;
    }

    void setImplPriorities(const std::vector<impl_desc_type>& implPriorities) {
        this->implPriorities = implPriorities;
    }

protected:
    MatMulAttrs mvnAttrs;

    dnnl::engine engine;
    std::vector<impl_desc_type> implPriorities;
    DnnlScratchPadPtr scratchPad = nullptr;
};

using MatMulExecutorPtr = std::shared_ptr<MatMulExecutor>;
using MatMulExecutorCPtr = std::shared_ptr<const MatMulExecutor>;

class MatMulExecutorBuilder {
public:
    ~MatMulExecutorBuilder() = default;
    virtual bool isSupported(const MatMulAttrs& MatMulAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) const = 0;
    virtual MatMulExecutorPtr makeExecutor() const = 0;
};

using MatMulExecutorBuilderPtr = std::shared_ptr<MatMulExecutorBuilder>;
using MatMulExecutorBuilderCPtr = std::shared_ptr<const MatMulExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov