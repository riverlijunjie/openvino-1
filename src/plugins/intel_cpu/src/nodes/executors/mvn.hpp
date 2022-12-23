// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"

namespace ov {
namespace intel_cpu {

// Defines way to add epsilon: inside sqrt or outside.
enum MVNEpsMode {
    INSIDE_SQRT,
    OUTSIDE_SQRT
};

struct MVNAttrs {
    bool initAcrossChannels_;
    bool normalizeVariance_;
    float epsValue_;
    MVNEpsMode epsMode_;
};

class MVNExecutor {
public:
    MVNExecutor();
    virtual bool init(const MVNAttrs& mvnAttrs,
                      const std::vector<MemoryDescCPtr>& srcDescs,
                      const std::vector<MemoryDescCPtr>& dstDescs,
                      const dnnl_primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual ~MVNExecutor() = default;

    static InferenceEngine::SizeVector transformTo5DCase(const InferenceEngine::SizeVector& shape, bool initAcrossChannels);

protected:
    MVNAttrs mvnAttrs;
};

using MVNExecutorPtr = std::shared_ptr<MVNExecutor>;
using MVNExecutorCPtr = std::shared_ptr<const MVNExecutor>;

class MVNExecutorBuilder {
public:
    ~MVNExecutorBuilder() = default;
    virtual bool isSupported(const MVNAttrs& mvnAttrs, const std::vector<MemoryDescCPtr>& srcDescs, const std::vector<MemoryDescCPtr>& dstDescs) const = 0;
    virtual MVNExecutorPtr makeExecutor() const = 0;
};

using MVNExecutorBuilderPtr = std::shared_ptr<MVNExecutorBuilder>;
using MVNExecutorBuilderCPtr = std::shared_ptr<const MVNExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov