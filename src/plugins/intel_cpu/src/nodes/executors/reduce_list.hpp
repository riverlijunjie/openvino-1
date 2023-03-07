// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "reduce.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_reduce.hpp"
#endif
#include "common/ref_reduce.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ReduceExecutorDesc {
    ExecutorType executorType;
    ReduceExecutorBuilderCPtr builder;
};

const std::vector<ReduceExecutorDesc>& getReduceExecutorsList();

class ReduceExecutorFactory : public ExecutorFactory {
public:
    ReduceExecutorFactory(const ReduceAttrs& reduceAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getReduceExecutorsList()) {
            if (desc.builder->isSupported(reduceAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ReduceExecutorFactory() = default;
    virtual ReduceExecutorPtr makeExecutor(const ReduceAttrs& reduceAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const ReduceExecutorDesc* desc) {
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(reduceAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ReduceExecutorPtr ptr = nullptr;
            return ptr;
        };


        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

private:
    std::vector<ReduceExecutorDesc> supportedDescs;
    const ReduceExecutorDesc* chosenDesc = nullptr;
};

using ReduceExecutorFactoryPtr = std::shared_ptr<ReduceExecutorFactory>;
using ReduceExecutorFactoryCPtr = std::shared_ptr<const ReduceExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov