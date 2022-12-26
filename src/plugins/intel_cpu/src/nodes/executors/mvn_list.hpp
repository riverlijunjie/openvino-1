// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn.hpp"
#include "x64/jit_mvn.hpp"
#include "common/ref_mvn.hpp"

#include "onednn/iml_type_mapper.h"
#include "cache/multi_cache.h"

#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ExecutorDesc {
    impl_desc_type implType;
    MVNExecutorBuilderCPtr builder;
};

const std::vector<ExecutorDesc>& getMVNExecutorsList();

class MVNExecutorFactory {
public:
    MVNExecutorFactory(const MVNAttrs& mvnAttrs,
                       const std::vector<MemoryDescCPtr>& srcDescs,
                       const std::vector<MemoryDescCPtr>& dstDescs) {
        for (auto& desc : getMVNExecutorsList()) {
            if (desc.builder->isSupported(mvnAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~MVNExecutorFactory() = default;
    virtual MVNExecutorPtr makeExecutor(const MVNAttrs& mvnAttrs,
                                        const std::vector<MemoryDescCPtr>& srcDescs,
                                        const std::vector<MemoryDescCPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const ExecutorDesc* desc) {
            switch (desc->implType) {
                case impl_desc_type::jit_uni: {
                    auto builder = [&](const MVNJitExecutor::Key& key) -> MVNExecutorPtr {
                        auto executor = desc->builder->makeExecutor();
                        if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };


                    auto key = MVNJitExecutor::Key(mvnAttrs, srcDescs, dstDescs, attr);
                    auto res = runtimeCache->getOrCreate(key, builder);
                    if (res.second == CacheEntryBase::LookUpStatus::Miss)
                        std::cerr << "Miss" << std::endl;
                    else
                        std::cerr << "Hit" << std::endl;

                    return res.first;
                } break;
                default: {
                    auto executor = desc->builder->makeExecutor();
                    if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            MVNExecutorPtr ptr = nullptr;
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

    virtual void setRuntimeCache(const MultiCachePtr& cache) {
        runtimeCache = cache;
    }

    std::vector<ExecutorDesc> supportedDescs;
    const ExecutorDesc* chosenDesc = nullptr;
    MultiCachePtr runtimeCache = nullptr;
};

using MVNExecutorFactoryPtr = std::shared_ptr<MVNExecutorFactory>;
using MVNExecutorFactoryCPtr = std::shared_ptr<const MVNExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov