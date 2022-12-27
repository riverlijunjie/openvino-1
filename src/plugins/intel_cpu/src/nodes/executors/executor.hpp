// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"

namespace ov {
namespace intel_cpu {

class ExecutorFactory {
public:
    ExecutorFactory() = default;
    ~ExecutorFactory() = default;

    virtual void setRuntimeCache(const MultiCachePtr& cache) {
        runtimeCache = cache;
    }

    MultiCachePtr runtimeCache = nullptr;
};

using ExecutorFactoryPtr = std::shared_ptr<ExecutorFactory>;
using ExecutorFactoryCPtr = std::shared_ptr<const ExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov