// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"

namespace ov {
namespace intel_cpu {

#if defined(OV_CPU_WITH_ACL)
#define OV_CPU_INSTANCE_ACL(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_ACL(...)
#endif

#if defined(OV_CPU_WITH_DNNL)
#define OV_CPU_INSTANCE_DNNL(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_DNNL(...)
#endif

#if defined(OV_CPU_X64)
#define OV_CPU_INSTANCE_X64(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_X64(...)
#endif

#define OV_CPU_INSTANCE_COMMON(...) \
    {__VA_ARGS__},

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