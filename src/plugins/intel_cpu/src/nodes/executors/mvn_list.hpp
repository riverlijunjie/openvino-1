// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn.hpp"
#include "x64/jit_mvn.hpp"
#include "common/ref_mvn.hpp"

#include "onednn/iml_type_mapper.h"

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
                                        const dnnl_primitive_attr &attr) const {
        for (auto& sd : supportedDescs) {
            auto executor = sd.builder->makeExecutor();
            if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

    std::vector<ExecutorDesc> supportedDescs;
};


}   // namespace intel_cpu
}   // namespace ov