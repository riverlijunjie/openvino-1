// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "matmul.hpp"
#include "dnnl/dnnl_matmul.hpp"

#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

struct MatMulExecutorDesc {
    impl_desc_type implType;
    MatMulExecutorBuilderCPtr builder;
};

const std::vector<MatMulExecutorDesc>& getMatMulExecutorsList();

class MatMulExecutorFactory : public ExecutorFactory {
public:
    MatMulExecutorFactory(const MatMulAttrs& MatMulAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) : ExecutorFactory() {
        for (auto& desc : getMatMulExecutorsList()) {
            if (desc.builder->isSupported(MatMulAttrs, srcDescs, dstDescs, attr)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~MatMulExecutorFactory() = default;
    virtual MatMulExecutorPtr makeExecutor(const MatMulAttrs& MatMulAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const MatMulExecutorDesc* desc) {
            switch (desc->implType) {
                case impl_desc_type::jit_uni: {
                    auto builder = [&](const DnnlMatMulExecutor::Key& key) -> MatMulExecutorPtr {
                        auto executor = desc->builder->makeExecutor();
                        executor->setEngine(engine);
                        executor->setScratchPad(scratchPad);
                        executor->setImplPriorities(implPriorities);
                        if (executor->init(MatMulAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = DnnlMatMulExecutor::Key(MatMulAttrs, srcDescs, dstDescs, attr);
                    auto res = runtimeCache->getOrCreate(key, builder);
                    return res.first;
                } break;
                default: {
                    auto executor = desc->builder->makeExecutor();
                    executor->setEngine(engine);
                    executor->setScratchPad(scratchPad);
                    executor->setImplPriorities(implPriorities);

                    if (executor->init(MatMulAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            MatMulExecutorPtr ptr = nullptr;
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

    void setEngine(const dnnl::engine& engine) {
        this->engine = engine;
    }

    void setScratchPad(const DnnlScratchPadPtr& scratchPad) {
        this->scratchPad = scratchPad;
    }

    void setImplPriorities(const std::vector<impl_desc_type>& implPriorities) {
        this->implPriorities = implPriorities;
    }

private:
    // TODO: remove dnnl dependency
    dnnl::engine engine;

    DnnlScratchPadPtr scratchPad = nullptr;

    std::vector<impl_desc_type> implPriorities;

    std::vector<MatMulExecutorDesc> supportedDescs;
    const MatMulExecutorDesc* chosenDesc = nullptr;
};

using MatMulExecutorFactoryPtr = std::shared_ptr<MatMulExecutorFactory>;
using MatMulExecutorFactoryCPtr = std::shared_ptr<const MatMulExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov