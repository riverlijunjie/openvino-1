// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../matmul.hpp"
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>

namespace ov {
namespace intel_cpu {

arm_compute::TensorShape ShapeCast(const VectorDims& dims);

class AclMatMulExecutor : public MatMulExecutor {
public:
    AclMatMulExecutor();

    bool init(const MatMulAttrs& matmulAttrs,
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
    MatMulAttrs matmulAttrs;
    impl_desc_type implType = impl_desc_type::undef;
    std::unique_ptr<arm_compute::NEFullyConnectedLayer> fc = nullptr;
};

class AclMatMulExecutorBuilder : public MatMulExecutorBuilder {
public:
    bool isSupported(const MatMulAttrs& matmulAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const dnnl::primitive_attr &attr) const override {
        // TODO: add correct conditions
        return true;
    }

    MatMulExecutorPtr makeExecutor() const override {
        return std::make_shared<AclMatMulExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov