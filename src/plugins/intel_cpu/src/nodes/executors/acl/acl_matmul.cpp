// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_matmul.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

TensorShape ShapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

AclMatMulExecutor::AclMatMulExecutor() : MatMulExecutor() {}

bool AclMatMulExecutor::init(const MatMulAttrs& matmulAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              const dnnl::primitive_attr &attr) {
    this->matmulAttrs = matmulAttrs;
    // _fconn = std::make_unique<NEFullyConnectedLayer>(_memory_manager);
    // _fconn = std::make_unique<NEFullyConnectedLayer>();
    // _fconn->configure(conv_input, conv_weights, biases, _qi ? &_outputqi : _output);
    return true;
}

void AclMatMulExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    // ARM_COMPUTE_ERROR_ON_MSG(!fc.get(), "Kernel isn't configured");

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor biaTensor;
    arm_compute::Tensor dstTensor;

    TensorInfo srcTensorInfo = TensorInfo(ShapeCast(src[0]->getStaticDims()), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo weiTensorInfo = TensorInfo(ShapeCast(src[1]->getStaticDims()), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo biaTensorInfo = TensorInfo(TensorShape(dst[0]->getStaticDims()[1]), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(ShapeCast(dst[0]->getStaticDims()), 1, DataType::F32, DataLayout::NCHW);

    ARM_COMPUTE_ERROR_THROW_ON(
        NEFullyConnectedLayer::validate(srcTensorInfo, weiTensorInfo, matmulAttrs.withBias ? biaTensorInfo : nullptr, dstTensor));

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    if (matmulAttrs.withBias)
        biaTensor.allocator()->init(biaTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    fc->configure(&srcTensor,
                  &weiTensor,
                  matmulAttrs.withBias ? &biaTensor : nullptr,
                  &dstTensor);

    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    weiTensor.allocator()->import_memory(src[1]->GetPtr());
    if (matmulAttrs.withBias)
        biaTensor.allocator()->import_memory(src[2]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    fc->run();

    srcTensor.allocator()->free();
    weiTensor.allocator()->free();
    if (matmulAttrs.withBias)
        biaTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
