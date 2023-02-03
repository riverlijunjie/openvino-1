// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "matmul.h"

#include "ngraph/opsets/opset1.hpp"
#include "ie_precision.hpp"
#include "cpu_types.h"
#include "eltwise.h"
#include "fake_quantize.h"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"

#include <numeric>
#include <string>
#include <vector>
#include <memory>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

bool canBeExecutedInInt8(const Precision& firstInput, const Precision& secondInput) {
    return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
}
} // namespace

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    matmulAttrs.transposeA = matMul->get_transpose_a();
    matmulAttrs.transposeB = matMul->get_transpose_b();
}

bool MatMul::canFuse(const NodePtr& node) const {
    // per channel binary post op for rank > 2D is supported only by oneDNN reference implementation because of unusual MatMul channel axis (issue 6669)
    if (getOutputShapeAtPort(0).getRank() > 2) {
        if (const auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get())) {
            if (one_of(eltwiseNode->getAlgorithm(), Algorithm::EltwiseAdd,
                                                    Algorithm::EltwiseMultiply,
                                                    Algorithm::EltwiseSubtract,
                                                    Algorithm::EltwiseDivide,
                                                    Algorithm::EltwisePrelu,
                                                    Algorithm::EltwiseMulAdd,
                                                    Algorithm::EltwisePowerStatic) &&
                eltwiseNode->getBroadcastingPolicy() != Eltwise::PerTensor) {
                return false;
            }
        } else if (const auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get())) {
            if (fakeQuantizeNode->getBroadcastingPolicy() != FakeQuantize::PerTensor) {
                return false;
            }
        }
    }

    // Todo:
    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32, but the FQ child will still has the int8 input precision.
    //  This information should be propagated! Note that we may need to propagate updated precision to child fused nodes.
    if (node->getType() == Type::FakeQuantize &&
        one_of(node->getOriginalOutputPrecisionAtPort(0), Precision::I8, Precision::U8) &&
        !canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1)))
        return false;
    return canFuseSimpleOperation(node);
}

void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
    dnnl::post_ops ops;

    dnnl::memory::data_type outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecisions[0]);

    bool isINT8 = canBeExecutedInInt8(getOriginalInputPrecisionAtPort(0), getOriginalInputPrecisionAtPort(1));

    DnnlPostOpsComposer dnnlpoc(getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8);

    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
            continue;
        }

        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
                   << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

    setPostOps(*attr, dims, true);

    return attr;
}

Node::AttrPtr MatMul::initPrimitiveAttr() {
    auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
    return initPrimitiveAttr(dummyShape.getStaticDims());
}

void MatMul::getSupportedDescriptors() {
}

void MatMul::initSupportedPrimitiveDescriptors() {
    matmulAttrs.withBias = getOriginalInputsNumber() == 3;

        inputPrecisions = getOriginalInputPrecisions();
        outputPrecisions = getOriginalOutputPrecisions();
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        if (inputPrecisions[0].size() != inputPrecisions[1].size())
            inputPrecisions[0] = inputPrecisions[1] = getMaxPrecision(getOriginalInputPrecisions());

        // fallback to fp32 for any precision that cannot be handled natively
        if ((!one_of(inputPrecisions[0] , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
            !one_of(inputPrecisions[1] , Precision::I8, Precision::BF16, Precision::FP32))) {
            outputPrecisions[0] = inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        }

        if (!fusedWith.empty()) {
            outputPrecisions[0] = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        }

        if (!canBeExecutedInInt8( inputPrecisions[0], inputPrecisions[1]) && one_of(outputPrecisions[0], Precision::U8, Precision::I8))
            outputPrecisions[0] = Precision::FP32; // INT output is not supported for non-INT inputs
    } else {
        inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        outputPrecisions[0] = Precision::FP32;
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    NodeConfig config;
    config.dynBatchSupport = true;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(-1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecisions[i], getInputShapeAtPort(i)));

        config.inConfs.push_back(portConfig);
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(canBeInPlace() ? 0 : -1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecisions[i], getOutputShapeAtPort(i)));

        config.outConfs.push_back(portConfig);
    }

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < config.inConfs.size(); i++) {
        srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < config.outConfs.size(); i++) {
        dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
    }

    auto attr = initPrimitiveAttr();
    auto factory = std::make_shared<MatMulExecutorFactory>(matmulAttrs, srcMemoryDescs, dstMemoryDescs, *attr.get(),
                                                           std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef, factory);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

size_t MatMul::getMaxBatch() const {
    if (!outputShapes.empty())
        return outputShapes[0].getStaticDims()[0];
    return 0;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

void MatMul::prepareParams() {
    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }

    AttrPtr attr = initPrimitiveAttr(dstMemoryDescs[0]->getShape().getStaticDims() );

    auto selectedPD = getSelectedPrimitiveDescriptor();
    execPtr = selectedPD->getExecutorFactoryAs<MatMulExecutorFactory>()->makeExecutor(matmulAttrs, srcMemoryDescs, dstMemoryDescs, *attr.get());
    selectedPD->setImplementationType(execPtr->getImplType());
}

void MatMul::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute MatMul node. Executor is not created";
    }

    std::vector<MemoryCPtr> srcMemory;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }

    execPtr->exec(srcMemory, dstMemory, postOpsArgs);
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getPrimitivesPriority() {
    return implPriorities;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
