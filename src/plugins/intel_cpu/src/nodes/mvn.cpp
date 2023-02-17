// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.h"

#include <algorithm>
#include <string>
#include <vector>

#include "fake_quantize.h"
#include "eltwise.h"
#include <dnnl_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"
#include "emitters/x64//jit_bf16_emitters.hpp"

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>

#include <ngraph/opsets/opset6.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl;
using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

} // namespace

bool MVN::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank < 1 || inDataRank > 5) {
            errorMessage = "First input accepts ranks from 1 to 5. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ngraph::as_type_ptr<const ngraph::op::v6::MVN>(op)) {
            auto axesOp = ngraph::as_type_ptr<ngraph::op::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ngraph::op::MVNEpsMode::INSIDE_SQRT &&
                    epsMode != ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                        std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVN node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + inDataRank : axe;
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                if (inDataRank > 5 || (inDataRank != axesVal.size() + 1 && inDataRank != axesVal.size() + 2)) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else if (auto mvnOp = ngraph::as_type_ptr<const ngraph::op::v0::MVN>(op)) {
        } else {
            errorMessage = "Node is not an instance of the MVN operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MVN::MVN(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    mvnAttrs.epsMode_ = INSIDE_SQRT;
    if (auto mvnOp = ngraph::as_type_ptr<ngraph::op::v6::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        if (mvnOp->get_eps_mode() == ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
            mvnAttrs.epsMode_ = OUTSIDE_SQRT;
        }

        mvnAttrs.initAcrossChannels_ = false;
        const auto& inDataShapeSize = getInputShapeAtPort(0).getRank();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1)
            mvnAttrs.initAcrossChannels_ = true;
    } else if (auto mvnOp = ngraph::as_type_ptr<ngraph::op::v0::MVN>(op)) {
        mvnAttrs.normalizeVariance_ = mvnOp->get_normalize_variance();
        mvnAttrs.epsValue_ = mvnOp->get_eps();
        mvnAttrs.initAcrossChannels_ = mvnOp->get_across_channels();
    }
}

void MVN::getSupportedDescriptors() {}

void MVN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrecision = getOriginalInputPrecisionAtPort(0);
    Precision outputPrecision = getOriginalOutputPrecisionAtPort(0);
    if (!mayiuse(avx512_core)) {
        if (outputPrecision == Precision::BF16)
            outputPrecision = Precision::FP32;
    }

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
    }

    // ref with float planar and no fusion
    if (!mayiuse(cpu::x64::sse41)) {
        inputPrecision = outputPrecision = Precision::FP32;
    }

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(inputsNum);
    config.outConfs.resize(1);
    config.inConfs[0].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);
    if (inputsNum == 2) {
        config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, getInputShapeAtPort(1)));
        config.inConfs[1].constant(true);
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format) {
        config.inConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(0)));
        config.outConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));

        std::vector<MemoryDescCPtr> srcMemoryDescs;
        for (int i = 0; i < config.inConfs.size(); i++) {
            srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
        }
        std::vector<MemoryDescCPtr> dstMemoryDescs;
        for (int i = 0; i < config.outConfs.size(); i++) {
            dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
        }

        auto factory = std::make_shared<MVNExecutorFactory>(mvnAttrs, srcMemoryDescs, dstMemoryDescs,
                                                            std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::undef, factory});
    };

    if (mayiuse(cpu::x64::sse41)) {
        // nspc
        if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
            pushDesc(LayoutType::nspc);

            // blk
            if (mayiuse(cpu::x64::avx512_core)) {
                pushDesc(LayoutType::nCsp16c);
            } else {
                pushDesc(LayoutType::nCsp8c);
            }
        }
    }

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    if (getInputShapeAtPort(0).getRank() == 4 || getInputShapeAtPort(0).getRank() == 5) {
        pushDesc(LayoutType::nspc);
    }
#endif

    // planar
    if (canBeInplace)
        config.inConfs[0].inPlace(0);
    pushDesc(LayoutType::ncsp);
}

void MVN::prepareParams() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    std::vector<MemoryDescCPtr> srcMemoryDescs;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }
    std::vector<MemoryDescCPtr> dstMemoryDescs;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemoryDescs.push_back(getChildEdgeAt(i)->getMemoryPtr()->getDescPtr());
    }

    dnnl::primitive_attr attr;
    setPostOps(attr, true);
    auto selectedPD = getSelectedPrimitiveDescriptor();
    execPtr = selectedPD->getExecutorFactoryAs<MVNExecutorFactory>()->makeExecutor(mvnAttrs, srcMemoryDescs, dstMemoryDescs, attr);
    selectedPD->setImplementationType(execPtr->getImplType());
}

void MVN::setPostOps(dnnl::primitive_attr &attr, bool initWeights) {
    dnnl::post_ops ops;
    auto postOpDims = MVNExecutor::transformTo5DCase(getParentEdgesAtPort(0)[0]->getMemoryPtr()->getStaticDims(), mvnAttrs.initAcrossChannels_);

    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, postOpDims, postOpsDataPtrs);
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }
    attr.set_post_ops(ops);
}

void MVN::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void MVN::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute MVN node. Executor is not created";
    }

    std::vector<MemoryCPtr> srcMemory;
    for (int i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (int i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }

    execPtr->exec(srcMemory, dstMemory, postOpsDataPtrs.data());
}

bool MVN::canFuse(const NodePtr& node) const {
    if (!mayiuse(cpu::x64::sse41)) {
        return false;
    }
    // limit post ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getInputShapeAtPort(0).getRank();
    bool unaryEltwise = one_of(node->getAlgorithm(), Algorithm::EltwiseRelu,
                                                     Algorithm::EltwiseGeluErf,
                                                     Algorithm::EltwiseGeluTanh,
                                                     Algorithm::EltwiseElu,
                                                     Algorithm::EltwiseSigmoid,
                                                     Algorithm::EltwiseClamp,
                                                     Algorithm::EltwiseTanh,
                                                     Algorithm::EltwiseSwish,
                                                     Algorithm::EltwiseHswish,
                                                     Algorithm::EltwiseMish,
                                                     Algorithm::EltwiseHsigmoid,
                                                     Algorithm::EltwiseRoundHalfToEven,
                                                     Algorithm::EltwiseRoundHalfAwayFromZero,
                                                     Algorithm::EltwiseAbs,
                                                     Algorithm::EltwiseSqrt,
                                                     Algorithm::EltwiseSoftRelu);
    if ((inputRank == 1 && !unaryEltwise) ||
        (inputRank == 2 && !unaryEltwise && mvnAttrs.initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVN::created() const {
    return getType() == Type::MVN;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
