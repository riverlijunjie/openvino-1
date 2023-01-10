// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_eltwise.hpp"
#include "ie_parallel.hpp"
#include "utils/general_utils.h"
#include "dnnl_extension_utils.h"
#include "cpu/primitive_attr_postops.hpp"

namespace ov {
namespace intel_cpu {

RefEltwiseExecutor::RefEltwiseExecutor() : EltwiseExecutor() {}

static void offset_out_calc(VectorDims& offset, const VectorDims& dims) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

static void offset_in_calc(VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

bool RefEltwiseExecutor::init(const EltwiseAttrs& eltwiseAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              const std::vector<EltwisePostOp>& postOps) {
    for (const auto& desc : srcDescs) {
        if (desc->getPrecision() != InferenceEngine::Precision::FP32)
            return false;
    }
    for (const auto& desc : dstDescs) {
        if (desc->getPrecision() != InferenceEngine::Precision::FP32)
            return false;
    }

    if (!postOps.empty())
        return false;

    this->eltwiseAttrs = eltwiseAttrs;

    auto outBlockingDesc = MemoryDescUtils::convertToBlockedMemoryDesc(dstDescs[0]);
    const auto &outOrder = outBlockingDesc->getOrder();
    const auto &currentOutBlkDims = outBlockingDesc->getBlockDims();

    size_t input_size = currentOutBlkDims.size();
    std::vector<VectorDims> inpDims;
    // init dims
    _inputNum = srcDescs.size();
    inpDims.resize(_inputNum);
    for (int i = 0; i < _inputNum; i++) {
        inpDims[i].resize(input_size, 1);
    }

    size_t outRank = currentOutBlkDims.size();

    std::vector<VectorDims> currentInBlkDims(_inputNum);
    for (int i = 0; i < _inputNum; i++) {
        auto inBlockingDesc = MemoryDescUtils::convertToBlockedMemoryDesc(srcDescs[i]);
        currentInBlkDims[i] = inBlockingDesc->getBlockDims();
        size_t inRank = currentInBlkDims[i].size();

        // WA to normalize blocked and planar layouts
        const auto &inOrder = inBlockingDesc->getOrder();
        size_t startOff = outOrder.size() != outBlockingDesc->getShape().getRank() &&
                          outOrder[outOrder.size() - 1] != inOrder[inOrder.size() - 1] ? 1 : 0;

        // WA to handle nspc layout with 1D tensors
        if (1 == inRank) {
            if (outRank > 2 && 1 == outOrder.back()) startOff = 1;
        }

        for (int j = 0; j < inRank; j++) {
            inpDims[i][inpDims[i].size() - 1 - j - startOff] = currentInBlkDims[i][inRank - 1 - j];
        }
    }

    _dims.resize(input_size, 1);
    for (int i = 0; i < currentOutBlkDims.size(); i++) {
        _dims[_dims.size() - 1 - i] = currentOutBlkDims[currentOutBlkDims.size() - 1 - i];
    }

    _fullWorkAmount = 1;
    for (int i = 0; i < _dims.size(); i++) {
        _fullWorkAmount *= _dims[i];
    }

    // init offset
    _dst_offsets.resize(input_size, 1);
    offset_out_calc(_dst_offsets, _dims);
    for (int j = 0; j < input_size; j++) {
        _dst_offsets[j] *= sizeof(float); // only FP32 out prc is supported
    }

    _src_offsets.resize(_inputNum);
    for (int i = 0; i < _inputNum; i++) {
        _src_offsets[i].resize(input_size, 1);
        offset_in_calc(_src_offsets[i], inpDims[i], _dims);
        for (int j = 0; j < input_size; j++) {
            _src_offsets[i][j] *= sizeof(float); // only FP32 inp prcs are supported
        }
    }

    return true;
}

void RefEltwiseExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    std::vector<const float*> src_ptrs;
    for (int i = 0; i < src.size(); i++) {
        src_ptrs.push_back(reinterpret_cast<const float*>(src[i]->GetPtr()));
    }
    float* dst_ptr = reinterpret_cast<float*>(dst[0]->GetPtr());

    // auto batchDimIdx = execPtr->getBatchDimIdx();
    VectorDims dims_out = _dims;

    std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
    if (one_of(eltwiseAttrs.algorithm,
            Algorithm::EltwiseRelu, Algorithm::EltwiseGeluErf, Algorithm::EltwiseGeluTanh, Algorithm::EltwiseElu, Algorithm::EltwiseTanh,
            Algorithm::EltwiseSigmoid, Algorithm::EltwiseAbs, Algorithm::EltwiseSqrt, Algorithm::EltwiseSoftRelu, Algorithm::EltwiseExp,
            Algorithm::EltwiseClamp, Algorithm::EltwiseSwish, Algorithm::EltwiseHswish, Algorithm::EltwiseMish, Algorithm::EltwiseHsigmoid,
            Algorithm::EltwiseRoundHalfToEven, Algorithm::EltwiseRoundHalfAwayFromZero)) {
        auto dnnlAlgorithm = DnnlExtensionUtils::convertToDnnlAlgorithm(eltwiseAttrs.algorithm);
        ref_eltwise_injector = std::make_shared<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>(
                static_cast<dnnl_alg_kind_t>(dnnlAlgorithm), eltwiseAttrs.alpha, eltwiseAttrs.beta, 1.f);
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0, end = 0;
        splitter(_fullWorkAmount, nthr, ithr, start, end);

        std::vector<size_t> counters(dims_out.size(), 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = dims_out.size() - 1; j >= 0; j--) {
                counters[j] = tmp % dims_out[j];
                tmp /= dims_out[j];
            }

            std::vector<size_t> index_in(_inputNum);
            for (int i = 0; i < _inputNum; i++) {
                index_in[i] = 0;
                for (int j = 0; j < counters.size(); j++) {
                    index_in[i] += counters[j] * _src_offsets[i][j];
                }
                index_in[i] /= sizeof(float);
            }

            size_t index_out = 0;
            for (int j = 0; j < counters.size(); j++) {
                index_out += counters[j] * _dst_offsets[j];
            }
            index_out /= sizeof(float);

            std::vector<float> src_f(_inputNum);
            for (int i = 0; i < _inputNum; i++) {
                src_f[i] = (src_ptrs[i] + index_in[i])[0];
            }
            float* dst_ptr_f = dst_ptr + index_out;

            if (ref_eltwise_injector) {
                    *dst_ptr_f = ref_eltwise_injector->compute_scalar(src_f[0]);
            } else {
                switch (eltwiseAttrs.algorithm) {
                    case Algorithm::EltwiseAdd:               *dst_ptr_f = src_f[0] + src_f[1]; break;
                    case Algorithm::EltwiseMulAdd:            *dst_ptr_f = src_f[0] * src_f[1] + src_f[2]; break;
                    case Algorithm::EltwiseSubtract:          *dst_ptr_f = src_f[0] - src_f[1]; break;
                    case Algorithm::EltwiseMultiply:          *dst_ptr_f = src_f[0] * src_f[1]; break;
                    case Algorithm::EltwiseDivide:            *dst_ptr_f = src_f[0] / src_f[1]; break;
                    case Algorithm::EltwiseFloorMod:          *dst_ptr_f = src_f[0] - floorf(src_f[0] / src_f[1]) * src_f[1]; break;
                    case Algorithm::EltwiseMod:               *dst_ptr_f = src_f[0] - truncf(src_f[0] / src_f[1]) * src_f[1]; break;
                    case Algorithm::EltwiseMaximum:           *dst_ptr_f = std::max(src_f[0], src_f[1]); break;
                    case Algorithm::EltwiseMinimum:           *dst_ptr_f = std::min(src_f[0], src_f[1]); break;
                    case Algorithm::EltwiseSquaredDifference: *dst_ptr_f = powf((src_f[0] - src_f[1]), 2.f); break;
                    case Algorithm::EltwisePowerDynamic:      *dst_ptr_f = powf(src_f[0], src_f[1]); break;
                    case Algorithm::EltwiseEqual:             *dst_ptr_f = src_f[0] == src_f[1]; break;
                    case Algorithm::EltwiseNotEqual:          *dst_ptr_f = src_f[0] != src_f[1]; break;
                    case Algorithm::EltwiseGreater:           *dst_ptr_f = src_f[0] > src_f[1]; break;
                    case Algorithm::EltwiseGreaterEqual:      *dst_ptr_f = src_f[0] >= src_f[1]; break;
                    case Algorithm::EltwiseLess:              *dst_ptr_f = src_f[0] < src_f[1]; break;
                    case Algorithm::EltwiseLessEqual:         *dst_ptr_f = src_f[0] <= src_f[1]; break;
                    case Algorithm::EltwiseLogicalAnd:        *dst_ptr_f = src_f[0] && src_f[1]; break;
                    case Algorithm::EltwiseLogicalOr:         *dst_ptr_f = src_f[0] || src_f[1]; break;
                    case Algorithm::EltwiseLogicalXor:        *dst_ptr_f = (src_f[0] || src_f[1]) - (src_f[0] && src_f[1]); break;
                    case Algorithm::EltwiseLogicalNot:        *dst_ptr_f = !src_f[0]; break;
                    case Algorithm::EltwisePowerStatic:       *dst_ptr_f = powf(eltwiseAttrs.beta * src_f[0] + eltwiseAttrs.gamma, eltwiseAttrs.alpha); break;
                    case Algorithm::EltwisePrelu:             *dst_ptr_f = src_f[0] > 0 ? src_f[0] : src_f[0] * src_f[1]; break;
                    case Algorithm::EltwiseErf:               *dst_ptr_f = std::erf(src_f[0]); break;
                    case Algorithm::EltwiseSoftSign:          *dst_ptr_f = src_f[0] / (1 + std::fabs(src_f[0])); break;
                    case Algorithm::EltwiseIsFinite:          *dst_ptr_f = std::isfinite(src_f[0]); break;
                    case Algorithm::EltwiseIsInf:
                        *dst_ptr_f = eltwiseAttrs.alpha && (src_f[0] == -std::numeric_limits<float>::infinity()) ||
                                     eltwiseAttrs.beta  && (src_f[0] == std::numeric_limits<float>::infinity());
                        break;
                    case Algorithm::EltwiseIsNaN:             *dst_ptr_f = std::isnan(src_f[0]); break;
                    default: IE_THROW() << "Unsupported operation type for Eltwise executor";
                }
            }
        }
    });
}

}   // namespace intel_cpu
}   // namespace ov
