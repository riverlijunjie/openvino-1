// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_reduce.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

RefReduceExecutor::RefReduceExecutor(const ExecutorContext::CPtr context) : ReduceExecutor(context) {}

bool RefReduceExecutor::init(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    this->reduceAttrs = reduceAttrs;

    if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 ||
        dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32)
        return false;

    if (!srcDescs[0]->hasLayoutType(LayoutType::ncsp) ||
        !dstDescs[0]->hasLayoutType(LayoutType::ncsp))
        return false;

    src_dims = srcDescs[0]->getShape().getDims();
    calc_process_dst_dims(dstDescs[0]->getShape().getStaticDims());
    return true;
}

void RefReduceExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    switch (reduceAttrs.operation) {
        case Algorithm::ReduceAnd:
            reduce_ref_process(src, dst, 1, [](float x, float y)->float { return x && y; });
            break;
        case Algorithm::ReduceL1:
            reduce_ref_process(src, dst, 0, [](float old, float y)->float { return old + (y >= 0 ? y : -y); });
            break;
        case Algorithm::ReduceL2:
            reduce_ref_process(src, dst, 0, [](float old, float y)->float { return old + y * y; });
            break;
        case Algorithm::ReduceLogSum:
            reduce_ref_process(src, dst, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceLogSumExp:
            reduce_ref_process(src, dst, 0, [](float old, float y)->float { return old + expf(y); });
            break;
        case Algorithm::ReduceMax:
            reduce_ref_process(src, dst, std::numeric_limits<float>::lowest(),
                                                    [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Algorithm::ReduceMean:
            reduce_ref_process(src, dst, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceMin:
            reduce_ref_process(src, dst, std::numeric_limits<float>::max(),
                                                    [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Algorithm::ReduceOr:
            reduce_ref_process(src, dst, 0, [](float x, float y)->float { return x || y; });
            break;
        case Algorithm::ReduceProd:
            reduce_ref_process(src, dst, 1, [](float x, float y)->float { return x * y; });
            break;
        case Algorithm::ReduceSum:
            reduce_ref_process(src, dst, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceSumSquare:
            reduce_ref_process(src, dst, 0, [](float old, float y)->float { return old + y * y; });
            break;
    default:
        IE_THROW() << "Reduce node gets unsupported reduce mode.";
    }
}

inline void RefReduceExecutor::calc_process_dst_dims(const InferenceEngine::SizeVector &dst_dims) {
    std::set<size_t> axes;
    InferenceEngine::SizeVector out_dims;
    process_dst_dims.clear();
    axes_for_reduction.clear();
    for (auto &axis : reduceAttrs.axes) {
        if (axis < 0)
            axis += src_dims.size();
        if (static_cast<size_t>(axis) > src_dims.size())
            IE_THROW() << "Reduce node " << axis << " " << src_dims.size() << " exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < src_dims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (reduceAttrs.keepDims) out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(src_dims[i]);
            process_dst_dims.push_back(src_dims[i]);
        }
    }
    for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
        if (out_dims[i] != dst_dims[i])
            IE_THROW() << "Reduce node gets incorrect number of output dimensions!";
    }
}

void RefReduceExecutor::reduce_ref_process(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst,
                                           float init_value, std::function<float(float, float)> func) {
    float *in_ptr = reinterpret_cast<float*>(src[0]->GetPtr());
    float *out_ptr = reinterpret_cast<float*>(dst[0]->GetPtr());

    size_t work_amount_dst = 1, reduced_dims_work_amount = 1;
    for (size_t i = 0; i < process_dst_dims.size(); i++)
        work_amount_dst *= process_dst_dims[i];
    for (size_t i = 0; i < src_dims.size(); i++)
        reduced_dims_work_amount *= src_dims[i];
    reduced_dims_work_amount /= work_amount_dst;

    InferenceEngine::SizeVector src_strides = src[0]->GetDescWithType<BlockedMemoryDesc>()->getStrides();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        InferenceEngine::SizeVector dst_counters(process_dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = process_dst_dims.size() - 1, i = start; j >= 0; j--) {
            dst_counters[j] = i % process_dst_dims[j];
            i /= process_dst_dims[j];
        }
        for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
            float reduce_prod = init_value;
            bool update_idx = true;
            InferenceEngine::SizeVector src_counters = dst_counters;
            for (i = 0; i < reduced_dims_work_amount; ++i) {
                if (update_idx) {
                    src_idx = 0;
                    for (j = 0; j < static_cast<int>(src_dims.size()); ++j)
                        src_idx += (src_counters[j] % src_dims[j]) * src_strides[j];
                    update_idx = false;
                }
                reduce_prod = func(reduce_prod, in_ptr[src_idx]);
                for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                    src_counters[axes_for_reduction[j]]++;
                    if (src_counters[axes_for_reduction[j]] < src_dims[axes_for_reduction[j]]) {
                        src_idx += src_strides[axes_for_reduction[j]];
                        break;
                    } else {
                        src_counters[axes_for_reduction[j]] = 0;
                        update_idx = true;
                    }
                }
            }
            out_ptr[dst_idx] = reduce_prod;
            for (j = process_dst_dims.size() - 1; j >= 0; j--) {
                dst_counters[j]++;
                if (dst_counters[j] < process_dst_dims[j])
                    break;
                else
                    dst_counters[j] = 0;
            }
        }
    });

    reduce_ref_map(out_ptr, work_amount_dst, reduced_dims_work_amount);
}

inline void RefReduceExecutor::reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount) {
    switch (reduceAttrs.operation) {
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceL1:
        case Algorithm::ReduceMax:
        case Algorithm::ReduceMin:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceProd:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            break;
        case Algorithm::ReduceL2:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = std::sqrt(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = logf(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceMean:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] /= reduced_dims_work_amount;
            });
            break;
        default:
            IE_THROW() << "Reduce node gets unsupported reduce mode.";
    }
}

}   // namespace intel_cpu
}   // namespace ov
