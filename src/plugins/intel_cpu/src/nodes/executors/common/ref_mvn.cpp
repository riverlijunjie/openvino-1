// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_mvn.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

MVNRefExecutor::MVNRefExecutor() : MVNExecutor() {}

bool MVNRefExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescCPtr>& srcDescs,
                          const std::vector<MemoryDescCPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    this->mvnAttrs = mvnAttrs;

    if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 ||
        dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32)
        return false;

    if (!srcDescs[0]->hasLayoutType(LayoutType::ncsp) ||
        !dstDescs[0]->hasLayoutType(LayoutType::ncsp))
        return false;

    return true;
}

void MVNRefExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    uint8_t *src_data = reinterpret_cast<uint8_t*>(src[0]->GetPtr());
    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dst[0]->GetPtr());

    const float *src_data_ptr = reinterpret_cast<const float *>(src_data);
    float *dst_data_ptr = reinterpret_cast<float *>(dst_data);

    auto shape5D = transformTo5DCase(src[0]->getStaticDims(), mvnAttrs.initAcrossChannels_);
    size_t N = shape5D[0];
    size_t C = shape5D[1];
    size_t D = shape5D[2];
    size_t H = shape5D[3];
    size_t W = shape5D[4];

    auto execAcrossChannels = mvnAttrs.initAcrossChannels_;
    if (one_of(src[0]->getStaticDims().size(), 1, 2) && mvnAttrs.initAcrossChannels_) {
        execAcrossChannels = false;
    }

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    parallel_for(N, [&](int b) {
        size_t cb = b * C3;
        if (execAcrossChannels) {
            // Parallel sum for each channel for mean
            float C3inv = 1.f / static_cast<float>(C3);
            float mean_temp = 0.0f;

            mean_temp = parallel_sum(C, mean_temp, [&](size_t c)->float {
                float mean_internal = 0.0f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean_internal += src_data_ptr[cc + sp];
                }
                return mean_internal;
            });

            float mean = mean_temp * C3inv;

            if (mvnAttrs.normalizeVariance_) {
                // parallel sum for each channel for variance
                float variance_temp = 0.0f;
                variance_temp = parallel_sum(C, variance_temp, [&](size_t c)->float {
                    float variance_internal = 0.0f;
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance_internal += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }
                    return variance_internal;
                });

                float variance = 1.f;
                if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                    variance = 1.f / sqrtf(variance_temp * C3inv + mvnAttrs.epsValue_);
                else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                    variance = 1.f / (sqrtf(variance_temp * C3inv) + mvnAttrs.epsValue_);

                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                });
            } else {
                parallel_for(C, [&](int c) {
                    size_t cc = cb + c * C2;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                });
            }
        } else {  // per channel
            float C2inv = 1.f / static_cast<float>(C2);
            parallel_for(C, [&](size_t c) {
                // mean for this channel
                float mean = 0.f;
                size_t cc = cb + c * C2;
                for (size_t sp = 0lu; sp < C2; sp++) {
                    mean += src_data_ptr[cc + sp];
                }
                mean *= C2inv;

                if (mvnAttrs.normalizeVariance_) {
                    // variance for this channel
                    float variance = 0.f;
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        variance += (src_data_ptr[cc + sp] - mean) * (src_data_ptr[cc + sp] - mean);
                    }

                    if (mvnAttrs.epsMode_ == INSIDE_SQRT)
                        variance = 1.f / sqrtf(variance * C2inv + mvnAttrs.epsValue_);
                    else if (mvnAttrs.epsMode_ == OUTSIDE_SQRT)
                        variance = 1.f / (sqrtf(variance * C2inv) + mvnAttrs.epsValue_);

                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = (src_data_ptr[cc + sp] - mean) * variance;
                    }
                } else {
                    // mvn for this channel
                    for (size_t sp = 0lu; sp < C2; sp++) {
                        dst_data_ptr[cc + sp] = src_data_ptr[cc + sp] - mean;
                    }
                }
            });
        }
    });
}

}   // namespace intel_cpu
}   // namespace ov
