// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn.hpp"

namespace ov {
namespace intel_cpu {

class MVNRefExecutor : public MVNExecutor {
public:
    MVNRefExecutor();

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescCPtr>& srcDescs,
              const std::vector<MemoryDescCPtr>& dstDescs,
              const dnnl_primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

private:
    void mvn_ref(const uint8_t *in_ptr_, uint8_t *out_ptr_);
};

class RefMVNExecutorBuilder : public MVNExecutorBuilder {
public:
    bool isSupported(const MVNAttrs& mvnAttrs, const std::vector<MemoryDescCPtr>& srcDescs, const std::vector<MemoryDescCPtr>& dstDescs) const override {
        return true;
    }

    MVNExecutorPtr makeExecutor() const override {
        return std::make_shared<MVNRefExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov