// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn.hpp"

namespace ov {
namespace intel_cpu {

class RefMVNExecutor : public MVNExecutor {
public:
    RefMVNExecutor();

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescCPtr>& srcDescs,
              const std::vector<MemoryDescCPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    void mvn_ref(const uint8_t *in_ptr_, uint8_t *out_ptr_);

    impl_desc_type implType = impl_desc_type::ref;
};

class RefMVNExecutorBuilder : public MVNExecutorBuilder {
public:
    bool isSupported(const MVNAttrs& mvnAttrs, const std::vector<MemoryDescCPtr>& srcDescs, const std::vector<MemoryDescCPtr>& dstDescs) const override {
        return true;
    }

    MVNExecutorPtr makeExecutor() const override {
        return std::make_shared<RefMVNExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov