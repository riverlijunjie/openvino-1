// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvn.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

namespace ov {
namespace intel_cpu {

enum MVNLayoutType {
    mvn_planar,
    mvn_block,
    mvn_by_channel
};

struct jit_mvn_config_params {
    MVNLayoutType layout;
    bool across_channels;
    bool normalize_variance;
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    int src_data_size;
    int dst_data_size;
    int N, C, D, H, W;
};

struct jit_mvn_call_args {
    const void *src;
    void *dst;
    float *sum;
    float *mean;
    float *variance;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
    const void* post_op_data;
};

struct jit_uni_mvn_mean_variance_kernel {
    void (*ker_)(const jit_mvn_call_args *);

    void operator()(const jit_mvn_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_mean_variance_kernel(jit_mvn_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_mvn_mean_variance_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
};

struct jit_uni_mvn_kernel {
    void (*ker_)(const jit_mvn_call_args *);

    void operator()(const jit_mvn_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_mvn_kernel(jit_mvn_config_params jcp, const dnnl::primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_mvn_kernel() {}

    virtual void create_ker() = 0;

    jit_mvn_config_params jcp_;
    const dnnl::primitive_attr &attr_;
    int optimized_scaleshift_num = 0;
};

class JitMVNExecutor : public MVNExecutor {
public:
    JitMVNExecutor(const ExecutorContext::CPtr context);

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescCPtr>& srcDescs,
              const std::vector<MemoryDescCPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    struct Key {
        MVNAttrs mvnAttrs;
        VectorDims srcDims;
        VectorDims srcOrder;
        InferenceEngine::Precision srcPrc;
        InferenceEngine::Precision dstPrc;
        dnnl::primitive_attr attr;

        Key(const MVNAttrs& mvnAttrs,
            const std::vector<MemoryDescCPtr>& srcDescs,
            const std::vector<MemoryDescCPtr>& dstDescs,
            const dnnl::primitive_attr &attr);
        size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

private:
    void mvn_pln(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_);
    void mvn_blk(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_);
    void mvn_nspc(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_);

    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_mean_kernel;
    std::shared_ptr<jit_uni_mvn_mean_variance_kernel> mvn_variance_kernel;
    std::shared_ptr<jit_uni_mvn_kernel> mvn_kernel;

    jit_mvn_config_params jcp;

    impl_desc_type implType = impl_desc_type::jit_uni;
};

class JitMVNExecutorBuilder : public MVNExecutorBuilder {
public:
    bool isSupported(const MVNAttrs& mvnAttrs, const std::vector<MemoryDescCPtr>& srcDescs, const std::vector<MemoryDescCPtr>& dstDescs) const override {
        return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
    }

    MVNExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<JitMVNExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov