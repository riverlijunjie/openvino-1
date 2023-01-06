// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<MatMulExecutorDesc>& getMatMulExecutorsList() {
    static std::vector<MatMulExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(impl_desc_type::unknown, std::make_shared<AclMatMulExecutorBuilder>())
        OV_CPU_INSTANCE_X64(impl_desc_type::jit_uni, std::make_shared<DnnlMatMulExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov