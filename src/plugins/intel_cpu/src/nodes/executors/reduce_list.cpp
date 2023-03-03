// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ReduceExecutorDesc>& getReduceExecutorsList() {
    static std::vector<ReduceExecutorDesc> descs = {
        //OV_CPU_INSTANCE_X64(ExecutorType::x64, std::make_shared<JitReduceExecutorBuilder>())
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclReduceExecutorBuilder>())
        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefReduceExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov