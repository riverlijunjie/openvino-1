// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<MVNExecutorDesc>& getMVNExecutorsList() {
    static std::vector<MVNExecutorDesc> descs = {
        OV_CPU_INSTANCE_X64(ExecutorType::x64, std::make_shared<JitMVNExecutorBuilder>())
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclMVNExecutorBuilder>())
        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefMVNExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov