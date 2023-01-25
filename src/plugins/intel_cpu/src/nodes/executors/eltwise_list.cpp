// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList() {
    static std::vector<EltwiseExecutorDesc> descs = {
        OV_CPU_INSTANCE_X64(ExecutorType::x64, std::make_shared<JitEltwiseExecutorBuilder>())
        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefEltwiseExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov