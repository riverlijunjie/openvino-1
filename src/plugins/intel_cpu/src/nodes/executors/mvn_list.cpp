// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<MVNExecutorDesc>& getMVNExecutorsList() {
    static std::vector<MVNExecutorDesc> descs = {
        { impl_desc_type::jit_uni, std::make_shared<JitMVNExecutorBuilder>() },
        { impl_desc_type::ref, std::make_shared<RefMVNExecutorBuilder>() },
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov