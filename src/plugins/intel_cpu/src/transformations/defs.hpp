// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace intel_cpu {

#if defined(OPENVINO_ARCH_X86_64)

#define OV_CPU_REGISTER_PASS_X64(MANAGER, PASS) \
    MANAGER.register_pass<PASS>();

#define OV_CPU_DISABLE_PASS_X64(MANAGER, PASS) \
    MANAGER.get_pass_config()->disable<PASS>();

#define OV_CPU_SET_CALLBACK_X64(MANAGER, CALLBACK, ...) \
    MANAGER.get_pass_config()->set_callback<__VA_ARGS__>(CALLBACK);

#else

#define OV_CPU_REGISTER_PASS_X64(MANAGER, PASS)
#define OV_CPU_DISABLE_PASS_X64(MANAGER, PASS)
#define OV_CPU_SET_CALLBACK_X64(MANAGER, CALLBACK, ...)

#endif

}   // namespace intel_cpu
}   // namespace ov
