// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/itt.hpp"
#include "pass_manager.h"
#include "program_helpers.h"

using namespace cldnn;

static size_t total_shared_count = 0;
static double total_duration = 0.0;
void build_implementations::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::build_implementations");
    if (p.get_config().get_partial_build_program()) {
        return;
    }

    auto& cache = p.get_kernels_cache();
    auto cache_shared = p.get_kernels_cache_shared();
    bool enable_cache_shared = p.is_internal_program() == false && p.is_body_program() == true;

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t shared_count = 0, built_count = 0;
    std::cout << "build_implementations: " << p.get_processing_order().size() << ", enable_cache_shared = " << enable_cache_shared << std::endl;

    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();
            if (enable_cache_shared) {
                // std::cout << "\tkernel_name = " << impl->get_kernel_name() << ", params: id = " << params->desc->id << std::endl;
                auto res = cache_shared->match_compiled_kernels(*params);
                if (res.first == false) {
                    cache.add_kernels_source(*params, impl->get_kernels_source());
                    built_count++;
                    // std::cout << "\tbuilt: " << n->id() << std::endl;
                    // std::cout << "\t\tadd kernels source to compile " << std::endl;
                } else {
                    shared_count++;
                    // std::cout << "\tshared: " << n->id() << std::endl;
                    // std::cout << "\t\tskip add kernels source to cache, reuse from cache_shared, cache_shared_compiled_kernel.size() =  " <<
                    // res.second.size()
                    //          << std::endl;
                }
            } else {
                cache.add_kernels_source(*params, impl->get_kernels_source());
                built_count++;
            }
        }
    }

    if (built_count > 0)
        cache.build_all();

    // std::cout << "build_implementations: after build_all" << std::endl;
    for (auto& n : p.get_processing_order()) {
        if (auto impl = n->get_selected_impl()) {
            auto params = n->get_kernel_impl_params();

            if (enable_cache_shared) {
                // std::cout << "\tkernel_name = " << impl->get_kernel_name() << ", params: id = " << params->desc->id << std::endl;

                auto cache_res = cache.get_compiled_kernels(*params);
                auto cache_shared_res = cache_shared->get_compiled_kernels(*params);

                if (cache_res.first == true) {
                    // Use local cache first
                    // Note: res.second may be empty, it means that the kernel is not compiled at all
                    // Not all impl will have compiled kernels,
                    // std::cout << "\t\tUse local cache first and add compiled kernels to cache_shared, compiled_kernels.size() = " << cache_res.second.size()
                    //          << std::endl;
                    cache_shared->add_compiled_kernels(*params, cache_res.second);
                } else if (cache_shared_res.first == true) {
                    // reuse kernel from cache_shared, maybe empty kernel
                    cache.add_compiled_kernels(*params, cache_shared_res.second);
                    // std::cout << "\t\treuse kernel " << std::endl;
                } else {
                    // Cannot find kernel in cache_shared and local cache, it means it doesn't need kernel
                    // let set it to empty or not set at all
                    kernels_cache::compiled_kernels_iter empty_kernel = {};
                    cache_shared->add_compiled_kernels(*params, empty_kernel);
                    cache.add_compiled_kernels(*params, empty_kernel);
                    // std::cout << "\t\tset empty kernel" << std::endl;
                }
            }
            impl->init_kernels(cache, *params);
            impl->reset_kernels_source();
        }
    }
    total_shared_count += shared_count;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    total_duration += duration;
    std::cout << "build_implementations: built_count = " << built_count << ", shared_count = " << shared_count << " total_shared_count = " << total_shared_count
              << ", total_cache_shared_size = " << cache_shared->get_kernel_size() << ", took " << duration << " ms" << ", total_cost = " << total_duration
              << " ms" << std::endl
              << std::endl;

    cache.reset();
}
