# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_CPU)
    return()
endif()

set(TARGET_NAME "openvino_intel_cpu_plugin")

# TODO: use 14 standart only for ARM build
set(CMAKE_CXX_STANDARD 14)

if(CMAKE_COMPILER_IS_GNUCC)
    ie_add_compiler_flags(-Wno-all)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from mkl-dnn headers conversion from 'XXX' to 'YYY', possible loss of data
    ie_add_compiler_flags(/wd4267)
    ie_add_compiler_flags(/wd4244)
    # mkldnn headers: '<<': result of 32-bit shift implicitly converted to 64 bits
    ie_add_compiler_flags(/wd4334)
    # plugin: inherits 'XXX' via dominance
    ie_add_compiler_flags(/wd4250)
    # '<': signed/unsigned mismatch
    ie_add_compiler_flags(/wd4018)
    ie_add_compiler_flags(/wd4309)
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

if(X86_64)
    add_definitions(-DOV_CPU_X64)
    set(OV_CPU_X64 ON)
elseif(AARCH64)
    add_definitions(-DOV_CPU_AARCH64)
    set(OV_CPU_AARCH64 ON)
else()
    message(FATAL_ERROR "Unsupported system processor ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if (ENABLE_CPU_DEBUG_CAPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_DEBUG_CAPS")
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h(pp))

addVersionDefines(${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp CI_BUILD_NUMBER)

add_subdirectory(thirdparty)
if(ENABLE_TESTS)
    add_subdirectory(tests/unit)
endif()
if(ENABLE_FUNCTIONAL_TESTS)
    add_subdirectory(tests/functional)
endif()

if(ACL_FOUND)
    add_definitions(-DOV_CPU_WITH_ACL)
    set(OV_CPU_WITH_ACL ON)
endif()

if (NOT OV_CPU_WITH_ACL)
    file(GLOB_RECURSE SOURCES_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*.cpp)
    file(GLOB_RECURSE HEADERS_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*.h
                                        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/acl/*.hpp)

    list(REMOVE_ITEM SOURCES ${SOURCES_TO_REMOVE})
    list(REMOVE_ITEM HEADERS ${HEADERS_TO_REMOVE})
endif()

if (NOT OV_CPU_X64)
    file(GLOB_RECURSE SOURCES_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*.cpp)
    file(GLOB_RECURSE HEADERS_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*.h
                                        ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors/x64/*.hpp)

    list(REMOVE_ITEM SOURCES ${SOURCES_TO_REMOVE})
    list(REMOVE_ITEM HEADERS ${HEADERS_TO_REMOVE})

    file(GLOB_RECURSE SOURCES_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/x64/*.cpp)
    file(GLOB_RECURSE HEADERS_TO_REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/x64/*.h
                                        ${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/x64/*.hpp)

    list(REMOVE_ITEM SOURCES ${SOURCES_TO_REMOVE})
    list(REMOVE_ITEM HEADERS ${HEADERS_TO_REMOVE})
endif()

# create plugin

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "CPU"
              AS_EXTENSION
              SOURCES ${SOURCES} ${HEADERS})

set_ie_threading_interface_for(${TARGET_NAME})

ie_mark_target_as_cc(${TARGET_NAME})

if(SELECTIVE_BUILD STREQUAL "ON")
    # After disabling a block of code, some variables might be unused.
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        target_compile_options(${TARGET_NAME} PRIVATE -Wno-unused-variable)
    endif()
endif()

target_link_libraries(${TARGET_NAME} PRIVATE dnnl
                                             ov_shape_inference
                                             openvino::pugixml
                                             inference_engine_snippets)

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_INFERENCE_EXTENSION_API)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

# Cross compiled function
# TODO: The same for proposal, proposalONNX, topk
cross_compiled_file(${TARGET_NAME}
        ARCH AVX2 ANY
                    src/nodes/proposal_imp.cpp
        API         src/nodes/proposal_imp.hpp
        NAME        proposal_exec
        NAMESPACE   InferenceEngine::Extensions::Cpu::XARCH
)

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#  add test object library

if(BUILD_SHARED_LIBS)
    add_library(${TARGET_NAME}_obj OBJECT ${SOURCES} ${HEADERS})
    link_system_libraries(${TARGET_NAME}_obj PUBLIC dnnl openvino::pugixml)

    target_include_directories(${TARGET_NAME}_obj
        PRIVATE
            $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:openvino::itt,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:ov_shape_inference,INTERFACE_INCLUDE_DIRECTORIES>
            $<TARGET_PROPERTY:inference_engine_snippets,INTERFACE_INCLUDE_DIRECTORIES>
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj SYSTEM PUBLIC $<TARGET_PROPERTY:dnnl,INCLUDE_DIRECTORIES>)

    target_include_directories(${TARGET_NAME}_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/nodes/executors)

    set_ie_threading_interface_for(${TARGET_NAME}_obj)

    target_compile_definitions(${TARGET_NAME}_obj PRIVATE
        USE_STATIC_IE IMPLEMENT_INFERENCE_ENGINE_PLUGIN IMPLEMENT_INFERENCE_EXTENSION_API
        $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_COMPILE_DEFINITIONS>)

    set_target_properties(${TARGET_NAME}_obj PROPERTIES EXCLUDE_FROM_ALL ON)

    # LTO
    set_target_properties(${TARGET_NAME}_obj PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
endif()

# LTO
set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
