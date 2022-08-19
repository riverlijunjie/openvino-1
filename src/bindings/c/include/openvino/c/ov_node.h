// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API, which is a C wrapper for ov::Node class.
 *
 * @file ov_node.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_partial_shape.h"
#include "openvino/c/ov_shape.h"

typedef struct ov_output_const_node ov_output_const_node_t;
typedef struct ov_output_node ov_output_node_t;

/**
 * @struct ov_output_node_list_t
 * @brief Reprents an array of ov_output_nodes.
 */
typedef struct {
    ov_output_const_node_t* output_nodes;
    size_t size;
} ov_output_node_list_t;

// Node
/**
 * @defgroup node node
 * @ingroup openvino_c
 * Set of functions representing of Model and Node.
 * @{
 */

/**
 * @brief Get the shape of port object.
 * @ingroup node
 * @param port A pointer to ov_output_const_node_t.
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_const_node_get_shape(ov_output_const_node_t* port, ov_shape_t* tensor_shape);

/**
 * @brief Get the shape of port object.
 * @ingroup node
 * @param port A pointer to ov_output_node_t.
 * @param tensor_shape tensor shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_node_get_shape(ov_output_node_t* port, ov_shape_t* tensor_shape);

/**
 * @brief Get the tensor name of port list by index.
 * @ingroup node
 * @param port_list A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param tensor_name A pointer to the tensor name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_any_name_by_index(ov_output_node_list_t* port_list, size_t idx, char** tensor_name);

/**
 * @brief Get the shape of port by index.
 * @ingroup node
 * @param port_list A pointer to the ov_output_node_list_t.
 * @param idx Index of the input tensor
 * @param shape The shape of the port.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_shape_by_index(ov_output_node_list_t* port_list, size_t idx, ov_shape_t* shape);

/**
 * @brief Get the partial shape of port with index in port_list.
 * @ingroup node
 * @param port_list A pointer to the ov_output_node_list_t.
 * @param idx Index in the port list.
 * @param partial_shape Partial shape.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_partial_shape_by_index(ov_output_node_list_t* port_list, size_t idx, ov_partial_shape_t** partial_shape);

/**
 * @brief Get the tensor type of port by index.
 * @ingroup node
 * @param nodes A pointer to the ov_output_node_list_t.
 * @param idx Index of port in port list.
 * @param tensor_type tensor type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_node_list_get_element_type_by_index(ov_output_node_list_t* port_list, size_t idx, ov_element_type_e* tensor_type);

/**
 * @brief free port list.
 * @ingroup node
 * @param port_list The pointer to the instance of the ov_output_node_list_t to free.
 */
OPENVINO_C_API(void) ov_output_node_list_free(ov_output_node_list_t* port_list);

/**
 * @brief free port object
 * @ingroup node
 * @param port The pointer to the instance of the ov_output_node_t to free.
 */
OPENVINO_C_API(void) ov_output_node_free(ov_output_node_t* port);

/**
 * @brief free const port
 * @ingroup node
 * @param port The pointer to the instance of the ov_output_const_node_t to free.
 */
OPENVINO_C_API(void) ov_output_const_node_free(ov_output_const_node_t* port);

/** @} */  // end of Node
