// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_model C API, which is a C wrapper for ov::Model class.
 * A user-defined model.
 * @file ov_model.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_partial_shape.h"

typedef struct ov_model ov_model_t;

// Model
/**
 * @defgroup model model
 * @ingroup openvino_c
 * Set of functions representing of Model and Node.
 * @{
 */

/**
 * @brief Release the memory allocated by ov_model_t.
 * @ingroup model
 * @param model A pointer to the ov_model_t to free memory.
 */
OPENVINO_C_API(void) ov_model_free(ov_model_t* model);

/**
 * @brief Get output ports list of ov_model_t.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param output_ports A pointer to the ov_output_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_outputs(const ov_model_t* model, ov_output_node_list_t* output_ports);

/**
 * @brief Get the outputs of ov_model_t.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param input_ports A pointer to the ov_input_nodes.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_model_inputs(const ov_model_t* model, ov_output_node_list_t* input_ports);

/**
 * @brief Get const input port of ov_model_t by name.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_port A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_node_t** input_port);

/**
 * @brief Get const input port of ov_model_t by port index.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_port A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_input_by_index(const ov_model_t* model, const size_t index, ov_output_const_node_t** input_port);

/**
 * @brief Get input port of ov_model_t by name.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param input_port A pointer to the ov_output_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_node_t** input_port);

/**
 * @brief Get input port of ov_model_t by port index.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param input_port A pointer to the ov_output_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_node_t** input_port);

/**
 * @brief Get const output port of ov_model_t by port index.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param output_port A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_output_by_index(const ov_model_t* model, const size_t index, ov_output_const_node_t** output_port);

/**
 * @brief Get const port of ov_model_t by name.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param output_port A pointer to the ov_output_const_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_const_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_const_node_t** output_port);

/**
 * @brief Get output port of ov_model_t by port index.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param index input tensor index.
 * @param output_port A pointer to the ov_output_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_output_by_index(const ov_model_t* model, const size_t index, ov_output_node_t** output_port);

/**
 * @brief Get output port of ov_model_t by name.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param output_port A pointer to the ov_output_node_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_model_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_node_t** output_port);

/**
 * @brief Returns true if any of the op's defined in the model contains partial shape.
 * @param model A pointer to the ov_model_t.
 */
OPENVINO_C_API(bool) ov_model_is_dynamic(const ov_model_t* model);

/**
 * @brief Do reshape in model with a list of <name, partial shape>.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_names input tensor name (char *) list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape(const ov_model_t* model,
                 const char* tensor_names[],
                 const ov_partial_shape_t* partial_shapes[],
                 size_t cnt);

/**
 * @brief Do reshape in model with partial shape for a specified name.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param tensor_name input tensor name (char *).
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_input_by_name(const ov_model_t* model,
                               const char* tensor_name,
                               const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model for one node(port 0).
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param partialShape A PartialShape.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_one_input(const ov_model_t* model, const ov_partial_shape_t* partial_shape);

/**
 * @brief Do reshape in model with a list of <port id, partial shape>.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param ports The port list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_ports(const ov_model_t* model, size_t* ports, const ov_partial_shape_t** partial_shape, size_t cnt);

/**
 * @brief Do reshape in model with a list of <ov_output_node_t, partial shape>.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param output_nodes The ov_output_node_t list.
 * @param partialShape A PartialShape list.
 * @param cnt The item count in the list.
 */
OPENVINO_C_API(ov_status_e)
ov_model_reshape_by_nodes(const ov_model_t* model,
                          const ov_output_node_t* output_nodes[],
                          const ov_partial_shape_t* partial_shapes[],
                          size_t cnt);

/**
 * @brief Gets the friendly name for a model.
 * @ingroup model
 * @param model A pointer to the ov_model_t.
 * @param friendly_name the model's friendly name.
 */
OPENVINO_C_API(ov_status_e) ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name);

/** @} */  // end of Model
