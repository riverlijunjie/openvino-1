// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_model.h"

#include "common.h"

ov_status_e ov_model_outputs(const ov_model_t* model, ov_output_node_list_t* output_ports) {
    if (!model || !output_ports) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->outputs();
        output_ports->size = results.size();
        std::unique_ptr<ov_output_const_node_t[]> tmp_output_ports(new ov_output_const_node_t[output_ports->size]);

        for (size_t i = 0; i < output_ports->size; i++) {
            tmp_output_ports[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        output_ports->output_ports = tmp_output_ports.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_inputs(const ov_model_t* model, ov_output_node_list_t* input_ports) {
    if (!model || !input_ports) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto results = std::const_pointer_cast<const ov::Model>(model->object)->inputs();
        input_ports->size = results.size();
        std::unique_ptr<ov_output_const_node_t[]> tmp_output_ports(new ov_output_const_node_t[input_ports->size]);

        for (size_t i = 0; i < input_ports->size; i++) {
            tmp_output_ports[i].object = std::make_shared<ov::Output<const ov::Node>>(std::move(results[i]));
        }
        input_ports->output_ports = tmp_output_ports.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_input_by_name(const ov_model_t* model,
                                         const char* tensor_name,
                                         ov_output_const_node_t** input_port) {
    if (!model || !tensor_name || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(tensor_name);
        std::unique_ptr<ov_output_const_node_t> _input_port(new ov_output_const_node_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_input_by_index(const ov_model_t* model,
                                          const size_t index,
                                          ov_output_const_node_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->input(index);
        std::unique_ptr<ov_output_const_node_t> _input_port(new ov_output_const_node_t);
        _input_port->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_node_t** input_port) {
    if (!model || !tensor_name || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->input(tensor_name);
        std::unique_ptr<ov_output_node_t> _input_port(new ov_output_node_t);
        _input_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_node_t** input_port) {
    if (!model || !input_port) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->input(index);
        std::unique_ptr<ov_output_node_t> _input_port(new ov_output_node_t);
        _input_port->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *input_port = _input_port.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_output_by_index(const ov_model_t* model,
                                           const size_t index,
                                           ov_output_const_node_t** output_node) {
    if (!model || !output_node) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->output(index);
        std::unique_ptr<ov_output_const_node_t> _output_node(new ov_output_const_node_t);
        _output_node->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_node = _output_node.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_const_output_by_name(const ov_model_t* model,
                                          const char* tensor_name,
                                          ov_output_const_node_t** output_node) {
    if (!model || !tensor_name || !output_node) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = std::const_pointer_cast<const ov::Model>(model->object)->output(tensor_name);
        std::unique_ptr<ov_output_const_node_t> _output_node(new ov_output_const_node_t);
        _output_node->object = std::make_shared<ov::Output<const ov::Node>>(std::move(result));
        *output_node = _output_node.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_output_by_index(const ov_model_t* model, const size_t index, ov_output_node_t** output_node) {
    if (!model || !output_node) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->output(index);
        std::unique_ptr<ov_output_node_t> _output_node(new ov_output_node_t);
        _output_node->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *output_node = _output_node.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_output_by_name(const ov_model_t* model, const char* tensor_name, ov_output_node_t** output_node) {
    if (!model || !tensor_name || !output_node) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto result = model->object->output(tensor_name);
        std::unique_ptr<ov_output_node_t> _output_node(new ov_output_node_t);
        _output_node->object = std::make_shared<ov::Output<ov::Node>>(std::move(result));
        *output_node = _output_node.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

bool ov_model_is_dynamic(const ov_model_t* model) {
    if (!model) {
        printf("[ERROR] The model is NULL!!!\n");
        return false;
    }
    return model->object->is_dynamic();
}

ov_status_e ov_model_reshape_input_by_name(const ov_model_t* model,
                                           const char* tensor_name,
                                           const ov_partial_shape_t* partial_shape) {
    if (!model || !tensor_name || !partial_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shape;
        if (partial_shape->rank.is_static() && (partial_shape->rank.get_length() == partial_shape->dims.size())) {
            in_shape[tensor_name] = partial_shape->dims;
        } else {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        model->object->reshape(in_shape);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape(const ov_model_t* model,
                             const char* tensor_names[],
                             const ov_partial_shape_t* partial_shapes[],
                             size_t cnt) {
    if (!model || !tensor_names || !partial_shapes || cnt < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<std::string, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto name = tensor_names[i];
            if (partial_shapes[i]->rank.is_static() &&
                (partial_shapes[i]->rank.get_length() == partial_shapes[i]->dims.size())) {
                in_shapes[name] = partial_shapes[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_by_ports(const ov_model_t* model,
                                      size_t* ports,
                                      const ov_partial_shape_t** partial_shape,
                                      size_t cnt) {
    if (!model || !ports || !partial_shape || cnt < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<size_t, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto port_id = ports[i];
            if (partial_shape[i]->rank.is_static() &&
                (partial_shape[i]->rank.get_length() == partial_shape[i]->dims.size())) {
                in_shapes[port_id] = partial_shape[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_reshape_one_input(const ov_model_t* model, const ov_partial_shape_t* partial_shape) {
    size_t port = 0;
    return ov_model_reshape_by_ports(model, &port, &partial_shape, 1);
}

ov_status_e ov_model_reshape_by_nodes(const ov_model_t* model,
                                      const ov_output_node_t* output_ports[],
                                      const ov_partial_shape_t* partial_shapes[],
                                      size_t cnt) {
    if (!model || !output_ports || !partial_shapes || cnt < 1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::map<ov::Output<ov::Node>, ov::PartialShape> in_shapes;
        for (auto i = 0; i < cnt; i++) {
            auto node = *output_ports[i]->object;
            if (partial_shapes[i]->rank.is_static() &&
                (partial_shapes[i]->rank.get_length() == partial_shapes[i]->dims.size())) {
                in_shapes[node] = partial_shapes[i]->dims;
            } else {
                return ov_status_e::PARAMETER_MISMATCH;
            }
        }
        model->object->reshape(in_shapes);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name) {
    if (!model || !friendly_name) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto& result = model->object->get_friendly_name();
        *friendly_name = str_to_char_array(result);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_model_free(ov_model_t* model) {
    if (model)
        delete model;
}
