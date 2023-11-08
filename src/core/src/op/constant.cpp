// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

#include "itt.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/type/nf4.hpp"
#include "openvino/runtime/shared_buffer.hpp"

template <typename T>
static inline std::string to_cpp_string(T value) {
    std::string rc;
    if (std::isnan(value)) {
        rc = "NAN";
    } else if (std::isinf(value)) {
        rc = (value > 0 ? "INFINITY" : "-INFINITY");
    } else {
        std::stringstream ss;
        ss << value;
        rc = ss.str();
    }
    return rc;
}
OPENVINO_SUPPRESS_DEPRECATED_START
std::shared_ptr<ov::AlignedBuffer> ov::op::v0::Constant::legacy_to_ov_aligned_buffer(
    const std::shared_ptr<ngraph::runtime::AlignedBuffer>& buffer) {
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(buffer->get_ptr<char>(),
                                                                                               buffer->size(),
                                                                                               buffer);
}
OPENVINO_SUPPRESS_DEPRECATED_END

OPENVINO_SUPPRESS_DEPRECATED_START
ov::op::v0::Constant::Constant(const std::shared_ptr<ngraph::runtime::Tensor>& tensor) {
    m_element_type = tensor->get_element_type();
    m_shape = tensor->get_shape();
    // Share data from HostTensor if we work with it
    // And copy data in other cas
    if (auto hostTensor = std::dynamic_pointer_cast<ngraph::runtime::HostTensor>(tensor)) {
        m_data = std::make_shared<ov::SharedBuffer<std::shared_ptr<ngraph::runtime::Tensor>>>(
            static_cast<char*>(hostTensor->get_data_ptr()),
            tensor->get_size_in_bytes(),
            tensor);
    } else {
        constructor_validate_and_infer_types();
        allocate_buffer(false);
        tensor->read(get_data_ptr_nc(), tensor->get_size_in_bytes());
    }
    constructor_validate_and_infer_types();
}
OPENVINO_SUPPRESS_DEPRECATED_END

ov::op::v0::Constant::Constant(const ov::Tensor& tensor) {
    m_element_type = tensor.get_element_type();
    m_shape = tensor.get_shape();
    // Share data from ov::Tensor
    m_data = std::make_shared<ov::SharedBuffer<ov::Tensor>>(static_cast<char*>(tensor.data()),
                                                            tensor.get_byte_size(),
                                                            tensor);

    constructor_validate_and_infer_types();
}

ov::op::v0::Constant::Constant(const element::Type& type,
                               const ov::Shape& shape,
                               const std::vector<std::string>& values)
    : Constant(false, type, shape) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    NODE_VALIDATION_CHECK(this,
                          values.size() == shape_size(m_shape) || values.size() == 1,
                          "Did not get the expected number of literals for a constant of shape ",
                          m_shape,
                          " (got ",
                          values.size(),
                          ", expected ",
                          shape_size(m_shape),
                          ".");

    using Type_t = element::Type_t;

    if (values.size() == 1 && shape_size(m_shape) != 1) {
        // broadcast single value
        switch (m_element_type) {
        case Type_t::boolean:
            fill_data<Type_t::boolean>(stoi(values[0]));
            break;
        case Type_t::bf16:
            fill_data<Type_t::bf16>(ngraph::parse_string<float>(values[0]));
            break;
        case Type_t::f16:
            fill_data<Type_t::f16>(ngraph::parse_string<float>(values[0]));
            break;
        case Type_t::f32:
            fill_data<Type_t::f32>(ngraph::parse_string<float>(values[0]));
            break;
        case Type_t::f64:
            fill_data<Type_t::f64>(ngraph::parse_string<double>(values[0]));
            break;
        case Type_t::i4:
            fill_data<Type_t::i4>(ngraph::parse_string<int64_t>(values[0]));
            break;
        case Type_t::i8:
            fill_data<Type_t::i8>(ngraph::parse_string<int64_t>(values[0]));
            break;
        case Type_t::i16:
            fill_data<Type_t::i16>(ngraph::parse_string<int64_t>(values[0]));
            break;
        case Type_t::i32:
            fill_data<Type_t::i32>(ngraph::parse_string<int64_t>(values[0]));
            break;
        case Type_t::i64:
            fill_data<Type_t::i64>(ngraph::parse_string<int64_t>(values[0]));
            break;
        case Type_t::u1:
            fill_data<Type_t::u1>(stoi(values[0]));
            break;
        case Type_t::u4:
            fill_data<Type_t::u4>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::u8:
            fill_data<Type_t::u8>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::u16:
            fill_data<Type_t::u16>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::u32:
            fill_data<Type_t::u32>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::u64:
            fill_data<Type_t::u64>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::nf4:
            fill_data<Type_t::nf4>(ngraph::parse_string<uint64_t>(values[0]));
            break;
        case Type_t::undefined:
            OPENVINO_THROW("deserialize unsupported type undefined");
        case Type_t::dynamic:
            OPENVINO_THROW("deserialize unsupported type dynamic");
        }
        update_identical_flags(true, true);
    } else {
        switch (m_element_type) {
        case Type_t::boolean:
            write_buffer<Type_t::boolean>(ngraph::parse_string<uint8_t>(values));
            break;
        case Type_t::bf16:
            write_buffer<Type_t::bf16>(ngraph::parse_string<float>(values));
            break;
        case Type_t::f16:
            write_buffer<Type_t::f16>(ngraph::parse_string<float>(values));
            break;
        case Type_t::f32:
            write_buffer<Type_t::f32>(ngraph::parse_string<float>(values));
            break;
        case Type_t::f64:
            write_buffer<Type_t::f64>(ngraph::parse_string<double>(values));
            break;
        case Type_t::i4:
            write_buffer<Type_t::i4>(ngraph::parse_string<int8_t>(values));
            break;
        case Type_t::i8:
            write_buffer<Type_t::i8>(ngraph::parse_string<int8_t>(values));
            break;
        case Type_t::i16:
            write_buffer<Type_t::i16>(ngraph::parse_string<int16_t>(values));
            break;
        case Type_t::i32:
            write_buffer<Type_t::i32>(ngraph::parse_string<int32_t>(values));
            break;
        case Type_t::i64:
            write_buffer<Type_t::i64>(ngraph::parse_string<int64_t>(values));
            break;
        case Type_t::u1:
            write_buffer<Type_t::u1>(ngraph::parse_string<uint8_t>(values));
            break;
        case Type_t::u4:
            write_buffer<Type_t::u4>(ngraph::parse_string<uint8_t>(values));
            break;
        case Type_t::u8:
            write_buffer<Type_t::u8>(ngraph::parse_string<uint8_t>(values));
            break;
        case Type_t::u16:
            write_buffer<Type_t::u16>(ngraph::parse_string<uint16_t>(values));
            break;
        case Type_t::u32:
            write_buffer<Type_t::u32>(ngraph::parse_string<uint32_t>(values));
            break;
        case Type_t::u64:
            write_buffer<Type_t::u64>(ngraph::parse_string<uint64_t>(values));
            break;
        case Type_t::nf4:
            write_buffer<Type_t::nf4>(ngraph::parse_string<uint8_t>(values));
            break;
        case Type_t::undefined:
            OPENVINO_THROW("deserialize unsupported type undefined");
        case Type_t::dynamic:
            OPENVINO_THROW("deserialize unsupported type dynamic");
        }
        update_identical_flags(false, false);
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}

ov::op::v0::Constant::Constant(const element::Type& type, const ov::Shape& shape) : Constant(true, type, shape) {}

ov::op::v0::Constant::Constant(bool memset_allocation, const element::Type& type, const ov::Shape& shape)
    : m_element_type(type),
      m_shape(shape) {
    allocate_buffer(memset_allocation);
    constructor_validate_and_infer_types();
}

void ov::op::v0::Constant::allocate_buffer(bool memset_allocation) {
    m_data = std::make_shared<ov::AlignedBuffer>(mem_size(), host_alignment());
    if (memset_allocation) {
        std::memset(m_data->get_ptr(), 0, m_data->size());
    }
}

ov::op::v0::Constant::Constant(const element::Type& type, const ov::Shape& shape, const void* data)
    : Constant(false, type, shape) {
    size_t size = (shape_size(m_shape) * m_element_type.bitwidth() + 7) >> 3;
    std::memcpy(get_data_ptr_nc(), data, size);
}

ov::op::v0::Constant::Constant(const Constant& other) {
    m_element_type = other.m_element_type;
    m_shape = other.m_shape;
    m_data = other.m_data;
    update_identical_flags(other.m_all_elements_bitwise_identical_checked, other.m_all_elements_bitwise_identical);
    constructor_validate_and_infer_types();
}

ov::op::v0::Constant::Constant(const Constant& other, const ov::Shape& new_shape) {
    OPENVINO_ASSERT(shape_size(other.m_shape) == shape_size(new_shape),
                    "ov::Shape size " + std::to_string(shape_size(new_shape)) + " is not equal to " +
                        std::to_string(shape_size(other.m_shape)));
    m_element_type = other.m_element_type;
    m_shape = new_shape;
    m_data = other.m_data;
    update_identical_flags(other.m_all_elements_bitwise_identical_checked, other.m_all_elements_bitwise_identical);
    constructor_validate_and_infer_types();
}

ov::op::v0::Constant::~Constant() = default;

std::string ov::op::v0::Constant::convert_value_to_string(size_t index) const {
    std::string rc;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
    using Type_t = element::Type_t;
    switch (get_element_type()) {
    case Type_t::boolean:
        rc = std::to_string(get_element_value<Type_t::boolean>(index));
        break;
    case Type_t::bf16:
        rc = to_cpp_string(static_cast<float>(get_element_value<Type_t::bf16>(index)));
        break;
    case Type_t::f16:
        rc = to_cpp_string(static_cast<float>(get_element_value<Type_t::f16>(index)));
        break;
    case Type_t::f32:
        rc = to_cpp_string(get_element_value<Type_t::f32>(index));
        break;
    case Type_t::f64:
        rc = to_cpp_string(get_element_value<Type_t::f64>(index));
        break;
    case Type_t::i4:
        rc = std::to_string(get_element_value<Type_t::i4>(index));
        break;
    case Type_t::i8:
        rc = std::to_string(get_element_value<Type_t::i8>(index));
        break;
    case Type_t::i16:
        rc = std::to_string(get_element_value<Type_t::i16>(index));
        break;
    case Type_t::i32:
        rc = std::to_string(get_element_value<Type_t::i32>(index));
        break;
    case Type_t::i64:
        rc = std::to_string(get_element_value<Type_t::i64>(index));
        break;
    case Type_t::u1:
        rc = std::to_string(get_element_value<Type_t::u1>(index));
        break;
    case Type_t::u4:
        rc = std::to_string(get_element_value<Type_t::u4>(index));
        break;
    case Type_t::u8:
        rc = std::to_string(get_element_value<Type_t::u8>(index));
        break;
    case Type_t::u16:
        rc = std::to_string(get_element_value<Type_t::u16>(index));
        break;
    case Type_t::u32:
        rc = std::to_string(get_element_value<Type_t::u32>(index));
        break;
    case Type_t::u64:
        rc = std::to_string(get_element_value<Type_t::u64>(index));
        break;
    case Type_t::nf4:
        rc = std::to_string(get_element_value<Type_t::nf4>(index));
        break;
    case Type_t::undefined:
    case Type_t::dynamic:
        OPENVINO_THROW("unsupported type");
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif
    return rc;
}

size_t ov::op::v0::Constant::get_byte_size() const {
    return m_data->size();
}

const void* ov::op::v0::Constant::get_data_ptr() const {
    return (m_data ? m_data->get_ptr() : nullptr);
}

void* ov::op::v0::Constant::get_data_ptr_nc() {
    return (m_data ? m_data->get_ptr() : nullptr);
}

std::vector<std::string> ov::op::v0::Constant::get_value_strings() const {
    std::vector<std::string> rc;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (get_element_type()) {
    case element::Type_t::boolean:
        for (int value : get_vector<char>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::bf16:
        for (bfloat16 value : get_vector<bfloat16>()) {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f16:
        for (float16 value : get_vector<float16>()) {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f32:
        for (float value : get_vector<float>()) {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::f64:
        for (double value : get_vector<double>()) {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::i4:
        for (auto value : cast_vector<int8_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::i8:
        for (int value : get_vector<int8_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::i16:
        for (int value : get_vector<int16_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::i32:
        for (int32_t value : get_vector<int32_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::i64:
        for (int64_t value : get_vector<int64_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::u1:
    case element::Type_t::u4:
    case element::Type_t::nf4:
        for (auto value : cast_vector<uint8_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::u8:
        for (uint32_t value : get_vector<uint8_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::u16:
        for (uint32_t value : get_vector<uint16_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::u32:
        for (uint32_t value : get_vector<uint32_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::u64:
        for (uint64_t value : get_vector<uint64_t>()) {
            rc.push_back(std::to_string(value));
        }
        break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
        OPENVINO_THROW("unsupported type");
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif

    return rc;
}

ov::Shape ov::op::v0::Constant::get_shape_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    std::vector<int64_t> out_shape = cast_vector<int64_t>();
    ov::Shape output_shape(shape_size(m_shape));
    std::transform(out_shape.begin(), out_shape.end(), output_shape.begin(), [&](const int64_t& v) {
        return (v > 0) ? v : 0;
    });
    return output_shape;
}

ov::Strides ov::op::v0::Constant::get_strides_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    std::vector<int64_t> out_strides = cast_vector<int64_t>();
    Strides output_strides(shape_size(m_shape));
    std::transform(out_strides.begin(), out_strides.end(), output_strides.begin(), [&](const int64_t& v) {
        return (v > 0) ? v : 0;
    });
    return output_strides;
}

ov::Coordinate ov::op::v0::Constant::get_coordinate_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    std::vector<int64_t> out_coordinate = cast_vector<int64_t>();
    Coordinate output_coordinate(shape_size(m_shape));
    std::transform(out_coordinate.begin(), out_coordinate.end(), output_coordinate.begin(), [&](const int64_t& v) {
        return (v > 0) ? v : 0;
    });
    return output_coordinate;
}

ov::CoordinateDiff ov::op::v0::Constant::get_coordinate_diff_val() const {
    OPENVINO_ASSERT(m_element_type == element::i64);
    std::vector<int64_t> out_coordinate_diff = cast_vector<int64_t>();
    CoordinateDiff output_coordinate_diff(shape_size(m_shape));
    std::transform(out_coordinate_diff.begin(),
                   out_coordinate_diff.end(),
                   output_coordinate_diff.begin(),
                   [&](const int64_t& v) {
                       return (v > 0) ? v : 0;
                   });
    return output_coordinate_diff;
}

ov::AxisVector ov::op::v0::Constant::get_axis_vector_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    std::vector<int64_t> out_axis_vector = cast_vector<int64_t>();
    AxisVector output_axis_vector(shape_size(m_shape));
    std::transform(out_axis_vector.begin(), out_axis_vector.end(), output_axis_vector.begin(), [&](const int64_t& v) {
        return (v > 0) ? v : 0;
    });
    return output_axis_vector;
}

ov::AxisSet ov::op::v0::Constant::get_axis_set_val() const {
    OPENVINO_ASSERT(m_element_type.is_integral_number());
    std::vector<int64_t> out_axis_set = cast_vector<int64_t>();
    AxisSet output_axis_set;
    for (auto& axis : out_axis_set) {
        output_axis_set.insert(axis > 0 ? axis : 0);
    }
    return output_axis_set;
}

std::shared_ptr<ov::Node> ov::op::v0::Constant::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Constant_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Constant>(*this);
}

template <typename T>
static bool test_bitwise_identical(const T* data, const size_t size) {
    bool data_is_constant = true;
    if (size > 0) {
        OPENVINO_ASSERT(data != nullptr);
        const T compare = data[0];
        for (size_t i = 1; i < size; i++) {
            if (data[i] != compare) {
                data_is_constant = false;
                break;
            }
        }
    }
    return data_is_constant;
}

bool ov::op::v0::Constant::are_all_data_elements_bitwise_identical() const {
    bool rc = false;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (m_element_type) {
    case element::Type_t::boolean:
    case element::Type_t::i8:
    case element::Type_t::u8: {
        rc = test_bitwise_identical<uint8_t>(get_data_ptr<uint8_t>(), shape_size(m_shape));
        break;
    }
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::i16:
    case element::Type_t::u16: {
        rc = test_bitwise_identical<uint16_t>(get_data_ptr<uint16_t>(), shape_size(m_shape));
        break;
    }
    case element::Type_t::f32:
    case element::Type_t::i32:
    case element::Type_t::u32: {
        rc = test_bitwise_identical<uint32_t>(get_data_ptr<uint32_t>(), shape_size(m_shape));
        break;
    }
    case element::Type_t::f64:
    case element::Type_t::i64:
    case element::Type_t::u64: {
        rc = test_bitwise_identical<uint64_t>(get_data_ptr<uint64_t>(), shape_size(m_shape));
        break;
    }
    case element::Type_t::i4:
    case element::Type_t::u1:
    case element::Type_t::u4:
    case element::Type_t::nf4:
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
        break;
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif
    return rc;
}

void ov::op::v0::Constant::update_identical_flags(bool is_checked, bool identical_value) const {
    m_all_elements_bitwise_identical_checked = is_checked;
    m_all_elements_bitwise_identical = identical_value;
}

bool ov::op::v0::Constant::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Constant_visit_attributes);
    ov::Shape prev_shape = m_shape;
    element::Type prev_type = m_element_type;
    visitor.on_attribute("element_type", m_element_type);
    visitor.on_attribute("shape", m_shape);

    bool need_to_reallocate = (m_shape != prev_shape || prev_type != m_element_type);
    if (m_alloc_buffer_on_visit_attributes && need_to_reallocate) {
        // Filling in a fresh constant
        allocate_buffer(false);
    }
    visitor.on_attribute("value", m_data);
    update_identical_flags(false, false);
    return true;
}

bool ov::op::v0::Constant::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Constant_evaluate);
    if (outputs.empty())
        outputs.emplace_back(ov::Tensor(m_element_type, m_shape));
    else
        outputs[0].set_shape(m_shape);
    std::memcpy(outputs[0].data(), get_data_ptr(), outputs[0].get_byte_size());
    return true;
}

bool ov::op::v0::Constant::has_evaluate() const {
    OV_OP_SCOPE(v0_Constant_has_evaluate);
    return true;
}

bool ov::op::v0::Constant::evaluate_lower(TensorVector& outputs) const {
    return evaluate(outputs, {});
}
bool ov::op::v0::Constant::evaluate_upper(TensorVector& outputs) const {
    return evaluate(outputs, {});
}

uint8_t ov::op::v0::Constant::quantize_nf4(float x) {
    return ov::ConvertNF4::quantize(x);
}
