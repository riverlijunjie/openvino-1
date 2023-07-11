// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request.h"

#include "async_infer_request.h"
#include "compiled_model.h"
#include "debug.h"
#include "dnnl_extension_utils.h"
#include "ie_common.h"
#include "ie_ngraph_utils.hpp"
#include "itt.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_state.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/concat.h"
#include "nodes/memory.hpp"
#include "nodes/split.h"
#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
SyncInferRequest::SyncInferRequest(std::shared_ptr<const CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(compiled_model) {
    m_is_legacy_api = m_compiled_model->GetGraph()._graph.getConfig().isLegacyApi;

    // Precision maybe changed after transformation, need store original input/output port information
    auto orig_model = m_compiled_model->get_orig_model();
    for (const auto& in : orig_model->inputs()) {
        auto port_name = get_port_name(in, m_is_legacy_api);
        m_orig_ports_map[port_name] = in;
    }
    for (const auto& out : orig_model->outputs()) {
        auto port_name = get_port_name(out, m_is_legacy_api);
        m_orig_ports_map[port_name] = out;
    }

    for (const auto& in : get_inputs()) {
        auto port_name = get_port_name(in, m_is_legacy_api);
        m_input_ports_map[port_name] = in;
        if (m_orig_ports_map.find(port_name) == m_orig_ports_map.end()) {
            OPENVINO_THROW("Input port's name has been changed, cannot find ", port_name);
        }
        m_port_precision_changed[port_name] =
            m_input_ports_map[port_name].get_element_type() != m_orig_ports_map[port_name].get_element_type();
        m_port_tensor_need_converted[port_name] = m_port_precision_changed[port_name];
    }
    for (const auto& out : get_outputs()) {
        auto port_name = get_port_name(out, m_is_legacy_api);
        m_output_ports_map[port_name] = out;
        if (m_orig_ports_map.find(port_name) == m_orig_ports_map.end()) {
            OPENVINO_THROW("Output port's name has been changed, cannot find ", port_name);
        }
        m_port_precision_changed[port_name] =
            m_output_ports_map[port_name].get_element_type() != m_orig_ports_map[port_name].get_element_type();
        m_port_tensor_need_converted[port_name] = m_port_precision_changed[port_name];
    }
    create_infer_request();
}

void SyncInferRequest::create_infer_request() {
    auto id = (m_compiled_model->m_numRequests)++;
    m_profiling_task = openvino::itt::handle("INTEL_CPU_INFER_" + m_compiled_model->m_name + "_" + std::to_string(id));

    if (m_compiled_model->m_graphs.size() == 0)
        OPENVINO_THROW("No graph was found");
    graph = &(m_compiled_model->GetGraph()._graph);

    // Alocate memory for each tensor if static shape
    for (const auto& it : m_input_ports_map) {
        init_tensor(it.first);
    }
    for (const auto& it : m_output_ports_map) {
        init_tensor(it.first);
        // allocate aux tensor for output if output precision has been changed
        get_tensor(it.second);
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto memoryNode = dynamic_cast<node::MemoryInput*>(node.get());
            if (!memoryNode) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
            }
            auto state_store = memoryNode->getStore();
            auto state_name = memoryNode->getId();

            // Remove suffix with pair ID. Internal information.
            auto suffix_idx = state_name.find("/id=");
            if (suffix_idx != std::string::npos)
                state_name = state_name.substr(0, suffix_idx);

            m_memory_states.emplace_back(new VariableState(state_name, state_store));
        }
    }
}

SyncInferRequest::~SyncInferRequest() {
    --(m_compiled_model->m_numRequests);
}

void SyncInferRequest::push_states() {
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : m_memory_states) {
                if (state->get_name() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->get_state().data();
                    auto data_size = state->get_state().get_byte_size();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(cur_state_mem_buf, data_ptr, data_size);
                }
            }
        }
    }
}

void SyncInferRequest::pull_states() {
    for (auto& node : graph->GetNodes()) {
        if (node->getType() == Type::MemoryInput) {
            auto cur_node = dynamic_cast<node::MemoryInput*>(node.get());
            if (!cur_node) {
                OPENVINO_THROW("Cannot cast ", node->getName(), " to MemoryInput");
            }
            auto cur_id = cur_node->getId();
            for (const auto& state : m_memory_states) {
                if (state->get_name() == cur_id) {
                    auto cur_state_mem = cur_node->getStore();
                    auto data_ptr = state->get_state().data();
                    auto data_size = state->get_state().get_byte_size();
                    auto cur_state_mem_buf = static_cast<uint8_t*>(cur_state_mem->GetPtr());

                    cpu_memcpy(data_ptr, cur_state_mem_buf, data_size);
                }
            }
        }
    }
}

void SyncInferRequest::redefine_memory_for_input_nodes() {
    const auto cpuInputNodes = graph->GetInputNodesMap();
    for (const auto& port : get_inputs()) {
        std::string name = get_port_name(port, m_is_legacy_api);
        if (name.empty()) {
            OPENVINO_THROW("compiled model doesn't contain this input port.");
        }
        const auto inputNode = cpuInputNodes.find(name);
        if (inputNode == cpuInputNodes.end())
            OPENVINO_THROW("CPU execution graph doesn't contain input node with name: ", name.c_str());
        if (inputNode->second->isDynamicNode()) {
            auto tensor = get_port_tensor(port);
            inputNode->second->redefineOutputMemory({tensor.get_shape()});
        }
    }
}

void SyncInferRequest::update_external_inputs() {
    // Update it due to batched_tensors case will update input tensor
    if (m_batched_tensors.size() == 0)
        return;
    for (auto input : get_inputs()) {
        std::string input_name = get_port_name(input, m_is_legacy_api);
        if (input_name.empty()) {
            OPENVINO_THROW("Input tensor map contains not registered during IPlugin::compile_model tensor with name ",
                           input_name);
        }
        if (external_ptr.find(input_name) != external_ptr.end()) {
            auto tensor = get_port_tensor(input);
            external_ptr[input_name] = tensor.data();
        }
    }
}

void SyncInferRequest::infer() {
    using namespace openvino::itt;
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, m_profiling_task);
    auto graphLock = m_compiled_model->GetGraph();
    graph = &(graphLock._graph);

    throw_if_canceled();
    convert_batched_tensors();
    update_external_inputs();

    if (graph->hasDynamicInput()) {
        redefine_memory_for_input_nodes();
    }

    change_default_ptr();

    throw_if_canceled();

    push_input_data();

    if (m_memory_states.size() != 0) {
        push_states();
    }

    graph->Infer(this);

    if (m_memory_states.size() != 0) {
        pull_states();
    }

    throw_if_canceled();

    graph->PullOutputData(m_outputs, m_port_precision_changed, m_aux_tensors);
}

std::vector<ov::ProfilingInfo> SyncInferRequest::get_profiling_info() const {
    if (!graph || !graph->IsReady())
        OPENVINO_THROW("Graph is not ready!");
    std::vector<ov::ProfilingInfo> perfMap;
    graph->GetPerfData(perfMap);
    return perfMap;
}

static inline void change_edge_ptr(const EdgePtr& edge, void* newPtr) {
    edge->getMemoryPtr()->setDataHandle(newPtr);
}

void SyncInferRequest::change_default_ptr() {
    for (auto& it : external_ptr) {
        const auto& inputNodesMap = graph->GetInputNodesMap();
        auto input = inputNodesMap.find(it.first);
        if (input != inputNodesMap.end()) {
            NodePtr inputNodePtr = input->second;
            if (inputNodePtr->getChildEdgeAt(0)->getMemory().GetData() == it.second)
                continue;
            auto& childEdges = inputNodePtr->getChildEdges();
            // Input cannot be in-place with other primitives
            bool canBeInPlace = true;
            for (auto& childEdge : childEdges) {
                auto ce = childEdge.lock();
                if (!ce)
                    OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");

                auto& child = ce->getChild();

                if (child->isConstant()) {
                    canBeInPlace = false;
                    break;
                }

                if (child->getType() == Type::Concatenation) {
                    auto concat = dynamic_cast<node::Concat*>(child.get());
                    if (concat && concat->isOptimized()) {
                        canBeInPlace = false;
                        break;
                    }
                }

                // Cannot be in-place before split because split is using different ptrs without offsets
                if (child->getType() == Type::Split) {
                    canBeInPlace = false;
                    break;
                }

                if (child->isInPlace()) {
                    canBeInPlace = false;
                    break;
                }

                auto& edges = child->getChildEdges();
                for (auto& edge : edges) {
                    auto e = edge.lock();
                    if (!e)
                        OPENVINO_THROW("Node ", child->getName(), " contains empty child edge");

                    if (e->getMemory().GetData() == ce->getMemory().GetData()) {
                        canBeInPlace = false;
                        break;
                    }
                }

                if (!canBeInPlace)
                    break;
            }
            if (canBeInPlace) {
                for (auto& edge : childEdges) {
                    auto e = edge.lock();
                    if (!e)
                        OPENVINO_THROW("Node ", inputNodePtr->getName(), " contains empty child edge");

                    change_edge_ptr(e, it.second);
                }
            }

            continue;
        }

        const auto& outputNodesMap = graph->GetOutputNodesMap();
        auto output = outputNodesMap.find(it.first);
        if (output != outputNodesMap.end()) {
            auto parentEdge = output->second->getParentEdgeAt(0);
            if (parentEdge->getMemory().GetData() == it.second)
                continue;

            bool canBeInPlace = true;
            void* defaultPtr = parentEdge->getMemory().GetData();
            // Cannot be in-place after concat because concat is using different ptrs without offsets
            auto parent = parentEdge->getParent();
            NodePtr previousParent;
            do {
                previousParent = parent;
                if (parent->getChildEdges().size() != 1 || parent->isConstant() || parent->isInPlace()) {
                    canBeInPlace = false;
                    break;
                }

                auto& parentEdges = parent->getParentEdges();
                for (auto& edge : parentEdges) {
                    auto e = edge.lock();
                    if (!e)
                        OPENVINO_THROW("Node ", parent->getName(), " contains empty parent edge");

                    if (e->getMemory().GetData() == defaultPtr) {
                        parent = e->getParent();
                        break;
                    }
                }
            } while (previousParent != parent);
            if (canBeInPlace)
                change_edge_ptr(parentEdge, it.second);
            continue;
        }
        OPENVINO_THROW("Cannot find input/output blob: ", it.first);
    }
}

std::vector<std::shared_ptr<ov::IVariableState>> SyncInferRequest::query_state() const {
    return m_memory_states;
}

void SyncInferRequest::set_async_request(AsyncInferRequest* asyncRequest) {
    m_asyncRequest = asyncRequest;
}

void SyncInferRequest::throw_if_canceled() const {
    if (m_asyncRequest != nullptr) {
        m_asyncRequest->throw_if_canceled();
    }
}

InferenceEngine::Precision SyncInferRequest::norm_to_input_supported_prec(
    const std::pair<const std::string, ov::Tensor>& input) const {
    auto inPrec = InferenceEngine::details::convertPrecision(input.second.get_element_type());
    if (graph->hasMeanImageFor(input.first) &&
        one_of(inPrec, InferenceEngine::Precision::U8, InferenceEngine::Precision::BOOL)) {
        inPrec = InferenceEngine::Precision::FP32;
    } else {
        inPrec = normalizeToSupportedPrecision(inPrec);
    }

    if (inPrec == InferenceEngine::Precision::UNSPECIFIED) {
        OPENVINO_THROW("Unsupported input precision ", input.second.get_element_type());
    }
    return inPrec;
}

bool SyncInferRequest::check_compiled_model_port(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port, m_is_legacy_api);
    if (name.empty()) {
        OPENVINO_THROW("cpu plugin checking port failed: cannot find this port with empty name.");
    }

    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        auto it = m_input_ports_map.find(name);
        if (it == m_input_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking input port failed: cannot find this port with name ", name);
        }

        if ((it->second.get_element_type() == port.get_element_type()) &&
            (it->second.get_partial_shape() == port.get_partial_shape())) {
            return true;
        }
        return false;
    } else {
        auto it = m_output_ports_map.find(name);
        if (it == m_output_ports_map.end()) {
            OPENVINO_THROW("cpu plugin checking output port failed: cannot find this port with name ", name);
        }
        if ((it->second.get_element_type() == port.get_element_type()) &&
            (it->second.get_partial_shape() == port.get_partial_shape())) {
            return true;
        }
        return false;
    }
}

InferenceEngine::TensorDesc SyncInferRequest::create_tensor_desc(const ov::Tensor& tensor) {
    auto element_type = tensor.get_element_type();
    auto shape = tensor.get_shape();
    std::vector<size_t> blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    std::vector<size_t> dim_offset(shape.size(), 0);
    std::vector<size_t> blk_strides;
    auto byte_strides = element_type.bitwidth() >= 8 ? tensor.get_strides() : Strides{};
    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else {
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " should be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    return ie::TensorDesc{ie::details::convertPrecision(element_type),
                          shape,
                          ie::BlockingDesc{shape, blk_order, 0, dim_offset, blk_strides}};
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::Tensor SyncInferRequest::get_port_tensor(const ov::Output<const ov::Node>& in_port) const {
    check_compiled_model_port(in_port);
    auto port = get_internal_port(in_port);
    auto tensor = ov::ISyncInferRequest::get_tensor(port);
    auto name = get_port_name(in_port, m_is_legacy_api);

    if (m_aux_tensors.find(name) != m_aux_tensors.end()) {
        auto& aux_tensor = m_aux_tensors[name];
        if (aux_tensor.get_shape() != tensor.get_shape()) {
            tensor.set_shape(aux_tensor.get_shape());
        }
    }
    return tensor;
}

ov::Tensor SyncInferRequest::get_tensor(const ov::Output<const ov::Node>& in_port) const {
    auto port_name = get_port_name(in_port, m_is_legacy_api);
    auto port = get_internal_port(in_port);
    auto port_tensor = ov::ISyncInferRequest::get_tensor(port);

    // No precision change
    auto is_precision_changed = m_port_precision_changed[port_name];
    if (!is_precision_changed)
        return port_tensor;

    // If precision has been changed, it need return original precision tensor.
    // Port's data will be stored in m_aux_tensors, and need converted to compiled tensor
    //     input  tensor: will be copied to compiled tensor before do graph inference
    //     output tensor: has been copied from graph's memory to aux tensor when inference done
    if (m_orig_ports_map.find(port_name) == m_orig_ports_map.end()) {
        OPENVINO_THROW("get_tensor: cannot find model port, name: ", port_name);
    }

    // Find aux tensor, will create one if cannot find
    auto port_shape = port.get_partial_shape();
    auto it = m_aux_tensors.find(port_name);
    ov::Shape aux_shape = port_tensor.get_shape();
    if (it == m_aux_tensors.end()) {
        m_aux_tensors[port_name] = ov::Tensor(m_orig_ports_map[port_name].get_element_type(), aux_shape);
    } else if (port_shape.is_dynamic()) {
        if (m_aux_tensors[port_name].get_shape() != aux_shape)
            m_aux_tensors[port_name].set_shape(aux_shape);
    }
    m_port_tensor_need_converted[port_name] = true;
    return m_aux_tensors[port_name];
}

std::vector<ov::Tensor> SyncInferRequest::get_tensors(const ov::Output<const ov::Node>& in_port) const {
    auto port = get_internal_port(in_port);
    return ov::ISyncInferRequest::get_tensors(port);
}

const ov::Output<const ov::Node>& SyncInferRequest::get_internal_port(const ov::Output<const ov::Node>& port) const {
    auto name = get_port_name(port, m_is_legacy_api);
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        return m_input_ports_map[name];
    } else {
        return m_output_ports_map[name];
    }
}

void SyncInferRequest::set_tensor(const ov::Output<const ov::Node>& in_port, const ov::Tensor& in_tensor) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "set_tensor");
    if (!in_tensor)
        OPENVINO_THROW("Failed to set empty tensor for port!");
    auto is_compiled_model_port = check_compiled_model_port(in_port);
    auto port = get_internal_port(in_port);
    auto tensor = in_tensor;

    // WA: legacy api create blob with ANY layout will not set BlockingDesc, which will lead to tensor.get_shape()
    // return empty shape but tensor.get_size() return correct value, and tensor.reshape() cannot update
    // BlockingDesc, so to construct new tensor with original tensor's data, which is only for ov legacy api usage.
    if (in_port.get_partial_shape().is_static() && in_tensor.get_size() > 0 && in_tensor.get_shape().size() == 0 &&
        in_tensor.get_size() == ov::shape_size(in_port.get_shape()) && in_port.get_shape().size() > 0) {
        tensor = ov::Tensor(in_tensor.get_element_type(), in_port.get_shape(), in_tensor.data());
    }
    auto name = get_port_name(in_port, m_is_legacy_api);
    auto is_precision_changed = m_port_precision_changed[name];

    // Precision has been changed
    if (is_precision_changed) {
        if (!is_compiled_model_port) {
            // Original port
            auto _orig_port = m_orig_ports_map[name];
            if (_orig_port.get_element_type() == in_tensor.get_element_type()) {
                // Original port + orig port's tensor
                m_aux_tensors[name] = in_tensor;
                m_port_tensor_need_converted[name] = true;
                tensor = ov::ISyncInferRequest::get_tensor(port);
                tensor.set_shape(in_tensor.get_shape());
            } else if (port.get_element_type() == in_tensor.get_element_type()) {
                // Original port + compiled port's tensor, should not reach here
                tensor = in_tensor;
                m_port_tensor_need_converted[name] = false;
            } else {
                OPENVINO_THROW("ParameterMismatch: failed to set input tensor with precision ",
                               in_tensor.get_element_type(),
                               ", if model input tensor precision is: ",
                               port.get_element_type(),
                               " or ",
                               _orig_port.get_element_type());
            }
        } else {
            // Compiled model port
            if (in_port.get_element_type() != in_tensor.get_element_type()) {
                if (m_orig_ports_map[name].get_element_type() == in_tensor.get_element_type()) {
                    // Original port precision tensor, likely reach here
                    m_aux_tensors[name] = in_tensor;
                    tensor = ov::ISyncInferRequest::get_tensor(port);
                    tensor.set_shape(in_tensor.get_shape());
                    m_port_tensor_need_converted[name] = true;
                } else {
                    OPENVINO_THROW("ParameterMismatch: failed to set input tensor with precision ",
                                   in_tensor.get_element_type(),
                                   ", if model input tensor precision is: ",
                                   in_port.get_element_type());
                }
            } else {
                // compiled model port + compiled model precision tensor
                m_port_tensor_need_converted[name] = false;
            }
        }
    }

    auto tensor_desc = create_tensor_desc(tensor);
    bool is_input = ov::op::util::is_parameter(port.get_node());
    if (is_input) {
        const auto netInPrc = port.get_element_type();
        if (netInPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set input tensor with precision: " << tensor.get_element_type()
                                        << ", if model input tensor precision is: " << netInPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();
        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            OPENVINO_THROW("The tensor size is not equal to model, can't set input tensor with name: ",
                           name,
                           ", because model input (shape=",
                           shape,
                           ") and tensor (shape=",
                           vec2str(tensor.get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ov::shape_size(shape.to_shape()) != tensor.get_size()) {
            OPENVINO_THROW("Can't set input tensor with name: ",
                           name,
                           ", because model input size = ",
                           ov::shape_size(shape.to_shape()),
                           " and tensor size = ",
                           tensor.get_size(),
                           " are different.");
        }

        MemoryDescPtr actualDesc = graph->getInputNodeByName(name)->getBaseMemDescAtOutputPort(0);
        if (!actualDesc->isDefined()) {
            // we must define desc for dynamic case
            // otherwise we got incorrect check on shape compatibility inside isCompatible
            // because lower and upper bound will be compared
            OPENVINO_SUPPRESS_DEPRECATED_START
            actualDesc = actualDesc->cloneWithNewDims(tensor_desc.getLayout() == InferenceEngine::Layout::SCALAR
                                                          ? InferenceEngine::SizeVector{1}
                                                          : tensor_desc.getDims());
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
        if (actualDesc->isCompatible(MemoryDescUtils::convertToCpuBlockedMemoryDesc(tensor_desc)) &&
            graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
            external_ptr[name] = tensor.data();
        } else if (external_ptr.find(name) != external_ptr.end()) {
            external_ptr.erase(name);
        }
    } else {
        const auto netOutPrc = port.get_element_type();
        if (netOutPrc != tensor.get_element_type()) {
            IE_THROW(ParameterMismatch) << "Failed to set output tensor with precision: " << tensor.get_element_type()
                                        << ", if model output tensor precision is: " << netOutPrc;
        }

        const auto shape = port.get_partial_shape();
        const bool isDynamic = shape.is_dynamic();

        if (!shape.compatible(ov::PartialShape(tensor.get_shape()))) {
            OPENVINO_THROW("The tensor size is not equal to model, can't set output tensor with name: ",
                           name,
                           ", because model output (shape=",
                           shape,
                           ") and blob (shape=",
                           vec2str(tensor.get_shape()),
                           ") are incompatible");
        }

        if (!isDynamic && ov::shape_size(shape.to_shape()) != tensor.get_size()) {
            OPENVINO_THROW("Can't set output tensor with name: ",
                           name,
                           ", because model output size = ",
                           ov::shape_size(shape.to_shape()),
                           " and blob size = ",
                           tensor.get_size(),
                           " are different.");
        }

        const auto& desc = graph->getOutputNodeByName(name)->getParentEdgesAtPort(0)[0]->getMemory().getDesc();
        if (!isDynamic && tensor_desc == MemoryDescUtils::convertToTensorDesc(desc)) {
            external_ptr[name] = tensor.data();
        } else if (external_ptr.find(name) != external_ptr.end()) {
            external_ptr.erase(name);
        }
        m_outputs[name] = tensor;
    }
    ov::ISyncInferRequest::set_tensor(port, tensor);
}

void SyncInferRequest::set_tensors_impl(const ov::Output<const ov::Node> port, const std::vector<ov::Tensor>& tensors) {
    for (const auto& input : get_inputs()) {
        if (input == port) {
            m_batched_tensors[input.get_tensor_ptr()] = tensors;
            return;
        }
    }
    OPENVINO_THROW("Cannot find port to set_tensors!");
}

void SyncInferRequest::init_tensor(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_cpu, "init_tensor");

    if (!graph || !graph->IsReady())
        OPENVINO_THROW("Graph is not ready!");

    if (name.empty())
        OPENVINO_THROW("Can't preapre tensor for empty name! ");

    ov::Tensor tensor;
    const auto& inMap = graph->inputNodesMap;
    auto input = inMap.find(name);
    if (input != inMap.end()) {
        auto input_port = m_input_ports_map.find(name);
        if (input_port != m_input_ports_map.end()) {
            auto& port = input_port->second;
            tensor = ov::ISyncInferRequest::get_tensor(port);

            if (!tensor) {
                const auto shape = port.get_partial_shape();
                const bool isDynamic = shape.is_dynamic();
                ov::Shape tensor_shape;
                if (isDynamic) {
                    tensor_shape = ov::Shape(shape.rank().get_length(), 0);
                } else {
                    tensor_shape = shape.to_shape();
                }

                tensor = ov::Tensor(port.get_element_type(), tensor_shape);
                ov::ISyncInferRequest::set_tensor(port, tensor);

                auto desc = create_tensor_desc(tensor);
                if (!isDynamic &&
                    desc == MemoryDescUtils::convertToTensorDesc(
                                graph->getInputNodeByName(name)->getChildEdgesAtPort(0)[0]->getMemory().getDesc()) &&
                    graph->_normalizePreprocMap.find(name) == graph->_normalizePreprocMap.end()) {
                    external_ptr[name] = tensor.data();
                }
            }
        } else {
            OPENVINO_THROW("Tensor with name: ", name, " exists in CPU plugin graph, but absents in network inputs");
        }
    }

    const auto& outMap = graph->outputNodesMap;
    auto output = outMap.find(name);
    if (output != outMap.end()) {
        auto output_port = m_output_ports_map.find(name);
        if (output_port != m_output_ports_map.end()) {
            auto& port = output_port->second;
            tensor = ov::ISyncInferRequest::get_tensor(port);
            const auto shape = port.get_partial_shape();
            const bool isDynamic = shape.is_dynamic();

            if (!tensor) {
                ov::Shape tensor_shape;
                if (isDynamic) {
                    tensor_shape = ov::Shape(shape.rank().get_length(), 0);
                } else {
                    tensor_shape = shape.to_shape();
                }
                tensor = ov::Tensor(port.get_element_type(), tensor_shape);
                ov::ISyncInferRequest::set_tensor(port, tensor);
            } else {
                const auto& blobDims = tensor.get_shape();
                // Static shape case is enough information that shapes are incompatible to throw exception
                // but in dynamic shape case we also need to handle following corner case:
                // on tensor initialization stage we create empty tensor with dimensions equal 0
                // so if we have tensor with all zero dimension we mustn't throw exception
                if (!shape.compatible(ov::PartialShape(blobDims)) &&
                    (!isDynamic || static_cast<int64_t>(blobDims.size()) != shape.rank().get_length() ||
                     std::any_of(blobDims.begin(), blobDims.end(), [](const size_t& dims) {
                         return dims != 0;
                     }))) {
                    IE_THROW(ParameterMismatch)
                        << "Network input and output use the same name: " << name
                        << ", but expect tensors with different shapes. Input shape: " << ov::PartialShape(blobDims)
                        << ", output shape: " << shape;
                }

                const auto netOutPrc = port.get_element_type();
                if (netOutPrc != tensor.get_element_type()) {
                    IE_THROW(ParameterMismatch)
                        << "Network input and output use the same name: " << name
                        << " but expect blobs with different precision: " << tensor.get_element_type()
                        << " for input and " << netOutPrc << " for output.";
                }
            }
            m_outputs[name] = tensor;
            auto desc = create_tensor_desc(tensor);
            if (!isDynamic && !external_ptr.count(name) &&
                desc == MemoryDescUtils::convertToTensorDesc(
                            output->second->getParentEdgesAtPort(0)[0]->getMemory().getDesc())) {
                external_ptr[name] = tensor.data();
            }
        } else {
            OPENVINO_THROW("Tensor with name: ", name, " exists in CPU plugin graph, but absents in network outputs");
        }
    }

    if (!tensor) {
        OPENVINO_THROW("Cannot find tensor with name: ", name);
    }
    return;
}

void SyncInferRequest::push_input_data() {
    for (auto input : get_inputs()) {
        std::string input_name = get_port_name(input, m_is_legacy_api);
        if (input_name.empty()) {
            OPENVINO_THROW("Input tensor map contains not registered during IPlugin::compile_model tensor with name ",
                           input_name);
        }
        auto tensor = get_port_tensor(input);
        if (m_aux_tensors.find(input_name) != m_aux_tensors.end() && m_port_tensor_need_converted[input_name]) {
            auto& aux_tensor = m_aux_tensors[input_name];

            if (aux_tensor.get_shape() != tensor.get_shape()) {
                tensor.set_shape(aux_tensor.get_shape());
            }
            const void* srcData = aux_tensor.data();
            void* dstData = tensor.data();
            if ((dstData == nullptr) || (srcData == nullptr)) {
                OPENVINO_THROW("Get tensor has no allocated memory");
            }
            cpu_convert(srcData,
                        dstData,
                        InferenceEngine::details::convertPrecision(aux_tensor.get_element_type()),
                        InferenceEngine::details::convertPrecision(tensor.get_element_type()),
                        tensor.get_size());
        }
        graph->PushInputData(input_name, tensor);
    }
}
}  // namespace intel_cpu
}  // namespace ov
