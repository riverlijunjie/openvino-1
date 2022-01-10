# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from typing import Any, List, Union

from openvino.pyopenvino import Model
from openvino.pyopenvino import Core as CoreBase
from openvino.pyopenvino import CompiledModel as CompiledModelBase
from openvino.pyopenvino import InferRequest as InferRequestBase
from openvino.pyopenvino import AsyncInferQueue as AsyncInferQueueBase
from openvino.pyopenvino import Output
from openvino.pyopenvino import Tensor
from openvino.pyopenvino import OVAny as OVAnyBase

from openvino.runtime.utils.types import get_dtype


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))


def normalize_inputs(py_dict: dict, py_types: dict) -> dict:
    """Normalize a dictionary of inputs to Tensors."""
    for k, val in py_dict.items():
        if not isinstance(k, (str, int)):
            raise TypeError("Incompatible key type for tensor named: {}".format(k))
        try:
            ov_type = py_types[k]
        except KeyError:
            raise KeyError("Port for tensor named {} was not found!".format(k))
        py_dict[k] = (
            val
            if isinstance(val, Tensor)
            else Tensor(np.array(val, get_dtype(ov_type)))
        )
    return py_dict


def get_input_types(obj: Union[InferRequestBase, CompiledModelBase]) -> dict:
    """Map all tensor names of all inputs to the data types of those tensors."""

    def map_tensor_names_to_types(input: Output) -> dict:
        return {n: input.get_element_type() for n in input.get_names()}

    input_types: dict = {}
    for idx, input in enumerate(obj.inputs):
        input_types.update(map_tensor_names_to_types(input))
        input_types[idx] = input.get_element_type()
    return input_types


class InferRequest(InferRequestBase):
    """InferRequest wrapper."""

    def infer(self, inputs: dict = None) -> dict:
        """Infer wrapper for InferRequest."""
        inputs = (
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )
        return super().infer(inputs)

    def start_async(self, inputs: dict = None, userdata: Any = None) -> None:
        """Asynchronous infer wrapper for InferRequest."""
        inputs = (
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )
        super().start_async(inputs, userdata)


class CompiledModel(CompiledModelBase):
    """CompiledModel wrapper."""

    def create_infer_request(self) -> InferRequest:
        """Create new InferRequest object."""
        return InferRequest(super().create_infer_request())

    def infer_new_request(self, inputs: dict = None) -> dict:
        """Infer wrapper for CompiledModel."""
        inputs = (
            {} if inputs is None else normalize_inputs(inputs, get_input_types(self))
        )
        return super().infer_new_request(inputs)


class AsyncInferQueue(AsyncInferQueueBase):
    """AsyncInferQueue wrapper."""

    def __getitem__(self, i: int) -> InferRequest:
        """Return i-th InferRequest from AsyncInferQueue."""
        return InferRequest(super().__getitem__(i))

    def start_async(self, inputs: dict = None, userdata: Any = None) -> None:
        """Asynchronous infer wrapper for AsyncInferQueue."""
        inputs = (
            {}
            if inputs is None
            else normalize_inputs(
                inputs, get_input_types(self[self.get_idle_request_id()])
            )
        )
        super().start_async(inputs, userdata)


class Core(CoreBase):
    """Core wrapper."""

    def compile_model(
        self, model: Union[Model, str], device_name: str, config: dict = None
    ) -> CompiledModel:
        """Compile a model from given Model."""
        return CompiledModel(
            super().compile_model(model, device_name, {} if config is None else config)
        )

    def compile_model(
        self, model: Union[Model, str], config: dict = None
    ) -> CompiledModel:
        """Compile a model from given Model with AUTO plugin as device device."""
        return CompiledModel(
            super().compile_model(model, {} if config is None else config)
        )

    def import_model(
        self, model_file: str, device_name: str, config: dict = None
    ) -> CompiledModel:
        """Compile a model from given model file path."""
        return CompiledModel(
            super().import_model(
                model_file, device_name, {} if config is None else config
            )
        )


class ExtendedNetwork(CompiledModel):
    """CompiledModel that additionally holds Core object."""

    def __init__(self, core: Core, net: CompiledModel):
        super().__init__(net)
        self.core = core  # needs to store Core object for CPU plugin


def compile_model(model_path: str) -> CompiledModel:
    """Compact method to compile model with AUTO plugin."""
    core = Core()
    return ExtendedNetwork(core, core.compile_model(model_path, "AUTO"))


class OVAny(OVAnyBase):
    """OVAny wrapper.

    Wrapper provides some useful overloads for simple built-in Python types.

    Access to the OVAny value is direct if it is a built-in Python data type.
    Example:
    @code{.py}
        any = OVAny([1, 2])
        print(any[0])

        Output: 2
    @endcode

    Otherwise if OVAny value is a custom data type (for example user class),
    access to the value is possible by 'get()' method or property 'value'.
    Example:
    @code{.py}
        class Test:
            def __init__(self):
                self.data = "test"

        any = OVAny(Test())
        print(any.value.data)
    @endcode
    """

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self.value[key]

    def __get__(self) -> Any:
        return self.value

    def __setitem__(self, key: Union[str, int], val: Any) -> None:
        self.value[key] = val

    def __set__(self, val: Any) -> None:
        self.value = val

    def __len__(self) -> int:
        return len(self.value)
