# Copyright (C) 2025 Asad Shafi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For issues and contributions, visit:
# https://github.com/asad-shafi/synapedge
# ============================================================================
import os
import onnx
from onnx import shape_inference, numpy_helper
from collections import defaultdict
import re
import numpy as np
from typing import List, Dict, Any
import logging

# -------------------------------------------------------------------
# Helper: Map ONNX element type (an integer) to a C type.
def get_c_type_from_elem_type(elem_type):
    dtype_map = {
        onnx.TensorProto.FLOAT: "float",
        onnx.TensorProto.UINT8: "uint8_t",
        onnx.TensorProto.INT8: "int8_t",
        onnx.TensorProto.UINT16: "uint16_t",
        onnx.TensorProto.INT16: "int16_t",
        onnx.TensorProto.INT32: "int32_t",
        onnx.TensorProto.INT64: "int64_t",
        onnx.TensorProto.STRING: "char*",  # Strings need special handling.
        onnx.TensorProto.BOOL: "bool",
        onnx.TensorProto.FLOAT16: "half",  # Adjust if you have a custom type.
        onnx.TensorProto.DOUBLE: "double",
        onnx.TensorProto.UINT32: "uint32_t",
        onnx.TensorProto.UINT64: "uint64_t",
        onnx.TensorProto.COMPLEX64: "complex_float",
        onnx.TensorProto.COMPLEX128: "complex_double",
    }
    return dtype_map.get(elem_type, "float")

# -------------------------------------------------------------------
# Helper: Format a single number as a C literal.
def format_number(x):
    if isinstance(x, np.floating):
        return f"{float(x):.7f}f"
    elif isinstance(x, np.integer):
        return str(int(x))
    else:
        return str(x)


import numpy as np

import numpy as np

def convert_to_c_array_dtyp(arr, level, c_type):
    # If it's a numpy array, convert to list
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
        
    indent = "    " * level  # 4 spaces per level
    
    # Base case: if not a list, it's a number so format it
    if not isinstance(arr, list):
        return f"{{{float(arr):.9f}f}}"

    # Check if the current list is "flat": no sublists
    if not any(isinstance(item, list) for item in arr):
        # Flat list: format each element on the same line
        if c_type == "float":
            items = [f"{float(item):.9f}f" for item in arr]# Format as float
        elif c_type == "int":
            items = [f"{int(item)}" for item in arr]# Format as int
        elif c_type == "int64_t":
            items = [f"{int(item)}" for item in arr]# Format as int
        #items = [f"{float(item):.5f}f" for item in arr]
        return "{" + ",".join(items) + "}"
    else:
        # For nested lists, add newlines and indent for each nested level.
        new_line_indent = "    " * (level + 1)
        inner_strings = []
        for item in arr:
            inner_str = convert_to_c_array(item, level + 1)
            inner_strings.append(new_line_indent + inner_str)
        # Compose the string with newlines at the start and before the closing brace
        return "{" + "\n" + ",\n".join(inner_strings) + "\n" + indent + "}"

#------------------------------------------------------
def convert_to_c_array(arr, level=0):
    logger = logging.getLogger(__name__)
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
        
    indent = "    " * level  # 4 spaces per level
    
    # Base case: if not a list, it's a number so format it
    if not isinstance(arr, list):
        logger.debug(f"Converting number: {arr}")
        return f"{{{float(arr):.9f}f}}"

    # Check if the current list is "flat": no sublists
    if not any(isinstance(item, list) for item in arr):
        # Flat list: format each element on the same line
        items = [f"{float(item):.9f}f" for item in arr]
        logger.debug(f"Flat list: {items}")
        return "{" + ",".join(items) + "}"
    else:
        # For nested lists, add newlines and indent for each nested level.
        new_line_indent = "    " * (level + 1)
        inner_strings = []
        for item in arr:
            inner_str = convert_to_c_array(item, level + 1)
            inner_strings.append(new_line_indent + inner_str)
        # Compose the string with newlines at the start and before the closing brace
        logger.debug(f"Nested list: {inner_strings}")
        return "{" + "\n" + ",\n".join(inner_strings) + "\n" + indent + "}"

# -------------------------------------------------------------------
# Helper: Recursively format a numpy array as a nested C initializer list.
def format_array(arr, indent=0):
    indent_str = "    " * indent
    if arr.ndim == 1:
        #return indent_str + ", ".join(format_number(x) for x in arr) 
        return indent_str + "{" + ",".join(format_number(x) for x in arr) + "}"
    else:
        lines = []
        lines.append(indent_str + "{")
        for sub_arr in arr:
            lines.append(format_array(np.array(sub_arr), indent + 1) + ",")
        lines.append(indent_str + "}")
        return "\n".join(lines)
    
def _get_tensor_shape(tensor) -> List[int]:
    """Extract shape dimensions from ONNX tensor."""
    return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

def _write_c_comment(buffer, comment: str, indent: int = 0) -> None:
    """Write a formatted C comment to the file."""
    buffer.write(" " * indent + f"/*{comment}*/\n")

def _search_intermediate_tensors_valve(name , intermediate_tensors: List[Dict[str, Any]],return_value) -> None:
    tensor_to_find = name
    filtered_tensor  = intermediate_tensors.get(tensor_to_find, None)
    if filtered_tensor is not None:
        return_value = filtered_tensor["shape"]
    return return_value

def _extract_intermediate_tensors_from_graph(graph, intermediate_tensors: List) -> None:
      # Extract intermediate_tensors tensor names and shapes
    for value_info in graph.value_info:
      tensor_name = value_info.name
      tensor_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
      intermediate_tensors[tensor_name] = tensor_shape