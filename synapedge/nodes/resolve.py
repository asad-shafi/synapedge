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

import nodes as nodes
from typing import IO, List, Dict, Any
from io import StringIO

import nodes.activation
import nodes.concatination
import nodes.convolution
import nodes.dropout
import nodes.elementwise
import nodes.gemm
import nodes.matmul
import nodes.pooling
import nodes.pow
import nodes.reshape
import nodes.resize
import nodes.softmax
import nodes.split
import nodes.transpose

def _dispatch_operator_handler(functions: Dict[str, Any], buffer: StringIO, op_type: str, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    
    """Dispatch to the appropriate operator handler based on the ONNX operator type."""
    op_type_lower = op_type.lower()  # Convert to lowercase for case-insensitive comparison
    if op_type_lower in ["conv", "convtranspose"]:
        nodes.convolution._write_convolution_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower in ["relu", "sigmoid", "tanh"]:
        nodes.activation._write_activation_function(functions,buffer,op_type, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "gemm":
        nodes.gemm._write_gemm_function(functions, buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "concat":
        nodes.concatination._write_concat_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "reshape":
        nodes.reshape._write_reshape_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower in ["add", "sub", "mul", "div"]:
        nodes.elementwise._write_elementwise_function(buffer,op_type, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "split":
        nodes.split._write_split_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "slice":
        nodes._write_slice_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "transpose":
        nodes.transpose._write_transpose_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "softmax":
        nodes.softmax._write_softmax_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "lstm":
        nodes._write_lstm_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "averagepool":
        nodes.pooling._write_pooling_function(buffer, func_name, inputs, outputs, attrs, "average",tensor_shape)
    elif op_type_lower == "maxpool":
        nodes.pooling._write_pooling_function(buffer, func_name, inputs, outputs, attrs, "max",tensor_shape)
    elif op_type_lower == "minpool":
        nodes.pooling._write_pooling_function(buffer, func_name, inputs, outputs, attrs, "min",tensor_shape)
    elif op_type_lower == "globalaveragepool":
        nodes.pooling._write_global_pooling_function(buffer, func_name, inputs, outputs, attrs, "average",tensor_shape)
    elif op_type_lower == "flatten":
        nodes._write_flatten_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "pad":
        nodes._write_pad_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "clip":
        nodes._write_clip_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "gather":
        nodes._write_gather_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower in ["upsample", "resize"]:  # Combine upsample and resize
        nodes.resize._write_resize_function(buffer, func_name, inputs, outputs, attrs, op_type,tensor_shape)  # Pass op_type for specific handling
    elif op_type_lower == "constant":
        nodes._write_constant_function(buffer, func_name, inputs, outputs, attrs,tensor_shape,tensor_shape)
    elif op_type_lower == "constantofshape":
        nodes._write_constant_of_shape_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "expand":
        nodes._write_expand_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "squeeze":
        nodes._write_squeeze_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "unsqueeze":
        nodes._write_unsqueeze_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "shape":
        nodes._write_shape_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "range":
        nodes._write_range_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "scatternd":
        nodes._write_scatternd_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "instancenorm":
        nodes._write_instancenorm_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "lrn":
        nodes._write_lrn_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "dropout":
        nodes.dropout._write_dropout_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "matmul":
        nodes.matmul._write_matmul_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "matmulinteger":
        nodes._write_matmul_integer_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "pow":
        nodes.pow._write_pow_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "dynamicquantizelinear":
        nodes._write_dynamic_quantize_linear_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower == "spatialfilter":
        nodes._write_spatial_filter_function(buffer, func_name, inputs, outputs, attrs,tensor_shape)
    elif op_type_lower in ["cast", "convinteger", "pooling"]:  # Placeholder for combined ops
        nodes._write_generic_stub(buffer, func_name, inputs, outputs, attrs, op_type,tensor_shape)  # Pass op_type for info
    else:
        nodes._write_generic_stub(buffer, func_name, inputs, outputs, attrs, op_type,tensor_shape)

