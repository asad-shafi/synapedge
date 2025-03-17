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

from typing import IO, List, Dict, Any
from io import StringIO
import nodes.helperfunc as helperfunc

def _write_matmul_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for MatMul operator using 2D array indexing."""
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "Matrix multiplication", indent=4)

    input0 = inputs[0]
    input1 = inputs[1]
    output = outputs[0]

    # Retrieve shapes from tensor_shape dictionary
    A_shape = tensor_shape.get(input0, [])
    B_shape = tensor_shape.get(input1, [])

    # Ensure both inputs are 2D tensors
    if len(A_shape) != 2 or len(B_shape) != 2:
        raise ValueError("MatMul currently only supports 2D tensors.")

    M, K = A_shape
    K_B, N = B_shape

    # Check if the inner dimensions match
    if K != K_B:
        raise ValueError(f"Inner dimensions do not match for MatMul: {K} vs {K_B}")

    # Generate matrix multiplication loops with 2D indexing
    buffer.write(f"    for (int i = 0; i < {M}; i++) {{\n")
    buffer.write(f"        for (int j = 0; j < {N}; j++) {{\n")
    buffer.write(f"            {output}[i][j] = 0.0f;\n")
    buffer.write(f"            for (int k = 0; k < {K}; k++) {{\n")
    buffer.write(f"                {output}[i][j] += {input0}[i][k] * {input1}[k][j];\n")
    buffer.write("            }\n")
    buffer.write("        }\n")
    buffer.write("    }\n")
    buffer.write("}\n")

def compute_broadcast_shape(shape1, shape2):
    """Compute broadcast shape for two shapes following NumPy broadcasting rules."""
    ndim = max(len(shape1), len(shape2))
    # Pad shapes with ones on the left if necessary.
    padded1 = [1] * (ndim - len(shape1)) + shape1
    padded2 = [1] * (ndim - len(shape2)) + shape2
    broadcast_shape = []
    for d1, d2 in zip(padded1, padded2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
        broadcast_shape.append(max(d1, d2))
    return broadcast_shape

