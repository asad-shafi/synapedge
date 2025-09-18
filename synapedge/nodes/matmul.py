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
''' 
this implementaion is working and have been tested on yolov5 it only support [AxB]@[CxD] 

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
        print(A_shape)
        print(B_shape)
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

'''

def _write_matmul_function(buffer: StringIO,func_name: str,inputs: List[str],outputs: List[str],attrs: Dict[str, Any],
                           tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for MatMul operator using 2D or 3D (batched) array indexing."""
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "Matrix multiplication", indent=4)

    A_name, B_name = inputs
    Y_name = outputs[0]

    A_shape = tensor_shape.get(A_name, [])
    B_shape = tensor_shape.get(B_name, [])

    # 2‑D case: exactly as before
    if len(A_shape) == 2 and len(B_shape) == 2:
        M, K = A_shape
        K_B, N = B_shape
        if K != K_B:
            raise ValueError(f"Inner dimensions do not match: {K} vs {K_B}")

        buffer.write(f"    for (int i = 0; i < {M}; i++) {{\n")
        buffer.write(f"        for (int j = 0; j < {N}; j++) {{\n")
        buffer.write(f"            {Y_name}[i][j] = 0.0f;\n")
        buffer.write(f"            for (int k = 0; k < {K}; k++) {{\n")
        buffer.write(f"                {Y_name}[i][j] += {A_name}[i][k] * {B_name}[k][j];\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
        buffer.write("}\n")
        return
    # batched A (3D) × plain B (2D) → batched Y (3D)
    if len(A_shape) == 3 and len(B_shape) == 2:
        B_dim, M, K = A_shape          # A_shape = [B, M, K]
        K_B, N = B_shape               # B_shape = [K, N]
        if K != K_B:
            raise ValueError(f"Inner dims must match: {K} vs {K_B}")

        # loops: batch, rows of A, cols of B
        buffer.write(f"    for (int b = 0; b < {B_dim}; b++) {{\n")
        buffer.write(f"        for (int i = 0; i < {M}; i++) {{\n")
        buffer.write(f"            for (int j = 0; j < {N}; j++) {{\n")
        #buffer.write(f"                {Y_name}[b][i][j] = 0.0f;\n")
        buffer.write(f"                float acc = 0.0f;\n")
        buffer.write(f"                for (int k = 0; k < {K}; k++) {{\n")
        buffer.write(
        #    f"                    {Y_name}[b][i][j] += {A_name}[b][i][k] * {B_name}[k][j];\n"
             f"                    acc += {A_name}[b][i][k] * {B_name}[k][j];\n"
        )
        buffer.write("                }\n")
        buffer.write(f"            {Y_name}[b][i][j] = acc;\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
        buffer.write("}\n")
        return

    # 3‑D case: batched matmul [B x M x K] @ [B x K x N] → [B x M x N]
    if len(A_shape) == 3 and len(B_shape) == 3:
        B_A, M, K = A_shape
        B_B, K_B, N = B_shape
        if K != K_B:
            raise ValueError(f"Inner dimensions do not match: {K} vs {K_B}")
        # broadcast batch dimension
        if not ((B_A == B_B) or (B_A == 1) or (B_B == 1)):
            raise ValueError(f"Batch dims not broadcastable: {B_A} vs {B_B}")
        B = max(B_A, B_B)

        buffer.write(f"    for (int b = 0; b < {B}; b++) {{\n")
        buffer.write(f"        for (int i = 0; i < {M}; i++) {{\n")
        buffer.write(f"            for (int j = 0; j < {N}; j++) {{\n")
        buffer.write(f"                {Y_name}[b][i][j] = 0.0f;\n")
        buffer.write(f"                for (int k = 0; k < {K}; k++) {{\n")
        # if one input has batch dim 1, always index [0] on that input
        a_idx = f"{A_name}[b]" if B_A != 1 else f"{A_name}[0]"
        b_idx = f"{B_name}[b]" if B_B != 1 else f"{B_name}[0]"
        buffer.write(
            f"                    {Y_name}[b][i][j] += {a_idx}[i][k] * {b_idx}[k][j];\n"
        )
        buffer.write("                }\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
        buffer.write("}\n")
        return
    # 4D batched MatMul: [B1 x B2 x M x K] @ [B1 x B2 x K x N] -> [B1 x B2 x M x N]
    if len(A_shape) == 4 and len(B_shape) == 4:
        B1_A, B2_A, M, K = A_shape
        B1_B, B2_B, K_B, N = B_shape

        if K != K_B:
            raise ValueError(f"Inner dimensions do not match: {K} vs {K_B}")
        if B1_A != B1_B or B2_A != B2_B:
            raise ValueError(f"Batch dimensions do not match: {B1_A, B2_A} vs {B1_B, B2_B}")

        buffer.write(f"    for (int b1 = 0; b1 < {B1_A}; b1++) {{\n")
        buffer.write(f"        for (int b2 = 0; b2 < {B2_A}; b2++) {{\n")
        buffer.write(f"            for (int i = 0; i < {M}; i++) {{\n")
        buffer.write(f"                for (int j = 0; j < {N}; j++) {{\n")
        buffer.write(f"                    float acc = 0.0f;\n")
        #buffer.write(f"                    {Y_name}[b1][b2][i][j] = 0.0f;\n")
        buffer.write(f"                    for (int k = 0; k < {K}; k++) {{\n")
        buffer.write(
            #f"                        {Y_name}[b1][b2][i][j] += "
            f"                        acc += "
            f"{A_name}[b1][b2][i][k] * {B_name}[b1][b2][k][j];\n"
        )
        buffer.write("                    }\n")
        buffer.write(f"                 {Y_name}[b1][b2][i][j] = acc; \n")
        buffer.write("                }\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
        buffer.write("}\n")
        return
    # anything else is unsupported
    raise ValueError(f"MatMul currently only supports 2D or 3D (batched) tensors; got shapes {A_shape} and {B_shape}.")
