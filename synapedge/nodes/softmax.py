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

def _write_softmax_function(buffer: StringIO, func_name: str, inputs: List[str],
                            outputs: List[str], attrs: Dict[str, Any],
                            tensor_shape: Dict[str, Any]) -> None:
    """
    Generates complete C code for the ONNX Softmax operator.
    """
    # Get softmax axis and adjust for negative values.
    axis = attrs.get('axis', -1)
    input_var = inputs[0]
    output_var = outputs[0]
    input_shape = tensor_shape[input_var]
    rank = len(input_shape)
    if axis < 0:
        axis += rank

    # Write the function signature and an introductory comment.
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"Softmax operator implementation along axis {axis}", indent=4)
    
    # Write dimension constants (one per dimension) for clarity.
    for idx, dim in enumerate(input_shape):
        buffer.write(f"    const int D{idx} = {dim};\n")
    buffer.write("\n");

    # Split dimensions into three groups: outer, softmax, and inner.
    outer_dims = input_shape[:axis]
    softmax_dim = input_shape[axis]
    inner_dims = input_shape[axis+1:]
    
    # Begin nested loops over outer dimensions.
    indent = "    "
    outer_vars = []
    for i, dim in enumerate(outer_dims):
        var = f"i{i}"
        outer_vars.append(var)
        buffer.write(f"{indent}for (int {var} = 0; {var} < D{i}; {var}++) {{\n")
        indent += "    "
    
    # Then, loop over the inner dimensions.
    inner_vars = []
    for j, dim in enumerate(inner_dims):
        var = f"j{j}"
        inner_vars.append(var)
        # The dimension constant for this inner loop is at index (axis+1+j)
        buffer.write(f"{indent}for (int {var} = 0; {var} < D{axis+1+j}; {var}++) {{\n")
        indent += "    "
    
    # Prepare the common indexing string for accessing the element.
    # The full index expression is constructed as:
    #   [outer indices] [a] [inner indices]
    def build_index_expr(loop_var_for_softmax: str) -> str:
        expr = ""
        for var in outer_vars:
            expr += f"[{var}]"
        expr += f"[{loop_var_for_softmax}]"
        for var in inner_vars:
            expr += f"[{var}]"
        return expr

    # Compute the maximum value along the softmax axis.
    buffer.write(f"{indent}// Find maximum value for numerical stability\n")
    buffer.write(f"{indent}float max_val = -INFINITY;\n")
    buffer.write(f"{indent}for (int a = 0; a < D{axis}; a++) {{\n")
    inner_indent = indent + "    "
    index_expr = build_index_expr("a")
    buffer.write(f"{inner_indent}if ({input_var}{index_expr} > max_val) {{ max_val = {input_var}{index_expr}; }}\n")
    buffer.write(f"{indent}}}\n\n")
    
    # Compute the sum of exponentials.
    buffer.write(f"{indent}// Compute sum of exponentials along the softmax axis\n")
    buffer.write(f"{indent}float sum = 0.0f;\n")
    buffer.write(f"{indent}for (int a = 0; a < D{axis}; a++) {{\n")
    buffer.write(f"{inner_indent}float exp_val = expf({input_var}{build_index_expr('a')} - max_val);\n")
    buffer.write(f"{inner_indent}sum += exp_val;\n")
    buffer.write(f"{indent}}}\n\n")
    
    # Write the normalization loop that computes the final softmax output.
    buffer.write(f"{indent}// Compute softmax by normalizing the exponentials\n")
    buffer.write(f"{indent}for (int a = 0; a < D{axis}; a++) {{\n")
    buffer.write(f"{inner_indent}{output_var}{build_index_expr('a')} = expf({input_var}{build_index_expr('a')} - max_val) / sum;\n")
    buffer.write(f"{indent}}}\n")
    
    # Close inner dimension loops.
    for _ in inner_dims:
        indent = indent[:-4]
        buffer.write(f"{indent}}}\n")
    # Close outer dimension loops.
    for _ in outer_dims:
        indent = indent[:-4]
        buffer.write(f"{indent}}}\n")
    
    buffer.write("}\n")
