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

def _write_transpose_function(buffer: StringIO, func_name: str, inputs: List[str],outputs: List[str], attrs: Dict[str, Any],tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for the ONNX Transpose operator.

    Raises:
      ValueError: Transpose operator must have exactly one input.
    """
    # Verify there is exactly one input.
    if len(inputs) != 1:
        raise ValueError("Transpose operator must have exactly one input.")
    input_name = inputs[0]

    # Get the input tensor shape and rank.
    input_dims = tensor_shape.get(input_name)
    if input_dims is None:
        raise ValueError("Input tensor shape not provided for transpose operator.")
    rank = len(input_dims)

    # Get the permutation attribute; if not provided, use reverse order.
    perm = attrs.get("perm", None)
    if perm is None:
        perm = list(reversed(range(rank)))
    else:
        if len(perm) != rank:
            raise ValueError("Permutation attribute length must equal input rank.")
        if sorted(perm) != list(range(rank)):
            raise ValueError("Invalid permutation attribute: must be a permutation of [0, ..., rank-1].")

    # Compute the output shape: output_dim[i] = input_dim[perm[i]]
    out_dims = [input_dims[p] for p in perm]
    tensor_shape[outputs[0]] = out_dims  # Update the output shape in tensor_shape

    # Write the function signature and a comment.
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"Transpose with perm = {perm}", indent=4)

    indent = "    "

    # Generate nested loops for each output dimension.
    # The output dimension i corresponds to input dimension perm[i]
    for i in range(rank):
        loop_var = f"i{i}"
        bound = input_dims[perm[i]]
        buffer.write(indent * (i + 1) + f"for (int {loop_var} = 0; {loop_var} < {bound}; {loop_var}++) {{\n")

    # In the innermost loop, generate the assignment.
    # Output is indexed as output[i0][i1]...[i{rank-1}]
    out_index = "".join(f"[i{j}]" for j in range(rank))
    # For the input, the rule is: output axis i comes from input axis perm[i].
    # Thus, for each input axis j, find k such that perm[k] == j,
    # then the corresponding index is i{k}.
    input_indices = [None] * rank
    for k in range(rank):
        j = perm[k]
        input_indices[j] = f"i{k}"
    in_index = "".join(f"[{idx}]" for idx in input_indices)

    buffer.write(indent * (rank + 1) + f"{outputs[0]}{out_index} = {inputs[0]}{in_index};\n")

    # Close all the nested loops.
    for i in range(rank, 0, -1):
        buffer.write(indent * i + "}\n")

    # Close the function.
    buffer.write("}\n")
