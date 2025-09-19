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

def _write_equal_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                         attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for the Equal operator with broadcasting support."""
    # Write the function signature and a comment
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "equal", indent=4)

    # Retrieve tensor shapes and compute ranks
    out_shape = tensor_shape[outputs[0]]
    out_rank = len(out_shape)
    in1_shape = tensor_shape[inputs[0]]
    in2_shape = tensor_shape[inputs[1]]
    in1_rank = len(in1_shape)
    in2_rank = len(in2_shape)

    indent = "    "
    # Generate nested loops over the output dimensions
    for d, dim in enumerate(out_shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # Helper to build multi-dimensional index strings for broadcasting
    def build_index(input_rank: int) -> str:
        if input_rank == out_rank:
            return ''.join([f"[i{j}]" for j in range(out_rank)])
        elif input_rank == 1:
            # Scalar broadcast
            return "[0]"
        else:
            # Align trailing dimensions
            offset = out_rank - input_rank
            return ''.join([f"[i{j}]" for j in range(offset, out_rank)])

    # Build index strings
    out_index = ''.join([f"[i{j}]" for j in range(out_rank)])
    in1_index = build_index(in1_rank)
    in2_index = build_index(in2_rank)

    # Write the element-wise comparison
    buffer.write(indent + f"{outputs[0]}{out_index} = ({inputs[0]}{in1_index} == {inputs[1]}{in2_index});\n")

    # Close loops
    for _ in range(out_rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")

    # Close function block
    buffer.write("}\n")
