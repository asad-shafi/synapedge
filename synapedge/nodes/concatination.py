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

def _write_concat_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for the ONNX Concat operator.
    """
    axis = attrs.get('axis', 0)

    # Validate that there are at least two input tensors.
    if len(inputs) < 2:
        raise ValueError("Concat operator requires at least two input tensors.")

    # Determine the rank from the first input and check dimension consistency.
    first_shape = tensor_shape[inputs[0]]
    rank = len(first_shape)
    for inp in inputs[1:]:
        shape = tensor_shape[inp]
        if len(shape) != rank:
            raise ValueError(f"All inputs must have the same number of dimensions. {inp} has {len(shape)} dimensions, expected {rank}.")
        for d in range(rank):
            if d == axis:
                continue
            if shape[d] != first_shape[d]:
                raise ValueError(
                    f"Mismatch in dimension {d} for input '{inp}'. "
                    f"Expected {first_shape[d]}, got {shape[d]}."
                )

    # Write the function signature and a descriptive comment.
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"Concat along axis={axis}", indent=4)

    # Begin function body: declare the axis offset variable.
    buffer.write("    int axis_offset = 0;\n")

    # For each input tensor, generate nested loops to copy its values.
    for input_name in inputs:
        shape = tensor_shape[input_name]
        buffer.write(f"    // Copy tensor '{input_name}' into the output\n")
        buffer.write("    {\n")
        indent = "        "
        loop_indices = []
        # Generate a loop for each dimension.
        for d in range(rank):
            idx = f"i{d}"
            loop_indices.append(idx)
            dim_size = shape[d]
            buffer.write(f"{indent}for (int {idx} = 0; {idx} < {dim_size}; {idx}++) {{\n")
            indent += "    "
        # Build the multidimensional indexing string.
        # For the input, all indices are used as-is.
        input_indices = "".join([f"[{idx}]" for idx in loop_indices])
        # For the output, add axis_offset to the index corresponding to the concat axis.
        output_indices_parts = []
        for d, idx in enumerate(loop_indices):
            if d == axis:
                output_indices_parts.append(f"[axis_offset + {idx}]")
            else:
                output_indices_parts.append(f"[{idx}]")
        output_indices = "".join(output_indices_parts)

        # Generate the copy statement (assuming a single output tensor, outputs[0]).
        buffer.write(f"{indent}{outputs[0]}{output_indices} = {input_name}{input_indices};\n")

        # Close all nested loops.
        for _ in range(rank):
            indent = indent[:-4]
            buffer.write(f"{indent}}}\n")
        buffer.write("    }\n")

        # Increment the axis_offset by the size of the current input along the concatenation axis.
        buffer.write(f"    axis_offset += {shape[axis]};\n")

    buffer.write("}\n")

