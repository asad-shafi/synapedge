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

def _write_split_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for the ONNX split operator.
    Raises:
        ValueError: If there are less than two outputs, inputs are insufficient, or split tensor is invalid.
    """
    if len(outputs) < 2:
        raise ValueError(f"Split must have at least 2 outputs {func_name}")
    if len(inputs) < 2:
        raise ValueError(f"Split requires two inputs: data and split {func_name}")

    axis = attrs.get('axis', 0)
    data_input = inputs[0]
    split_input = inputs[1]
    input_shape = tensor_shape.get(data_input, [])
    split_shape = tensor_shape.get(split_input, [])
    rank = len(input_shape)

    # Validate rank
    if rank == 0:
        raise ValueError(f"Input shape for {data_input} is empty for {func_name}")

    # Validate split tensor shape: must be 1D with length equal to number of outputs
    if len(split_shape) != 1 or split_shape[0] != len(outputs):
        raise ValueError("Split input must be 1D with length equal to number of outputs")

    # Normalize negative axis
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Axis {attrs.get('axis')} out of range for input rank {rank} in {func_name}")

    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    buffer.write(f"    // Split along axis={attrs.get('axis', 0)} (normalized to {axis})\n")
    buffer.write(f"    const int64_t* split = (const int64_t*){split_input};\n\n")

    for k in range(len(outputs)):
        output_name = outputs[k]
        buffer.write(f"    // Processing output {k}: {output_name}\n")

        # Calculate start index for this split
        if k == 0:
            buffer.write(f"    const int64_t start_{k} = 0;\n")
        else:
            sum_terms = " + ".join([f"split[{i}]" for i in range(k)])
            buffer.write(f"    const int64_t start_{k} = {sum_terms};\n")
        buffer.write(f"    const int64_t split_size_{k} = split[{k}];\n\n")

        indent = "    "
        loop_vars = []
        # Generate nested loops for each dimension
        for d in range(rank):
            if d == axis:
                loop_bound = f"split_size_{k}"
            else:
                loop_bound = str(input_shape[d])
            buffer.write(f"{indent}for (int i{d} = 0; i{d} < {loop_bound}; i{d}++) {{\n")
            loop_vars.append(f"i{d}")
            indent += "    "

        # Generate assignment line with multidimensional indices
        input_indices = []
        for d in range(rank):
            if d == axis:
                # Use start offset + local index for the split axis
                input_idx = f"start_{k} + i{d}"
            else:
                input_idx = f"i{d}"
            input_indices.append(input_idx)

        input_indices_str = "][".join(input_indices)
        output_indices_str = "][".join(loop_vars)
        buffer.write(f"{indent}{output_name}[{output_indices_str}] = {data_input}[{input_indices_str}];\n")

        # Close loops in reverse order
        for _ in range(rank):
            indent = indent[:-4]  # remove 4 spaces per level
            buffer.write(f"{indent}}}\n")
        buffer.write("\n")

    buffer.write("}\n")
