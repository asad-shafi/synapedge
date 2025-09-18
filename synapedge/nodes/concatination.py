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

def _write_concat_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                           attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for the ONNX Concat operator using a single loop group
    and an if/else chain in the innermost loop to pick the source input.
    """
    axis = attrs.get('axis', 0)

    if len(inputs) < 2:
        raise ValueError("Concat operator requires at least two input tensors.")

    first_shape = tensor_shape[inputs[0]]
    rank = len(first_shape)

    # normalize negative axis
    if axis < 0:
        axis += rank

    # validate shapes (all dims except axis must match)
    for inp in inputs[1:]:
        shape = tensor_shape[inp]
        if len(shape) != rank:
            raise ValueError(f"All inputs must have same number of dimensions. {inp} has {len(shape)} != {rank}.")
        for d in range(rank):
            if d == axis:
                continue
            if shape[d] != first_shape[d]:
                raise ValueError(
                    f"Mismatch in dimension {d} for input '{inp}'. Expected {first_shape[d]}, got {shape[d]}."
                )

    # output shape (assume single output)
    out_name = outputs[0]
    out_shape = tensor_shape[out_name]

    # compute sizes along concat axis and offsets
    axis_sizes = [tensor_shape[inp][axis] for inp in inputs]
    offsets = []
    cum = 0
    for s in axis_sizes:
        offsets.append(cum)
        cum += s

    # Write signature and comment
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"Concat along axis={axis} (single loop with innermost if-chain)", indent=4)

    # Optionally write the axis sizes / offsets as const arrays (not necessary, but useful for debugging)
    n_inputs = len(inputs)
    sizes_list = ", ".join(str(s) for s in axis_sizes)
    offs_list = ", ".join(str(o) for o in offsets)
    buffer.write(f"    const int input_axis_sizes[{n_inputs}] = {{ {sizes_list} }};\n")
    buffer.write(f"    const int input_axis_offsets[{n_inputs}] = {{ {offs_list} }};\n\n")

    # Single nested loops over the output tensor
    loop_indices = []
    indent = "    "
    for d in range(rank):
        idx = f"i{d}"
        loop_indices.append(idx)
        dim_size = out_shape[d]
        buffer.write(f"{indent}for (int {idx} = 0; {idx} < {dim_size}; {idx}++) {{\n")
        indent += "    "

    # Build the output index string like [i0][i1][i2]...
    output_indices = "".join(f"[{idx}]" for idx in loop_indices)

    # Innermost: choose which input to read from based on i{axis}
    idx_axis = f"i{axis}"
    # Start if/else if chain
    for k, inp in enumerate(inputs):
        off = offsets[k]
        sizek = axis_sizes[k]
        cond = f"{idx_axis} >= {off} && {idx_axis} < {off + sizek}"
        if k == 0:
            buffer.write(f"{indent}if ({cond}) {{\n")
        else:
            buffer.write(f"{indent}else if ({cond}) {{\n")

        # Build input indices: same as output, except for axis we use [i{axis} - offset]
        input_indices_parts = []
        for d, idx in enumerate(loop_indices):
            if d == axis:
                # use parenthesis to be safe: (iAxis - offset)
                input_indices_parts.append(f"[{idx} - {off}]")
            else:
                input_indices_parts.append(f"[{idx}]")
        input_indices = "".join(input_indices_parts)

        # assignment
        buffer.write(f"{indent}    {out_name}{output_indices} = {inp}{input_indices};\n")
        buffer.write(f"{indent}}}\n")

    # Close all loops
    for _ in range(rank):
        indent = indent[:-4]
        buffer.write(f"{indent}}}\n")

    buffer.write("}\n")


def _write_concat_function_loopwise(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
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

