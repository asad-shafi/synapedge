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
import logging

logger = logging.getLogger(__name__)
def _write_elementwise_function(buffer: StringIO, op_type: str, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for element-wise operations with dynamic loop variables and broadcasting support."""
    c_op = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}.get(op_type)
    if not c_op:
        raise ValueError(f"Unsupported operation type: {op_type}")

    input_shapes = [tensor_shape[inp] for inp in inputs]
    output_shape = tensor_shape[outputs[0]]
    output_rank = len(output_shape)

    # Check for broadcasting and generate warning
    broadcast_needed = False
    for shape in input_shapes:
        padded_shape = [1] * (output_rank - len(shape)) + shape
        if padded_shape != output_shape:
            broadcast_needed = True
            break
    if broadcast_needed:
        logger.warning(f"Broadcasting will be applied. {op_type} ({func_name})")
        buffer.write("    // Warning: Broadcasting is applied.\n")

    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)

    # Generate dynamic loop variables (d0, d1, ..., dn)
    loop_vars = [f'd{i}' for i in range(output_rank)]

    # Generate nested loops
    for d in range(output_rank):
        buffer.write(f"    for (int {loop_vars[d]} = 0; {loop_vars[d]} < {output_shape[d]}; {loop_vars[d]}++) {{\n")

    # Generate input expressions with correct broadcasting indices
    input_exprs = []
    for inp in inputs:
        original_shape = tensor_shape[inp]
        original_rank = len(original_shape)
        padded_shape = [1] * (output_rank - original_rank) + original_shape
        index_parts = []
        for i in range(original_rank):
            # Calculate position in padded shape
            padded_dim = (output_rank - original_rank) + i
            input_dim = padded_shape[padded_dim]
            output_dim = output_shape[padded_dim]
            # Broadcast only if input dim is 1 and output dim > 1
            if input_dim == 1 and output_dim != 1:
                index_parts.append('0')
            else:
                index_parts.append(loop_vars[padded_dim])
        # Build index string (e.g., [d0][d1][0][0])
        index_str = '[' + ']['.join(index_parts) + ']' if index_parts else ''
        input_exprs.append(f"{inp}{index_str}")

    # Output index uses all loop variables
    output_index = '[' + ']['.join(loop_vars) + ']'
    buffer.write(f"        {outputs[0]}{output_index} = {input_exprs[0]} {c_op} {input_exprs[1]};\n")

    # Close loops
    for _ in range(output_rank):
        buffer.write("    }\n")

    buffer.write("}\n")