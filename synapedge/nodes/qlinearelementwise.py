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
def _write_qlinearelementwise_function(functions: Dict[str, Any], buffer: StringIO, op_type: str, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for element-wise operations with dynamic loop variables and broadcasting support."""
    c_op = {"QLinearAdd": "+", "QLinearSub": "-", "QLinearMul": "*", "QLinearDiv": "/"}.get(op_type)
    if not c_op:
        raise ValueError(f"Unsupported elementwise operation type: {op_type}")

    input_shapes = [tensor_shape[inp] for inp in inputs]
    output_shape = tensor_shape.get(outputs[0], [])
    output_name = outputs[0]

# If the output shape is empty, use the input shape instead
    if not output_shape:
        print(f"Warning: Output size for {outputs[0]} not provided. Using  input size {inputs[0]}.")
        tensor_shape[outputs[0]] = input_shapes[0] # input A
        output_shape = input_shapes[0]
        # Update 'outputs[0]' in functions
        for i in range(len(functions)):  # Iterate using index
            func = functions[i]  # Get function dictionary by index
            if helperfunc._sanitize_name(func['name']) == func_name: # Update output shape of the producer node (or curent node)
                functions[i]['intermediate_tensors_shape'][outputs[0]] = output_shape  # Update using index
                functions[i]['intermediate_tensors_shape'][f"{outputs[0]}_dtype"] = tensor_shape.get(inputs[0]+"_dtype", "uint8_t") # suport only uint8_t 

                dims = ''.join([f"[{dim}]" for dim in output_shape])
                na = f"tensor_{output_name}"
                functions[i]['computed_shape'] = f"{tensor_shape.get(output_name+'_dtype', 'uint8_t')} {helperfunc._sanitize_name(na)}{dims}"
            for ins in func['inputs']:
              if (helperfunc._sanitize_name(ins)) == output_name: # Update inputs shape of the consumers node in graph
                functions[i]['intermediate_tensors_shape'][helperfunc._sanitize_name(ins)] = output_shape # Update inputs shapes of the consumer node with producer node output shape.
                functions[i]['intermediate_tensors_shape'][f"{helperfunc._sanitize_name(ins)}_dtype"] = tensor_shape.get(inputs[0]+"_dtype", "uint8_t") # suport only uint8_t 
                #break  # Exit the loop after updating
    # Check if input and output shapes are the same
    if input_shapes[0] != output_shape:
        raise ValueError(f"#error Input and output shapes must be the same for ElementWise functions. Input shape: {input_shapes[0]}, Output shape: {output_shape}")
        buffer.write(f"#error Input and output shapes must be the same. Input shape: {input_shape}, Output shape: {output_shape}\n")
        return
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
    
    buffer.write(f"    float A_scale_val = {inputs[1]}[0]; ")
    buffer.write(f" int32_t A_zp = (int32_t){inputs[2]}[0];\n")
    buffer.write(f"    float B_scale_val = {inputs[4]}[0]; ")
    buffer.write(f" int32_t B_zp = (int32_t){inputs[5]}[0];\n")

    buffer.write(f"    float C_scale_val = {inputs[6]}[0]; ")
    buffer.write(f"float C_zp = (float){inputs[7]}[0];\n")
    buffer.write(f"     float multiplier = (A_scale_val * B_scale_val)/C_scale_val;\n\n")
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

    buffer.write(f"        int32_t prod = ({input_exprs[0]} - A_zp) {c_op} ({input_exprs[3]} - B_zp);\n")
    buffer.write(f"        float y = (float)prod * multiplier;\n") 
    buffer.write(f"        float scaled = y  + C_zp;\n")

    buffer.write(f"        int32_t q = (int32_t)roundf(scaled);\n")
    buffer.write(f"        if (q < 0) q = 0;\n")
    buffer.write(f"        if (q > 255) q = 255;\n")
    buffer.write(f"        {outputs[0]}{output_index} = (uint8_t)q;\n")
    # Close loops
    for _ in range(output_rank):
        buffer.write("    }\n")

    buffer.write("}\n")