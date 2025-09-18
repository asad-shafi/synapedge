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

def _write_qlinearactivation_function(functions: Dict[str, Any], buffer: StringIO, op_type: str, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for activation functions (ReLU, Sigmoid, etc.).
    """
    # abs to calculate sigmoid, it might be faster than exp
    abs = True
    input_name = inputs[0]
    X_scale = inputs[1] 
    X_zero_point = inputs[2] 
    y_scale = inputs[3] 
    y_zero_point = inputs[4]   


    output_name = outputs[0]
    input_shape = tensor_shape.get(input_name, [])
    output_shape = tensor_shape.get(output_name, [])

    # If the output shape is empty, use the input shape instead
    if not output_shape:
        print(f"Warning: Output size for {outputs[0]} not provided. Using  input size {inputs[0]}.")
        tensor_shape[outputs[0]] = input_shape
        output_shape = input_shape
        # Update 'outputs[0]' in functions
        for i in range(len(functions)):  # Iterate using index
            func = functions[i]  # Get function dictionary by index
            if helperfunc._sanitize_name(func['name']) == func_name: # Update output shape of the producer node (or curent node)
                functions[i]['intermediate_tensors_shape'][outputs[0]] = output_shape  # Update using index
            for ins in func['inputs']:
              if (helperfunc._sanitize_name(ins)) == output_name: # Update inputs shape of the consumers node in graph
                functions[i]['intermediate_tensors_shape'][helperfunc._sanitize_name(ins)] = output_shape # Update inputs shapes of the consumer node with producer node output shape.
                #break  # Exit the loop after updating
    # Check if input and output shapes are the same
    if input_shape != output_shape:
        raise ValueError(f"#error Input and output shapes must be the same for activation functions. Input shape: {input_shape}, Output shape: {output_shape}")
        buffer.write(f"#error Input and output shapes must be the same. Input shape: {input_shape}, Output shape: {output_shape}\n")
        return
    # Calculate total number of elements
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim
    # Write function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    # Flatten the input and output arrays
    buffer.write(f"    float *X_ptr = (float *){input_name};\n")
    buffer.write(f"    float *Y_ptr = (float *){output_name};\n")
    # Write loop with the correct termination condition
    buffer.write(f"    for (int i = 0; i < {total_elements}; i++) {{\n")
    # Write the corresponding operation based on op_type
    if op_type == "Relu":
        buffer.write("        Y_ptr[i] = fmaxf(0.0f, X_ptr[i]);\n")
    elif op_type == "Sigmoid":
        buffer.write("        Y_ptr[i] = 1.0f / (1.0f + expf(-X_ptr[i]));\n")
    elif op_type == "Tanh":
        buffer.write("        Y_ptr[i] = tanhf(X_ptr[i]);\n")
    # Close the loop and function
    buffer.write("    }\n")
    buffer.write("}\n")