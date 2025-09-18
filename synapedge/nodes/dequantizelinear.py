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

from typing import Any, Dict, List
from io import StringIO


def _write_dequantizelinear_function(functions: Dict[str, Any],buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for DeQuantizeLinear operator that works for both uint8_t and int8_t."""
    x_name = inputs[0]
    y_scale_name = inputs[1]
    y_zp_name = inputs[2]
    y_name = outputs[0]
    axis = int(attrs.get("axis", 1))
    out_dtype = tensor_shape.get(y_name+"_dtype", [])

    # Get tensor shapes
    x_shape = tensor_shape[x_name]
    scale_shape = tensor_shape[y_scale_name]
    zp_shape = tensor_shape[y_zp_name]
    y_shape = tensor_shape[y_name]


   # Calculate output dimensions if not provided
    if not y_shape:
        #print(output_dims)
        print(f"Warning: Output size for {outputs[0]} not provided. Using  input size {inputs[0]}.")
        tensor_shape[outputs[0]] = x_shape # input A
        y_shape = x_shape
        # Update 'outputs[0]' in functions
        for i in range(len(functions)):  # Iterate using index
            func = functions[i]  # Get function dictionary by index
            if helperfunc._sanitize_name(func['name']) == func_name: # Update output shape of the producer node (or curent node)
                functions[i]['intermediate_tensors_shape'][outputs[0]] = y_shape  # Update using index
                # quantize out uint8_t or int8_t, so hardcode data type
                functions[i]['intermediate_tensors_shape'][f"{outputs[0]}_dtype"] = "float"#tensor_shape.get(x_name+"_dtype", "uint8_t")

                dims = ''.join([f"[{dim}]" for dim in y_shape])
                na = f"tensor_{y_name}"
                functions[i]['computed_shape'] = f"{tensor_shape.get(y_name+'_dtype', 'float')} {helperfunc._sanitize_name(na)}{dims}" 
            for ins in func['inputs']:
              if (helperfunc._sanitize_name(ins)) == y_name: # Update inputs shape of the consumers node in graph
                functions[i]['intermediate_tensors_shape'][helperfunc._sanitize_name(ins)] = y_shape # Update inputs shapes of the consumer node with producer node output shape.
                functions[i]['intermediate_tensors_shape'][f"{helperfunc._sanitize_name(ins)}_dtype"] = "float" # dequantize out flaot

                #break  # Exit the loop after updating

    # Compute total elements in scale tensor
    total_scale_elements = 1
    for dim in scale_shape:
        total_scale_elements *= dim

    # Convert axis to positive index
    rank = len(x_shape)
    if axis < 0:
        axis += rank

    # Write function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "QuantizeLinear", indent=4)
    buffer.write(f"    int32_t zp = (int32_t){y_zp_name}[0];\n")
    buffer.write(f"    float z_scale = {y_scale_name}[0];\n")
    # Per-tensor quantization
    if total_scale_elements == 1:
        indent = "    "
        for d, dim in enumerate(x_shape):
            buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
            indent += "    "
        
        # Build index string
        idx_str = ''.join([f"[i{d}]" for d in range(rank)])
        
        # Write quantization operation float q = (float)images_quantized[i0][i1][i2][i3];
        buffer.write(indent + f"int32_t q = (int32_t){x_name}{idx_str};\n")
        buffer.write(indent + f"{y_name}{idx_str} = (float)(q - zp) * z_scale;\n")

        
        # Close loops
        for _ in range(rank):
            indent = indent[:-4]
            buffer.write(indent + "}\n")
    
    # Per-axis quantization
    else:
        indent = "    "
        for d, dim in enumerate(x_shape):
            buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
            indent += "    "
        
        # Build index string
        idx_str = ''.join([f"[i{d}]" for d in range(rank)])
        axis_idx = f"i{axis}"
        
        # Write quantization operation
        buffer.write(indent + f"float q = (float){x_name}{idx_str};\n")
        buffer.write(indent + f"{y_name}{idx_str} = (q - (float){y_zp_name}[{axis_idx}]) * {y_scale_name}[{axis_idx}];\n")
        # Close loops
        for _ in range(rank):
            indent = indent[:-4]
            buffer.write(indent + "}\n")
    
    buffer.write("}\n")

    #print(buffer.getvalue())

    
