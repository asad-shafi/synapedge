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

def _write_dropout_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                            attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for the ONNX Dropout operator in inference mode.
    
    In inference, Dropout is a no-op:
      - The output tensor is identical to the input tensor.
      - If a mask output is provided, it is filled with ones.
      
    """
    # Retrieve dropout ratio (though not used in inference, it is noted in the comments)
    ratio = attrs.get('ratio', 0.5)
    if len(outputs) > 1: # warnif mask output is not provided
        if outputs[1] not in tensor_shape or tensor_shape[outputs[1]] is None:
            outputs.pop(1)
            #raise ValueError("Mask output tensor shape must be provided.")
    # Write the function signature using an assumed helper function.
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    
    # Write a C-style comment about the dropout operator.
    helperfunc._write_c_comment(buffer, f"Dropout (Inference): ratio={ratio}. Pass-through with mask=1 if provided.", indent=4)
    
    # Retrieve the shape for the primary input tensor.
    in_shape = tensor_shape[inputs[0]]
    ndim = len(in_shape)
    
    # Declare dimension constants based on the input shape.
    for idx, dim in enumerate(in_shape):
        buffer.write(f"    const int dim{idx} = {dim};\n")
    buffer.write("\n")
    
    # Generate nested loops to iterate over every element in the tensor.
    indent = "    "
    for idx in range(ndim):
        buffer.write(indent + f"for (int i{idx} = 0; i{idx} < dim{idx}; i{idx}++) {{\n")
        indent += "    "
    
    # Build the index string (e.g., "[i0][i1]...[in]").
    index_str = "".join(f"[i{idx}]" for idx in range(ndim))
    input_name = inputs[0]
    output_name = outputs[0]
    
    # Copy each element from input to output.
    buffer.write(indent + f"{output_name}{index_str} = {input_name}{index_str};\n")
    
    # If a mask output is provided, fill it with 1.
    if len(outputs) > 1:
        mask_name = outputs[1]
        buffer.write(indent + f"{mask_name}{index_str} = 1;\n")
    
    # Close all nested loops.
    for _ in range(ndim):
        indent = indent[:-4]
        buffer.write(indent + "}\n")
    
    # End of function.
    buffer.write("}\n")
