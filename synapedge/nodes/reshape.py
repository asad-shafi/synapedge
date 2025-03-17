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

def _write_reshape_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for ONNX Reshape operator."""
    input_name = inputs[0]
    output_name = outputs[0]

    # Calculate element count from input shape
    input_dims = tensor_shape.get(input_name, [])
    output_dims = tensor_shape.get(output_name, [])

    # Verify total elements match
    input_size = 1
    for dim in input_dims:
        input_size *= dim

    output_size = 1
    for dim in output_dims:
        output_size *= dim

    if input_size != output_size:
        raise ValueError(f"Reshape mismatch: Input {input_size} elements vs Output {output_size} elements")
        buffer.write(f"#error Reshape mismatch: Input {input_size} elements vs Output {output_size} elements\n")
        return

    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    buffer.write("    // Reshape does not modify data, only strides/shapes\n")
    buffer.write(f"    float *src = (float*){input_name};\n")
    buffer.write(f"    float *dst = (float*){output_name};\n")
    buffer.write(f"    memcpy(dst, src, {input_size} * sizeof(float));\n")
    buffer.write("}\n")
