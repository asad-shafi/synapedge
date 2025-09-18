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


def _write_squeeze_node(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for the ONNX Squeeze node, which removes dimensions of size 1.
    Like Unsqueeze, Squeeze only changes the shape, not the data layout — so the
    generated C is a simple element-wise copy. This function updates `tensor_shape`
    for downstream nodes and emits a C wrapper that copies the input buffer to the
    output buffer.
    """
    input_name = inputs[0]
    output_name = outputs[0]

    # 1. Fetch and compute shapes
    in_shape = tensor_shape.get(input_name, [])
    in_shape= tensor_shape.get(output_name, [])
    # axes may be absent -> remove all dims of size 1
    axes = attrs.get("axes", None)


    # 3. Emit the C function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)

    # 4. Pointer aliasing (squeeze does not change data layout)
    buffer.write(f"    float *X_ptr = (float *){input_name};\n")
    buffer.write(f"    float *Y_ptr = (float *){output_name};\n")

    # 5. Compute total number of elements in the *input* (same as output)
    total_elems = 1
    for d in in_shape:
        total_elems *= d

    # 6. Emit a simple copy loop
    buffer.write(f"    for (int i = 0; i < {total_elems}; i++) {{\n")
    buffer.write("        Y_ptr[i] = X_ptr[i];\n")
    buffer.write("    }\n")
    buffer.write("}\n")
