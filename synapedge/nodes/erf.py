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
from io import StringIO
import nodes.helperfunc as helperfunc
import logging
from typing import List, Dict, Any

def _write_erf_node(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """
    Generate C code for the ONNX Erf node (elementwise erf).
    - Assumes input and output are contiguous 1D arrays in C (flattened).
    - If tensor_shape contains concrete integer dimensions for the input tensor,
      this generates a simple for-loop over the flattened element count.
    - If shape is not fully known at generation time, it emits a TODO placeholder
      where you should plug your runtime-size expression.
    """

    input_name = inputs[0]
    output_name = outputs[0]

    # 1. Fetch and compute shapes (make a working copy)
    in_shape = tensor_shape.get(input_name, [])
    out_shape = list(in_shape)

    # Write the function signature using your helper (keeps consistency with other nodes)
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)

    # Begin function body (helperfunc likely wrote an opening brace; if not adjust accordingly)
    buffer.write("    /* element-wise error function (erf) */\n")
    buffer.write("    /* Note: requires <math.h> at top-level for erf/erff */\n\n")
    buffer.write(f"   float *src = (float*){input_name};\n")
    buffer.write(f"   float *dst = (float*){output_name};\n")

    # Try to compute total elements at generation time if all dims are concrete integers
    all_int_dims = True
    total_elems = 1
    for d in out_shape:
        if not isinstance(d, int) or d <= 0:
            all_int_dims = False
            break
        total_elems *= d

    if all_int_dims:
        # Emit a fast flattened loop with compile-time-computed total
        buffer.write(f"    size_t total = {total_elems}u;\n")
        buffer.write("    if (total == 0) return;\n")
        buffer.write("    for (size_t i = 0; i < total; ++i) {\n")
        buffer.write(f"        dst[i] = erff(src[i]);\n")
        buffer.write("    }\n")
    else:
        # Fallback: shape unknown at generation time — emit runtime-size placeholder
        buffer.write("    /* WARNING: some or all output dimensions are not concrete at generation time. */\n")
        buffer.write("    /* Replace 'runtime_total' below with your runtime-computed element count. */\n")
        buffer.write("    size_t runtime_total = 0; /* TODO: compute product of output shape at runtime */\n")
        buffer.write("    if (runtime_total == 0) return;\n")
        buffer.write("    for (size_t i = 0; i < runtime_total; ++i) {\n")
        buffer.write(f"        {output_name}[i] = erf({input_name}[i]);\n")
        buffer.write("    }\n")

    # Close function body (match helperfunc._write_function_signature's opening)
    buffer.write("}\n")
