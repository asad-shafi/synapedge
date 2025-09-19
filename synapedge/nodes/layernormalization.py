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


def _write_layernormalization_node(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                                   attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:

    input_name = inputs[0]
    scale = inputs[1]
    bias = inputs[2]
    output_name = outputs[0]

    # 1. Fetch and compute shapes
    in_shape = tensor_shape.get(input_name, [])
    # ONNX LayerNormalization attributes
    axis = int(attrs.get("axis", -1))
    epsilon = float(attrs.get("epsilon", 1e-5))

    # Write function signature (helper will open the brace)
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)

    if in_shape:
        nd = len(in_shape)
        # convert negative axis to positive index relative to nd
        axis_index = axis if axis >= 0 else (nd + axis)
        # clamp axis_index to [0, nd-1]
        if axis_index < 0:
            axis_index = 0
        if axis_index > nd - 1:
            axis_index = nd - 1

        # pre-normalization dims (we will loop over them, including batch)
        pre_dims = in_shape[:axis_index]
        norm_dims = in_shape[axis_index:]  # dims to normalize over (axis..end)

        # Emit constants for dims
        for idx, d in enumerate(in_shape):
            buffer.write(f"    const int D{idx} = {int(d)};\n")

        # Compute INNER_SIZE = product of normalized dims
        inner_size = 1
        for d in norm_dims:
            inner_size *= int(d)
        buffer.write(f"\n    // LayerNormalization: axis={axis}, epsilon={epsilon}\n")
        buffer.write(f"    const int INNER_SIZE = {inner_size};\n")

        # If there are pre-dims, emit nested loops; ensure batch loop is present (pre_dims[0])
        if pre_dims:
            # emit outer loops for each pre-dim
            loop_vars = []
            for idx in range(len(pre_dims)):
                var = f"d{idx}"
                loop_vars.append(var)
                buffer.write(f"    for (int {var} = 0; {var} < D{idx}; ++{var}) {{\n")

            # build index prefix for bracketed indexing like [d0][d1]...[i]
            index_prefix = "".join(f'[{v}]' for v in loop_vars)
            # compute mean
            buffer.write("        /* compute mean */\n")
            buffer.write("        float mean = 0.0f;\n")
            buffer.write("        for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"            mean += {input_name}{index_prefix}[i];\n")
            buffer.write("        }\n")
            buffer.write("        mean /= (float)INNER_SIZE;\n\n")

            # variance
            buffer.write("        /* compute variance */\n")
            buffer.write("        float var = 0.0f;\n")
            buffer.write("        for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"            float diff = {input_name}{index_prefix}[i] - mean;\n")
            buffer.write("            var += diff * diff;\n")
            buffer.write("        }\n")
            buffer.write("        var /= (float)INNER_SIZE;\n")
            buffer.write(f"        float inv_std = 1.0f / sqrtf(var + {epsilon}f);\n\n")

            # normalize, scale & bias
            buffer.write("        /* normalize, scale and bias */\n")
            buffer.write("        for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"            float normalized = ({input_name}{index_prefix}[i] - mean) * inv_std;\n")
            buffer.write(f"            {output_name}{index_prefix}[i] = normalized * {scale}[i] + {bias}[i];\n")
            buffer.write("        }\n")

            # close the pre-dim loops
            for _ in loop_vars:
                buffer.write("    }\n")
        else:
            # No pre-dims: normalize across entire tensor (single outer group)
            buffer.write("    /* No pre-normalization dims: normalize across entire tensor */\n")
            buffer.write("    float mean = 0.0f;\n")
            buffer.write("    for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"        mean += {input_name}[i];\n")
            buffer.write("    }\n")
            buffer.write("    mean /= (float)INNER_SIZE;\n\n")

            buffer.write("    float var = 0.0f;\n")
            buffer.write("    for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"        float diff = {input_name}[i] - mean;\n")
            buffer.write("        var += diff * diff;\n")
            buffer.write("    }\n")
            buffer.write("    var /= (float)INNER_SIZE;\n")
            buffer.write(f"    float inv_std = 1.0f / sqrtf(var + {epsilon}f);\n\n")

            buffer.write("    for (int i = 0; i < INNER_SIZE; ++i) {\n")
            buffer.write(f"        float normalized = ({input_name}[i] - mean) * inv_std;\n")
            buffer.write(f"        {output_name}[i] = normalized * {scale}[i] + {bias}[i];\n")
            buffer.write("    }\n")
    else:
        # Fallback: no static shape available. Generate runtime-style code.
        buffer.write("    // LayerNormalization: tensor shape unknown at codegen time.\n")
        buffer.write("    // Fallback implementation: normalize over the last dimension by assuming\n")
        buffer.write("    // the caller provides proper sizes / stride computation.\n\n")
        buffer.write("    int total_elems = 1; /* TODO: fill with runtime total elements */\n")
        buffer.write("    int inner_size = 1; /* TODO: fill with runtime normalized-axis size */\n")
        buffer.write("    int outer_size = total_elems / inner_size; /* assume exact division */\n\n")
        buffer.write("    for (int o = 0; o < outer_size; ++o) {\n")
        buffer.write("        float mean = 0.0f;\n")
        buffer.write("        for (int i = 0; i < inner_size; ++i) {\n")
        buffer.write(f"            mean += {input_name}[o * inner_size + i];\n")
        buffer.write("        }\n")
        buffer.write("        mean /= (float)inner_size;\n\n")
        buffer.write("        float var = 0.0f;\n")
        buffer.write("        for (int i = 0; i < inner_size; ++i) {\n")
        buffer.write(f"            float diff = {input_name}[o * inner_size + i] - mean;\n")
        buffer.write("            var += diff * diff;\n")
        buffer.write("        }\n")
        buffer.write("        var /= (float)inner_size;\n")
        buffer.write(f"        float inv_std = 1.0f / sqrtf(var + {epsilon}f);\n\n")
        buffer.write("        for (int i = 0; i < inner_size; ++i) {\n")
        buffer.write(f"            float normalized = ({input_name}[o * inner_size + i] - mean) * inv_std;\n")
        buffer.write(f"            {output_name}[o * inner_size + i] = normalized * {scale}[i] + {bias}[i];\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    # Note: the caller is expected to write the closing brace "}\n"
    buffer.write("}\n")
