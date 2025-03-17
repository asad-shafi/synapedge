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

def _write_resize_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], op_type: str, tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for Resize/Upsample operator."""
    mode = attrs.get('mode', 'nearest')
    coordinate_transformation_mode = attrs.get('coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = attrs.get('cubic_coeff_a', -0.75)
    exclude_outside = attrs.get('exclude_outside', 0)
    extrapolation_value = attrs.get('extrapolation_value', 0.0)
    nearest_mode = attrs.get('nearest_mode', 'round_prefer_floor')

    comment = f"{op_type}: mode={mode}, coord_transform={coordinate_transformation_mode}, cubic_a={cubic_coeff_a}, exclude_outside={exclude_outside}, extrapolation={extrapolation_value}, nearest_mode={nearest_mode}"
        # Remove the ROI tensor if four inputs are provided (ONNX > 10)
    if len(inputs) == 4:
        # Remove the second input (ROI) to avoid naming issues in the generated code
        inputs.pop(1)

    input_name = inputs[0]
    input_shape = tensor_shape[input_name]
    N = input_shape[0]
    C = input_shape[1]
    H_in = input_shape[2]
    W_in = input_shape[3]

    # Determine H_out and W_out based on sizes or scales
    if inputs[2] != '': # if size present use size 
        size_shape = tensor_shape[inputs[2]] 
        #H_out = size_shape[2]
        #W_out = size_shape[3]
        H_out = f"{inputs[2]}[2]; // {tensor_shape[outputs[0]][2]}"
        W_out = f"{inputs[2]}[3]; // {tensor_shape[outputs[0]][3]}"
    elif inputs[1] != '': # if scales present use scales
        scales_data = tensor_shape[inputs[2]]
        H_out = int(H_in * scales_data[2])
        W_out = int(W_in * scales_data[3])
    else:
        raise ValueError("Resize: Either sizes or scales must be provided")
    if inputs[1] == '':
        inputs.pop(1) # remove scales if not present it wil caiuse error in the code because no name is present

        
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, comment, indent=4)
    # Write dimension constants
    buffer.write(f"    const int N = {N};\n")
    buffer.write(f"    const int C = {C};\n")
    buffer.write(f"    const int H_in = {H_in};\n")
    buffer.write(f"    const int W_in = {W_in};\n")
    buffer.write(f"    const int H_out = {H_out};\n")
    buffer.write(f"    const int W_out = {W_out};\n\n")

    buffer.write (f"    float scale_h = (float)H_out / H_in;\n")
    buffer.write (f"    float scale_w = (float)W_out / W_in;\n")

    buffer.write("    for (int n = 0; n < N; ++n) {\n")
    buffer.write("        for (int c = 0; c < C; ++c) {\n")
    buffer.write("            for (int y = 0; y < H_out; ++y) {\n")
    buffer.write("                for (int x = 0; x < W_out; ++x) {\n")

    # Compute input coordinates based on coordinate transformation mode
    if coordinate_transformation_mode == "half_pixel" or "b'half_pixel'":
        #buffer.write("                    float scale_h = (float)H_out / H_in;\n")
        #buffer.write("                    float scale_w = (float)W_out / W_in;\n")
        buffer.write("                    float y_in = ((float)y + 0.5f) / scale_h - 0.5f;\n")
        buffer.write("                    float x_in = ((float)x + 0.5f) / scale_w - 0.5f;\n")
    elif coordinate_transformation_mode == "pytorch_half_pixel" or "b'pytorch_half_pixel'":
        ##buffer.write("                    float scale_h = (float)H_out / H_in;\n")
        buffer.write("                    float scale_w = (float)W_out / W_in;\n")
        buffer.write("                    float y_in = (H_out > 1) ? ((float)y + 0.5f) / scale_h - 0.5f : 0.0f;\n")
        buffer.write("                    float x_in = (W_out > 1) ? ((float)x + 0.5f) / scale_w - 0.5f : 0.0f;\n")
    elif coordinate_transformation_mode == "align_corners" or "b'align_corners'":
        buffer.write("                    float scale_h = (H_in > 1 && H_out > 1) ? (float)(H_in - 1) / (H_out - 1) : 0.0f;\n")
        buffer.write("                    float scale_w = (W_in > 1 && W_out > 1) ? (float)(W_in - 1) / (W_out - 1) : 0.0f;\n")
        buffer.write("                    float y_in = y * scale_h;\n")
        buffer.write("                    float x_in = x * scale_w;\n")
    elif coordinate_transformation_mode == "asymmetric" or "b'asymmetric'":
        #buffer.write("                    float scale_h = (float)H_out / H_in;\n")
        #buffer.write("                    float scale_w = (float)W_out / W_in;\n")
        buffer.write("                    float y_in = (float)y / scale_h;\n")
        buffer.write("                    float x_in = (float)x / scale_w;\n")
    elif coordinate_transformation_mode == "tf_half_pixel_for_nn" or "b'tf_half_pixel_for_nn'":
        #buffer.write("                    float scale_h = (float)H_out / H_in;\n")
        #buffer.write("                    float scale_w = (float)W_out / W_in;\n")
        buffer.write("                    float y_in = (y + 0.5f) / scale_h;\n")
        buffer.write("                    float x_in = (x + 0.5f) / scale_w;\n")
    else:
        raise ValueError(f"Unsupported coordinate transformation mode: {coordinate_transformation_mode} in node {func_name}")
        buffer.write("                    float y_in = 0.0f, x_in = 0.0f;\n")

    # Handle interpolation mode
    if mode == 'nearest' or "b'nearest'":
        # Nearest neighbor interpolation
        if nearest_mode == 'floor' or "b'floor'":
            buffer.write("                    int y_index = (int)floorf(y_in);\n")
            buffer.write("                    int x_index = (int)floorf(x_in);\n")
        elif nearest_mode == 'ceil' or "b'ceil'":
            buffer.write("                    int y_index = (int)ceilf(y_in);\n")
            buffer.write("                    int x_index = (int)ceilf(x_in);\n")
        elif nearest_mode == 'round_prefer_floor' or "b'round_prefer_floor'":
            buffer.write("                    int y_index = (int)roundf(y_in);\n")
            buffer.write("                    if (y_in - floorf(y_in) == 0.5f) y_index = (int)floorf(y_in);\n")
            buffer.write("                    int x_index = (int)roundf(x_in);\n")
            buffer.write("                    if (x_in - floorf(x_in) == 0.5f) x_index = (int)floorf(x_in);\n")
        else:
            buffer.write("                    int y_index = (int)roundf(y_in);\n")
            buffer.write("                    int x_index = (int)roundf(x_in);\n")

        # Clamp indices
        buffer.write("                    y_index = y_index < 0 ? 0 : (y_index >= H_in ? H_in - 1 : y_index);\n")
        buffer.write("                    x_index = x_index < 0 ? 0 : (x_index >= W_in ? W_in - 1 : x_index);\n")
        buffer.write(f"                    {outputs[0]}[n][c][y][x] = {inputs[0]}[n][c][y_index][x_index];\n")
    elif mode == 'linear' or "b'linear'":
        # Bilinear interpolation
        buffer.write("                    int y0 = (int)floorf(y_in);\n")
        buffer.write("                    int y1 = y0 + 1;\n")
        buffer.write("                    float dy = y_in - y0;\n")
        buffer.write("                    y0 = y0 < 0 ? 0 : (y0 >= H_in ? H_in - 1 : y0);\n")
        buffer.write("                    y1 = y1 < 0 ? 0 : (y1 >= H_in ? H_in - 1 : y1);\n")
        buffer.write("                    int x0 = (int)floorf(x_in);\n")
        buffer.write("                    int x1 = x0 + 1;\n")
        buffer.write("                    float dx = x_in - x0;\n")
        buffer.write("                    x0 = x0 < 0 ? 0 : (x0 >= W_in ? W_in - 1 : x0);\n")
        buffer.write("                    x1 = x1 < 0 ? 0 : (x1 >= W_in ? W_in - 1 : x1);\n")
        buffer.write(f"                    float v00 = {inputs[0]}[n][c][y0][x0];\n")
        buffer.write(f"                    float v01 = {inputs[0]}[n][c][y0][x1];\n")
        buffer.write(f"                    float v10 = {inputs[0]}[n][c][y1][x0];\n")
        buffer.write(f"                    float v11 = {inputs[0]}[n][c][y1][x1];\n")
        buffer.write("                    float val = (1 - dy) * (1 - dx) * v00 + (1 - dy) * dx * v01 + dy * (1 - dx) * v10 + dy * dx * v11;\n")
        buffer.write(f"                    {outputs[0]}[n][c][y][x] = val;\n")
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode} in node {func_name}")
        buffer.write("                    // Unsupported interpolation mode\n")
        buffer.write(f"                    {outputs[0]}[n][c][y][x] = 0.0f;\n")

    buffer.write("                }\n")
    buffer.write("            }\n")
    buffer.write("        }\n")
    buffer.write("    }\n")
    buffer.write("}\n")

