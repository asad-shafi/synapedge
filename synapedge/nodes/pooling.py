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

def _write_pooling_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                            attrs: Dict[str, Any], pool_type: str, tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for Pooling operators (AveragePool, MaxPool, MinPool)
    using multi-dimensional indexing.

    Raises:
      ValueError: If there are less than two outputs, inputs are insufficient,
            or split tensor is invalid.
    """
    # Prepare the attribute comment block
    str_attr = f"    /* {pool_type.capitalize()}Pool\n"
    for attr, value in attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        elif isinstance(value, list):
            value = " ".join(map(str, value))
        str_attr += f"     * {attr}: {value}\n"
    str_attr += "     */"

    # Validate pool_type
    if pool_type not in ['average', 'max', 'min']:
        raise ValueError(f"Invalid pool_type: {pool_type}")

    # Extract and validate attributes
    kernel_shape = attrs.get('kernel_shape', [1, 1])
    if len(kernel_shape) != 2 or any(k <= 0 for k in kernel_shape):
        raise ValueError(f"Invalid kernel_shape: {kernel_shape}")

    strides = attrs.get('strides', [1, 1])
    if len(strides) != 2 or any(s <= 0 for s in strides):
        raise ValueError(f"Invalid strides: {strides}")

    pads = attrs.get('pads', [0] * 4)
    if len(pads) != 4 or any(p < 0 for p in pads):
        raise ValueError(f"Invalid pads: {pads}")

    # Validate input and output shapes
    input_name = inputs[0]
    output_name = outputs[0]
    input_shape = tensor_shape.get(input_name)
    output_shape = tensor_shape.get(output_name)

    if not input_shape or len(input_shape) != 4:
        raise ValueError(f"Input shape must be 4D, got {input_shape}")
    if not output_shape or len(output_shape) != 4:
        raise ValueError(f"Output shape must be 4D, got {output_shape}")

    batch, channels, in_h, in_w = input_shape
    out_batch, out_channels, out_h, out_w = output_shape

    if batch != out_batch or channels != out_channels:
        raise ValueError("Batch and channel dimensions must match between input and output")

    # Additional checks for AveragePool
    count_include_pad = 0
    if pool_type == 'average':
        count_include_pad = attrs.get('count_include_pad', 0)
        if count_include_pad not in (0, 1):
            raise ValueError("count_include_pad must be 0 or 1 for AveragePool")

    # Generate the C code
    buffer.write(f"{str_attr}\n")
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    buffer.write(f"    const int batch = {batch}, channels = {channels};\n")
    buffer.write(f"    const int in_h = {in_h}, in_w = {in_w};\n")
    buffer.write(f"    const int out_h = {out_h}, out_w = {out_w};\n")
    buffer.write(f"    const int kernel_h = {kernel_shape[0]}, kernel_w = {kernel_shape[1]};\n")
    buffer.write(f"    const int stride_h = {strides[0]}, stride_w = {strides[1]};\n")
    buffer.write(f"    const int pad_begin_h = {pads[0]}, pad_end_h = {pads[1]}, "
                 f"pad_begin_w = {pads[2]}, pad_end_w = {pads[3]};\n")

    if pool_type == 'average':
        buffer.write(f"    const int count_include_pad = {count_include_pad};\n")
    buffer.write("\n")

    buffer.write("    for (int n = 0; n < batch; ++n) {\n")
    buffer.write("        for (int c = 0; c < channels; ++c) {\n")
    buffer.write("            for (int oh = 0; oh < out_h; ++oh) {\n")
    buffer.write("                for (int ow = 0; ow < out_w; ++ow) {\n")

    if pool_type == 'max':
        buffer.write("                    float max_val = -INFINITY;\n")
        buffer.write("                    for (int kh = 0; kh < kernel_h; ++kh) {\n")
        buffer.write("                        for (int kw = 0; kw < kernel_w; ++kw) {\n")
        buffer.write("                            int h_in = oh * stride_h - pad_begin_h + kh;\n")
        buffer.write("                            int w_in = ow * stride_w - pad_begin_w + kw;\n")
        buffer.write("                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {\n")
        buffer.write(f"                                float val = {input_name}[n][c][h_in][w_in];\n")
        buffer.write("                                if (val > max_val) max_val = val;\n")
        buffer.write("                            }\n")
        buffer.write("                        }\n")
        buffer.write("                    }\n")
        buffer.write(f"                    {output_name}[n][c][oh][ow] = max_val;\n")
    elif pool_type == 'min':
        buffer.write("                    float min_val = INFINITY;\n")
        buffer.write("                    for (int kh = 0; kh < kernel_h; ++kh) {\n")
        buffer.write("                        for (int kw = 0; kw < kernel_w; ++kw) {\n")
        buffer.write("                            int h_in = oh * stride_h - pad_begin_h + kh;\n")
        buffer.write("                            int w_in = ow * stride_w - pad_begin_w + kw;\n")
        buffer.write("                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {\n")
        buffer.write(f"                                float val = {input_name}[n][c][h_in][w_in];\n")
        buffer.write("                                if (val < min_val) min_val = val;\n")
        buffer.write("                            }\n")
        buffer.write("                        }\n")
        buffer.write("                    }\n")
        buffer.write(f"                    {output_name}[n][c][oh][ow] = min_val;\n")
    elif pool_type == 'average':
        buffer.write("                    float sum = 0.0f;\n")
        buffer.write("                    int count = 0;\n")
        buffer.write("                    int total = kernel_h * kernel_w;\n")
        buffer.write("                    for (int kh = 0; kh < kernel_h; ++kh) {\n")
        buffer.write("                        for (int kw = 0; kw < kernel_w; ++kw) {\n")
        buffer.write("                            int h_in = oh * stride_h - pad_begin_h + kh;\n")
        buffer.write("                            int w_in = ow * stride_w - pad_begin_w + kw;\n")
        buffer.write("                            if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {\n")
        buffer.write(f"                                float val = {input_name}[n][c][h_in][w_in];\n")
        buffer.write("                                sum += val;\n")
        buffer.write("                                count++;\n")
        buffer.write("                            } else if (count_include_pad) {\n")
        buffer.write("                                sum += 0.0f;\n")
        buffer.write("                                count++;\n")
        buffer.write("                            }\n")
        buffer.write("                        }\n")
        buffer.write("                    }\n")
        buffer.write("                    if (count_include_pad) {\n")
        buffer.write("                        sum /= total;\n")
        buffer.write("                    } else {\n")
        buffer.write("                        if (count > 0) sum /= count;\n")
        buffer.write("                        else sum = 0.0f;\n")
        buffer.write("                    }\n")
        buffer.write(f"                    {output_name}[n][c][oh][ow] = sum;\n")

    buffer.write("                }\n")
    buffer.write("            }\n")
    buffer.write("        }\n")
    buffer.write("    }\n")
    buffer.write("}\n")


def _write_global_pooling_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                                   attrs: Dict[str, Any], pool_type: str, tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for Global Pooling operators (GlobalAveragePool, GlobalMaxPool, GlobalMinPool).

    This function writes the complete C implementation for a global pooling operator.
    It assumes that the input tensor is multidimensional (e.g., [N, C, H, W]) and that
    the output tensor uses multidimensional indexing as well (e.g., [N, C, 1, 1] for global pooling).
    
    All loop counters and auxiliary variables are declared locally within the function.
    Arrays are accessed using multidimensional indexing, avoiding any casting or linear indexing.
    
    Parameters:
        buffer (StringIO): The buffer where the C code is written.
        func_name (str): Name of the generated C function.
        inputs (List[str]): List of input tensor variable names.
        outputs (List[str]): List of output tensor variable names.
        attrs (Dict[str, Any]): Dictionary of operator attributes.
        pool_type (str): Type of pooling ("average", "max", or "min").
        tensor_shape (Dict[str, Any]): Dictionary mapping tensor names to their shapes.
                                       Expected to have at least one input and one output shape.
    """
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"Global{pool_type.capitalize()}Pool", indent=4)
    
    # Extract input and output shapes.
    # Assumes input tensor shape is [N, C, H, W] and output tensor shape is [N, C, 1, 1]
    input_shape = tensor_shape[inputs[0]]
    output_shape = tensor_shape[outputs[0]]
    buffer.write("    // Dimension constants extracted from tensor shapes\n")
    buffer.write(f"    const int N = {input_shape[0]};\n")
    buffer.write(f"    const int C = {input_shape[1]};\n")
    buffer.write(f"    const int H = {input_shape[2]};\n")
    buffer.write(f"    const int W = {input_shape[3]};\n")
    buffer.write("\n")
    
    # Global pooling implementation for different pool types
    if pool_type.lower() == "average":
        buffer.write("    // Global average pooling: compute the average value over spatial dimensions (H, W) for each channel\n")
        buffer.write("    for (int n = 0; n < N; n++) {\n")
        buffer.write("        for (int c = 0; c < C; c++) {\n")
        buffer.write("            float sum = 0.0f;  // Accumulate sum for current (n, c)\n")
        buffer.write("            for (int h = 0; h < H; h++) {\n")
        buffer.write("                for (int w = 0; w < W; w++) {\n")
        buffer.write(f"                    sum += {inputs[0]}[n][c][h][w];  // Multidimensional access to input\n")
        buffer.write("                }\n")
        buffer.write("            }\n")
        buffer.write("            // Compute average over all spatial elements and store result\n")
        buffer.write(f"            {outputs[0]}[n][c][0][0] = sum / (H * W);\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    elif pool_type.lower() == "max":
        buffer.write("    // Global max pooling: compute the maximum value over spatial dimensions (H, W) for each channel\n")
        buffer.write("    for (int n = 0; n < N; n++) {\n")
        buffer.write("        for (int c = 0; c < C; c++) {\n")
        buffer.write(f"            float max_val = {inputs[0]}[n][c][0][0];  // Initialize max with the first element\n")
        buffer.write("            for (int h = 0; h < H; h++) {\n")
        buffer.write("                for (int w = 0; w < W; w++) {\n")
        buffer.write(f"                    if ({inputs[0]}[n][c][h][w] > max_val) {{\n")
        buffer.write(f"                        max_val = {inputs[0]}[n][c][h][w];\n")
        buffer.write("                    }\n")
        buffer.write("                }\n")
        buffer.write("            }\n")
        buffer.write(f"            {outputs[0]}[n][c][0][0] = max_val;\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    elif pool_type.lower() == "min":
        buffer.write("    // Global min pooling: compute the minimum value over spatial dimensions (H, W) for each channel\n")
        buffer.write("    for (int n = 0; n < N; n++) {\n")
        buffer.write("        for (int c = 0; c < C; c++) {\n")
        buffer.write(f"            float min_val = {inputs[0]}[n][c][0][0];  // Initialize min with the first element\n")
        buffer.write("            for (int h = 0; h < H; h++) {\n")
        buffer.write("                for (int w = 0; w < W; w++) {\n")
        buffer.write(f"                    if ({inputs[0]}[n][c][h][w] < min_val) {{\n")
        buffer.write(f"                        min_val = {inputs[0]}[n][c][h][w];\n")
        buffer.write("                    }\n")
        buffer.write("                }\n")
        buffer.write("            }\n")
        buffer.write(f"            {outputs[0]}[n][c][0][0] = min_val;\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    else:
        raise ValueError(f"Invalid pool_type: {pool_type}")
        buffer.write("    // TODO: Implement global pooling logic for pool types other than average, max, or min\n")
    
    buffer.write("}\n")
