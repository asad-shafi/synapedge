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

def _write_convolution_function(buffer: StringIO, func_name: str, inputs: List[str],outputs: List[str], attrs: Dict[str, Any],tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for 2D convolution with complete variable declarations."""
    # Validate input/output counts
    min_inputs = 2  # Input and kernel
    has_bias = len(inputs) >= 3
    if len(inputs) < min_inputs or len(outputs) != 1:
        raise ValueError(f"#error Invalid number of inputs/outputs in conv node {func_name}")      
        buffer.write("#error Invalid number of inputs/outputs\n")
        return

    input_name = inputs[0]
    kernel_name = inputs[1]
    output_name = outputs[0]
    bias_name = inputs[2] if has_bias else None

    # Tensor shape extraction and validation
    try:
        input_dims = tensor_shape[input_name]
        kernel_dims = tensor_shape[kernel_name]
        output_dims = tensor_shape[output_name]
        N, C_in, H_in, W_in = input_dims
        K, C_in_kernel, K_h, K_w = kernel_dims
        N_out, C_out, H_out, W_out = output_dims
    except (KeyError, ValueError) as e:
        raise ValueError(f"#error conv Invalid tensor dimensions: {str(e)}") 
        buffer.write(f"#error Invalid tensor dimensions: {str(e)}\n")
        return

    # Dimension validation
    if C_in != C_in_kernel:
        raise ValueError(f"#error conv Input channel mismatch: {C_in} vs {C_in_kernel}") 
        buffer.write(f"#error Input channel mismatch: {C_in} vs {C_in_kernel}\n")
        return
    if C_out != K:
        raise ValueError(f"#error conv Output channel mismatch: {C_out} vs {K}") 
        buffer.write(f"#error Output channel mismatch: {C_out} vs {K}\n")
        return
    if N != N_out:
        raise ValueError(f"#error conv Batch size mismatch: {N} vs {N_out}") 
        buffer.write(f"#error Batch size mismatch: {N} vs {N_out}\n")
        return

    # Bias validation
    if has_bias:
        try:
            bias_dims = tensor_shape[bias_name]
            if len(bias_dims) != 1 or bias_dims[0] != K:
                raise ValueError
        except (KeyError, ValueError):
            raise ValueError(f"#error conv Bias must be 1D tensor with {K} elements") 
            buffer.write(f"#error Bias must be 1D tensor with {K} elements\n")
            return

    # Attribute handling with validation
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', None)
    dilations = attrs.get('dilations', [1, 1])

    # Calculate padding if not provided
    if pads is None:
        # Auto-calculate padding to maintain output shape
        logger.warning(f"No padding found in node ({func_name}), Auto-calculated padding applied")
        buffer.write("  /*No padding found in node, Auto-calculated padding applied */ \n")
        pad_h_total = ((H_out - 1)*strides[0] + dilations[0]*(K_h - 1) + 1 - H_in)
        pad_w_total = ((W_out - 1)*strides[1] + dilations[1]*(K_w - 1) + 1 - W_in)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
        pads = [pad_h, pad_w, pad_h_total - pad_h, pad_w_total - pad_w]
    elif len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]  # Symmetric padding

    # Validate attribute dimensions
    if len(strides) != 2 or any(s <= 0 for s in strides):
        raise ValueError(f"#error conv ({func_name}) Invalid strides") 
        buffer.write("#error Invalid strides\n")
        return
    if len(pads) != 4 or any(p < 0 for p in pads):
        raise ValueError(f"#error conv ({func_name}) Invalid padding") 
        buffer.write("#error Invalid padding\n")
        return
    if len(dilations) != 2 or any(d <= 0 for d in dilations):
        raise ValueError(f"#error conv ({func_name}) Invalid dilations") 
        buffer.write("#error Invalid dilations\n")
        return

    # Write function signature with flat pointers
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)

    # Write dimension constants
    buffer.write("    //Ddimension constants\n")
    buffer.write(f"    const int N = {N}, C_in = {C_in},  H_in = {H_in},  W_in = {W_in};\n")
    buffer.write(f"    const int K = {K}, K_h = {K_h},  K_w = {K_w};\n")
    buffer.write(f"    const int H_out = {H_out},W_out = {W_out};\n")

    # Write convolution parameters
    buffer.write("    // Convolution parameters\n")
    buffer.write(f"    const int stride_h = {strides[0]}, stride_w = {strides[1]};\n")
    buffer.write(f"    const int pad_t = {pads[0]}, pad_b = {pads[2]},  pad_l = {pads[1]},  pad_r = {pads[3]};\n")
    buffer.write(f"    const int dilation_h = {dilations[0]}, dilation_w = {dilations[1]};\n")

    # Pointer declarations
    buffer.write("    // Tensor pointers\n")
    buffer.write(f"    float* input = (float *){input_name};\n")
    buffer.write(f"    float* kernel = (float *){kernel_name};\n")
    buffer.write(f"    float* output = (float *){output_name};\n")
    if has_bias:
        buffer.write(f"    float* bias = (float *){bias_name};\n\n")

    # Nested convolution loops
    buffer.write("    // Convolution computation\n")
    buffer.write("    for (int n = 0; n < N; ++n) {\n")
    buffer.write("        for (int k = 0; k < K; ++k) {\n")
    buffer.write("            for (int h = 0; h < H_out; ++h) {\n")
    buffer.write("                for (int w = 0; w < W_out; ++w) {\n")
    buffer.write("                    float sum = 0.0f;\n")

    # Convolution kernel computation
    buffer.write("                    for (int c = 0; c < C_in; ++c) {\n")
    buffer.write("                        for (int kh = 0; kh < K_h; ++kh) {\n")
    buffer.write("                            for (int kw = 0; kw < K_w; ++kw) {\n")
    buffer.write("                                const int h_in = h * stride_h - pad_t + kh * dilation_h;\n")
    buffer.write("                                const int w_in = w * stride_w - pad_l + kw * dilation_w;\n")
    buffer.write("                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {\n")
    buffer.write("                                    const int in_idx = ((n * C_in + c) * H_in + h_in) * W_in + w_in;\n")
    buffer.write("                                    const int ker_idx = ((k * C_in + c) * K_h + kh) * K_w + kw;\n")
    buffer.write("                                    sum += input[in_idx] * kernel[ker_idx];\n")
    buffer.write("                                }\n")
    buffer.write("                            }\n")
    buffer.write("                        }\n")
    buffer.write("                    }\n")

    # Bias addition
    if has_bias:
        buffer.write("                    sum += bias[k];\n")
    else:
        buffer.write("                    sum += 0.0f;  // No bias\n")

    # Store result
    buffer.write("                    const int out_idx = ((n * K + k) * H_out + h) * W_out + w;\n")
    buffer.write("                    output[out_idx] = sum;\n")

    # Close loops
    buffer.write("                }\n")
    buffer.write("            }\n")
    buffer.write("        }\n")
    buffer.write("    }\n")
    buffer.write("}\n")