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
from platform import node
from typing import IO, List, Dict, Any
from io import StringIO

from numpy import pad
import nodes.helperfunc as helperfunc
import logging

logger = logging.getLogger(__name__)

def conv2d_output_shape(input_dims, kernel_dims, strides=[1, 1], pads=None, dilations=[1, 1]):
    """
    Calculate output shape of 2D convolution.

    Parameters:
        input_dims   : (N, C_in, H_in, W_in)
        kernel_dims  : (C_out, C_in, K_h, K_w)
        strides      : [stride_h, stride_w]
        pads         : [pad_top, pad_left, pad_bottom, pad_right] or None
        dilations    : [dilation_h, dilation_w]
    Returns:
        (N, C_out, H_out, W_out)
    """
    N, C_in, H_in, W_in = input_dims
    C_out, C_in_k, K_h, K_w = kernel_dims

    assert C_in == C_in_k, "Input channels must match kernel's C_in"

    S_h, S_w = strides
    D_h, D_w = dilations

    if pads is None:
        pad_top = pad_bottom = pad_left = pad_right = 0
    else:
        pad_top, pad_left, pad_bottom, pad_right = pads

    H_out = ((H_in + pad_top + pad_bottom - D_h * (K_h - 1) - 1) // S_h) + 1
    W_out = ((W_in + pad_left + pad_right - D_w * (K_w - 1) - 1) // S_w) + 1

    return [N, C_out, H_out, W_out]


def _write_qlinearconvolution_function(functions: Dict[str, Any], buffer: StringIO, func_name: str,inputs: List[str], outputs: List[str], attrs: Dict[str, Any], 
                                      tensor_shape: Dict[str, Any]) -> None:
    # Attribute handling
    strides = attrs.get('strides', [1, 1])
    pads = attrs.get('pads', None)
    dilations = attrs.get('dilations', [1, 1])
    group = attrs.get('group', 1)
    
    # Validate input/output counts
    min_inputs = 8  # Required inputs: X, x_scale, x_zp, W, w_scale, w_zp, y_scale, y_zp
    has_bias = len(inputs) >= 9
    if len(inputs) < min_inputs or len(outputs) != 1:
        raise ValueError("QLinearConv requires at least 8 inputs and exactly 1 output")
    
    # Extract input/output names
    x_name, x_scale, x_zero_point = inputs[0], inputs[1], inputs[2]
    w_name, w_scale, w_zero_point = inputs[3], inputs[4], inputs[5]
    w_name_dtype = tensor_shape.get(w_name+"_dtype", [])
    w_zero_point_dtype = tensor_shape.get(w_zero_point+"_dtype", [])

    if w_name_dtype == [] or w_zero_point_dtype == []:
        raise ValueError(f"qlinearconv weigth data types are not given in {func_name}")

    y_scale, y_zero_point = inputs[6], inputs[7]
    bias_name = inputs[8] if has_bias else None
    y_name = outputs[0]

    # Get tensor shapes
    input_dims = tensor_shape.get(x_name, [])
    kernel_dims = tensor_shape.get(w_name, [])
    output_dims = tensor_shape.get(y_name, [])

    
    
    # Calculate output dimensions if not provided
    if not output_dims:
        output_dims = conv2d_output_shape(input_dims=input_dims,kernel_dims=kernel_dims,strides=strides,pads=pads)  
        #print(output_dims)
        print(f"Warning: Output size for {outputs[0]} not provided. Using  input size {inputs[0]}.")
        tensor_shape[outputs[0]] = output_dims # input A
        # Update 'outputs[0]' in functions
        for i in range(len(functions)):  # Iterate using index
            func = functions[i]  # Get function dictionary by index
            if helperfunc._sanitize_name(func['name']) == func_name: # Update output shape of the producer node (or curent node)
                functions[i]['intermediate_tensors_shape'][outputs[0]] = output_dims  # Update using index
                functions[i]['intermediate_tensors_shape'][f"{outputs[0]}_dtype"] = tensor_shape.get(x_name+"_dtype", "uint8_t")

                dims = ''.join([f"[{dim}]" for dim in output_dims])
                na = f"tensor_{y_name}"
                functions[i]['computed_shape'] = f"{tensor_shape.get(y_name+'_dtype', 'uint8_t')} {helperfunc._sanitize_name(na)}{dims}" 
            for ins in func['inputs']:
              if (helperfunc._sanitize_name(ins)) == y_name: # Update inputs shape of the consumers node in graph
                functions[i]['intermediate_tensors_shape'][helperfunc._sanitize_name(ins)] = output_dims # Update inputs shapes of the consumer node with producer node output shape.
                functions[i]['intermediate_tensors_shape'][f"{helperfunc._sanitize_name(ins)}_dtype"] = tensor_shape.get(x_name+"_dtype", "uint8_t") # Update inputs data type of the consumer node with producer node output shape.
   
                #break  # Exit the loop after updating
    # Tensor shape extraction and validation
    # Extract dimensions
    tensors_dtype = tensor_shape.get(y_name+"_dtype", "uint8_t")
    print(tensors_dtype)
    print(tensor_shape.get(x_name+"_dtype", "-'-"))

    try:
        N, C_in, H_in, W_in = input_dims
        K, C_in_kernel, K_h, K_w = kernel_dims
        N_out, C_out, H_out, W_out = output_dims
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid tensor dimensions: {str(e)}")

    # Check if per-channel quantization (w_scale is an array)
    w_scale_dims = tensor_shape.get(w_scale, [])
    w_zp_dims = tensor_shape.get(w_zero_point, [])
    is_per_channel = (len(w_scale_dims) == 1 and w_scale_dims[0] == K and
                     len(w_zp_dims) == 1 and w_zp_dims[0] == K)
    
    # Validate dimensions
    if C_in != C_in_kernel:
        raise ValueError(f"Input channel mismatch: {C_in} vs {C_in_kernel}")
    if C_out != K:
        raise ValueError(f"Output channel mismatch: {C_out} vs {K}")
    if N != N_out:
        raise ValueError(f"Batch size mismatch: {N} vs {N_out}")
    if group != 1:
        raise NotImplementedError("Grouped convolution not yet implemented")

    # Calculate padding if not provided
    if pads is None:
        pad_h_total = ((H_out - 1) * strides[0] + dilations[0] * (K_h - 1) + 1 - H_in)
        pad_w_total = ((W_out - 1) * strides[1] + dilations[1] * (K_w - 1) + 1 - W_in)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
        pads = [pad_h, pad_w, pad_h_total - pad_h, pad_w_total - pad_w]
    elif len(pads) == 2:
        pads = [pads[0], pads[1], pads[0], pads[1]]

    # Write function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    
    # Write dimensions and parameters
    buffer.write("    // Dimensions\n")
    buffer.write(f"    const int N = {N}, C_in = {C_in}, H_in = {H_in}, W_in = {W_in};\n")
    buffer.write(f"    const int K = {K}, K_h = {K_h}, K_w = {K_w};\n")
    buffer.write(f"    const int H_out = {H_out}, W_out = {W_out};\n")
    buffer.write("    // Convolution parameters\n")
    buffer.write(f"    const int stride_h = {strides[0]}, stride_w = {strides[1]};\n")
    buffer.write(f"    const int pad_t = {pads[0]}, pad_b = {pads[2]}, pad_l = {pads[1]}, pad_r = {pads[3]};\n")
    buffer.write(f"    const int dilation_h = {dilations[0]}, dilation_w = {dilations[1]};\n")
    buffer.write(f"    const int group = {group};\n\n")
    
    # Write quantization parameters
    buffer.write("    // Quantization parameters\n")
    buffer.write(f"    uint8_t x_zp = {x_zero_point}[0];\n")
    buffer.write(f"    float x_scale_val = {x_scale}[0];\n")
    buffer.write(f"    uint8_t y_zp = {y_zero_point}[0];\n")
    buffer.write(f"    float y_scale_val = {y_scale}[0];\n")
    
    if is_per_channel:
        buffer.write("    // Per-channel quantization for weights\n")
        buffer.write(f"    float* w_scale_arr = (float*){w_scale};\n")
        buffer.write(f"    {w_zero_point_dtype}* w_zp_arr = ({w_zero_point_dtype}*){w_zero_point};\n")
    else:
        buffer.write("    // Per-tensor quantization for weights\n")
        buffer.write(f"    float w_scale_val = *((float*){w_scale});\n")
        buffer.write(f"    {w_zero_point_dtype} w_zp = *(({w_zero_point_dtype}*){w_zero_point});\n")
    
    if has_bias:
        buffer.write(f"    int32_t* bias = (int32_t*){bias_name};\n\n")
    else:
        buffer.write("    // No bias\n\n")
    
    # Main convolution loop
    buffer.write("    // Convolution computation\n")
    buffer.write("    for (int n = 0; n < N; ++n) {\n")
    buffer.write("        for (int k = 0; k < K; ++k) {\n")
    
    # Per-channel multiplier calculation
    if is_per_channel:
        buffer.write("            float w_scale_k = w_scale_arr[k];\n")
        buffer.write(f"            {w_zero_point_dtype} w_zp_k = w_zp_arr[k];\n")
        buffer.write("            double multiplier = (double)x_scale_val * (double)w_scale_k / (double)y_scale_val;\n")
    else:
        buffer.write("            double multiplier = (double)x_scale_val * (double)w_scale_val / (double)y_scale_val;\n")
    
    buffer.write("            for (int h = 0; h < H_out; ++h) {\n")
    buffer.write("                for (int w = 0; w < W_out; ++w) {\n")
    buffer.write("                    int32_t acc = 0;\n")
    
    # Initialize with bias if present
    if has_bias:
        buffer.write("                    acc = bias[k];\n")
    
   # buffer.write("                    for (int c = 0; c < C_in; ++c) {\n")
    buffer.write("                     for (int kh = 0; kh < K_h; ++kh) {\n")
    buffer.write("                         int h_in = h * stride_h - pad_t + kh * dilation_h;\n")
    buffer.write("                         if (h_in < 0 || h_in >= H_in) continue;\n")
    buffer.write("                         for (int kw = 0; kw < K_w; ++kw) {\n")
    buffer.write("                             int w_in = w * stride_w - pad_l + kw * dilation_w;\n")
    buffer.write("                             if (w_in < 0 || w_in >= W_in) continue;\n")
    buffer.write("                             for (int c = 0; c < C_in; ++c) {\n")
    
    # Tensor access with bounds checking
    buffer.write(f"                                uint8_t x_val = {x_name}[n][c][h_in][w_in];\n")
    
    # Weight access - handle per-channel vs per-tensor
    if is_per_channel:
        buffer.write(f"                                {w_name_dtype} w_val = {w_name}[k][c][kh][kw];\n")
        buffer.write("                             acc += (int32_t)(x_val - x_zp) * (int32_t)(w_val - w_zp_k);\n")
    else:
        buffer.write(f"                                {w_name_dtype} w_val = {w_name}[k][c][kh][kw];\n")
        buffer.write("                             acc += (int32_t)(x_val - x_zp) * (int32_t)(w_val - w_zp);\n")
    
    buffer.write("                            }\n")  # kw loop
    buffer.write("                        }\n")  # kh loop
    buffer.write("                    }\n")  # c loop
    
    # Requantization
    buffer.write("                    // Requantization\n")
    buffer.write("                    float scaled_val = (float)acc * (float)multiplier;\n")
    buffer.write("                    int32_t result_val;\n")
    buffer.write("                    if (scaled_val >= 0) {\n")
    buffer.write("                        result_val = (int32_t)(scaled_val + 0.5f);\n")
    buffer.write("                    } else {\n")
    buffer.write("                        result_val = (int32_t)(scaled_val - 0.5f);\n")
    buffer.write("                    }\n")
    buffer.write("                    result_val += y_zp;\n")
    
    # Clamping
    buffer.write("                    // Clamping\n")
    buffer.write("                    if (result_val < 0) result_val = 0;\n")
    buffer.write("                    if (result_val > 255) result_val = 255;\n")
    
    # Store result
    buffer.write(f"                    {y_name}[n][k][h][w] = (uint8_t)result_val;\n")
    
    buffer.write("                }\n")  # w loop
    buffer.write("            }\n")  # h loop
    buffer.write("        }\n")  # k loop
    buffer.write("    }\n")  # n loop
    buffer.write("}\n")