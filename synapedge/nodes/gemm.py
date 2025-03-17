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

def _write_gemm_function(functions: Dict[str, Any], buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                         attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates complete C code for the GEMM operator with matrix multiplication implementation.
    
    Args:
        buffer: StringIO buffer to write the code to.
        func_name: Name of the function to generate.
        inputs: List of input tensor names.
        outputs: List of output tensor names.
        attrs: Dictionary of operator attributes.
        tensor_shape: Dictionary mapping tensor names to their shapes.

    Computes:
        output = alpha * op(A) * op(B) + beta * bias,
    where op(A) and op(B) may be transposed according to transA and transB.

    If the output tensor shape is not provided in tensor_shape, it is calculated as [M, N],
    and a warning is printed to the user. also update shape in functions, so sub sequent functions have correct shapes. i am not sure wheather generatr error or calculate, but for now i pick up calculation

    Parameters:
        buffer: StringIO object to which the generated C code is written.
        func_name: The name of the generated C function.
        inputs: List of input variable names. Expected order is:
                [A (e.g., "input"), B (e.g., "fc1_weight"), bias (e.g., "fc1_bias")].
        outputs: List of output variable names (e.g., ["_fc1_Gemm_output_0"]).
        attrs: Dictionary of operator attributes. Recognized keys are:
               'alpha' (default 1.0), 'beta' (default 1.0),
               'transA' (default 0), and 'transB' (default 0).
        tensor_shape: Dictionary mapping tensor names to their shapes.
    """
    # Retrieve attributes with defaults.
    alpha = attrs.get('alpha', 1.0)
    beta = attrs.get('beta', 1.0)
    transA = int(attrs.get('transA', 0))
    transB = int(attrs.get('transB', 0))

    # Determine dimensions for matrix A (first input).
    if transA == 0:
        M = tensor_shape[inputs[0]][0]
        K = tensor_shape[inputs[0]][1]
    else:
        M = tensor_shape[inputs[0]][1]
        K = tensor_shape[inputs[0]][0]

    # Determine dimensions for matrix B (second input).
    if transB == 0:
        K2 = tensor_shape[inputs[1]][0]
        N = tensor_shape[inputs[1]][1]
    else:
        K2 = tensor_shape[inputs[1]][1]
        N = tensor_shape[inputs[1]][0]

    if K != K2:
        raise ValueError("Inner dimensions of A and B do not match for GEMM operation.")

    # Check if output size is provided; if not, compute and warn the user.
    if outputs[0] not in tensor_shape or not tensor_shape[outputs[0]]:
        computed_output_shape = [M, N]
        #print(functions)
        #print(functions['outputs'] )
        print(f"Warning: Output size for {outputs[0]} not provided. Using computed output size {computed_output_shape}.")
        tensor_shape[(outputs[0])] = computed_output_shape
        # Update 'outputs[0]' in functions
        for i in range(len(functions)):  # Iterate using index
            func = functions[i]  # Get function dictionary by index
            #print(outputs[0])
            #print(func)
            if helperfunc._sanitize_name(func['name']) == func_name:
                #print(functions[i])
                #print("----------")
                functions[i]['intermediate_tensors_shape'][outputs[0]] = computed_output_shape  # Update using index
                #print(functions[i])
                # Update output shape of the producer node
            for ins in func['inputs']:
              if (helperfunc._sanitize_name(ins)) == outputs[0]:
                #print(ins)
                functions[i]['intermediate_tensors_shape'][helperfunc._sanitize_name(ins)] = computed_output_shape # Update inputs shapes of the consumer node with producer node output shape.
                #break  # Exit the loop after updating
    #print(tensor_shape)
    # Determine how to index the bias input.
    if len(tensor_shape[inputs[2]]) == 1:
        bias_expr = f"{inputs[2]}[j]"
    else:
        bias_expr = f"{inputs[2]}[i][j]"

    # Write the function signature and a comment with operator parameters.
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, f"GEMM: alpha={alpha}, beta={beta}, transA={transA}, transB={transB}", indent=4)

    # Write dimension definitions.
    buffer.write(f"    int M = {M};\n")
    buffer.write(f"    int N = {N};\n")
    buffer.write(f"    int K = {K};\n")
    buffer.write("    int i, j, k;\n\n")

    # Generate the GEMM loops based on the transposition flags.
    if transA == 0 and transB == 0:
        # A: [M x K], B: [K x N]
        buffer.write("    for (i = 0; i < M; i++) {\n")
        buffer.write("        for (j = 0; j < N; j++) {\n")
        buffer.write(f"            {outputs[0]}[i][j] = {beta} * {bias_expr};\n")
        buffer.write("            for (k = 0; k < K; k++) {\n")
        buffer.write(f"                {outputs[0]}[i][j] += {alpha} * {inputs[0]}[i][k] * {inputs[1]}[k][j];\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    elif transA == 0 and transB == 1:
        # A: [M x K], B: originally [N x K] so op(B) is B transposed -> [K x N]
        buffer.write("    for (i = 0; i < M; i++) {\n")
        buffer.write("        for (j = 0; j < N; j++) {\n")
        buffer.write(f"            {outputs[0]}[i][j] = {beta} * {bias_expr};\n")
        buffer.write("            for (k = 0; k < K; k++) {\n")
        buffer.write(f"                {outputs[0]}[i][j] += {alpha} * {inputs[0]}[i][k] * {inputs[1]}[j][k];\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    elif transA == 1 and transB == 0:
        # A: originally [K x M] so op(A) is A transposed -> [M x K], B: [K x N]
        buffer.write("    for (i = 0; i < M; i++) {\n")
        buffer.write("        for (j = 0; j < N; j++) {\n")
        buffer.write(f"            {outputs[0]}[i][j] = {beta} * {bias_expr};\n")
        buffer.write("            for (k = 0; k < K; k++) {\n")
        buffer.write(f"                {outputs[0]}[i][j] += {alpha} * {inputs[0]}[k][i] * {inputs[1]}[k][j];\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    elif transA == 1 and transB == 1:
        # A: originally [K x M] so op(A) is A transposed -> [M x K],
        # B: originally [N x K] so op(B) is B transposed -> [K x N]
        buffer.write("    for (i = 0; i < M; i++) {\n")
        buffer.write("        for (j = 0; j < N; j++) {\n")
        buffer.write(f"            {outputs[0]}[i][j] = {beta} * {bias_expr};\n")
        buffer.write("            for (k = 0; k < K; k++) {\n")
        buffer.write(f"                {outputs[0]}[i][j] += {alpha} * {inputs[0]}[k][i] * {inputs[1]}[j][k];\n")
        buffer.write("            }\n")
        buffer.write("        }\n")
        buffer.write("    }\n")
    else:
        buffer.write("    // Unsupported transposition configuration\n")

    buffer.write("}\n")
