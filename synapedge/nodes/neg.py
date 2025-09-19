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

def _write_neg_function(buffer, func_name, inputs, outputs, attrs, tensor_shape):
    """Generates C code for the Neg operator (unary negation)."""
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "neg", indent=4)

    out_shape = tensor_shape[outputs[0]]
    out_rank = len(out_shape)
    in_shape = tensor_shape[inputs[0]]

    # Sanity check: ONNX enforces input/output shapes must match for Neg
    assert in_shape == out_shape, "Neg operator requires input and output shapes to match."

    indent = "    "
    # Generate nested loops for all dimensions
    for d, dim in enumerate(out_shape):
        buffer.write(f"{indent}for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # Build index string (e.g., [i0][i1][i2]...)
    index_str = "".join([f"[i{d}]" for d in range(out_rank)])
    
    # Write the negation operation
    buffer.write(f"{indent}{outputs[0]}{index_str} = -{inputs[0]}{index_str};\n")

    # Close all loops
    for _ in range(out_rank):
        indent = indent[:-4]
        buffer.write(f"{indent}}}\n")

    buffer.write("}\n")
