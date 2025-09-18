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

def _write_sin_function(buffer: StringIO,
                        func_name: str,
                        inputs: List[str],
                        outputs: List[str],
                        attrs: Dict[str, Any],
                        tensor_shape: Dict[str, Any]) -> None:
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "sin", indent=4)

    shape = tensor_shape[inputs[0]]
    rank  = len(shape)
    indent = "    "

    # one loop per dimension
    for d, dim in enumerate(shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # build index string ["[i0][i1]…"]
    idx = ''.join(f"[i{j}]" for j in range(rank))
    buffer.write(indent +
                 f"{outputs[0]}{idx} = sin({inputs[0]}{idx});\n")

    # close loops
    for _ in range(rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")
    buffer.write("}\n")

def _write_cos_function(buffer: StringIO,
                        func_name: str,
                        inputs: List[str],
                        outputs: List[str],
                        attrs: Dict[str, Any],
                        tensor_shape: Dict[str, Any]) -> None:
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "sin", indent=4)

    shape = tensor_shape[inputs[0]]
    rank  = len(shape)
    indent = "    "

    # one loop per dimension
    for d, dim in enumerate(shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # build index string ["[i0][i1]…"]
    idx = ''.join(f"[i{j}]" for j in range(rank))
    buffer.write(indent +
                 f"{outputs[0]}{idx} = cos({inputs[0]}{idx});\n")

    # close loops
    for _ in range(rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")
    buffer.write("}\n")