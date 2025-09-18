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

def _write_scatternd_function(
    buffer: StringIO,
    func_name: str,
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
    tensor_shape: Dict[str, Any]
) -> None:
    "Generates C code for the ScatterND operator."
    # function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "ScatterND", indent=4)

    data_shape = tensor_shape[inputs[0]]
    idx_shape = tensor_shape[inputs[1]]
    upd_shape = tensor_shape[inputs[2]]
    out_rank = len(data_shape)

    # 1) First, copy data into output
    buffer.write("    // Copy input data to output\n")
    for d, dim in enumerate(data_shape):
        buffer.write("    " * (d+1) + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
    idx = ''.join(f"[i{d}]" for d in range(out_rank))
    buffer.write("    " * (out_rank+1) +
                 f"{outputs[0]}{idx} = {inputs[0]}{idx};\n")
    for d in reversed(range(out_rank)):
        buffer.write("    " * (d+1) + "}\n")
    buffer.write("\n")

    # 2) Now apply updates
    # index tensor has rank n+1, with last dim = K
    n = len(idx_shape) - 1
    K = idx_shape[-1]

    # loops over the first n dims of indices/updates
    buffer.write("    // Scatter updates\n")
    for d in range(n):
        buffer.write("    " * (d+1) + f"for (int i_idx{d} = 0; i_idx{d} < {idx_shape[d]}; i_idx{d}++) {{\n")

    # Build C expressions to read the index tuple
    # e.g., int idx0 = indices[i_idx0][i_idx1]...[0];
    for k in range(K):
        idx_access = ''.join(f"[i_idx{d}]" for d in range(n)) + f"[{k}]"
        buffer.write("    " * (n+1) +
                     f"int idx{k} = {inputs[1]}{idx_access};\n")

    # loops over the remaining 'slice' dims of updates/data beyond K
    slice_dims = data_shape[K:]
    slice_rank = len(slice_dims)
    for s in range(slice_rank):
        dim = slice_dims[s]
        buffer.write("    " * (n+1+s) + f"for (int i_s{s} = 0; i_s{s} < {dim}; i_s{s}}}++) {{\n")

    # Build output index: [idx0][idx1]...[idxK-1][i_s0]...[i_s{slice_rank-1}]
    out_index = ''.join(f"[idx{k}]" for k in range(K))
    out_index += ''.join(f"[i_s{s}]" for s in range(slice_rank))

    # Build updates index: same loops as above, but indices loops first
    upd_index = ''.join(f"[i_idx{d}]" for d in range(n))
    upd_index += ''.join(f"[i_s{s}]" for s in range(slice_rank))

    buffer.write("    " * (n+1+slice_rank) +
                 f"{outputs[0]}{out_index} = {inputs[2]}{upd_index};\n")

    # Close slice loops
    for s in reversed(range(slice_rank)):
        buffer.write("    " * (n+1+s) + "}\n")
    # Close index loops
    for d in reversed(range(n)):
        buffer.write("    " * (d+1) + "}\n")

    buffer.write("}\n")
