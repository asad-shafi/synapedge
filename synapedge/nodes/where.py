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
'''def _write_where_function(buffer: StringIO,
                          func_name: str,
                          inputs: List[str],
                          outputs: List[str],
                          attrs: Dict[str, Any],
                          tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for the ONNX 'Where' operator:
      output[i…] = condition[i…] ? X[i…] : Y[i…]
    with broadcasting support.
    """
    # function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "where", indent=4)

    # retrieve shapes and ranks
    out_shape = tensor_shape[outputs[0]]
    out_rank  = len(out_shape)
    cond_shape = tensor_shape[inputs[0]]
    x_shape    = tensor_shape[inputs[1]]
    y_shape    = tensor_shape[inputs[2]]
    cond_rank = len(cond_shape)
    x_rank    = len(x_shape)
    y_rank    = len(y_shape)

    indent = "    "
    # open loops for each output dimension
    for d, dim in enumerate(out_shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # helper to build a broadcast-aware index string
    def build_index(input_rank: int) -> str:
        if input_rank == out_rank:
            # same rank → use all i0…i{out_rank-1}
            return ''.join(f"[i{j}]" for j in range(out_rank))
        elif input_rank == 1:
            # scalar broadcast
            return "[0]"
        else:
            # align trailing dims
            offset = out_rank - input_rank
            return ''.join(f"[i{j}]" for j in range(offset, out_rank))

    # build the three index expressions
    out_idx  = ''.join(f"[i{j}]" for j in range(out_rank))
    cond_idx = build_index(cond_rank)
    x_idx    = build_index(x_rank)
    y_idx    = build_index(y_rank)

    # emit the core ternary
    buffer.write(indent +
        f"{outputs[0]}{out_idx} = {inputs[0]}{cond_idx} ? {inputs[1]}{x_idx} : {inputs[2]}{y_idx};\n"
    )

    # close loops
    for _ in range(out_rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")
    
    buffer.write(indent + "}\n")
'''

def _write_where_function(buffer: StringIO,
                          func_name: str,
                          inputs: List[str],
                          outputs: List[str],
                          attrs: Dict[str, Any],
                          tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for the ONNX 'Where' operator with correct C-style indexing
    that matches each input's declared rank (handles broadcasting right-aligned).
    """
    # function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "where", indent=4)

    # retrieve shapes and ranks
    out_shape = tensor_shape[outputs[0]]
    out_rank = len(out_shape)

    cond_shape = tensor_shape[inputs[0]]
    x_shape = tensor_shape[inputs[1]]
    y_shape = tensor_shape[inputs[2]]

    # validate shapes exist
    if out_rank == 0:
        raise ValueError(f"Output shape for {outputs[0]} must be non-empty")

    # pad helper (keeps original shape too)
    def _pad_shape(shape: List[int]) -> List[int]:
        if len(shape) > out_rank:
            raise ValueError(f"Input shape {shape} has rank > output rank {out_rank}")
        return [1] * (out_rank - len(shape)) + list(shape)

    cond_padded = _pad_shape(cond_shape)
    x_padded = _pad_shape(x_shape)
    y_padded = _pad_shape(y_shape)

    # NEW: build_index returns an index string matching the *declared* rank
    # (i.e., len(original_shape) bracketed indices), mapping those indices
    # to the RIGHTMOST dimensions of the output.
    def build_index_from_original(original_shape: List[int], padded_shape: List[int]) -> str:
        """
        original_shape: the declared shape for the input (e.g. [1] or [1,64,64])
        padded_shape: padded to length out_rank (e.g. [1,1,1,1] or [1,1,64,64])
        Returns something like '[i0][0][i2]' but with exactly len(original_shape) parts,
        corresponding to the rightmost original axes.
        """
        orig_rank = len(original_shape)
        # if the input is a true scalar / single-element (product == 1), access with single [0]
        prod = 1
        for s in original_shape:
            prod *= s
        if prod == 1:
            return "[0]"

        # indices correspond to the rightmost orig_rank positions of the padded_shape
        start = out_rank - orig_rank  # index in padded_shape where original dims begin
        parts = []
        for j in range(start, out_rank):
            src_dim = padded_shape[j]
            tgt_dim = out_shape[j]
            if src_dim == tgt_dim:
                parts.append(f"i{j}")   # use the output loop variable for that axis
            elif src_dim == 1:
                parts.append("0")       # broadcasted axis -> index 0
            else:
                raise ValueError(f"Cannot broadcast dimension {j}: input dim {src_dim} -> output dim {tgt_dim}")
        return ''.join(f"[{p}]" for p in parts)

    # open loops for each output dimension (these set i0..iN)
    indent = "    "
    for d, dim in enumerate(out_shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # build index expressions: note we pass original shapes so the index matches declared rank
    out_idx = ''.join(f"[i{j}]" for j in range(out_rank))
    cond_idx = build_index_from_original(cond_shape, cond_padded)
    x_idx = build_index_from_original(x_shape, x_padded)
    y_idx = build_index_from_original(y_shape, y_padded)

    # emit the core ternary
    buffer.write(indent +
        f"{outputs[0]}{out_idx} = {inputs[0]}{cond_idx} ? {inputs[1]}{x_idx} : {inputs[2]}{y_idx};\n"
    )

    # close loops
    for _ in range(out_rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")

    # close function
    buffer.write("}\n")
