from typing import IO, List, Dict, Any
from io import StringIO
import nodes.helperfunc as helperfunc
import logging


def _write_expand_function(
    buffer: StringIO,
    func_name: str,
    inputs: List[str],
    outputs: List[str],
    attrs: Dict[str, Any],
    tensor_shape: Dict[str, Any]
) -> None:
    """
    Generates C code for the ONNX Expand operator, replicating elements of the
    input tensor across any dimensions of size 1 to match the target shape.
    """
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "Expand", indent=4)

    # Retrieve shapes and ranks
    out_shape = tensor_shape[outputs[0]]
    out_rank = len(out_shape)
    in_shape = tensor_shape[inputs[0]]
    in_rank = len(in_shape)

    indent = "    "
    # Open nested loops over each dimension of the output
    for d, dim in enumerate(out_shape):
        buffer.write(indent + f"for (int i{d} = 0; i{d} < {dim}; i{d}++) {{\n")
        indent += "    "

    # Build an indexing string for the input tensor with broadcasting rules
    def build_input_index() -> str:
        idx = []
        offset = out_rank - in_rank
        for j in range(out_rank):
            if j < offset:
                # input has no corresponding dimension: broadcast scalar
                idx.append("[0]")
            else:
                in_dim = in_shape[j - offset]
                if in_dim == 1:
                    # size-1 dimension: always index 0
                    idx.append("[0]")
                else:
                    # normal dimension: use loop index
                    idx.append(f"[i{j}]")
        return "".join(idx)

    out_index = "".join(f"[i{j}]" for j in range(out_rank))
    in_index  = build_input_index()

    # Emit the expansion assignment
    buffer.write(indent +
        f"{outputs[0]}{out_index} = {inputs[0]}{in_index};\n"
    )

    # Close all the loops
    for _ in range(out_rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")

    # Final closing brace (if your signature wrote one open brace)
    buffer.write("}\n")
