from typing import IO, List, Dict, Any
from io import StringIO
import nodes.helperfunc as helperfunc

def _write_gather_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],
                           attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for Gather operator (robust handling of indices scalar vs vector)."""

    axis = attrs.get('axis', 0)
    data_name = inputs[0]
    indices_name = inputs[1]
    output_name = outputs[0]

    input_shape = tensor_shape.get(data_name, [])
    indices_shape = tensor_shape.get(indices_name, [])
    output_shape = tensor_shape.get(output_name, [])

    if not input_shape:
        raise ValueError(f"Input shape for {data_name} not provided")
    if not indices_shape:
        raise ValueError(f"Indices shape for {indices_name} not provided")
    if not output_shape:
        raise ValueError(f"Output shape for {output_name} not provided")

    input_rank = len(input_shape)
    indices_rank = len(indices_shape)

    # normalize negative axis
    if axis < 0:
        axis += input_rank

    # expected output by ONNX: input[:axis] + indices_shape + input[axis+1:]
    expected_output = input_shape[:axis] + indices_shape + input_shape[axis+1:]

    # Decide whether indices are "treated as scalar" (common if indices_shape == [1] but output omitted it)
    indices_treated_as_scalar = False
    if expected_output != output_shape:
        # If indices is shape [1] and output == input without the indices dim, treat indices as scalar
        collapsed_output_if_scalar = input_shape[:axis] + input_shape[axis+1:]
        if indices_shape == [1] and output_shape == collapsed_output_if_scalar:
            indices_treated_as_scalar = True
        else:
            # Emit an informative error in generated C (or raise here)
            raise ValueError(f"Output shape mismatch. Expected {expected_output} but got {output_shape}."
                             " If you intended indices to be a scalar, set indices shape to [] or"
                             " ensure provided output shape includes the indices dimension.")

    # Write signature + comment
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    buffer.write(f"    // Gather along axis={axis}\n")
    buffer.write(f"    // Input shape: {input_shape}, Indices shape: {indices_shape}, Output shape: {output_shape}\n")

    # write nested loops for all output dims
    output_rank = len(output_shape)
    for i in range(output_rank):
        buffer.write("    " * (i + 1) + f"for (int d{i} = 0; d{i} < {output_shape[i]}; d{i}++) {{\n")

    body_indent = "    " * (output_rank + 1)

    # build index access expression
    if indices_treated_as_scalar:
        buffer.write(body_indent + f"int64_t index_val = {indices_name}[0];\n")
    else:
        # indices occupy output positions [axis, axis+indices_rank-1]
        indices_access = ''.join([f"[d{axis + j}]" for j in range(indices_rank)])
        buffer.write(body_indent + f"int64_t index_val = {indices_name}{indices_access};\n")

    # handle negative indices
    axis_dim = input_shape[axis]
    buffer.write(body_indent + f"if (index_val < 0) index_val += {axis_dim};\n")

    # bounds check
    buffer.write(body_indent + f"if (index_val < 0 || index_val >= {axis_dim}) {{\n")
    buffer.write(body_indent + "    " + "// Index out of bounds, set to 0\n")

    # build output access string
    output_access = ''.join([f"[d{i}]" for i in range(output_rank)])
    buffer.write(body_indent + "    " + f"{output_name}{output_access} = 0.0f;\n")
    buffer.write(body_indent + f"}} else {{\n")

    # Build input access by mapping each input dimension to either dX or index_val.
    input_access_parts = []
    # For each input dim i, choose the correct output loop variable or index_val
    for i in range(input_rank):
        if i < axis:
            # corresponds to output variable d{i}
            input_access_parts.append(f"d{i}")
        elif i == axis:
            input_access_parts.append("index_val")
        else:
            # i > axis: map to output variable:
            # position in output = axis + indices_rank + (i - (axis+1))
            out_pos = axis + (0 if indices_treated_as_scalar else indices_rank) + (i - (axis + 1))
            input_access_parts.append(f"d{out_pos}")

    input_access = ''.join([f"[{p}]" for p in input_access_parts])
    buffer.write(body_indent + "    " + f"{output_name}{output_access} = {data_name}{input_access};\n")
    buffer.write(body_indent + "}\n")

    # close loops
    for i in range(output_rank - 1, -1, -1):
        buffer.write("    " * (i + 1) + "}\n")

    buffer.write("}\n")
