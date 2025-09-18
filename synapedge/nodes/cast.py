from typing import IO, List, Dict, Any
from io import StringIO
import nodes.helperfunc as helperfunc

def _write_cast_function(buffer: StringIO,
                         func_name: str,
                         inputs: List[str],
                         outputs: List[str],
                         attrs: Dict[str, Any],
                         tensor_shape: Dict[str, Any]) -> None:
    """
    Generates C code for the ONNX Cast operator:
      output = (target_type) input
    """
    # Function signature
    helperfunc._write_function_signature(buffer, func_name, inputs, outputs, tensor_shape)
    helperfunc._write_c_comment(buffer, "Cast", indent=4)

    # Determine the output shape & rank (same as input for Cast)
    out_shape = tensor_shape[outputs[0]]
    rank = len(out_shape)

    # Map ONNX 'to' enum to C type name
    type_map = {
        # common ONNX dtypes (TensorProto.DataType.*)
        1:  "float",    # FLOAT
        2:  "uint8_t",  # UINT8
        3:  "int8_t",   # INT8
        4:  "uint16_t", # UINT16
        5:  "int16_t",  # INT16
        6:  "int32_t",  # INT32
        7:  "int64_t",  # INT64
        10: "double",   # DOUBLE
        # add more as needed…
    }
    to_enum = attrs.get("to")
    c_type = type_map.get(to_enum, "float")  # default to float if unknown

    # Generate nested loops
    indent = "    "
    for dim_idx, dim_size in enumerate(out_shape):
        buffer.write(indent + f"for (int i{dim_idx} = 0; i{dim_idx} < {dim_size}; i{dim_idx}++) {{\n")
        indent += "    "

    # Build index strings
    idx = ''.join(f"[i{d}]" for d in range(rank))

    # Emit the cast assignment
    buffer.write(indent + f"{outputs[0]}{idx} = ({c_type}) {inputs[0]}{idx};\n")

    # Close loops
    for _ in range(rank):
        indent = indent[:-4]
        buffer.write(indent + "}\n")

    # End of function
    buffer.write("}\n")
