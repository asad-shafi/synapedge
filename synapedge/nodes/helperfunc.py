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
import re
import logging

logger = logging.getLogger(__name__)

def _sanitize_name(name: str) -> str:
    """Replace invalid characters (e.g., '/') in names with underscores and print replacements."""
    sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Collapse two or more consecutive underscores into a single underscore
    sanitized_name = re.sub(r'_{2,}', '_', sanitized_name)
    # Check if any character was replaced
    if sanitized_name != name:
        logger.debug(f"Replaced invalid characters in '{name}' -> '{sanitized_name}' with underscore")
    return sanitized_name

def _write_c_comment(buffer, comment: str, indent: int = 0) -> None:
    """Write a formatted C comment to the file."""
    buffer.write(" " * indent + f"/*{comment}*/\n")


def _write_function_signature(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str],tensors_shape: List[str]) -> None:
    """
    Dynamically generates a function signature with all inputs and outputs.
    Args:
        f: File object to write to.
        func_name: Name of the function.
        inputs: List of input tensor names.
        outputs: List of output tensor names.
    """
    input_args = []
    output_args = []

    # Generate input arguments
    for input_name in inputs:
        tensors = tensors_shape.get(input_name, None)

        tensors_dtype = tensors_shape.get(input_name+"_dtype", "floatt")
        if input_name.isdigit(): # show warning if input name is a number
            logger.warning(f"Input name {input_name} is a number, it should be a string, replacing with tensor_{input_name}")
            input_name = f"tensor_{input_name}"
        if tensors is None:
            input_args.append(f"const {tensors_dtype} {input_name}")
        else:
            dims = ''.join([f"[{dim}]" for dim in tensors])
            input_args.append(f"const {tensors_dtype} {input_name}{dims}")

    # Generate output arguments
    for output_name in outputs:
        tensors = tensors_shape.get(output_name, None)
        tensors_dtype = tensors_shape.get(output_name+"_dtype", "float")
        if output_name.isdigit(): # show warning if output name is a number
            logger.warning(f"Output name {output_name} is a number, it should be a string, replacing with tensor_{output_name}")
            output_name = f"tensor_{output_name}"
        tensors_dtype = tensors_shape.get(output_name+"_dtype", "float")
        #print(tensors)
        if tensors is None:
            output_args.append(f"{tensors_dtype} {output_name}")
        else:
            dims = ''.join([f"[{dim}]" for dim in tensors])
            output_args.append(f"{tensors_dtype} {output_name}{dims}")
    #output_args = [f"float* {output_name}" for output_name in outputs]
    # Combine inputs and outputs
    all_args = ", ".join(input_args + output_args)
    temp = f"node_{func_name}"
    # Write the function signature
    buffer.write(f"void {_sanitize_name(temp)}({all_args}) {{\n")