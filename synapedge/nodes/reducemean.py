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

def _write_reduce_mean(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
#def _write_reduce_mean(buffer: StringIO, op_type: str, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for ReduceMean operator using multidimensional arrays."""
    input_name = inputs[0]
    output_name = outputs[0]
    input_shape = tensor_shape.get(input_name, [])
    input_rank = len(input_shape)
    
    # Get attributes with defaults
    axes = attrs.get('axes', list(range(input_rank)))  # Default: reduce all axes
    keepdims = attrs.get('keepdims', 1)  # Default: keep reduced dimensions
    
    # Convert negative axes to positive indices
    axes = [ax if ax >= 0 else input_rank + ax for ax in axes]
    
    # Calculate output shape
    if keepdims:
        output_shape = [1 if i in axes else dim for i, dim in enumerate(input_shape)]
    else:
        output_shape = [dim for i, dim in enumerate(input_shape) if i not in axes]
    
    # Update tensor shape information
    tensor_shape[output_name] = output_shape
    output_rank = len(output_shape)
    
    # Identify reduced and non-reduced axes
    non_reduced_axes = [i for i in range(input_rank) if i not in axes]
    reduced_axes = sorted(axes)
    
    # Calculate reduction size
    reduction_size = 1
    for axis in reduced_axes:
        reduction_size *= input_shape[axis]
    
    # Write function signature
    buffer.write(f"void {func_name}(const float {input_name}")
    if input_rank > 0:
        buffer.write("[" + "][".join(map(str, input_shape)) + "]")
    buffer.write(f", float {output_name}")
    if output_rank > 0:
        buffer.write("[" + "][".join(map(str, output_shape)) + "]")
    buffer.write(") {\n")
    
    # Generate loops for non-reduced axes (output dimensions)
    for i, axis in enumerate(non_reduced_axes):
        dim_size = input_shape[axis]
        buffer.write(f"    for (int i{i} = 0; i{i} < {dim_size}; i{i}++) {{\n")
    
    # Initialize sum
    buffer.write("        float sum = 0.0f;\n")
    
    # Generate loops for reduced axes
    for i, axis in enumerate(reduced_axes):
        dim_size = input_shape[axis]
        buffer.write(f"        for (int r{i} = 0; r{i} < {dim_size}; r{i}++) {{\n")
    
    # Create input index string
    input_index = []
    for axis in range(input_rank):
        if axis in non_reduced_axes:
            idx = non_reduced_axes.index(axis)
            input_index.append(f"i{idx}")
        else:
            idx = reduced_axes.index(axis)
            input_index.append(f"r{idx}")
    input_index_str = "][".join(input_index)
    
    # Accumulate sum
    buffer.write(f"            sum += {input_name}[{input_index_str}];\n")
    
    # Close reduction loops
    for _ in reduced_axes:
        buffer.write("        }\n")
    
    # Create output index string
    output_index = []
    if keepdims:
        for axis in range(input_rank):
            if axis in non_reduced_axes:
                idx = non_reduced_axes.index(axis)
                output_index.append(f"i{idx}")
            else:
                output_index.append("0")
    else:
        for i in range(len(non_reduced_axes)):
            output_index.append(f"i{i}")
    output_index_str = "][".join(output_index)
    
    # Calculate and store mean
    buffer.write(f"        float mean = sum / {reduction_size}.0f;\n")
    buffer.write(f"        {output_name}[{output_index_str}] = mean;\n")
    
    # Close non-reduced loops
    for _ in non_reduced_axes:
        buffer.write("    }\n")
    
    buffer.write("}\n")