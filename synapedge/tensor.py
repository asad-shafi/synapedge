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


import os
import onnx
from onnx import shape_inference, numpy_helper
from collections import defaultdict
import re
import numpy as np
import helpers as helpers
from nodes import helperfunc 
import logging
from typing import List
 

logger = logging.getLogger(__name__)
# -------------------------------------------------------------------
# Main function: Generate Code header and source code pieces from an ONNX model.
def generate_code_from_model(model, model_filename):
    """
    Returns:
        header_lines: List of strings containing the header file content.
        fp_lines: List of strings containing the forward pass source file content.
        weights_lines:
        header_directrives:
    """
    # Run shape inference on the model.
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    nodes = list(graph.node)

    # Build a mapping from tensor name to consumer node indices.
    tensor_consumers = defaultdict(list)
    for node_idx, node in enumerate(nodes):
        for input_name in node.input:
            tensor_consumers[input_name].append(node_idx)
        for output_name in node.output:
            tensor_consumers[output_name]  # Ensure key exists.
    # -------------------------------------------------------------------
    # Collect tensor information for node outputs.
    tensor_info = {}

    for node_idx, node in enumerate(nodes): 
        if not node.name:
            logger.warning(f"Node {node.op_type} at index {node_idx} does not have a name. replacing with {node.op_type}_{node_idx}")
            node.name = f"{node.op_type}_{node_idx}"
        node_id = node.name if node.name else node.op_type
        #print(f"{node}--{node.name}-->{node.op_type}")
        for out_idx, output_name in enumerate(node.output):
            #gen_name = f"tensor_{node_id}_Output_{out_idx}"
            gen_name = f"tensor_{output_name}"#_Output_{out_idx}"
            tensor_info[gen_name] = {
                'producer': node_idx,
                'consumers': tensor_consumers.get(output_name, []),
                'original': output_name,  # Original ONNX tensor name.
            }
    # Add initializer tensors (e.g., weights).
    initializers = {init.name: init for init in graph.initializer}
    for init_name, init in initializers.items():
        tensor_name = f"tensor_{init_name}"
        tensor_info[tensor_name] = {
            'producer': None,
            'consumers': tensor_consumers.get(init_name, []),
            'original': init_name,
        }
    
    # -------------------------------------------------------------------
    # Helper: Retrieve tensor shape from value_info, graph inputs/outputs, or initializer.
    def get_shape(info):
        original = info.get('original')
        for vi in graph.value_info:
            if vi.name == original:
                dims = []
                for dim in vi.type.tensor_type.shape.dim:
                    dims.append(dim.dim_value if dim.HasField('dim_value') else 0)
                return dims
        for inp in graph.input:
            if inp.name == original:
                dims = []
                for dim in inp.type.tensor_type.shape.dim:
                    dims.append(dim.dim_value if dim.HasField('dim_value') else 0)
                return dims
        for out in graph.output:
            if out.name == original:
                dims = []
                for dim in out.type.tensor_type.shape.dim:
                    dims.append(dim.dim_value if dim.HasField('dim_value') else 0)
                return dims
        if original in initializers:
            return list(initializers[original].dims)
        return []

    # -------------------------------------------------------------------
    # Helper: Retrieve tensor C type from value_info, graph inputs/outputs, or initializer.
    def get_c_type_for_tensor(info):
        original = info.get('original')
        for vi in graph.value_info:
            if vi.name == original:
                return helpers.get_c_type_from_elem_type(vi.type.tensor_type.elem_type)
        for inp in graph.input:
            if inp.name == original:
                return helpers.get_c_type_from_elem_type(inp.type.tensor_type.elem_type)
        for out in graph.output:
            if out.name == original:
                return helpers.get_c_type_from_elem_type(out.type.tensor_type.elem_type)
        if original in initializers:
            return helpers.get_c_type_from_elem_type(initializers[original].data_type)
        else:
            return "unkown"

    # Compute lifetimes (start/end indices) for each tensor.
    tensor_intervals = {}
    for tensor_name, info in tensor_info.items():
        if info['producer'] is None:
            if info['consumers']:
                start = min(info['consumers'])
                end = max(info['consumers'])
            else:
                start = end = -1
        else:
            start = info['producer']
            if info['consumers']:
                end = max(info['consumers'])
            else:
                end = len(nodes) - 1
        shape = get_shape(info)
        tensor_intervals[tensor_name] = {
            'start': start,
            'end': end,
            'shape': shape
        }

    # -------------------------------------------------------------------
    # Exclude external outputs from union grouping.
    external_output_names = {out.name for out in graph.output}
    # Group tensors (excluding initializers and external outputs) into unions using greedy interval-partitioning.
    sorted_tensors = sorted(
        (
            (tensor_name, interval)
            for tensor_name, interval in tensor_intervals.items()
            if tensor_name.replace("tensor_", "") not in initializers and
               tensor_info[tensor_name]['original'] not in external_output_names
        ),
        key=lambda kv: kv[1]['start']
    )

    union_groups = []  # Each group is a list of dicts with keys: 'name', 'original', 'interval', 'c_type'
    for tensor_name, interval in sorted_tensors:
        if interval['start'] == -1:
            continue
        placed = False
        for group in union_groups:
            if all((interval['end'] < other['interval']['start'] or interval['start'] > other['interval']['end'])
                   for other in group):
                group.append({
                    'name': helperfunc._sanitize_name(tensor_name),
                    'original': tensor_info[tensor_name]['original'],
                    'interval': interval,
                    'c_type': get_c_type_for_tensor(tensor_info[tensor_name])
                })
                placed = True
                break
        if not placed:
            union_groups.append([{
                'name': helperfunc._sanitize_name(tensor_name),
                'original': tensor_info[tensor_name]['original'],
                'interval': interval,
                'c_type': get_c_type_for_tensor(tensor_info[tensor_name])
            }])

    # -------------------------------------------------------------------
    # Generate C unions for memory reuse and extract weight arrays.

    # Generate union definitions.
    union_lines = []
    for group_idx, group in enumerate(union_groups):
        union_name = f"tensor_union_{group_idx}"
        union_lines.append(f"union {union_name}")
        union_lines.append("\n{\n")
        for tensor in group:
            shape = tensor['interval']['shape']
            shape_str = "".join(f"[{dim}]" if dim != 0 else "[]" for dim in shape) if shape else "[]"
            union_lines.append(f"    {tensor['c_type']} {tensor['name']}{shape_str}; // {np.prod(shape)}\n")
        union_lines.append("};\n")
        union_lines.append(f"static union {union_name} tu{group_idx};\n")

    # Build a mapping from the original tensor name to its union reference.
    union_mapping = {}
    for group_idx, group in enumerate(union_groups):
        for tensor in group:
            union_mapping[tensor['original']] = f"tu{group_idx}.{tensor['name']}"

    # Extract weight arrays from initializers.
    # Maximum chunk size in bytes (5MB)
    MAX_CHUNK_SIZE = 5 * 1024 * 1024  
    weights_chunk=""
    current_chunk_size = 0
    weights_lines = []
    header_lines ="// Weight arrays extracted from model initializers\n"
    weights_chunk = header_lines
    # Dictionary of initializers    
    for init_name, init in initializers.items():
        arr = numpy_helper.to_array(init)

        
        #print(f"---------------{init_name}")
        #print(arr.max())
        #print(arr.min())
        #arr_size_bytes = arr.nbytes
        #print(f"------------{arr.size}------------- ")
        #print(arr)
        if arr.shape == () and arr.size == 1:
            shape_str = "[1]"
        else:
            shape_str = "".join(f"[{dim}]" for dim in arr.shape)
        #shape_str = "".join(f"[{dim}]" for dim in arr.shape)
        c_type = helpers.get_c_type_from_elem_type(init.data_type)
        #formatted_array = helpers.format_array(arr, indent=0)
        c_array_string = helpers.convert_to_c_array_dtyp(arr,0,c_type)
        if arr.size > 0:
            min_max = f"// min: {arr.min()}, max: {arr.max()} \n"
        #weights_lines.append(f"static const {c_type} tensor_{helperfunc._sanitize_name(init_name)}{shape_str} =\n{c_array_string};\n")
        array_declaration = (f"{min_max}static const {c_type} tensor_{helperfunc._sanitize_name(init_name)}"f"{shape_str} =\n{c_array_string};\n")
        weights_chunk += array_declaration
        arr_size_bytes = len(array_declaration)
        if current_chunk_size > MAX_CHUNK_SIZE:
            weights_lines.append(weights_chunk)
            weights_chunk = header_lines
            current_chunk_size = 0

        current_chunk_size += arr_size_bytes
    if weights_chunk :
        weights_lines.append(weights_chunk)
    #print(len(weights_lines[0]))
    # -------------------------------------------------------------------
    # Determine external network inputs (graph inputs not in initializers) and outputs.
    external_inputs = []
    for inp in graph.input:
        if inp.name not in initializers:
            dims = [dim.dim_value if dim.HasField('dim_value') else 0 for dim in inp.type.tensor_type.shape.dim]
            c_type = helpers.get_c_type_from_elem_type(inp.type.tensor_type.elem_type)
            external_inputs.append((inp.name, dims, c_type))
    external_outputs = []
    for out in graph.output:
        dims = [dim.dim_value if dim.HasField('dim_value') else 0 for dim in out.type.tensor_type.shape.dim]
        c_type = helpers.get_c_type_from_elem_type(out.type.tensor_type.elem_type)
        external_outputs.append((out.name, dims, c_type))

    def format_c_param(tensor_name, shape, c_type, is_input=True):
        shape_str = "".join(f"[{dim}]" if dim != 0 else "[]" for dim in shape)
        qualifier = "const " if is_input else ""
        return f"{qualifier}{c_type} {tensor_name}{shape_str}"

    params = []
    for name, shape, c_type in external_inputs:
        params.append(format_c_param(name, shape, c_type, is_input=True))
    for name, shape, c_type in external_outputs:
        params.append(format_c_param(name, shape, c_type, is_input=False))
    forward_pass_signature = f"void forward_pass({', '.join(params)})"

    # -------------------------------------------------------------------
    # Generate the forward_pass function.
    fp_lines = []
    fp_lines.append(forward_pass_signature)
    fp_lines.append("{")
    for node in nodes:
        name = f"node_{(node.name)}"
        node_fn = f"{helperfunc._sanitize_name(name) if node.name else helperfunc._sanitize_name(node.op_type)}"
        args = []
        # Process inputs
        for in_name in node.input:
            if not in_name:
                continue
            if in_name in union_mapping:
                args.append(union_mapping[in_name])
            elif in_name in initializers:
                # For initializers, use the declared weight array variable name.
                args.append(f"tensor_{helperfunc._sanitize_name(in_name)}")
            else:
                args.append(in_name)
        # Process outputs
        for out_name in node.output:
            if not out_name:
                continue
            # If the output is external, do not use a union mapping.
            if out_name in union_mapping and out_name not in external_output_names:
                args.append(union_mapping[out_name])
            else:
                args.append(helperfunc._sanitize_name(out_name))
        fp_lines.append(f"    {node_fn}({', '.join(args)});")
    fp_lines.append("}\n")

    # -------------------------------------------------------------------
    # Prepare data for generating node function prototypes.
    # Create dictionaries for external inputs, outputs, and initializers info.
    external_input_dict = {name: (c_type, dims) for name, dims, c_type in external_inputs}
    external_output_dict = {name: (c_type, dims) for name, dims, c_type in external_outputs}
    initializer_dict = {}
    for init_name, init in initializers.items():
        initializer_dict[init_name] = (helpers.get_c_type_from_elem_type(init.data_type), list(init.dims))
    union_tensor_info = {}
    for group in union_groups:
        for tensor in group:
            union_tensor_info[tensor['original']] = (tensor['c_type'], tensor['interval']['shape'])

    # -------------------------------------------------------------------
    # Helper to get parameter type and name for function prototypes using the actual tensor names.
    def get_prototype_param(tensor_name, is_input=True):
        if tensor_name in union_tensor_info:
            c_type, _ = union_tensor_info[tensor_name]
        elif is_input and tensor_name in external_input_dict:
            c_type, _ = external_input_dict[tensor_name]
        elif (not is_input) and tensor_name in external_output_dict:
            c_type, _ = external_output_dict[tensor_name]
        elif tensor_name in initializer_dict:
            c_type, _ = initializer_dict[tensor_name]
        else:
            c_type = "float"
        param_type = f"const {c_type}" if is_input else f"{c_type}"
        # Use the sanitized tensor name as the parameter name.
        param_name = helperfunc._sanitize_name(tensor_name)
        return f"{param_type} {param_name}"

    # Generate node function prototypes.
    node_prototypes = set()
    for node in nodes:
        node_fn = f"node_{helperfunc._sanitize_name(node.name) if node.name else helperfunc._sanitize_name(node.op_type)}"
        proto_params = []
        # Use actual input tensor names.
        for in_name in node.input:
            if not in_name:
                continue
            proto_params.append(get_prototype_param(in_name, is_input=True))
        # Use actual output tensor names.
        for out_name in node.output:
            if not out_name:
                continue
            proto_params.append(get_prototype_param(out_name, is_input=False))
        proto = f"void {node_fn}({', '.join(proto_params)});"
        node_prototypes.add(proto)
    node_prototypes = sorted(node_prototypes)

    # -------------------------------------------------------------------
    # Generate header file content with include guards and necessary includes.
    base_name = os.path.splitext(os.path.basename(model_filename))[0]
    header_guard = re.sub(r'\W+', '_', base_name).upper() + "_H"

    header_lines = []
    header_lines.append(f"#ifndef {header_guard}\n")
    header_lines.append(f"#define {header_guard}\n")
    header_lines.append("")
    header_lines.append("#include <stdint.h>\n")
    header_lines.append("#include <stdbool.h>\n")
    header_lines.append("#include <math.h>\n")
    header_lines.append("")
    # Add weight arrays
    header_directrives =""
    if len(weights_lines) > 1:
        for i, chunk in enumerate(weights_lines):
            _header_guard = re.sub(r'\W+', '_', base_name).upper() + f"_WEIGHTS_{i}_H"
            weights_lines[i] = f"#ifndef {_header_guard}\n#define {_header_guard}\n\n{weights_lines[i]}\n#endif // {_header_guard}" 
            header_directrives +=f"#include \"{base_name}_weights_{i}.h\"\n"
            #header_lines.append(f"#include \"{base_name}_weights_{i}.h\"\n")
        #header_lines.append("") 
        #header_directrives += "\n"
    else :
        header_lines.extend(weights_lines)
        header_lines.append("")
    # Add union definitions
    header_lines.extend(union_lines)
    header_lines.append("")
    # Add node function prototypes
    #header_lines.append("// Node function prototypes\n")
    #header_lines.extend(node_prototypes)
    #header_lines.append("")
    # Forward pass prototype (declaration)
    header_lines.append(f"{forward_pass_signature};")
    header_lines.append("")
    header_lines.append(f"\n#endif // {header_guard}")

    # -------------------------------------------------------------------
    # Generate source file content that includes the header.
    source_lines = []
    #source_lines.append(f'#include "{base_name}.h"')
    #source_lines.append("")
    source_lines.extend(fp_lines)

    return header_lines, fp_lines,weights_lines,header_directrives
