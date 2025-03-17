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

import numpy as np
import argparse
import re
from io import StringIO
from typing import List, Dict, Any
import onnx
import tensor as tensor

import helpers as helpers
import nodes.helperfunc as helperfunc
import nodes.resolve as resolve
import os
import logging



logger = logging.getLogger(__name__)

def generate_c_file(functions: List[Dict[str, Any]], output_c_path: str,header_directrives:str) -> None:
    """
    Generates a C source file with implementations for ONNX operators.
    Args:
        functions: List of extracted ONNX nodes/operators
        output_c_path: Output file path for the C code
    """
    buffer = StringIO()
    #with open(output_c_path, 'w') as f:
        # Header includes and basic setup
    base_name = os.path.splitext(os.path.basename(output_c_path))[0]

    buffer.write(f'#include "{base_name}.h"\n')
    buffer.write(header_directrives)
    buffer.write("#include <stdio.h>\n")
    buffer.write("#include <math.h>\n")
    buffer.write("#include <string.h>\n\n")

        # Write implementations for each operator
    for func in functions:
            op_type = func['op_type']
            inputs = func['inputs']
            outputs = func['outputs']
            attrs = func['attributes']
            tensor = func['intermediate_tensors_shape']

            # Sanitize function and variable names
            func_name = helperfunc._sanitize_name(func['name'])
            input_names = [helperfunc._sanitize_name(i) for i in inputs]
            output_names = [helperfunc._sanitize_name(o) for o in outputs]

            #print(input_names)
            # Write function comment with original names
            comment =f"  Operator: {op_type} \n    Name in model: {func['name']}\n"
            #_write_c_comment(buffer, f"Operator: {op_type} \n   Name in model: {func['name']}", indent=0)
            for input_name in input_names:
                node_intermediate_tensors = tensor.get(input_name, None)
                #_write_c_comment(buffer, f"Input: {input_name} { node_intermediate_tensors}", indent=4)
                comment +=f"    Input: {input_name} { node_intermediate_tensors}\n"

           # _write_c_comment(buffer, f"Inputs: {', '.join(inputs)}", indent=4)
            for output_name in output_names:
                node_intermediate_tensors = tensor.get(output_name, None)
                #_write_c_comment(buffer, f"Output: {output_name} { node_intermediate_tensors}", indent=4)
                comment +=f"    Output: {output_name} { node_intermediate_tensors}"
           # _write_c_comment(buffer, f"Outputs: {', '.join(outputs)}", indent=4)
            comment+= "\n"
            helpers._write_c_comment(buffer, comment, indent=0)
            #print(func)
            # Sanitize function and variable names in functions
            func['name'] = func_name
            func['inputs'] = [helperfunc._sanitize_name(i) for i in inputs] # Sanitize input names directly in func['inputs']
            func['outputs'] = [helperfunc._sanitize_name(o) for o in outputs] # Sanitize output names directly in func['outputs']
            #func['intermediate_tensors_shape'] = [helperfunc._sanitize_name(t) for t in tensor] # Sanitize output names directly in func['outputs']


            #print(func)
            # Dispatch to operator-specific handlers
            # passing functions here to update shape of in/out when not present in input/output
            resolve._dispatch_operator_handler(functions,buffer, op_type, func_name, input_names, output_names, attrs,tensor)


            buffer.write("\n")  # Add spacing between functions

    # Get final string and close buffer
    result = buffer.getvalue()
    buffer.close()
    return result

def save_to_file(code: str, filename: str) -> None:
    with open(filename, 'w') as f:
        f.write(code)

def parse(onnx_model_path: str, output_c_path: str,output_file_name: str ,verbose, optimizations ) -> None:
    # Load model and extract metadata
    model = onnx.load(onnx_model_path)
    weights = {
        init.name: onnx.numpy_helper.to_array(init)
        for init in model.graph.initializer
    }
        # Extract model metadata
    logger.info(f"Model IR Version: {model.ir_version}")
    logger.info(f"Producer Name:, {model.producer_name}")
    logger.info(f"Producer Version:, {model.producer_version}")
    logger.info(f"Opset Version:, {model.opset_import[0].version}")
    logger.info(f"Graph Name:, {model.graph.name}")
    
    if model.producer_name == "onnx.quantize":
       raise ValueError("Quantized models are not supported.")
    #Generate header file
    header_lines, source_lines, weights_tensors,header_directrives = tensor.generate_code_from_model(model=model,model_filename=output_c_path)


    # Extract computational graph nodes 
    functions = []
    _extract_functions_from_graph(model.graph, functions)



    # Generate C code
    c_code = generate_c_file(functions, output_c_path,header_directrives)
    c_code = c_code + "\n".join(source_lines) 
          
    #base_name = os.path.splitext(os.path.basename(model_path))[0]
    #header_filename = f"/content/{base_name}.h"
    save_to_file(c_code, output_c_path)
    if len(weights_tensors) > 1:
      logger.info("Multiple weights files are generated")
      save_to_file("".join(header_lines), output_c_path.replace(".c",".h"))
      for i, chunk in enumerate(weights_tensors):
        save_to_file(chunk, output_c_path.replace(".c",f"_weights_{i}.h"))
    else:
        logger.info("Single header file is generated")
        save_to_file("".join(header_lines), output_c_path.replace(".c",".h"))

def _extract_functions_from_graph(graph, functions: List) -> None:
    """Recursively extract nodes and subgraphs from ONNX graph."""

    # Extract metadata input and output of model
    input_shapes = {}
    output_shapes = {}
    for input in graph.input:
        input_shapes[input.name] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    for output in graph.output:
        output_shapes[output.name] = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    #print(input_shapes)
    #print(output_shapes)
    weights_dict = {}
    for initializer in graph.initializer:
        # Extract the name and shape of each weight (initializer)
        weights_dtype = helpers.get_c_type_from_elem_type( initializer.data_type)
        weight_name = initializer.name
        weight_shape = list(initializer.dims)  # Convert the dimensions to a list
        if not weight_shape :
            #print("no value")
            weight_shape.append(1) # found,some cases return weight_shape=[] netron confirm there is single value
            #print(weight_shape)
        #print(f"{weight_name} {weight_shape}")
        # Store the weight name and shape in the dictionary
        weights_dict[weight_name] = weight_shape
        weights_dict[weight_name+"_dtype"] = weights_dtype

    intermediate_tensors = {}
    for value_info in graph.value_info:
      tensor_name = value_info.name
      tensor_shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
      tensor_dtype = helpers.get_c_type_from_elem_type(value_info.type.tensor_type.elem_type)
      #tensor_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[value_info.type.tensor_type.elem_type]
      intermediate_tensors[tensor_name] = tensor_shape
      intermediate_tensors[tensor_name+"_dtype"] = tensor_dtype
      #intermediate_tensors[tensor_name] = {
      #      'shape': tensor_shape,
      #      'dtype': tensor_dtype
      # }

    for node_indx, node in enumerate(graph.node): 
        if node.name == "": # need warning message here
            logger.warning(f"Node name is empty at index {node_indx}")
            node.name = f"{node.op_type}_{node_indx}"
        func = {
            'name': node.name,
            'op_type': node.op_type,
            'inputs': node.input,
            'outputs': node.output,
            'attributes': {
                attr.name: onnx.helper.get_attribute_value(attr)
                for attr in node.attribute
            },
            'intermediate_tensors_shape' : {
            }
        }
        # Store the shape of each input/output and param tensor in 'intermediate_tensors'
        for output in node.output:
          if output in intermediate_tensors:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(output)] = intermediate_tensors[output]
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(output)+"_dtype"] = intermediate_tensors[output+"_dtype"]
          elif output in weights_dict:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(output)] = weights_dict[output]
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(output)+"_dtype"] = weights_dict[output+"_dtype"]
          elif output in output_shapes:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(output)] = output_shapes[output]
          else :
             logger.info(f"unknown shap in output at :{node.name}")  

        for input in node.input:
          if input in intermediate_tensors:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(input)] = intermediate_tensors[input]
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(input)+"_dtype"] = intermediate_tensors[input+"_dtype"]
          elif input in weights_dict:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(input)] = weights_dict[input]
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(input)+"_dtype"] = weights_dict[input+"_dtype"]
          elif input in input_shapes:
            func['intermediate_tensors_shape'][helperfunc._sanitize_name(input)] = input_shapes[input]
          else :
            logger.info(f"unknown shap in input at :{node.name}")
        functions.append(func)
        #print(func['intermediate_tensors
        # Handle subgraphs (e.g., If/Loop bodies)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _extract_functions_from_graph(attr.g, functions)



#parse('/content/yolov5n6.onnx', '/content/model.c','/content/model.h')

