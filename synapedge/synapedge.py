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

import argparse
import logging
import os
import graph as graph

str_help = """Synapedge is ONNX to C Code Generator
------------------------
Not all ONNX operators are supported yet. The following operators are supported:
- Tensor operations (Reshape, Concat.)
- Mathematical operations (Add,Sub,Div, Mul, Gemm,,Pow etc.)
- Neural network layers (Conv, etc.)
- Control flow (If, Loop) and other advanced operators (not supported yet)
- Activation functions (ReLU, Sigmoid, Tanh.)
- Pooling operations (MaxPool,MinPool, AveragePool, GlobalAveragePool, etc.)
- Element-wise operations (Add, Sub, Mul, Div,Pow etc.)
- Data manipulation (Transpose, etc.)
- RNN and LSTM (not supported yet)
- Loss functions (Softmax, etc.)
------------------------
Please check the documentation for the latest updates.

"""

optimizations = """ Apply Optimizations: 
check for updates for supported optimizations
- FP16 : Convert float32 to float16. reduce memory usage (not in this version).
- FXP16 : Convert float32 to fixed point 16, reduce memory usage and increase speed good for non-FPU supported devices (not in this version).
"""

def main(model_path, output_path, verbose, optimizations):
    logger = logging.getLogger(__name__)
    logger.info("Starting ONNX to C conversion...")
    logger.info(f"ONNX Model Path: {model_path}")
    logger.info(f"Output C File Path: {output_path}")
    
    if optimizations:
        raise NotImplementedError("Optimizations are not implemented yet")
        logger.info(f"Applied Optimizations: {', '.join(optimizations)}")
    else:
        logger.info("No optimizations applied.")

    print(f"Converting ONNX model: {model_path}")
    graph.parse(model_path, output_path, output_path, verbose=verbose, optimizations=optimizations)
    print(f"Saving generated C code to: {output_path}")

    if verbose:
        logger.info("Conversion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapedge ONNX to C Compiler",epilog=str_help,formatter_class=argparse.RawTextHelpFormatter)

    # Positional argument for the ONNX model path
    parser.add_argument("model_path", help="Path to ONNX model file")

    # Optional argument for output file path or directory
    parser.add_argument("-o", "--output", help="Output C file path or directory (default: same as ONNX model)")

    # Version argument
    parser.add_argument("--version", action="version", version="Synapedge Version = 0.0.1")

    # Verbose argument
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (e.g., -v, -vv, -vvv)")

    # Optimization argument (multiple values allowed)
    parser.add_argument("--optimize", nargs="*", help=optimizations)

    args = parser.parse_args()

    if args.optimize and "FP16" in args.optimize:
        print("FP16 not in this version fall back to FP32")
    if args.optimize and "FXP16" in args.optimize:
        print("FXP16 not in this version fall back to FP32")

    # Set up logging configuration based on verbosity level

    # Default log level
    log_level = logging.ERROR # Only display errors by default
    if args.verbose == 1:
        log_level = logging.INFO
    if args.verbose == 2:
        log_level = logging.WARNING
    elif args.verbose > 2:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

    # Get directory of the ONNX model
    model_dir = os.path.dirname(os.path.abspath(args.model_path))

    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)  # Convert to absolute path

        if os.path.isdir(output_path):  # If output is a directory, save as "model.c"
            output_path = os.path.join(output_path, "model.c")
        elif not os.path.isabs(args.output):  # If it's just a filename (e.g., "test.c")
            output_path = os.path.join(model_dir, args.output)  # Save in ONNX model directory
    else:
        output_path = os.path.join(model_dir, "model.c")  # Default output file

    main(args.model_path, output_path, args.verbose, args.optimize or [])

