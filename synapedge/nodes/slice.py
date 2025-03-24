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

def _write_slice_function(buffer: StringIO, func_name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any], tensor_shape: Dict[str, Any]) -> None:
    """Generates C code for Slice operator."""
    starts = attrs.get('starts', [0])
    ends = attrs.get('ends', [0])
    axes = attrs.get('axes', [0])
    steps = attrs.get('steps', [1])

    helperfunc._write_function_signature(buffer, func_name, inputs, outputs,tensor_shape)
    helperfunc._write_c_comment(buffer, f"Slice: starts={starts}, ends={ends}, axes={axes}, steps={steps}", indent=4)
    buffer.write("    // TODO: Implement slicing logic\n")
    buffer.write("}\n")