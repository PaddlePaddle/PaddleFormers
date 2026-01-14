# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import sys
from contextlib import suppress
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "formatter": [
        "Formatter",
        "EmptyFormatter",
        "StringFormatter",
        "FunctionFormatter",
        "ToolFormatter",
    ],
    "grounding_plugin": [
        "BaseGroundingPlugin",
        "register_grounding_plugin",
        "get_grounding_plugin",
    ],
    "mm_plugin": [
        "_make_batched_images",
        "_check_video_is_nested_images",
        "MMPluginMixin",
        "BasePlugin",
        "ErnieVLPlugin",
        "PaddleOCRVLPlugin",
        "Qwen2VLPlugin",
        "Qwen3VLPlugin",
        "GLM4VPlugin",
        "Gemma3Plugin",
        "register_mm_plugin",
        "get_mm_plugin",
    ],
    "template": [
        "Role",
        "Template",
        "ReasoningTemplate",
        "Llama2Template",
        "register_template",
        "parse_template",
        "get_template_and_fix_tokenizer",
    ],
    "tool_utils": [
        "FunctionCall",
        "ToolUtils",
        "DefaultToolUtils",
        "QwenToolUtils",
        "GLM4ToolUtils",
        "GLM4MOEToolUtils",
        "Llama3ToolUtils",
        "ERNIEToolUtils",
        "get_tool_utils",
    ],
    "augment_utils": [
        "RandomApply",
        "RandomDiscreteRotation",
        "JpegCompression",
        "RandomScale",
        "RandomSingleSidePadding",
    ],
}

if TYPE_CHECKING:
    from .augment_utils import *
    from .formatter import *
    from .grounding_plugin import *
    from .mm_plugin import *
    from .template import *
    from .tool_utils import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
