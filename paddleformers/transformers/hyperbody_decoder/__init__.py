# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""HyperBody decoder（LLM 主干）。

用 ``_LazyModule``（与 ``deepseek_v4/__init__.py`` 同形）而不是直接 import：
``transformers/__init__.py`` 会把本包登记进 import_structure，
真正 import ``modeling`` 会连带拉起 ``paddlefleet``，那是几秒级的开销，
不该在 ``import paddleformers`` 时就付。
"""
import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["HyperBodyDecoderConfig"],
    "modeling": [
        "HyperBodyDecoderForCausalLM",
        "HyperBodyDecoderForCausalLMPipe",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
