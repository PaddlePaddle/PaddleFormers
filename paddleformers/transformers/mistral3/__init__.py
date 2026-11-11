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

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule
from ..pixtral import PixtralImageProcessor, PixtralProcessor

_import_structure = {
    "configuration": ["Mistral3Config"],
    "modeling": [
        "Mistral3ForCausalLM",
        "Mistral3ForConditionalGeneration",
        "Mistral3Model",
        "Mistral3PretrainedModel",
        "Mistral3PatchMerger",
        "Mistral3MultiModalProjector",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        # Auto mappings look up processors through the config's model package.
        # Keep aliases to the canonical Pixtral classes, not duplicate modules
        # or top-level registrations.
        extra_objects={"PixtralImageProcessor": PixtralImageProcessor, "PixtralProcessor": PixtralProcessor},
    )
