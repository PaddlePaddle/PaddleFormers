# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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

import_structure = {
    "configuration": ["Glm4vMoeConfig", "Glm4vMoeTextConfig", "Glm4vMoeVisionConfig"],
    "modeling": [
        "Glm4vMoeForConditionalGeneration",
        "Glm4vMoeModel",
        "Glm4vMoePreTrainedModel",
        "Glm4vMoeTextModel",
        "Glm4vMoeVisionModel",
    ],
    # TODO: might be moved to glm4v in the future
    "image_processor": ["Glm4vImageProcessor"],
    "image_processor_fast": ["Glm4vImageProcessorFast"],
    "processor": ["Glm4vProcessor"],
    "video_processor": ["Glm4vVideoProcessor"],
}

if TYPE_CHECKING:
    from .configuration import *
    from .image_processor import *
    from .image_processor_fast import *
    from .modeling import *
    from .processor import *
    from .video_processor import *
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
