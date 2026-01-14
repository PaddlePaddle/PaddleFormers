# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    "tokenizer": ["Ernie4_5_VLTokenizer"],
    "configuration": [
        "Ernie4_5_VLConfig",
    ],
    "image_processor": [
        "Ernie4_5_VLImageProcessor",
    ],
    "processor": [
        "Ernie4_5_VLProcessor",
    ],
    "modeling": [
        "Ernie4_5_VLMoeForConditionalGenerationModel",
        "Ernie4_5_VLMoeForConditionalGeneration",
        "Ernie4_5_VLMoeForConditionalGenerationPipe",
    ],
    "vision_process": [
        "read_frames_decord",
        "read_video_decord",
        "RAW_IMAGE_DIR",
        "get_downloadable",
        "render_frame_timestamp",
    ],
    "model": [],
}

if TYPE_CHECKING:
    from .configuration import *
    from .image_processor import Ernie4_5_VLImageProcessor
    from .model import *
    from .modeling import *
    from .processor import Ernie4_5_VLProcessor
    from .tokenizer import Ernie4_5_VLTokenizer
    from .vision_process import (
        RAW_IMAGE_DIR,
        get_downloadable,
        read_frames_decord,
        read_video_decord,
        render_frame_timestamp,
    )
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
