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

import transformers as hf
from transformers.models.gemma3.image_processing_gemma3 import (
    Gemma3ImageProcessorKwargs,
)

from ..image_processing_utils import warp_base_image_processor

Gemma3ImageProcessor = warp_base_image_processor(hf.Gemma3ImageProcessor)

__all__ = ["Gemma3ImageProcessor", "Gemma3ImageProcessorKwargs"]
