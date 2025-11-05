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

"""Paddle Ernie4_5_VL model"""

from .model.modeling_moe_vl import (
    Ernie4_5_VLMoeForConditionalGeneration as Ernie4_5_VLMoeForConditionalGenerationModel,
)
from .model.modeling_moe_vl_pp import Ernie4_5_VLMoeForConditionalGenerationPipe

__all__ = [
    "Ernie4_5_VLMoeForConditionalGenerationModel",
    "Ernie4_5_VLMoeForConditionalGenerationPipe",
]
