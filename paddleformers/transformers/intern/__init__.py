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

"""
InternLM2 Common Module

This module provides unified access to both InternLM2 2.0 and 2.5 models.
It automatically routes to the correct implementation based on the model configuration.
"""

from .configuration import InternLM2Config
from .modeling import (
    InternLM2ForCausalLM,
    InternLM2ForQuestionAnswering,
    InternLM2ForSequenceClassification,
    InternLM2ForTokenClassification,
    InternLM2Model,
    InternLM2PretrainedModel,
)

# Alias for auto system compatibility
InternLM2 = InternLM2Model

__all__ = [
    "InternLM2Config",
    "InternLM2Model",
    "InternLM2",
    "InternLM2PretrainedModel",
    "InternLM2ForCausalLM",
    "InternLM2ForSequenceClassification",
    "InternLM2ForQuestionAnswering",
    "InternLM2ForTokenClassification",
]
