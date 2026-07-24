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

from .gemma4_rotary_pos_embedding import (
    DualRoPEOutput as DualRoPEOutput,
    Gemma4DualRotaryEmbedding as Gemma4DualRotaryEmbedding,
    Gemma4ProportionalRotaryEmbedding as Gemma4ProportionalRotaryEmbedding,
)
from .language_model_embedding import (
    Gemma4Embedding as Gemma4Embedding,
    LanguageModelEmbedding as LanguageModelEmbedding,
)
from .rope_utils import (
    apply_rotary_pos_emb as apply_rotary_pos_emb,
)
from .rotary_pos_embedding import (
    MultimodalRotaryEmbedding as MultimodalRotaryEmbedding,
    Rope2DPosEmbRepeated as Rope2DPosEmbRepeated,
    RotaryEmbedding as RotaryEmbedding,
)
from .yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding as YarnRotaryEmbedding,
)
