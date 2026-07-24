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

from dataclasses import dataclass

from paddleformers.fleet.transformer.transformer_config import TransformerConfig


@dataclass
class GPTConfig(TransformerConfig):
    """Configuration object for gpt."""

    vocab_size: int = 1024

    position_embedding_type: str = "rope"

    rotary_percent: float = 1.0

    rotary_base: int = 10000

    rope_scaling: float = 1.0

    max_sequence_length: int = 64

    tie_word_embeddings: bool = False

    moe_expert_fusion: bool = False

    parallel_output: bool = True

    layer_types: list = None
