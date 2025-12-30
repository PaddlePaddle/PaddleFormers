# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from paddlefleet.transformer.transformer_config import TransformerConfig
from transformers import PretrainedConfig

from .base import Qwen3VLTextProvider, Qwen3VLProvider, Qwen3VLVisionProvider


@dataclass
class Qwen3VLTextProvider2B(Qwen3VLTextProvider):
    num_hidden_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    intermediate_size: int = 6144
    num_key_value_heads: int = 8


@dataclass
class Qwen3VLTextProvider4B(Qwen3VLTextProvider):
    num_hidden_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    intermediate_size: int = 9728

@dataclass
class Qwen3VLTextProvider8B(Qwen3VLTextProvider):
    num_hidden_layers: int = 36
    hidden_size: int = 4096
    num_attention_heads: int = 48
    intermediate_size: int = 12288

@dataclass
class Qwen3VLTextProvider32B(Qwen3VLTextProvider):
    num_hidden_layers: int = 64
    num_attention_heads: int = 64
    intermediate_size: int = 25600
    hidden_size: int = 5120


@dataclass
class Qwen3VLTextProvider30BA3B(Qwen3VLTextProvider):
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    intermediate_size: int = 6144
    hidden_size: int = 2048


@dataclass
class Qwen3VLTextProvider235BA22B(Qwen3VLTextProvider):
    num_hidden_layers: int = 94
    num_attention_heads: int = 64
    n_toute_experts: int = 128
    num_experts_per_tok: int = 8
    intermediate_size: int = 12288
    hidden_size: int = 4096


@dataclass
class Qwen3VLProvider2B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider2B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4096, hidden_size=1024, num_hidden_layers=24,
            deepstack_visual_indexes=[5, 11, 17], out_hidden_size=2048,
        )
    )
    

@dataclass
class Qwen3VLProvider4B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider4B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4096, hidden_size=1024, num_hidden_layers=24,
            deepstack_visual_indexes=[5, 11, 17], out_hidden_size=2560,
        )
    )


@dataclass
class Qwen3VLProvider8B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider8B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4096, hidden_size=1024, num_hidden_layers=24,
            out_hidden_size=4096,
        )
    )

@dataclass
class Qwen3VLProvider32B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider32B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4304, hidden_size=1152, num_hidden_layers=27,
            out_hidden_size=5120
        )
    )


@dataclass
class Qwen3VLProvider30B_A3B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider30BA3B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4304, hidden_size=1152, num_hidden_layers=27,
            out_hidden_size=2048
        )
    )


@dataclass
class Qwen3VLProvider235B_A22B(Qwen3VLProvider):
    language_transformer_config: TransformerConfig = field(
        default_factory=lambda: Qwen3VLTextProvider235BA22B()
    )
    vision_transformer_config: TransformerConfig | PretrainedConfig = field(
        default_factory=lambda: Qwen3VLVisionProvider(
            num_attention_heads=16, intermediate_size=4304, hidden_size=1152, num_hidden_layers=27,
            out_hidden_size=4096
        )
    )