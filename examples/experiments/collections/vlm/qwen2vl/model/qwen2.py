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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import paddle
import paddle.nn.functional as F
from paddle import nn

from .base import Qwen2Provider


@dataclass
class Qwen2Provider500M(Qwen2Provider):
    """
    Provider for Qwen 2 0.5B: https://huggingface.co/Qwen/Qwen2-0.5B
    """

    num_hidden_layers: int = 24
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864


@dataclass
class Qwen25Config500M(Qwen2Provider500M):
    """
    Config for Qwen 2.5 0.5B: https://huggingface.co/Qwen/Qwen2.5-0.5B
    """

    seq_length: int = 32768


@dataclass
class Qwen2Provider1P5B(Qwen2Provider):
    """
    Config for Qwen 2 1.5B: https://huggingface.co/Qwen/Qwen2-1.5B
    """

    num_hidden_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960


@dataclass
class Qwen25Provider3B(Qwen2Provider):
    """
    Config for Qwen 2.5 3B: https://huggingface.co/Qwen/Qwen2.5-3B
    """

    num_hidden_layers: int = 36
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    intermediate_size: int = 11008
    vocab_size: int = 151936
    share_embeddings_and_output_weights: bool = True


@dataclass
class Qwen25Provider1P5B(Qwen2Provider1P5B):
    """
    Config for Qwen 2.5 1.5B: https://huggingface.co/Qwen/Qwen2.5-1.5B
    """

    seq_length: int = 131072


@dataclass
class Qwen2Provider7B(Qwen2Provider):
    """
    Config for Qwen 2 7B: https://huggingface.co/Qwen/Qwen2-7B
    """

    num_hidden_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    vocab_size: int = 152064


@dataclass
class Qwen25Provider7B(Qwen2Provider7B):
    """
    Config for Qwen 2.5 7B: https://huggingface.co/Qwen/Qwen2.5-7B
    """

    seq_length: int = 131072


@dataclass
class Qwen25Provider14B(Qwen2Provider):
    """
    Config for Qwen 2.5 14B: https://huggingface.co/Qwen/Qwen2.5-14B
    """

    num_hidden_layers: int = 48
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    intermediate_size: int = 13824
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    seq_length: int = 131072


@dataclass
class Qwen25Provider32B(Qwen2Provider):
    """
    Config for Qwen 2.5 32B: https://huggingface.co/Qwen/Qwen2.5-32B
    """

    num_hidden_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    intermediate_size: int = 27648
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    seq_length: int = 131072


@dataclass
class Qwen2Provider72B(Qwen2Provider):
    """
    Config for Qwen 2 72B: https://huggingface.co/Qwen/Qwen2-72B
    """

    num_hidden_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    intermediate_size: int = 29568
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6


@dataclass
class Qwen25Provider72B(Qwen2Provider72B):
    """
    Config for Qwen 2.5 72B: https://huggingface.co/Qwen/Qwen2.5-72B
    """

    seq_length: int = 131072



__all__ = [
    "Qwen2Provider",
    "Qwen2Provider500M",
    "Qwen2Provider1P5B",
    "Qwen25Provider3B",
    "Qwen2Provider7B",
    "Qwen2Provider72B",
    "Qwen25Provider500M",
    "Qwen25Provider1P5B",
    "Qwen25Provider7B",
    "Qwen25Provider14B",
    "Qwen25Provider32B",
    "Qwen25Provider72B",
]