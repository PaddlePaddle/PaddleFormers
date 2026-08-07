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

"""Generation configuration for Fleet models."""

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Args:
        max_length: Maximum length of the generated sequence (including input)
        max_new_tokens: Maximum number of new tokens to generate
        min_length: Minimum length of the generated sequence
        decode_strategy: Decoding strategy, one of ['greedy_search', 'sampling']
        temperature: Temperature for sampling (1.0 = no change, <1.0 = more conservative, >1.0 = more random)
        top_k: Top-k sampling parameter (1 = greedy, >1 = sampling)
        top_p: Top-p (nucleus) sampling parameter (1.0 = no filtering)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        eos_token_id: End of sequence token id
        pad_token_id: Padding token id
        bos_token_id: Beginning of sequence token id
        use_cache: Whether to use KV cache (reserved for V2)
        stop_words: List of stop words (reserved for V2)
        streamer: Streamer object for streaming output (reserved for V2)
    """

    # Sequence length control
    max_length: int = 2048
    max_new_tokens: int = 512
    min_length: int = 0

    # Decoding strategy
    decode_strategy: str = "greedy_search"

    # Sampling parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    # Logits processing
    repetition_penalty: float = 1.0

    # Special tokens
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None

    # Advanced features (reserved for V2)
    use_cache: bool = True
    stop_words: list[str] | None = None
    streamer: Any | None = None

    def __post_init__(self):
        """Validate configuration."""
        if self.decode_strategy not in ["greedy_search", "sampling"]:
            raise ValueError(
                f"decode_strategy must be one of ['greedy_search', 'sampling'], "
                f"got {self.decode_strategy}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not (0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1.0], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive, got {self.repetition_penalty}"
            )
