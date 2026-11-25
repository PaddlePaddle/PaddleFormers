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

"""Tokenization classes for InternLM2.5."""

from typing import List

from ..tokenizer_utils import PretrainedTokenizer
from ..tokenizer_utils_internlm import (
    VOCAB_FILES_NAMES,
    InternLMSentencePieceTokenizerMixin,
)


class InternLM25Tokenizer(InternLMSentencePieceTokenizerMixin, PretrainedTokenizer):
    resource_files_names = VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def encode(
        self,
        text: None = None,
        text_pair: None = None,
        add_special_tokens: bool = True,
        padding: bool | str = False,
        truncation: bool | str | None = None,
        max_length: int | None = None,
        stride: int = 0,
        padding_side: str | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> List[int]:
        padding_strategy, truncation_strategy, max_length, kwargs_updated = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )

        kwargs.update(kwargs_updated)

        encoded_inputs = self._encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )

        return encoded_inputs["input_ids"]
