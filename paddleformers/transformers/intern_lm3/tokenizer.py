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
Tokenization classes for InternLM3.
"""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.utils.log import logger

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"


class InternLM3Tokenizer(PretrainedTokenizer):
    resource_files_names = VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        spaces_for_interleaved_special_tokens=False,
        add_prefix_space=True,
        **kwargs,
    ):
        if spm is None:
            raise ImportError(
                "You need to install sentencepiece to use InternLM3Tokenizer. "
                "See https://github.com/google/sentencepiece for installation."
            )
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self.add_prefix_space = add_prefix_space
        self.spaces_for_interleaved_special_tokens = spaces_for_interleaved_special_tokens

        vocab_size = self.sp_model.get_piece_size()
        self.decoder = {i: self.sp_model.id_to_piece(i) for i in range(vocab_size)}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            sp_model_kwargs=sp_model_kwargs,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text, **kwargs):
        return super().tokenize(text, **kwargs)

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, "")

    def convert_tokens_to_string(self, tokens):
        if tokens and tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.spaces_for_interleaved_special_tokens:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                if (
                    prev_is_special
                    and i == 1
                    and self.add_prefix_space
                    and not token.startswith(SPIECE_UNDERLINE)
                    and self.spaces_for_interleaved_special_tokens
                ):
                    out_string += " "
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

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


__all__ = ["InternLM3Tokenizer"]
