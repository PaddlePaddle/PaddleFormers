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

# Original source: phi4-code-raw/tokenizer_config.json
# tokenizer_class: GPT2Tokenizer, BPE vocab_size=200064
# Special tokens: <|endoftext|>(199999), <|assistant|>(200019), <|end|>(200020),
#                 <|user|>(200021), <|system|>(200022), <|tag|>(200028)

from typing import Dict, List, Optional, Tuple, Union

from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers import decoders as tokenizers_decoders
from tokenizers import pre_tokenizers as tokenizers_pre_tokenizers
from tokenizers.models import BPE

from ..tokenizer_utils import PreTrainedTokenizerFast

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# phi4 regex pattern from tokenizer.json pre_tokenizer
_PHI4_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _build_phi4_backend(vocab: Dict[str, int], merges: List[Tuple[str, str]]) -> Tokenizer:
    bpe_model = BPE(
        vocab=vocab,
        merges=merges,
        dropout=None,
        continuing_subword_prefix="",
        end_of_word_suffix="",
        fuse_unk=False,
    )
    backend = Tokenizer(bpe_model)
    backend.pre_tokenizer = tokenizers_pre_tokenizers.Sequence(
        [
            tokenizers_pre_tokenizers.Split(
                pattern=Regex(_PHI4_REGEX),
                behavior="removed",
                invert=True,
            ),
            tokenizers_pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    backend.decoder = tokenizers_decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    return backend


class Phi4Tokenizer(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, List]] = None,
        tokenizer_file: Optional[str] = None,
        tokenizer_object: Optional[Tokenizer] = None,
        bos_token: Union[str, AddedToken] = "<|endoftext|>",
        eos_token: Union[str, AddedToken] = "<|endoftext|>",
        unk_token: Union[str, AddedToken] = "<|endoftext|>",
        pad_token: Union[str, AddedToken] = "<|endoftext|>",
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        add_prefix_space: bool = False,
        **kwargs,
    ):
        if tokenizer_object is None and vocab is not None:
            _vocab = vocab if isinstance(vocab, dict) else {}
            _merges = [tuple(m) if isinstance(m, list) else m for m in (merges or [])]
            tokenizer_object = _build_phi4_backend(_vocab, _merges)

        self._vocab = vocab
        self._merges = merges

        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_object=tokenizer_object,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + bos + token_ids_1 + eos

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id] if self.add_bos_token else []
        eos = [self.eos_token_id] if self.add_eos_token else []
        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos + bos + token_ids_1 + eos) * [0]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        bos = [1] if self.add_bos_token else []
        eos = [1] if self.add_eos_token else []
        if token_ids_1 is None:
            return bos + ([0] * len(token_ids_0)) + eos
        return bos + ([0] * len(token_ids_0)) + eos + bos + ([0] * len(token_ids_1)) + eos

    def encode(
        self,
        text=None,
        text_pair=None,
        add_special_tokens: bool = True,
        padding=False,
        truncation=None,
        max_length: Optional[int] = None,
        stride: int = 0,
        padding_side: Optional[str] = None,
        return_tensors: Optional[str] = None,
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
