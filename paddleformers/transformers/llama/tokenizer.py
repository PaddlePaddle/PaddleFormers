# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as spm

from ...utils.log import logger
from .. import PretrainedTokenizer
from ..convert_slow_tokenizer import import_protobuf

__all__ = ["LlamaTokenizer", "Llama3Tokenizer"]


class LlamaTokenizer(PretrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    resource_files_names = {
        "vocab_file": "sentencepiece.bpe.model",
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "__internal_testing__/micro-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "__internal_testing__/tiny-random-llama": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-7b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-13b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-30b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
            "facebook/llama-65b": "https://bj.bcebos.com/paddlenlp/models/transformers/llama/sentencepiece.bpe.model",
        },
    }

    pretrained_init_configuration = {
        "__internal_testing__/micro-random-llama": {},
        "__internal_testing__/tiny-random-llama": {},
        "facebook/llama-7b": {},
        "facebook/llama-13b": {},
        "facebook/llama-30b": {},
        "facebook/llama-65b": {},
    }
    padding_side = "left"

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=False,
        sp_model_kwargs=None,
        decode_with_prefix_space=False,
        **kwargs
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", True))

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def __len__(self):
        """
        Returns the vocabulary size. added_tokens_encoder has to be added in the sp_model
        """
        added_size = 0

        for id in self.added_tokens_decoder:
            if id >= self.sp_model.get_piece_size():
                added_size += 1

        return self.vocab_size + added_size

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_spm_processor(self, from_slow=True):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.id_to_piece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]


"""Copied Tokenization classes for QWen."""

import base64
import unicodedata
from typing import Collection, Set

from ...utils.import_utils import is_tiktoken_available
from .. import PretrainedTokenizer
from ..tokenizer_utils_base import AddedToken

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PAT_STR = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
BEGINOFTEXT = "<|begin_of_text|>"
ENDOFTEXT = "<|end_of_text|>"
IMSTART = "<|start_header_id|>"
IMEND = "<|end_header_id|>"
EOTID = "<|eot_id|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|reserved_special_token_{i}|>" for i in range(251)))
SPECIAL_TOKENS = (BEGINOFTEXT, ENDOFTEXT) + EXTRAS[0:4] + (IMSTART, IMEND, EXTRAS[4], EOTID) + EXTRAS[5:]

tiktoken = None


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)
    }


class Llama3Tokenizer(PretrainedTokenizer):
    """QWen tokenizer."""

    model_input_names = ["input_ids", "attention_mask", "position_ids"]
    resource_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        errors="replace",
        padding_side="left",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        if not is_tiktoken_available():
            raise ValueError("tiktoken is not installed, please install it use: pip install tiktoken")

        import tiktoken as tk

        tiktoken = tk

        self.errors = errors  # how to handle errors in decoding

        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)  # type: dict[bytes, int]
        self.special_tokens = {
            token: index for index, token in enumerate(SPECIAL_TOKENS, start=len(self.mergeable_ranks))
        }
        enc = tiktoken.Encoding(
            "Llama3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}  # type: dict[int, bytes|str]
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc  # type: tiktoken.Encoding

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.bod_id = self.special_tokens[BEGINOFTEXT]
        self.eod_id = self.special_tokens[ENDOFTEXT]
        self.start_header_id = self.special_tokens[IMSTART]
        self.end_header_id = self.special_tokens[IMEND]
        self.eot_id = self.special_tokens[EOTID]

        if "pad_token_id" in kwargs:
            self.pad_token_id = kwargs["pad_token_id"]
        if "eos_token_id" in kwargs:
            self.eos_token_id = kwargs["eos_token_id"]

        self.bos_token = BEGINOFTEXT
        self.eos_token = ENDOFTEXT
        self.bos_token_id = self.bod_id
        self.eos_token_id = self.eod_id
        if "pad_token" not in kwargs:
            self.pad_token = self.convert_ids_to_tokens(self.eos_token_id)
            kwargs["pad_token"] = self.pad_token

        super().__init__(**kwargs)

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return {**self.mergeable_ranks, **self.special_tokens}

    def convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.mergeable_ranks.get(token))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.decoder[ids]
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index >= len(self.mergeable_ranks):
                continue
            if index in self.decoder:
                tokens.append(self.decoder[index])
        return tokens

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS:
                logger.info(f"adding a special token '{surface_form}'.")
                token_id = len(self.mergeable_ranks) + len(self.special_tokens)
                self.special_tokens[surface_form] = token_id
                self.decoder[token_id] = surface_form

        import tiktoken as tk

        tiktoken = tk
        enc = tiktoken.Encoding(
            "Llama3",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.tokenizer = enc  # type: tiktoken.Encoding

        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "tokenizer.model")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bod_id] if self.add_bos_token else []
        eos_token_id = [self.eod_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i <= len(self.mergeable_ranks)]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)
