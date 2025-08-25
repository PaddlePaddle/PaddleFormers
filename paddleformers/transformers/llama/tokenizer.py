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
import transformers as hf

from ..tokenizer_utils import warp_tokenizer

LlamaTokenizer = warp_tokenizer(hf.LlamaTokenizer)


# Legacy PretrainedTokenizer, will be deprecated in the future.
import base64
import os
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union

from ...utils.import_utils import is_tiktoken_available
from ...utils.log import logger
from ..legacy.tokenizer_utils_base import AddedToken
from ..tokenizer_utils import PretrainedTokenizer

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
    resource_files_names = {"vocab_file": "tokenizer.model"}

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
