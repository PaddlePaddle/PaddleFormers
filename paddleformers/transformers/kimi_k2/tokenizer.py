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

import os
from logging import getLogger
from pathlib import Path
from shutil import copyfile
from typing import Dict, Iterator, List, Optional, Tuple, Union, cast

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import AddedToken
from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.tokenization_utils import PreTrainedTokenizer

logger = getLogger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tiktoken.model"}


class KimiK2TikTokenTokenizer(PreTrainedTokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer. See megatron/tokenizer/tiktoken_tokenizer.py.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            The path to the Tiktoken model file.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<|begin_of_text|>",`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<|end_of_text|>"`):
            The end of sequence token.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<|reserved_special_token_249|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. The second to last item in special_tokens.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<|reserved_special_token_250|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (list of `str`, *optional*):
            A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be
            skipped when decoding if `skip_special_tokens` is set to `True`.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    model_input_names = ["input_ids", "attention_mask"]

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    def __init__(
        self,
        vocab_file,
        bos_token: Union[str, AddedToken] = "[BOS]",
        eos_token: Union[str, AddedToken] = "[EOS]",
        unk_token: Union[str, AddedToken, None] = None,
        pad_token: Union[str, AddedToken, None] = None,
        additional_special_tokens: List[str] = None,
        added_tokens_decoder: Optional[dict] = None,
        **kwargs,
    ):
        assert os.path.isfile(vocab_file), vocab_file

        if additional_special_tokens is None:
            additional_special_tokens = [
                "<|im_end|>",
                "<|im_user|>",
                "<|im_assistant|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "[EOT]",
                "<|im_system|>",
                "<|im_middle|>",
            ]

        special_tokens_mapping = {i: added_tokens_decoder[i].content for i in added_tokens_decoder}

        self.vocab_file = vocab_file
        mergeable_ranks = load_tiktoken_bpe(vocab_file)
        num_base_tokens = len(mergeable_ranks)
        self.special_tokens = {
            special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
            for i in range(num_base_tokens, num_base_tokens + self.num_reserved_special_tokens + 2)
        }

        self.model = tiktoken.Encoding(
            name=Path(vocab_file).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {vocab_file}")

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens[str(bos_token)]
        self.eos_id: int = self.special_tokens[str(eos_token)]
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")

        self.pad_id: int = self.special_tokens[str(pad_token)]
        self.unk_id: int = self.special_tokens[str(unk_token)]

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.decoder = {}
        for i in range(self.n_words):
            # Taken from https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
            decoding = "".join(
                [self.byte_encoder[ord(char)] for char in self.model.decode_single_token_bytes(i).decode("latin-1")]
            )
            self.decoder[i] = decoding

        self.encoder = {}
        for i in range(self.n_words):
            if i in self.decoder:
                self.encoder[self.decoder[i]] = i

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.all_special_ids_set = set(self.all_special_ids)

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encodes a string into a list of token IDs.
        Args:
            text (str): The input string to be encoded.
        Returns:
            list[int]: A list of token IDs.
        """
        # If there are other args, we should call super().encode because there are a lot of code
        # to handle those args. supper().encode finally will call _tokenize and _convert_token_to_id.
        # NOTE: our encode method is not compatible with the super().encode method,
        #   e.g. split_special_tokens' default is True in our encode method.
        if len(kwargs) > 0:
            logger.warning(f"Calling super().encode with {kwargs}")
            return super().encode(text, **kwargs)

        assert type(text) is str

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        texts = self.pre_tokenizer_process(text)

        all_substrs = []
        for text in texts:
            substrs = (
                substr
                for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)
                for substr in self._split_whitespaces_or_nonwhitespaces(
                    text[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
                )
            )
            all_substrs.extend(substrs)

        t: List[int] = []
        for substr in all_substrs:
            t.extend(
                # we should consider special token as a common token
                self.model.encode(
                    substr,
                    disallowed_special=(),
                )
            )

        return t

    def decode(self, token_ids: Union[int, List[int]], **kwargs) -> str:
        """
        Decodes a list of token IDs into a string.
        Args:
            token_ids (List[int]): The list of token IDs to be decoded.
        Returns:
            str: The decoded string.
        """
        # If there are other args, we should call super().decode because there are a lot of code
        # to handle those args. supper().encode finally will call convert_tokens_to_string and _convert_id_to_token.
        if len(kwargs) > 0:
            return super().decode(token_ids, **kwargs)

        if type(token_ids) is int:
            token_ids = [token_ids]

        return self.model.decode(cast(List[int], token_ids))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def pre_tokenizer_process(self, text: str) -> List[str]:
        """
        pre-tokenizes the input text into a list of tokens.
        This method is used to split the input text into smaller chunks for internal processing.
        """
        return [text]

    """ ----- Below are the abstract methods required by PreTrainedTokenizer ----- """

    @property
    def vocab_size(self) -> int:
        return self.n_words

    def get_vocab(self) -> Dict[str, int]:
        return self.encoder

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return [self.decoder[t] for t in self.encode(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        return out_string

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", "replace")
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            raise ValueError(f"vocabulary path ({save_directory}) should be a directory")
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
