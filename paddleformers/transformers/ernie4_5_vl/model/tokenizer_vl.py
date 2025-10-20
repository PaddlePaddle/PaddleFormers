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

"""
Ernie4_5_VLTokenizer
"""

import random
import os
import re
from shutil import copyfile
from typing import Dict, Optional, Tuple, List
import numpy as np
import sentencepiece as spm

import paddle


from paddleformers.utils.log import logger
from paddleformers.transformers import PretrainedTokenizer
from paddleformers.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    TextInput,
)

coor_num = 1001
NOT_FOUND_TOKEN_ID = -101
NUM_IMAGE_SPECIAL_TOKEN = 2048
NUM_AUDIO_SPECIAL_TOKEN = 1024
SFT_IMAGE_START_TOKEN = "<|IMAGE_START|>"
SFT_IMAGE_END_TOKEN = "<|IMAGE_END|>"
SFT_VIDEO_START_TOKEN = "<|VIDEO_START|>"
SFT_VIDEO_END_TOKEN = "<|VIDEO_END|>"
SFT_ASR_START_TOKEN = "<|ASR_START|>"
SFT_ASR_END_TOKEN = "<|ASR_END|>"

special_tokens_info = {
    "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
    "loc_coor": [f"<|LOC_{i}|>" for i in range(coor_num)],
    "loc_begin_end": ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"],
    "image_begin_end": ["<|BOI|>", "<|EOI|>"],
    "video_begin_end": ["<|BOV|>", "<|EOV|>"],
    "sft_video_begin_end": [SFT_VIDEO_START_TOKEN, SFT_VIDEO_END_TOKEN],
}


class Ernie4_5_VLTokenizer(PretrainedTokenizer):
    """
    Ernie4_5_VLTokenizer

    Attributes:
        resource_files_names (dict): Mapping of resource file names.
        pretrained_resource_files_map (dict): Mapping of pretrained resources.
        pretrained_init_configuration (dict): Mapping of pretrained init configuration.
        model_input_names (list): Model input names expected by the tokenizer
        padding_side (str): Padding side (where to add padding tokens)
    """

    resource_files_names = {
        "vocab_file": "tokenizer.model",
    }
    pretrained_resource_files_map = {"vocab_file": {"ernie-bot-10b": None}}
    pretrained_init_configuration = {
        "ernie-bot-10b": {},
    }
    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        **kwargs,
    ):
        """doc"""
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            verbose=False,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def space_token(self):
        """Return the space token"""
        return "<mask:1>"

    @property
    def space_token_id(self):
        """Return the ID of the space token"""
        return self.sp_model.piece_to_id("<mask:1>")

    @property
    def gend_token(self):
        """Return the gender token"""
        return "<mask:7>"

    @property
    def gend_token_id(self):
        """Return the ID of the gender token"""
        return self.sp_model.piece_to_id("<mask:7>")

    @property
    def im_start_id(self):
        """Return the ID of the image start token"""
        return self.sp_model.piece_to_id("<|im_start|>")

    @property
    def im_end_id(self):
        """Return the ID of the image end token"""
        return self.sp_model.piece_to_id("<|im_end|>")

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary.

        Returns:
            int: The number of tokens in the vocabulary.
        """
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """Get the vocabulary as a dictionary mapping tokens to their IDs.

        Returns:
            dict: A dictionary mapping tokens to their corresponding IDs.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize text using SentencePiece.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of tokens.
        """
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Convert a token (str) to an ID using the vocabulary.

        Args:
            token (str): The token to convert.

        Returns:
            int: The corresponding token ID.
        """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """Convert an ID to a token (str) using the vocabulary.

        Args:
            id (int): The token ID to convert.

        Returns:
            str: The corresponding token.
        """
        return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a single string.

        Args:
            tokens (List[str]): A list of tokens to convert.

        Returns:
            str: The reconstructed string.
        """
        current_sub_tokens = []
        out_string = ""
        # prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                # if not prev_is_special:
                #     out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                # prev_is_special = True

                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                # prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string  # .strip()

    def prepare_for_model(self, *args, **kwargs):
        """doc"""
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
            # logger.warning(f'ErnieBotTokenizer v2 does not support `add_special_tokens`')
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
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
            (filename_prefix + "-" if filename_prefix else "")
            + self.resource_files_names["vocab_file"],
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        return (out_vocab_file,)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok)
                for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(
                pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text
            )

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)

        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _decode(self, *args, **kwargs):
        """doc"""
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)
        return super()._decode(
            *args,
            **kwargs,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )

    def _pad(
        self,
        encoded_inputs: Dict,
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """doc"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if (
                max_length is not None
                and pad_to_multiple_of is not None
                and (max_length % pad_to_multiple_of != 0)
            ):
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and len(required_input) != max_length
            )
            if (
                "attention_mask" in encoded_inputs
                and encoded_inputs["attention_mask"] is not None
            ):
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, paddle.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(
                        f"Unexpected type {type(attention_mask)} of attention_mask, "
                    )
            else:
                attention_mask = np.tril(
                    np.ones((len(required_input), len(required_input)), dtype=np.int64)
                )
                attention_mask = np.expand_dims(attention_mask, axis=0)
            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError(
                        "Invalid padding strategy:" + str(self.padding_side)
                    )
                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs


_tokenizer_cache = {}


def _get_serialized_args(args):
    resulted_args = (
        ("random_seed", args.random_seed),
        ("tokenizer", args.model_name_or_path),  # args.tokenizer
    )
    return resulted_args


def get_image_special_tokens(
    special_tokens_info,
):
    """
    add image special token

    placeholder [<|IMAGE_PLACEHOLDER|>, <|AUDIO_PLACEHOLDER|>, <|VIDEO_PLACEHOLDER|>]

    special tokens [<|BOI|> <|EOI|> <|BOA|> <|EOA|> <|BOV|> <|EOV|>]

    location special tokens [<|LOC_0|> <|LOC_1|> ... <|LOC_1000|>] 1001

    other <|IMAGE_UNUSE:xx|>

    Args:
        tokenizer (ErnieTokenizer): tokenizer
        special_token_ids_start (int, optional): special token begin ids. Defaults to 254208.
        special_token_ids_end (int, optional): special token end ids. Defaults to 256256.
    """
    special_tokens = [
        special_tokens_info["image_placeholder"],
        special_tokens_info["audio_placeholder"],
    ]

    image_unuse_token_num = max(0, NUM_IMAGE_SPECIAL_TOKEN - len(special_tokens))
    for i in range(image_unuse_token_num):
        special_tokens.append(f"<|IMAGE_UNUSE:{i}|>")

    assert NUM_IMAGE_SPECIAL_TOKEN == len(
        special_tokens
    ), f"Image Special Tokens number is not as expected. expected={NUM_IMAGE_SPECIAL_TOKEN}, now={len(special_tokens)}"
    return special_tokens


def get_tokenizer(args):
    """
    return tokenizer
    """

    serialized_args = _get_serialized_args(args)
    if serialized_args in _tokenizer_cache:
        return _tokenizer_cache[serialized_args]
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if os.path.isdir(args.model_name_or_path):
        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            args.model_name_or_path,
            verbose=False,
            padding_side="right",
            model_max_length=args.max_seq_length,
        )
    else:
        tokenizer = Ernie4_5_VLTokenizer(
            args.model_name_or_path,
            verbose=False,
            padding_side="right",
            model_max_length=args.max_seq_length,
        )
        origin_vocab_size = len(tokenizer.get_vocab())
        image_special_tokens = get_image_special_tokens(
            special_tokens_info=special_tokens_info,
        )
        # check vocab_size
        expect_final_vocab_size = (
            origin_vocab_size + NUM_IMAGE_SPECIAL_TOKEN + NUM_AUDIO_SPECIAL_TOKEN
        )
        real_final_vocab_size = len(tokenizer.get_vocab())
        assert (
            real_final_vocab_size == expect_final_vocab_size
        ), f"[ERROR] vocab_size = {real_final_vocab_size} != {expect_final_vocab_size} add too many special tokens!"

        # check image
        image_first_special_tokens = tokenizer.encode(image_special_tokens[0])[
            "input_ids"
        ]
        image_last_special_tokens = tokenizer.encode(image_special_tokens[-1])[
            "input_ids"
        ]

        assert (
            image_first_special_tokens[0] == origin_vocab_size
        ), f"[ERROR] image_first_special_tokens={image_first_special_tokens}"
        assert (
            image_last_special_tokens[0]
            == origin_vocab_size + NUM_IMAGE_SPECIAL_TOKEN - 1
        ), f"[ERROR] image_last_special_tokens={image_last_special_tokens}"

    tokenizer.ignored_index = -100
    _tokenizer_cache[serialized_args] = tokenizer

    return tokenizer
