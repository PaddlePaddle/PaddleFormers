# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Tokenization classes for Ernie_45T_VL."""

import os
from shutil import copyfile
from typing import Dict, Optional, Tuple

import numpy as np
import paddle
import sentencepiece as spm

from ...utils.log import logger
from ..tokenizer_utils import PreTrainedTokenizer
from ..tokenizer_utils_base import PaddingStrategy


class Ernie4_5_VLTokenizer(PreTrainedTokenizer):
    """
    Ernie4_5_VLTokenizer
    """

    vocab_files_names = {
        "vocab_file": "tokenizer.model",
    }
    # Model input names expected by the tokenizer
    model_input_names = ["input_ids", "position_ids"]
    # Padding side (where to add padding tokens)
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
        """
        Initialize the Ernie4_5_VLTokenizer

        Args:
            vocab_file (str): Path to the tokenizer vocabulary model.
            bos_token (str, optional): The beginning of sequence token. Defaults to `"<s>"`.
            cls_token (str, optional): The classifier token. Defaults to `"<cls>"`.
            eos_token (str, optional): The end of sequence token. Defaults to `"</s>"`.
            mask_token (str, optional): The masking token. Defaults to `"<mask:0>"`.
            pad_token (str, optional): The padding token. Defaults to `"<pad>"`.
            sep_token (str, optional): The separation token. Defaults to `"<sep>"`.
            unk_token (str, optional): The unknown tokens symbol. Defaults to `"<unk>"`.
            additional_special_tokens (List[str], optional): Additional special tokens to use.
                Defaults to `["<mask:1>", "<mask:7>"]`.
            **kwargs (dict): Additional keyword arguments passed along to the superclass.
        """

        # Store vocabulary file path
        self.vocab_file = vocab_file
        # Initialize SentencePiece processor
        self.sp_model = spm.SentencePieceProcessor()
        # Load the vocabulary model
        self.sp_model.Load(vocab_file)

        # Set default additional special tokens if none provided
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
            **kwargs,
        )

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
        return self.encode("<|IMAGE_START|>")[0]

    @property
    def im_end_id(self):
        """Return the ID of the image end token"""
        return self.encode("<|IMAGE_END|>")[0]

    @property
    def vocab_size(self):
        """Return the size of the vocabulary"""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """Return the vocabulary as a dictionary mapping tokens to IDs"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize the input text into pieces"""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Convert a token to its corresponding ID"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """Convert an ID to its corresponding token"""
        return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a string"""
        current_sub_tokens = []
        out_string = ""

        for token in tokens:
            # Handle special tokens differently
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)

        # Add any remaining sub-tokens
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def prepare_for_model(self, *args, **kwargs):
        """Prepare the tokenized inputs for the model"""
        # Remove add_special_tokens if present (not supported)
        if "add_special_tokens" in kwargs:
            kwargs.pop("add_special_tokens")
        return super().prepare_for_model(*args, **kwargs)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`): The directory to save the vocabulary to
            filename_prefix (`str`, optional): Prefix to add to the filename

        Returns:
            `Tuple(str)`: Paths to the saved files
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Construct output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"],
        )

        # Copy or create vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _decode(self, *args, **kwargs):
        """Decode token_id back to text"""
        # Remove some parameters that aren't used
        kwargs.pop("clean_up_tokenization_spaces", None)
        kwargs.pop("spaces_between_special_tokens", None)

        # Call parent decode method with specific parameters
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
        **kwargs
    ) -> dict:
        """Pad the encoded inputs to the specified length"""
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)

            # Adjust max_length if needed for multiple of padding
            if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
                max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

            # Check if padding is needed
            needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

            # Handle attention mask if present
            if "attention_mask" in encoded_inputs and encoded_inputs["attention_mask"] is not None:
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, paddle.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(f"Unexpected type {type(attention_mask)} of attention_mask, ")
            else:
                # Create default attention mask if none provided
                attention_mask = np.tril(np.ones((len(required_input), len(required_input)), dtype=np.int64))
                attention_mask = np.expand_dims(attention_mask, axis=0)

            # Perform padding if needed
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
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )

        # Call parent padding method
        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )

        # Add attention mask back if needed
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()

        return encoded_inputs


__all__ = ["Ernie4_5_VLTokenizer"]
