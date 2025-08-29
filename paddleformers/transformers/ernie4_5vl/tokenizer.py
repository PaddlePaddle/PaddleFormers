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

"""Tokenization classes for Ernie4_5_VL."""

import os

import sentencepiece as spm
from transformers.utils import logging

# Fix relative import issues
from ..tokenizer_utils import PreTrainedTokenizer

logger = logging.get_logger(__name__)

__all__ = [
    "Ernie4_5_VLTokenizer",
]


class Ernie4_5_VLTokenizer(PreTrainedTokenizer):
    """
    ERNIE 4.5 VL Tokenizer based on SentencePiece with smart tensor support.

    This tokenizer is designed for multimodal inputs including text, images, and videos.
    It inherits from PaddleTokenizerMixin for smart tensor conversion and PreTrainedTokenizer
    for standard HuggingFace functionality.

    Features:
    - SentencePiece-based tokenization
    - Multimodal token support (image/video placeholders)
    - Smart tensor conversion (Paddle/PyTorch/NumPy)
    - Chat template support
    - Enhanced encoding methods including encode_chat_input
    """

    vocab_files_names = {
        "vocab_file": "tokenizer.model",
    }

    model_input_names = ["input_ids", "position_ids", "attention_mask", "labels"]
    padding_side = "right"

    # ERNIE 4.5 VL specific tokens
    SPECIAL_TOKENS = {
        "space": "<mask:1>",
        "gender": "<mask:7>",
        "image_start": "<|im_start|>",
        "image_end": "<|im_end|>",
        "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
        "video_start": "<|VIDEO_START|>",
        "video_end": "<|VIDEO_END|>",
        "video_placeholder": "<|VIDEO_PLACEHOLDER|>",
    }

    def __init__(
        self,
        vocab_file=None,
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
        Initialize the ERNIE 4.5 VL Tokenizer.

        Args:
            vocab_file: Path to the SentencePiece vocabulary file
            bos_token: Beginning of sequence token
            cls_token: Classification token
            eos_token: End of sequence token
            mask_token: Masking token
            pad_token: Padding token
            sep_token: Separator token
            unk_token: Unknown token
            additional_special_tokens: Additional special tokens
            **kwargs: Additional keyword arguments
        """
        # Handle possible parameter renaming

        if vocab_file is None:
            for key in ["tokenizer_file", "model_file", "spm_file"]:
                if key in kwargs:
                    vocab_file = kwargs.pop(key)
                    break

        if vocab_file is None:
            raise ValueError(
                "vocab_file is required. Please provide the path to the tokenizer.model file "
                "or ensure it's available in the model directory."
            )

        # Initialize SentencePiece model first, as parent __init__ might call get_vocab()
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        # Set default additional special tokens
        if additional_special_tokens is None:
            additional_special_tokens = [self.SPECIAL_TOKENS["space"], self.SPECIAL_TOKENS["gender"]]

        # Call PreTrainedTokenizer's __init__
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

        # Save initialization parameters for save_pretrained
        self.init_kwargs = {
            "vocab_file": vocab_file,
            "bos_token": bos_token,
            "cls_token": cls_token,
            "eos_token": eos_token,
            "mask_token": mask_token,
            "pad_token": pad_token,
            "sep_token": sep_token,
            "unk_token": unk_token,
            "additional_special_tokens": additional_special_tokens,
        }
        self.init_kwargs.update(kwargs)

        # Set default attributes
        self.split_special_tokens = False

        # Set internal attributes
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._cls_token = cls_token
        self._sep_token = sep_token
        self._mask_token = mask_token
        self._additional_special_tokens = additional_special_tokens

        # Set context manager attributes
        self._in_target_context_manager = False

        # Set chat template
        self.chat_template = None

        # Set initialization inputs
        self.init_inputs = []

        # Set added tokens decoder
        self._added_tokens_decoder = {}

    # ==================== Pickle Support ====================

    def __getstate__(self):
        """Support for pickle serialization."""
        state = self.__dict__.copy()
        del state["sp_model"]
        return state

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.__dict__.update(state)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    # ==================== Special Token Properties ====================

    @property
    def space_token(self):
        """Return the space token."""
        return self.SPECIAL_TOKENS["space"]

    @property
    def space_token_id(self):
        """Return the ID of the space token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["space"])

    @property
    def gend_token(self):
        """Return the gender token."""
        return self.SPECIAL_TOKENS["gender"]

    @property
    def gend_token_id(self):
        """Return the ID of the gender token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["gender"])

    @property
    def im_start_id(self):
        """Return the ID of the image start token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["image_start"])

    @property
    def im_end_id(self):
        """Return the ID of the image end token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["image_end"])

    @property
    def image_placeholder_id(self):
        """Return the ID of the image placeholder token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["image_placeholder"])

    @property
    def video_placeholder_id(self):
        """Return the ID of the video placeholder token."""
        return self.sp_model.piece_to_id(self.SPECIAL_TOKENS["video_placeholder"])

    # ==================== Core Tokenization Methods ====================

    @property
    def vocab_size(self):
        """Return the size of the vocabulary."""
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """Return the vocabulary as a dictionary mapping tokens to IDs."""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize the input text into pieces."""
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Convert a token to its corresponding ID."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """Convert an ID to its corresponding token."""
        return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens back to a string."""
        current_sub_tokens = []
        out_string = ""

        for token in tokens:
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)

        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # ==================== Core Tokenization Methods ====================

    def convert_tokens_to_ids(self, tokens):
        """Convert a sequence of tokens to a sequence of IDs."""
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        elif isinstance(tokens, list):
            return [self._convert_token_to_id(token) for token in tokens]
        else:
            raise TypeError(f"tokens should be a string or a list of strings, got {type(tokens)}")

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Convert a sequence of IDs to a sequence of tokens."""
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif isinstance(ids, list):
            return [self._convert_id_to_token(id) for id in ids]
        else:
            raise TypeError(f"ids should be an int or a list of ints, got {type(ids)}")

    def _add_tokens(self, new_tokens, special_tokens=False):
        """Add new tokens to the tokenizer."""
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported for SentencePiece tokenizers")
        return 0

    def tokenize(self, text, **kwargs):
        """Tokenize the input text."""
        return self._tokenize(text)

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode a sequence of token IDs to a string."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        # Filter out special tokens if requested
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i not in self.all_special_ids]

        # Convert IDs to tokens
        tokens = [self._convert_id_to_token(id) for id in token_ids]

        # Convert tokens to string
        return self.convert_tokens_to_string(tokens)

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode a sequence of token IDs to a string."""
        return self._decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def encode(self, text, **kwargs):
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def encode_plus(self, text, **kwargs):
        """Encode text to token IDs with additional information."""
        # Get basic encoding
        input_ids = self.encode(text, **kwargs)

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Handle padding if requested
        padding = kwargs.get("padding", False)
        if padding:
            max_length = kwargs.get("max_length", None)
            if max_length is not None:
                # Pad to max_length
                while len(input_ids) < max_length:
                    input_ids.append(self.pad_token_id)
                    attention_mask.append(0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add token_type_ids if requested
        if kwargs.get("return_token_type_ids", False):
            result["token_type_ids"] = [0] * len(input_ids)

        return result

    # ==================== Enhanced Encoding Methods ====================

    def encode_chat_input(self, messages, add_generation_prompt=False, **kwargs):
        """
        Encode chat messages into token IDs.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            add_generation_prompt: Whether to add a generation prompt
            **kwargs: Additional arguments for encoding

        Returns:
            Encoded token IDs or encoding result
        """
        # Apply chat template
        chat_text = self.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)

        # Encode text
        if kwargs.get("return_tensors") is not None:
            return self(chat_text, **kwargs)
        else:
            return self.encode(chat_text, **kwargs)

    def encode_multimodal(self, text, images=None, videos=None, **kwargs):
        """
        Encode multimodal input including text, images, and videos.

        Args:
            text: Input text
            images: List of image paths or URLs
            videos: List of video paths or URLs
            **kwargs: Additional encoding arguments

        Returns:
            Encoding result with multimodal tokens
        """
        # Build multimodal text
        multimodal_text = text

        if images:
            for i, image in enumerate(images):
                multimodal_text += f" {self.SPECIAL_TOKENS['image_start']} {self.SPECIAL_TOKENS['image_placeholder']} {self.SPECIAL_TOKENS['image_end']}"

        if videos:
            for i, video in enumerate(videos):
                multimodal_text += f" {self.SPECIAL_TOKENS['video_start']} {self.SPECIAL_TOKENS['video_placeholder']} {self.SPECIAL_TOKENS['video_end']}"

        # Encode multimodal text
        return self(multimodal_text, **kwargs)

    # ==================== Additional Utility Methods ====================

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        """Get the special tokens mask."""
        if already_has_special_tokens:
            return [0] * len(token_ids)

        special_tokens_mask = [0] * len(token_ids)
        for i, token_id in enumerate(token_ids):
            if token_id in self.all_special_ids:
                special_tokens_mask[i] = 1

        return special_tokens_mask

    def num_special_tokens_to_add(self, pair=False):
        """Return the number of special tokens that will be added."""
        return 0

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence or a pair of sequences."""
        if token_ids_1 is None:
            return token_ids_0
        else:
            return token_ids_0 + [self.sep_token_id] + token_ids_1

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """Create token type IDs from a sequence or a pair of sequences."""
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        else:
            return [0] * len(token_ids_0) + [1] * (len(token_ids_1) + 1)

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """Prepare text for tokenization."""
        return text, kwargs

    # ==================== Utility Methods ====================

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory: The directory to save the vocabulary to
            filename_prefix: Prefix to add to the filename

        Returns:
            Paths to the saved files
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
            import shutil

            shutil.copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load tokenizer from pretrained model

        Args:
            pretrained_model_name_or_path: Model name or path
            *args: Other positional arguments
            **kwargs: Other keyword arguments

        Returns:
            Loaded tokenizer instance
        """
        # Call parent's from_pretrained method to handle all file downloads and path resolution
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
