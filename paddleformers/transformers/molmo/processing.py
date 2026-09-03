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
"""Processor for Molmo."""

from __future__ import annotations

from typing import Optional

import numpy as np
import paddle
from PIL import ImageOps
from PIL.Image import Image

from ..auto.tokenizer import AutoTokenizer
from ..processing_utils import ProcessorMixin
from .image_processing import MolmoImageProcessor

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_IM_COL_TOKEN = "<im_col>"
IMAGE_PROMPT = "<|image|>"
EXTRA_TOKENS = (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_COL_TOKEN,
    IMAGE_PROMPT,
)


def get_special_token_ids(tokenizer):
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    if len(ids) != len(EXTRA_TOKENS):
        raise ValueError("Molmo tokenizer special image tokens are not encoded as single tokens.")
    return {token: token_id for token, token_id in zip(EXTRA_TOKENS, ids)}


class MolmoProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MolmoImageProcessor"
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast")
    model_input_names = ["input_ids", "images", "image_masks", "image_input_idx"]

    def __init__(self, image_processor: Optional[MolmoImageProcessor] = None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)
        self._special_tokens = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        image_processor_kwargs = dict(kwargs)
        image_processor_kwargs.pop("use_fast", None)
        try:
            image_processor = MolmoImageProcessor.from_pretrained(
                pretrained_model_name_or_path, **image_processor_kwargs
            )
        except Exception:
            image_processor = MolmoImageProcessor()
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens

    def get_tokens_input(self, prompt, message_format, always_start_with_space):
        prompt = "" if prompt is None else prompt
        if message_format == "none" or message_format is None:
            pass
        elif message_format == "role":
            prompt = "User: " + prompt + " Assistant:"
        else:
            raise NotImplementedError(f"Message format {message_format} not implemented")

        if always_start_with_space:
            prompt = " " + prompt

        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def process(
        self,
        text=None,
        images=None,
        *,
        tokens=None,
        max_crops: int = 12,
        overlap_margins: Optional[list[int]] = None,
        base_image_input_size: Optional[list[int]] = None,
        image_token_length_w: int = 12,
        image_token_length_h: int = 12,
        image_patch_size: int = 14,
        image_padding_mask: bool = True,
        style: str = "long_caption",
        system_prompt: str = "none",
        message_format: str = "role",
        always_start_with_space: bool = True,
        sequence_length: int = 1536,
        return_tensors: str = "pd",
        **kwargs,
    ):
        del style, system_prompt, kwargs
        if tokens is None:
            tokens = self.get_tokens_input(text, message_format, always_start_with_space)

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            image_arrays = []
            for image in images:
                if isinstance(image, Image):
                    image = image.convert("RGB")
                    image = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    image = np.asarray(image)
                    if not (len(image.shape) == 3 and image.shape[-1] == 3):
                        raise ValueError("Molmo images should be RGB images with shape [H, W, 3].")
                    image_arrays.append(image.astype(np.uint8))
            images = image_arrays
            image_idx = [-1] * len(images)
        else:
            image_idx = None

        out = self.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=sequence_length,
            image_patch_token_id=self.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN],
            image_col_token_id=self.special_token_ids[DEFAULT_IM_COL_TOKEN],
            image_start_token_id=self.special_token_ids[DEFAULT_IM_START_TOKEN],
            image_end_token_id=self.special_token_ids[DEFAULT_IM_END_TOKEN],
            max_crops=max_crops,
            overlap_margins=overlap_margins or [4, 4],
            base_image_input_size=base_image_input_size or [336, 336],
            image_token_length_w=image_token_length_w,
            image_token_length_h=image_token_length_h,
            image_patch_size=image_patch_size,
            image_padding_mask=image_padding_mask,
        )

        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        out["input_ids"] = np.pad(out["input_ids"], [[1, 0]], constant_values=bos)
        if "image_input_idx" in out:
            image_input_idx = out["image_input_idx"]
            out["image_input_idx"] = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)

        if return_tensors == "np":
            return out
        if return_tensors != "pd":
            raise ValueError("MolmoProcessor only supports return_tensors='pd' or 'np'.")

        tensor_out = {}
        for key, value in out.items():
            if key in {"input_ids", "image_input_idx"}:
                tensor_out[key] = paddle.to_tensor(value, dtype="int64")
            else:
                tensor_out[key] = paddle.to_tensor(value, dtype="float32")
        return tensor_out


__all__ = ["MolmoProcessor", "MolmoImageProcessor"]
