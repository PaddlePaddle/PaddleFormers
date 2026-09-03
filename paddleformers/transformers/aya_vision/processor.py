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
"""Processor class for AyaVision."""

from typing import Optional, Union

import numpy as np

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput, make_flat_list_of_images
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput


class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
    }


class AyaVisionProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 28,
        img_size: int = 364,
        image_token="<image>",
        downsample_factor: int = 1,
        start_of_img_token="<|START_OF_IMG|>",
        end_of_img_token="<|END_OF_IMG|>",
        img_patch_token="<|IMG_PATCH|>",
        img_line_break_token="<|IMG_LINE_BREAK|>",
        tile_token="TILE",
        tile_global_token="TILE_GLOBAL",
        chat_template=None,
        **kwargs,
    ):
        del kwargs
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token = image_token
        self.patch_size = patch_size * downsample_factor
        self.img_size = img_size

        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token
        self.tile_token = tile_token
        self.tile_global_token = tile_global_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.img_patch_token)
        self._set_image_token_ids(max_num_patches=1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.pop("_from_auto", None)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _prompt_split_image(self, num_patches):
        num_patches = int(num_patches)
        img_patches_per_tile = (self.img_size // self.patch_size) ** 2
        img_string = self.start_of_img_token
        if num_patches > 1:
            for idx in range(1, num_patches):
                img_string += f"{self.tile_token}_{idx}" + self.img_patch_token * img_patches_per_tile

        img_string += self.tile_global_token + self.img_patch_token * img_patches_per_tile
        img_string += self.end_of_img_token
        return img_string

    def _set_image_token_ids(self, max_num_patches):
        tile_tokens = [self.tile_global_token] + [f"{self.tile_token}_{idx}" for idx in range(1, int(max_num_patches))]
        self.image_token_ids = self.tokenizer.convert_tokens_to_ids(
            [
                self.img_patch_token,
                self.start_of_img_token,
                self.end_of_img_token,
                *tile_tokens,
            ]
        )

    @staticmethod
    def _to_int_list(value):
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            return [int(x) for x in value.reshape([-1]).tolist()]
        if isinstance(value, (list, tuple)):
            return [int(x) for x in value]
        return [int(value)]

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        **kwargs: Unpack[AyaVisionProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            AyaVisionProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = list(text)

        image_inputs = {}
        if images is not None:
            if hasattr(self.image_processor, "fetch_images"):
                images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            num_patches = self._to_int_list(image_inputs.pop("num_patches"))

            image_index = 0
            max_num_patches = 1
            processed_text = []
            for prompt in text:
                new_prompt = prompt
                while self.image_token in new_prompt:
                    if image_index >= len(num_patches):
                        raise ValueError("Number of image placeholders exceeds processed images.")
                    max_num_patches = max(max_num_patches, int(num_patches[image_index]))
                    image_tokens = self._prompt_split_image(num_patches[image_index])
                    new_prompt = new_prompt.replace(self.image_token, image_tokens, 1)
                    image_index += 1
                processed_text.append(new_prompt)

            if image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")

            self._set_image_token_ids(max_num_patches)
            text = processed_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_token_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = AyaVisionProcessorKwargs._defaults.get("images_kwargs", {}).copy()
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]

            token_per_patch = (self.img_size // self.patch_size) ** 2
            num_image_tokens = [
                token_per_patch + 3 + sum(token_per_patch + 1 for _ in range(1, num_patches))
                for num_patches in num_image_patches
            ]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


__all__ = ["AyaVisionProcessor"]
