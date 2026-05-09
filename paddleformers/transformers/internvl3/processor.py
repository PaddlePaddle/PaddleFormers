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

import numpy as np

from ..image_processing_utils import BatchFeature
from ..image_utils import ImageInput, make_flat_list_of_images
from ..processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput


def concatenate_list(items):
    if len(items) == 0:
        return items

    first = items[0]

    if hasattr(first, "ndim") and hasattr(first, "dtype") and first.__class__.__module__.startswith("paddle"):
        import paddle

        return paddle.concat(items, axis=0)

    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis=0)

    if isinstance(first, list):
        out = []
        for x in items:
            out.extend(x)
        return out

    try:
        return np.concatenate(items, axis=0)
    except Exception:
        out = []
        for x in items:
            if isinstance(x, (list, tuple)):
                out.extend(x)
            else:
                out.append(x)
        return out


class InternVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
    }


class InternVL3ImageProcessorAdapter:
    def __init__(self, image_processor, image_seq_length: int):
        self.image_processor = image_processor
        self.image_seq_length = image_seq_length
        self.merge_size = 1
        self.model_input_names = list(getattr(image_processor, "model_input_names", ["pixel_values"])) + [
            "image_grid_thw"
        ]

    def __getattr__(self, name):
        return getattr(self.image_processor, name)

    def fetch_images(self, images):
        if hasattr(self.image_processor, "fetch_images"):
            return self.image_processor.fetch_images(images)
        return images

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            return BatchFeature(data={}, tensor_type=return_tensors)

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
        pixel_values = image_inputs["pixel_values"]
        num_images = pixel_values.shape[0]
        image_grid_thw = np.asarray([[1, 1, self.image_seq_length]] * num_images, dtype="int64")
        num_patches = [1] * num_images
        image_inputs["image_grid_thw"] = image_grid_thw
        image_inputs["num_patches"] = num_patches
        return image_inputs


class InternVL3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_seq_length: int = 256,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

        self.image_seq_length = image_seq_length
        self.image_processor = InternVL3ImageProcessorAdapter(self.image_processor, image_seq_length=image_seq_length)
        self.start_image_token = getattr(tokenizer, "start_image_token", "<img>")
        self.end_image_token = getattr(tokenizer, "end_image_token", "</img>")
        self.image_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
        self.start_image_token_id = tokenizer.convert_tokens_to_ids(self.start_image_token)
        self.end_image_token_id = tokenizer.convert_tokens_to_ids(self.end_image_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        if min(self.start_image_token_id, self.end_image_token_id, self.image_token_id) < 0:
            raise ValueError(
                "Required multimodal special tokens are missing from tokenizer vocabulary: "
                f"{self.start_image_token}, {self.end_image_token}, {self.image_token}"
            )
        self.image_ids = [self.image_token_id, self.start_image_token_id, self.end_image_token_id]

    def _insert_image_placeholders(
        self,
        text: list[str],
        image_pixel_values,
        image_num_patches: list[int],
        image_num_patches_indices: np.ndarray,
    ):
        image_index = 0
        processed_text = []
        image_patches = []

        for prompt in text:
            new_prompt = prompt
            while self.image_token in new_prompt:
                start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                end_index = image_num_patches_indices[image_index]
                image_patches.append(image_pixel_values[start_index:end_index])
                replace_str = (
                    f"{self.start_image_token}"
                    f"{self.image_token * self.image_seq_length * image_num_patches[image_index]}"
                    f"{self.end_image_token}"
                )
                new_prompt = new_prompt.replace(self.image_token, replace_str, 1)
                image_index += 1
            processed_text.append(new_prompt)

        return processed_text, image_patches, image_index

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[InternVLProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            InternVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, (list, tuple)):
            text = [text]

        image_num_patches = []
        image_pixel_values = None
        image_num_patches_indices = np.array([0])
        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_patches = image_inputs.pop("num_patches")
            image_pixel_values = image_inputs.pop("pixel_values")
            image_num_patches_indices = np.cumsum(image_num_patches)

        image_inputs = {}
        if images is not None:
            text, image_patches, image_index = self._insert_image_placeholders(
                text,
                image_pixel_values,
                image_num_patches,
                image_num_patches_indices,
            )
            if image_index != len(images):
                raise ValueError("Number of image placeholders in the prompt does not match the number of images.")
            image_inputs = {"pixel_values": concatenate_list(image_patches)}

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        data = {**text_inputs, **image_inputs}

        if return_mm_token_type_ids:
            input_ids = data["input_ids"]
            mm_token_type_ids = []
            for tokenizer_input in input_ids:
                tokenizer_input = np.array(tokenizer_input)
                token_types = np.zeros_like(tokenizer_input)
                token_types[np.isin(tokenizer_input, self.image_ids)] = 1
                mm_token_type_ids.append(token_types.tolist())
            data["mm_token_type_ids"] = mm_token_type_ids

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return tokenizer_input_names + image_processor_input_names

    def to_dict(self, legacy_serialization=True):
        output = {
            "image_seq_length": self.image_seq_length,
            "processor_class": self.__class__.__name__,
        }
        if hasattr(self, "auto_map"):
            output["auto_map"] = self.auto_map
        return output


InternVLProcessor = InternVL3Processor

__all__ = ["InternVL3Processor", "InternVLProcessor"]
