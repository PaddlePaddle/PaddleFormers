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

"""Processor for Ovis2."""

import numpy as np
import paddle

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput, make_nested_list_of_images
from ..processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import TextInput


class Ovis2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}, "images_kwargs": {}}


def _to_list(value):
    if isinstance(value, paddle.Tensor):
        return value.numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class Ovis2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    valid_processor_kwargs = Ovis2ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<IMG_ATOM>",
        image_seq_length=256,
        prompt_image_token="<image>",
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        # HF checkpoints store the internal image token separately from the
        # public prompt marker used before expanding image patches.
        self.image_token = image_token
        self.prompt_image_token = prompt_image_token
        if not self.prompt_image_token:
            raise ValueError("`prompt_image_token` must not be empty.")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _image_placeholder(self, grid):
        num_rows, num_columns = [int(value) for value in _to_list(grid)]
        placeholder = f"<IMG_START>{self.image_token * self.image_seq_length}<IMG_GRID>"
        if num_rows * num_columns > 1:
            for row in range(num_rows):
                for column in range(num_columns):
                    placeholder += self.image_token * self.image_seq_length
                    if column < num_columns - 1:
                        placeholder += "<IMG_COL>"
                if row < num_rows - 1:
                    placeholder += "<IMG_ROW>"
        return placeholder + "<IMG_END>"

    def expand_image_placeholders(
        self, text: list[TextInput], grids, *, prompt_image_token: str | None = None
    ) -> list[TextInput]:
        """Expand images using the processor marker or an explicit template marker."""
        prompt_image_token = self.prompt_image_token if prompt_image_token is None else prompt_image_token
        if not prompt_image_token:
            raise ValueError("`prompt_image_token` must not be empty.")
        grids = list(_to_list(grids))
        num_images = sum(prompt.count(prompt_image_token) for prompt in text)
        if num_images != len(grids):
            raise ValueError(f"Prompt contained {num_images} image tokens but received {len(grids)} image grids.")
        image_index = 0
        expanded_text = []
        for prompt in text:
            parts = prompt.split(prompt_image_token)
            expanded = parts[0]
            for part in parts[1:]:
                expanded += self._image_placeholder(grids[image_index]) + part
                image_index += 1
            expanded_text.append(expanded)
        return expanded_text

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[Ovis2ProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Ovis2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if isinstance(text, str):
            text = [text]
        elif text is not None and (not isinstance(text, list) or (text and not isinstance(text[0], str))):
            raise TypeError("`text` must be a string or a list of strings.")

        image_inputs = {}
        if images is not None:
            images = self.image_processor.fetch_images(images)
            batched_images = make_nested_list_of_images(images)
            if text is None:
                text = [" ".join([self.prompt_image_token] * len(batch)) for batch in batched_images]
            if len(batched_images) != len(text):
                raise ValueError(f"Received {len(batched_images)} image batches but {len(text)} text prompts.")

            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            grids = list(_to_list(image_inputs.pop("grids")))
            for prompt, batch in zip(text, batched_images):
                if prompt.count(self.prompt_image_token) != len(batch):
                    raise ValueError(
                        f"Prompt contained {prompt.count(self.prompt_image_token)} image tokens but received {len(batch)} images."
                    )
            text = self.expand_image_placeholders(text, grids)

        return_tensors = output_kwargs["text_kwargs"].get("return_tensors")
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        image_names = [name for name in self.image_processor.model_input_names if name != "grids"]
        return list(dict.fromkeys(self.tokenizer.model_input_names + image_names))


__all__ = ["Ovis2Processor", "Ovis2ProcessorKwargs"]
