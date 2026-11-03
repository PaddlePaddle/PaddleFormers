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

from ..feature_extraction_utils import BatchFeature
from ..image_utils import ImageInput, make_flat_list_of_images
from ..processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ..tokenizer_utils_base import PreTokenizedInput, TextInput
from .image_processor import find_supported_resolutions, get_best_fit


class Llama4ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding_side": "left"}}


class Llama4Processor(ProcessorMixin):
    """Combine Llama 4 image preprocessing, visual-token expansion, and tokenization."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        pixel_shuffle_ratio: float = 0.5,
        fake_image_token: str = "<|image|>",
        image_token: str = "<|image|>",
        start_of_image_token: str = "<|image_start|>",
        end_of_image_token: str = "<|image_end|>",
        patch_token: str = "<|patch|>",
        tile_x_separator_token: str = "<|tile_x_separator|>",
        tile_y_separator_token: str = "<|tile_y_separator|>",
        chat_template=None,
        **kwargs,
    ):
        self.downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        self.patch_size = patch_size
        self.pixel_shuffle_ratio = pixel_shuffle_ratio
        self.fake_image_token = fake_image_token
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.start_of_img_token = start_of_image_token
        self.end_of_img_token = end_of_image_token
        self.img_patch_token = patch_token
        self.tile_token = tile_x_separator_token
        self.tile_global_token = tile_y_separator_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _prompt_split_image(self, aspect_ratio, num_patches_per_chunk: int) -> str:
        ratio_h, ratio_w = (int(value) for value in aspect_ratio)
        image_string = self.start_of_img_token
        if ratio_h * ratio_w > 1:
            for row in range(ratio_h):
                for column in range(ratio_w):
                    image_string += self.img_patch_token * num_patches_per_chunk
                    if column < ratio_w - 1:
                        image_string += self.tile_token
                image_string += self.tile_global_token

        image_string += self.image_token
        image_string += self.img_patch_token * num_patches_per_chunk
        image_string += self.end_of_img_token
        return image_string

    def _num_patches_per_chunk(self) -> int:
        image_height = self.image_processor.size["height"]
        image_width = self.image_processor.size["width"]
        return (image_height // self.patch_size) * (image_width // self.patch_size) // self.downsample_ratio

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[Llama4ProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You have to specify text.")

        output_kwargs = self._merge_kwargs(
            Llama4ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not isinstance(text, (list, tuple)):
            text = [text]
        else:
            text = list(text)

        image_inputs = {}
        if images is not None:
            images = make_flat_list_of_images(self.image_processor.fetch_images(images))
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            aspect_ratios = image_inputs.pop("aspect_ratios")
            total_placeholders = sum(prompt.count(self.fake_image_token) for prompt in text)
            if total_placeholders != len(images):
                raise ValueError(
                    f"Found {total_placeholders} placeholders across the batch, but have {len(images)} flattened images."
                )

            image_index = 0
            processed_text = []
            num_patches_per_chunk = self._num_patches_per_chunk()
            for prompt in text:
                prompt_parts = prompt.split(self.fake_image_token)
                expanded_prompt = []
                for part_index, prompt_part in enumerate(prompt_parts):
                    expanded_prompt.append(prompt_part)
                    if part_index < len(prompt_parts) - 1:
                        expanded_prompt.append(
                            self._prompt_split_image(aspect_ratios[image_index], num_patches_per_chunk)
                        )
                        image_index += 1
                processed_text.append("".join(expanded_prompt))
            text = processed_text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """Compute exact visual token counts for ``(height, width)`` image sizes."""
        vision_data = {}
        if image_sizes is not None:
            num_patches_per_chunk = self._num_patches_per_chunk()
            image_height = self.image_processor.size["height"]
            image_width = self.image_processor.size["width"]
            supported_resolutions = find_supported_resolutions(
                self.image_processor.max_patches, image_height, image_width
            )
            num_image_patches = []
            for height, width in image_sizes:
                target_height, target_width = get_best_fit(
                    (height, width),
                    supported_resolutions,
                    self.image_processor.resize_to_max_canvas,
                )
                num_local_tiles = (target_height // image_height) * (target_width // image_width)
                num_image_patches.append(num_local_tiles + int(num_local_tiles > 1))
            vision_data["num_image_tokens"] = [
                num_patches * num_patches_per_chunk for num_patches in num_image_patches
            ]
            vision_data["num_image_patches"] = num_image_patches
        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        image_processor_input_names = [
            name for name in self.image_processor.model_input_names if name != "aspect_ratios"
        ]
        return list(dict.fromkeys(self.tokenizer.model_input_names + image_processor_input_names))


__all__ = ["Llama4Processor", "Llama4ProcessorKwargs"]
