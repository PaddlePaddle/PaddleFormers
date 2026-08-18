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
from PIL import Image

from ..image_processing_utils import BatchFeature
from ..image_transforms import to_pil_image
from ..image_utils import ImageInput, load_image, make_flat_list_of_images
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
            "dynamic_image_size": True,
            "min_dynamic_patch": 1,
            "max_dynamic_patch": 12,
            "use_thumbnail": True,
        },
    }


class InternVL3ImageProcessorAdapter:
    def __init__(
        self,
        image_processor,
        image_seq_length: int,
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 12,
        use_thumbnail: bool = True,
        dynamic_image_size: bool = True,
    ):
        self.image_processor = image_processor
        self.image_seq_length = image_seq_length
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.dynamic_image_size = dynamic_image_size
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

    @staticmethod
    def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @classmethod
    def _dynamic_preprocess(cls, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = cls._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        resized_img = image.resize((target_width, target_height), resample=bicubic)
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size), resample=bicubic))

        return processed_images

    @staticmethod
    def _to_pil_rgb(image):
        if isinstance(image, str):
            return load_image(image)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return to_pil_image(image).convert("RGB")

    @staticmethod
    def _extract_square_size(size):
        if size is None:
            return None
        if isinstance(size, int):
            return size
        if isinstance(size, dict):
            height = size.get("height")
            width = size.get("width")
            if height is not None and width is not None and height == width:
                return height
            if "shortest_edge" in size:
                return size["shortest_edge"]
            return height or width
        if isinstance(size, (list, tuple)):
            if len(size) == 0:
                return None
            if len(size) == 1 or size[0] == size[1]:
                return size[0]
            return min(size)
        return None

    def _resolve_image_size(self, kwargs, image_size):
        if image_size is not None:
            return int(image_size)
        return int(
            self._extract_square_size(kwargs.get("crop_size"))
            or self._extract_square_size(getattr(self.image_processor, "crop_size", None))
            or self._extract_square_size(kwargs.get("size"))
            or self._extract_square_size(getattr(self.image_processor, "size", None))
            or 448
        )

    def __call__(self, images=None, return_tensors=None, **kwargs):
        if images is None:
            return BatchFeature(data={}, tensor_type=return_tensors)

        crop_to_patches = kwargs.pop("crop_to_patches", True)
        dynamic_image_size = kwargs.pop("dynamic_image_size", self.dynamic_image_size)
        min_dynamic_patch = int(kwargs.pop("min_dynamic_patch", self.min_dynamic_patch))
        max_dynamic_patch = int(kwargs.pop("max_dynamic_patch", self.max_dynamic_patch))
        use_thumbnail = kwargs.pop("use_thumbnail", self.use_thumbnail)
        image_size = kwargs.pop("image_size", None) or kwargs.pop("input_size", None)
        image_size = self._resolve_image_size(kwargs, image_size)

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)
        num_patches = [1] * len(images)
        if crop_to_patches and dynamic_image_size:
            patched_images = []
            num_patches = []
            for image in images:
                patches = self._dynamic_preprocess(
                    self._to_pil_rgb(image),
                    min_num=min_dynamic_patch,
                    max_num=max_dynamic_patch,
                    image_size=image_size,
                    use_thumbnail=use_thumbnail,
                )
                patched_images.extend(patches)
                num_patches.append(len(patches))
            images = patched_images

        image_inputs = self.image_processor(images=images, return_tensors=return_tensors, **kwargs)
        image_grid_thw = np.asarray(
            [[num_patch, 1, self.image_seq_length] for num_patch in num_patches], dtype="int64"
        )
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
        image_placeholder: str = "<image>",
        min_dynamic_patch: int = 1,
        max_dynamic_patch: int = 12,
        use_thumbnail: bool = True,
        dynamic_image_size: bool = True,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

        self.image_seq_length = image_seq_length
        self.image_placeholder = image_placeholder
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.dynamic_image_size = dynamic_image_size
        self.image_processor = InternVL3ImageProcessorAdapter(
            self.image_processor,
            image_seq_length=image_seq_length,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            use_thumbnail=use_thumbnail,
            dynamic_image_size=dynamic_image_size,
        )
        self.start_image_token = getattr(tokenizer, "start_image_token", "<img>")
        self.end_image_token = getattr(tokenizer, "end_image_token", "</img>")
        self.image_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
        if self.image_placeholder == self.image_token:
            raise ValueError(
                f"image_placeholder ({self.image_placeholder}) must be different from image_token ({self.image_token})."
            )
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
            while self.image_placeholder in new_prompt:
                if image_index >= len(image_num_patches):
                    raise ValueError("Number of image placeholders in the prompt exceeds the number of images.")
                start_index = image_num_patches_indices[image_index - 1] if image_index > 0 else 0
                end_index = image_num_patches_indices[image_index]
                image_patches.append(image_pixel_values[start_index:end_index])
                replace_str = (
                    f"{self.start_image_token}"
                    f"{self.image_token * self.image_seq_length * image_num_patches[image_index]}"
                    f"{self.end_image_token}"
                )
                new_prompt = new_prompt.replace(self.image_placeholder, replace_str, 1)
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
            "image_placeholder": self.image_placeholder,
            "min_dynamic_patch": self.min_dynamic_patch,
            "max_dynamic_patch": self.max_dynamic_patch,
            "use_thumbnail": self.use_thumbnail,
            "dynamic_image_size": self.dynamic_image_size,
            "processor_class": self.__class__.__name__,
        }
        if hasattr(self, "auto_map"):
            output["auto_map"] = self.auto_map
        return output


InternVLProcessor = InternVL3Processor

__all__ = ["InternVL3Processor", "InternVLProcessor"]
