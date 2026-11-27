# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Native global/local image inputs and real image-token expansion for OCR-2."""

import math

import numpy as np
from PIL import Image, ImageOps
from transformers.image_transforms import to_pil_image
from transformers.image_utils import make_list_of_images

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..processing_utils import ProcessorMixin
from .image_processing import find_closest_aspect_ratio


class DeepseekOcr2ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "pixel_values_local", "num_local_patches"]

    def __init__(
        self,
        size=None,
        tile_size=768,
        crop_to_patches=True,
        min_patches=2,
        max_patches=6,
        image_mean=None,
        image_std=None,
        background_color=None,
        resample=Image.Resampling.BICUBIC,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.size = {"height": 1024, "width": 1024} if size is None else size
        self.tile_size = tile_size
        self.crop_to_patches = crop_to_patches
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.image_mean = [0.5] * 3 if image_mean is None else image_mean
        self.image_std = [0.5] * 3 if image_std is None else image_std
        self.background_color = [127] * 3 if background_color is None else background_color
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

    def preprocess(self, images, return_tensors=None, **kwargs):
        crop = kwargs.pop("crop_to_patches", self.crop_to_patches)
        tile = kwargs.pop("tile_size", self.tile_size)
        low, high = kwargs.pop("min_patches", self.min_patches), kwargs.pop("max_patches", self.max_patches)
        size = kwargs.pop("size", self.size)
        resample = kwargs.pop("resample", self.resample)
        mean, std = kwargs.pop("image_mean", self.image_mean), kwargs.pop("image_std", self.image_std)
        do_rescale, do_normalize = kwargs.pop("do_rescale", self.do_rescale), kwargs.pop(
            "do_normalize", self.do_normalize
        )
        factor = kwargs.pop("rescale_factor", self.rescale_factor)
        background = kwargs.pop("background_color", self.background_color)
        input_format = kwargs.pop("input_data_format", None)
        if tile <= 0 or not 1 <= low <= high:
            raise ValueError("Invalid tile size or patch count range")

        def normalize(image):
            value = np.asarray(image, dtype=np.float32)
            if do_rescale:
                value = (value.astype(np.float64) * factor).astype(np.float32)
            if do_normalize:
                value = (value - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
            return value.transpose(2, 0, 1)

        global_views, local_views, counts = [], [], []
        for image in make_list_of_images(images):
            if not isinstance(image, Image.Image):
                image = to_pil_image(image, input_data_format=input_format)
            image = image.convert("RGB")
            width, height = image.size
            count = 0
            if crop and max(width, height) > tile:
                ratios = sorted(
                    [(w, h) for w in range(1, high + 1) for h in range(1, high + 1) if low <= w * h <= high],
                    key=lambda pair: pair[0] * pair[1],
                )
                cols, rows = find_closest_aspect_ratio(width / height, ratios, width, height, tile)
                resized = image.resize((cols * tile, rows * tile), resample=resample)
                for y in range(rows):
                    for x in range(cols):
                        local_views.append(
                            normalize(resized.crop((x * tile, y * tile, (x + 1) * tile, (y + 1) * tile)))
                        )
                count = cols * rows
            counts.append(count)
            target = size["height"] if crop else tile
            scale = target / max(width, height)
            global_view = image.resize((round(width * scale), round(height * scale)), resample=resample)
            global_view = ImageOps.expand(
                global_view,
                border=(
                    (target - global_view.width) // 2,
                    (target - global_view.height) // 2,
                    target - global_view.width - (target - global_view.width) // 2,
                    target - global_view.height - (target - global_view.height) // 2,
                ),
                fill=tuple(background),
            )
            global_views.append(normalize(global_view))
        data = {"pixel_values": np.stack(global_views), "num_local_patches": counts}
        if local_views:
            data["pixel_values_local"] = np.stack(local_views)
        return BatchFeature(data=data, tensor_type=return_tensors)


class DeepseekOcr2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "DeepseekOcr2ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, patch_size=16, downsample_ratio=4, chat_template=None, **kwargs):
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def image_token_counts(self, image_inputs):
        def query_count(pixels):
            height, width = pixels.shape[-2:]
            patches = math.ceil(height / self.patch_size / self.downsample_ratio) * math.ceil(
                width / self.patch_size / self.downsample_ratio
            )
            return 144 if patches <= 144 else 256

        counts = image_inputs["num_local_patches"]
        global_count = query_count(image_inputs["pixel_values"])
        local_count = query_count(image_inputs["pixel_values_local"]) if "pixel_values_local" in image_inputs else 0
        return [global_count + int(count) * local_count + 1 for count in counts]

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        if images is None or text is None:
            raise ValueError("DeepseekOcr2Processor requires images and text with image placeholders")
        texts = [text] if isinstance(text, str) else list(text)
        image_kwargs = kwargs.pop("images_kwargs", {})
        for key in ("crop_to_patches", "min_patches", "max_patches", "tile_size", "size"):
            if key in kwargs:
                image_kwargs[key] = kwargs.pop(key)
        image_inputs = self.image_processor(images, return_tensors="np", **image_kwargs)
        counts = image_inputs["num_local_patches"].tolist()
        if sum(sample.count(self.image_token) for sample in texts) != len(counts):
            raise ValueError("Supply one image for each image placeholder")

        token_counts = self.image_token_counts(image_inputs)
        index, expanded, expected = 0, [], 0
        for sample in texts:
            parts = sample.split(self.image_token)
            result = parts[0]
            for part in parts[1:]:
                number = token_counts[index]
                result += self.image_token * number + part
                index += 1
                expected += number
            expanded.append(result)
        text_kwargs = kwargs.pop("text_kwargs", {})
        text_kwargs.update(kwargs)
        text_inputs = self.tokenizer(expanded, return_tensors=None, **text_kwargs)
        if sum(row.count(self.image_token_id) for row in text_inputs["input_ids"]) != expected:
            raise ValueError("Tokenization or truncation changed the number of image tokens")
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))


__all__ = ["DeepseekOcr2ImageProcessor", "DeepseekOcr2Processor"]
