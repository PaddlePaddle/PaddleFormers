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

"""Image preprocessing for FastVLM."""

import numpy as np
from PIL import Image
from transformers import CLIPImageProcessor
from transformers.image_utils import (
    ChannelDimension,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
)

from ..image_processing_utils import PaddleImageProcessingMixin


class FastVLMImageProcessor(PaddleImageProcessingMixin, CLIPImageProcessor):
    def __init__(self, image_size=1024, image_aspect_ratio="pad", **kwargs):
        self.image_aspect_ratio = image_aspect_ratio
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": image_size})
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("crop_size", {"height": image_size, "width": image_size})
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.0, 0.0, 0.0])
        kwargs.setdefault("image_std", [1.0, 1.0, 1.0])
        super().__init__(**kwargs)

    def preprocess(self, images, **kwargs):
        if self.image_aspect_ratio == "pad":
            padded_images = []
            image_mean = kwargs.get("image_mean", self.image_mean)
            for image in make_list_of_images(images):
                if isinstance(image, Image.Image):
                    image = image.convert("RGB")
                image = to_numpy_array(image)
                channel_format = kwargs.get("input_data_format") or infer_channel_dimension_format(image)
                channels_first = channel_format == ChannelDimension.FIRST
                if channels_first:
                    image = np.moveaxis(image, 0, -1)
                height, width, channels = image.shape
                size = max(height, width)
                # Match upstream expand2square: fill with int(mean * 255).
                # Preserve floating point [0, 1] image inputs without quantizing them.
                background = np.asarray([int(value * 255) for value in image_mean])
                if np.issubdtype(image.dtype, np.floating) and is_scaled_image(image):
                    background = background / 255.0
                padded = np.empty((size, size, channels), dtype=image.dtype)
                padded[...] = background
                top, left = (size - height) // 2, (size - width) // 2
                padded[top : top + height, left : left + width] = image
                padded_images.append(np.moveaxis(padded, -1, 0) if channels_first else padded)
            images = padded_images
        return super().preprocess(images, **kwargs)


__all__ = ["FastVLMImageProcessor"]
