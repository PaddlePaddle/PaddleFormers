# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Image processor class for PaddleOCR-VL."""

import io
import random

from paddle.vision import transforms
from PIL import Image, ImageOps


class RandomApply:
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


class RandomDiscreteRotation:
    def __init__(self, degrees, interpolation="nearest", expand=True):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand

    def __call__(self, img):
        angle = random.choice(self.degrees)
        return img.rotate(angle, self.interpolation, self.expand)


class JpegCompression:
    def __init__(self, quality_range=(20, 80)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        output = io.BytesIO()
        img.convert("RGB").save(output, "JPEG", quality=quality)
        output.seek(0)
        return Image.open(output)


class RandomScale:
    def __init__(self, scale_range=(0.7, 1.3), interpolation="bicubic"):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        original_width, original_height = img.size
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        new_size = (new_height, new_width)  # transforms.Resize需要 (h, w)

        return transforms.functional.resize(img, new_size, self.interpolation)


class RandomSingleSidePadding:
    def __init__(self, padding_range=(0, 20), fill="white"):
        assert (
            isinstance(padding_range, (tuple, list)) and len(padding_range) == 2
        ), "padding_range must be the tuple or list like (min, max)"
        self.min_pad, self.max_pad = padding_range
        self.fill = fill

    def __call__(self, img):

        pad_amount = random.randint(self.min_pad, self.max_pad)
        if pad_amount == 0:
            return img

        chosen_edge = random.choice(["left", "top", "right", "bottom"])

        pad_left, pad_top, pad_right, pad_bottom = 0, 0, 0, 0

        if chosen_edge == "left":
            pad_left = pad_amount
        elif chosen_edge == "top":
            pad_top = pad_amount
        elif chosen_edge == "right":
            pad_right = pad_amount
        else:  # 'bottom'
            pad_bottom = pad_amount

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return ImageOps.expand(img, border=padding, fill=self.fill)
