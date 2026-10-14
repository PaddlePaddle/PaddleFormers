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

import math
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import paddle

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ..image_utils import ImageInput, PILImageResampling, SizeDict
from ..processing_utils import Unpack
from ..tokenizer_utils import TensorType


def get_factors(dividend: int) -> set[int]:
    """Return every integer factor of ``dividend``."""
    factors = set()
    for factor in range(1, int(dividend**0.5) + 1):
        if dividend % factor == 0:
            factors.add(factor)
            factors.add(dividend // factor)
    return factors


@lru_cache(maxsize=8)
def find_supported_resolutions(
    max_num_chunks: int, patch_height: int, patch_width: int
) -> tuple[tuple[int, int], ...]:
    """Build the same ordered canvas list used by Transformers Llama 4."""
    if patch_height != patch_width:
        raise ValueError("`size` must be square.")

    aspect_ratios = defaultdict(list)
    for chunk_size in range(max_num_chunks, 0, -1):
        for factor in sorted(get_factors(chunk_size)):
            height, width = factor, chunk_size // factor
            aspect_ratios[height / width].append((height, width))

    return tuple(
        (height * patch_height, width * patch_width)
        for resolutions in aspect_ratios.values()
        for height, width in resolutions
    )


def get_best_fit(
    image_size: tuple[int, int],
    possible_resolutions: tuple[tuple[int, int], ...],
    resize_to_max_canvas: bool = False,
) -> tuple[int, int]:
    """Select the least-distorting Llama 4 tile canvas."""
    image_height, image_width = image_size
    scales = [min(height / image_height, width / image_width) for height, width in possible_resolutions]
    upscaling_options = [scale for scale in scales if scale >= 1]
    if upscaling_options:
        selected_scale = max(upscaling_options) if resize_to_max_canvas else min(upscaling_options)
    else:
        selected_scale = max(scales)

    candidates = [resolution for resolution, scale in zip(possible_resolutions, scales) if scale == selected_scale]
    return min(candidates, key=lambda resolution: resolution[0] * resolution[1])


def get_max_res_without_distortion(image_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int]:
    """Fit an image inside a target canvas while retaining its aspect ratio."""
    image_height, image_width = image_size
    target_height, target_width = target_size
    scale_width = target_width / image_width
    scale_height = target_height / image_height
    if scale_width < scale_height:
        return min(math.floor(image_height * scale_width), target_height), target_width
    return target_height, min(math.floor(image_width * scale_height), target_width)


class Llama4ImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    max_patches: Optional[int]
    resize_to_max_canvas: Optional[bool]


class Llama4ImageProcessor(BaseImageProcessorFast):
    """Paddle implementation of the Transformers Llama 4 image tiling pipeline."""

    methods_to_wrap = []
    resample = PILImageResampling.BILINEAR
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 336, "width": 336}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_patches = 16
    resize_to_max_canvas = False
    valid_kwargs = Llama4ImageProcessorKwargs
    model_input_names = ["pixel_values", "aspect_ratios"]

    def __init__(self, **kwargs: Unpack[Llama4ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def rescale_and_normalize(
        self,
        images: paddle.Tensor,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
    ) -> paddle.Tensor:
        # The reference implementation performs the combined path in bfloat16.
        if do_rescale and do_normalize:
            target_dtype = "float32" if images.place.is_cpu_place() else "bfloat16"
            images = images.astype(target_dtype) * rescale_factor
            images = self.normalize(images, image_mean, image_std)
        elif do_rescale:
            images = self.rescale(images, rescale_factor)
        elif do_normalize:
            images = self.normalize(images, image_mean, image_std)
        return images

    @staticmethod
    def _split_to_tiles(images: paddle.Tensor, ratio_h: int, ratio_w: int) -> paddle.Tensor:
        batch_size, channels, height, width = images.shape
        tile_height, tile_width = height // ratio_h, width // ratio_w
        return (
            images.reshape([batch_size, channels, ratio_h, tile_height, ratio_w, tile_width])
            .transpose([0, 2, 4, 1, 3, 5])
            .reshape([batch_size, ratio_h * ratio_w, channels, tile_height, tile_width])
        )

    def _preprocess(
        self,
        images: list[paddle.Tensor],
        size: SizeDict,
        max_patches: int,
        resize_to_max_canvas: bool,
        interpolation: Optional[str],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        possible_resolutions = find_supported_resolutions(max_patches, size.height, size.width)
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        grouped_processed_images = {}
        grouped_aspect_ratios = {}

        for shape, stacked_images in grouped_images.items():
            image_size = tuple(stacked_images.shape[-2:])
            target_size = get_best_fit(image_size, possible_resolutions, resize_to_max_canvas)
            resize_canvas = target_size
            if not resize_to_max_canvas:
                resize_canvas = (
                    min(max(image_size[0], size.height), target_size[0]),
                    min(max(image_size[1], size.width), target_size[1]),
                )

            resized_size = get_max_res_without_distortion(image_size, resize_canvas)
            resized_size = (max(resized_size[0], 1), max(resized_size[1], 1))
            processed_images = self.resize(
                stacked_images,
                SizeDict(height=resized_size[0], width=resized_size[1]),
                interpolation=interpolation,
            )
            pad_height = target_size[0] - resized_size[0]
            pad_width = target_size[1] - resized_size[1]
            if processed_images.dtype == paddle.uint8:
                processed_images = processed_images.astype("float32")
            processed_images = paddle.nn.functional.pad(processed_images, [0, pad_width, 0, pad_height])
            processed_images = self.rescale_and_normalize(
                processed_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            ratio_h, ratio_w = target_size[0] // size.height, target_size[1] // size.width
            processed_images = self._split_to_tiles(processed_images, ratio_h, ratio_w)
            if ratio_h * ratio_w > 1:
                global_tiles = self.resize(stacked_images, size, interpolation=interpolation)
                global_tiles = self.rescale_and_normalize(
                    global_tiles, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_images = paddle.concat([processed_images, global_tiles.unsqueeze(1)], axis=1)

            grouped_processed_images[shape] = processed_images
            grouped_aspect_ratios[shape] = paddle.to_tensor(
                [[ratio_h, ratio_w]] * stacked_images.shape[0], dtype="int64"
            )

        processed_images = reorder_images(grouped_processed_images, grouped_images_index)
        aspect_ratios = reorder_images(grouped_aspect_ratios, grouped_images_index)
        if return_tensors:
            processed_images = paddle.concat(processed_images, axis=0)
            aspect_ratios = paddle.stack(aspect_ratios, axis=0)

        return BatchFeature(
            data={"pixel_values": processed_images, "aspect_ratios": aspect_ratios}, tensor_type=return_tensors
        )

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Llama4ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)


__all__ = [
    "Llama4ImageProcessor",
    "Llama4ImageProcessorKwargs",
    "find_supported_resolutions",
    "get_best_fit",
    "get_factors",
    "get_max_res_without_distortion",
]
