# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Paddle/PIL image preprocessing for LFM2-VL."""

import math

import numpy as np
import paddle
from PIL import Image

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import ImageProcessingMixin


def _round_by_factor(number, factor):
    return round(number / factor) * factor


class Lfm2VlImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(
        self,
        downsample_factor=2,
        do_image_splitting=True,
        min_tiles=2,
        max_tiles=10,
        use_thumbnail=True,
        min_image_tokens=64,
        max_image_tokens=256,
        encoder_patch_size=16,
        tile_size=512,
        max_pixels_tolerance=2.0,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        self.downsample_factor = downsample_factor
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.max_num_patches = max(
            max_image_tokens * downsample_factor**2,
            (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        import os

        path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        config = {}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                config = json.load(file).get("image_processor", {})
        config.update(kwargs)
        return cls(**config)

    def smart_resize(self, height, width):
        factor = self.encoder_patch_size * self.downsample_factor
        minimum = self.min_image_tokens * factor**2
        maximum = self.max_image_tokens * factor**2
        target_height = max(factor, _round_by_factor(height, factor))
        target_width = max(factor, _round_by_factor(width, factor))
        if target_height * target_width > maximum:
            beta = math.sqrt((height * width) / maximum)
            target_height = max(factor, math.floor(height / beta / factor) * factor)
            target_width = max(factor, math.floor(width / beta / factor) * factor)
        elif target_height * target_width < minimum:
            beta = math.sqrt(minimum / (height * width))
            target_height = math.ceil(height * beta / factor) * factor
            target_width = math.ceil(width * beta / factor) * factor
        return target_height, target_width

    def _is_too_large(self, height, width):
        factor = self.encoder_patch_size * self.downsample_factor
        target_height = max(self.encoder_patch_size, _round_by_factor(height, factor))
        target_width = max(self.encoder_patch_size, _round_by_factor(width, factor))
        limit = self.max_image_tokens * factor**2 * self.max_pixels_tolerance
        return target_height * target_width > limit

    def _grid_layout(self, height, width):
        ratios = sorted(
            {
                (grid_width, grid_height)
                for count in range(self.min_tiles, self.max_tiles + 1)
                for grid_width in range(1, count + 1)
                for grid_height in range(1, count + 1)
                if self.min_tiles <= grid_width * grid_height <= self.max_tiles
            },
            key=lambda item: item[0] * item[1],
        )
        aspect_ratio = width / height
        best = (1, 1)
        best_difference = float("inf")
        for ratio in ratios:
            difference = abs(aspect_ratio - ratio[0] / ratio[1])
            if difference < best_difference:
                best, best_difference = ratio, difference
            elif difference == best_difference:
                target_area = self.tile_size**2 * ratio[0] * ratio[1]
                if width * height > 0.5 * target_area:
                    best = ratio
        return best

    def _split_and_resize(self, image):
        original_height, original_width = image.height, image.width
        smart_height, smart_width = self.smart_resize(original_height, original_width)
        if self.do_image_splitting and self._is_too_large(original_height, original_width):
            columns, rows = self._grid_layout(original_height, original_width)
            resized = image.resize((columns * self.tile_size, rows * self.tile_size), Image.Resampling.BILINEAR)
            crops = []
            for row in range(rows):
                for column in range(columns):
                    crops.append(
                        resized.crop(
                            (
                                column * self.tile_size,
                                row * self.tile_size,
                                (column + 1) * self.tile_size,
                                (row + 1) * self.tile_size,
                            )
                        )
                    )
            if self.use_thumbnail and rows * columns != 1:
                crops.append(image.resize((smart_width, smart_height), Image.Resampling.BILINEAR))
            return crops, rows, columns, [smart_height, smart_width]
        resized = image.resize((smart_width, smart_height), Image.Resampling.BILINEAR)
        return [resized], 1, 1, [smart_height, smart_width]

    def _patchify(self, image):
        array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        array = (array - np.asarray(self.image_mean, dtype="float32")) / np.asarray(self.image_std, dtype="float32")
        height, width, channels = array.shape
        patch = self.encoder_patch_size
        patch_height, patch_width = height // patch, width // patch
        array = array.reshape([patch_height, patch, patch_width, patch, channels])
        array = array.transpose([0, 2, 1, 3, 4]).reshape([patch_height * patch_width, -1])
        mask = np.zeros([self.max_num_patches], dtype="int64")
        mask[: array.shape[0]] = 1
        padded = np.zeros([self.max_num_patches, array.shape[1]], dtype="float32")
        padded[: array.shape[0]] = array
        return padded, mask, [patch_height, patch_width]

    def __call__(self, images, return_tensors=None, return_row_col_info=True, **kwargs):
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        pixel_values, masks, spatial_shapes = [], [], []
        image_rows, image_cols, image_sizes = [], [], []
        for image in images:
            crops, rows, columns, image_size = self._split_and_resize(image.convert("RGB"))
            image_rows.append(rows)
            image_cols.append(columns)
            image_sizes.append(image_size)
            for crop in crops:
                patches, mask, spatial_shape = self._patchify(crop)
                pixel_values.append(patches)
                masks.append(mask)
                spatial_shapes.append(spatial_shape)
        data = {
            "pixel_values": np.stack(pixel_values),
            "pixel_attention_mask": np.stack(masks),
            "spatial_shapes": np.asarray(spatial_shapes, dtype="int64"),
        }
        if return_row_col_info:
            data.update({"image_rows": image_rows, "image_cols": image_cols, "image_sizes": image_sizes})
        if return_tensors == "pd":
            for key in ["pixel_values", "pixel_attention_mask", "spatial_shapes"]:
                data[key] = paddle.to_tensor(data[key])
        return BatchFeature(data=data)


__all__ = ["Lfm2VlImageProcessor"]
