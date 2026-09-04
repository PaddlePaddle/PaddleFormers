# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Image processor for Llava-NeXT."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_transforms import to_pil_image

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def select_best_resolution(original_size: tuple[int, int], possible_resolutions: list[list[int]]) -> tuple[int, int]:
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution
        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)
    return best_fit


def _to_pil(image, do_convert_rgb=True, input_data_format=None) -> Image.Image:
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB") if do_convert_rgb else image
    # 参考其他多模态模型（paddleocr_vl / glm_ocr 等）的迁移风格：用通用 to_pil_image
    # 把 numpy/tensor 转成 PIL Image，它对灰度/RGBA 会自动设置正确 mode，再由
    # convert("RGB") 统一转成 RGB。
    image = to_pil_image(image)
    return image.convert("RGB") if do_convert_rgb else image


def _resize_keep_ratio(image: Image.Image, target_resolution: tuple[int, int], resample) -> Image.Image:
    target_height, target_width = target_resolution
    width, height = image.size
    scale_w = target_width / width
    scale_h = target_height / height
    # 与 HF `get_patch_output_size` 对齐：用 math.ceil 而非 int 截断。
    # 否则 resize 尺寸会差 1 像素，pad/patch 整体错位，放大成 loss 差异。
    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(width * scale_h), target_width)
    return image.resize((new_width, new_height), resample=resample)


def _pad_to_resolution(image: Image.Image, target_resolution: tuple[int, int]) -> Image.Image:
    target_height, target_width = target_resolution
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    left = (target_width - image.size[0]) // 2
    top = (target_height - image.size[1]) // 2
    canvas.paste(image, (left, top))
    return canvas


def _divide_to_patches(image: Image.Image, patch_size: int) -> list[Image.Image]:
    patches = []
    width, height = image.size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patches.append(image.crop((x, y, x + patch_size, y + patch_size)))
    return patches


def _center_crop(image: Image.Image, crop_size: tuple[int, int]) -> Image.Image:
    crop_height, crop_width = crop_size
    width, height = image.size
    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    right = min(left + crop_width, width)
    bottom = min(top + crop_height, height)
    cropped = image.crop((left, top, right, bottom))
    if cropped.size == (crop_width, crop_height):
        return cropped
    canvas = Image.new("RGB", (crop_width, crop_height), (0, 0, 0))
    paste_left = (crop_width - cropped.size[0]) // 2
    paste_top = (crop_height - cropped.size[1]) // 2
    canvas.paste(cropped, (paste_left, paste_top))
    return canvas


def _normalize(
    image: Image.Image,
    image_mean: Iterable[float],
    image_std: Iterable[float],
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: float,
):
    array = np.asarray(image).astype("float32")
    if do_rescale:
        array = array * rescale_factor
    if do_normalize:
        array = (array - np.asarray(image_mean, dtype="float32")) / np.asarray(image_std, dtype="float32")
    return array


def _normalize_and_format(
    image: Image.Image,
    image_mean: Iterable[float],
    image_std: Iterable[float],
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: float,
    data_format: str,
):
    array = _normalize(image, image_mean, image_std, do_rescale, do_normalize, rescale_factor)
    if data_format == "channels_first":
        return array.transpose(2, 0, 1)
    if data_format == "channels_last":
        return array
    raise ValueError("LlavaNextImageProcessor supports `data_format` values 'channels_first' and 'channels_last'.")


def _to_size_dict(size, default_to_square=False):
    if isinstance(size, int):
        return {"height": size, "width": size} if default_to_square else {"shortest_edge": size}
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            return {"height": size[0], "width": size[0]} if default_to_square else {"shortest_edge": size[0]}
        return {"height": size[0], "width": size[1]}
    return dict(size)


class LlavaNextImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_sizes", "image_grid_thw"]

    def __init__(
        self,
        image_grid_pinpoints=None,
        size=None,
        crop_size=None,
        image_mean=None,
        image_std=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        do_resize=True,
        do_center_crop=True,
        do_convert_rgb=True,
        do_pad=True,
        data_format="channels_first",
        input_data_format=None,
        default_to_square=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_grid_pinpoints = image_grid_pinpoints or [
            [336, 672],
            [672, 336],
            [672, 672],
            [1008, 336],
            [336, 1008],
        ]
        self.size = size or {"shortest_edge": 336}
        self.crop_size = crop_size or {"height": 336, "width": 336}
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_convert_rgb = do_convert_rgb
        self.do_pad = do_pad
        self.data_format = data_format
        self.input_data_format = input_data_format
        self.default_to_square = default_to_square
        self.resample = Image.Resampling.BICUBIC

    def preprocess(self, images, return_tensors=None, **kwargs):
        if images is None:
            return BatchFeature(data={}, tensor_type=return_tensors)
        if not isinstance(images, (list, tuple)):
            images = [images]

        default_to_square = kwargs.get("default_to_square", self.default_to_square)
        size = _to_size_dict(kwargs.get("size", self.size), default_to_square=default_to_square)
        crop_size = _to_size_dict(kwargs.get("crop_size", self.crop_size), default_to_square=True)
        grid_pinpoints = kwargs.get("image_grid_pinpoints", self.image_grid_pinpoints)
        do_pad = kwargs.get("do_pad", self.do_pad)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        rescale_factor = kwargs.get("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        do_resize = kwargs.get("do_resize", self.do_resize)
        do_center_crop = kwargs.get("do_center_crop", self.do_center_crop)
        do_convert_rgb = kwargs.get("do_convert_rgb", self.do_convert_rgb)
        data_format = kwargs.get("data_format", self.data_format)
        input_data_format = kwargs.get("input_data_format", self.input_data_format)
        resample = kwargs.get("resample", self.resample)
        if not do_resize:
            raise ValueError("LlavaNextImageProcessor currently requires `do_resize=True` for any-resolution patches.")
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                "LlavaNextImageProcessor supports `data_format` values 'channels_first' and 'channels_last'."
            )

        if "height" in size and "width" in size:
            base_size = (size["height"], size["width"])
        else:
            base = size.get("shortest_edge", 336)
            base_size = (base, base)
        crop_height = crop_size.get("height", base_size[0])
        crop_width = crop_size.get("width", crop_height)
        patch_size = crop_height

        processed_images = []
        image_sizes = []
        for image in images:
            image = _to_pil(image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format)
            original_size = (image.size[1], image.size[0])
            image_sizes.append(list(original_size))
            best_resolution = select_best_resolution(original_size, grid_pinpoints)
            resized = _resize_keep_ratio(image, best_resolution, resample)
            padded = _pad_to_resolution(resized, best_resolution)
            original = image.resize((base_size[1], base_size[0]), resample=resample)
            patches = [original] + [
                patch.resize((base_size[1], base_size[0]), resample=resample)
                for patch in _divide_to_patches(padded, patch_size)
            ]
            if do_center_crop:
                patches = [_center_crop(patch, (crop_height, crop_width)) for patch in patches]
            processed_images.append(
                np.stack(
                    [
                        _normalize_and_format(
                            p,
                            image_mean,
                            image_std,
                            do_rescale,
                            do_normalize,
                            rescale_factor,
                            data_format,
                        )
                        for p in patches
                    ]
                )
            )

        if do_pad:
            max_patches = max(x.shape[0] for x in processed_images)
            processed_images = [
                np.pad(x, ((0, max_patches - x.shape[0]), (0, 0), (0, 0), (0, 0))).astype("float32")
                for x in processed_images
            ]

        return BatchFeature(
            data={
                "pixel_values": processed_images,
                "image_sizes": np.asarray(image_sizes, dtype="int64"),
                "image_grid_thw": np.asarray(image_sizes, dtype="int64"),
            },
            tensor_type=return_tensors,
        )

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)


__all__ = ["LlavaNextImageProcessor", "select_best_resolution"]
