# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Paddle Idefics3 image processor."""

from __future__ import annotations

import math
from io import BytesIO
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]
MAX_IMAGE_SIZE = 1456


def _to_pil(image) -> Image.Image:
    if isinstance(image, str):
        with urlopen(image) as response:
            image = Image.open(BytesIO(response.read()))
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = array.transpose(1, 2, 0)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Idefics3ImageProcessor expects RGB images with shape [H, W, 3].")
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _make_nested_images(images):
    if images is None:
        return None
    if isinstance(images, (str, Image.Image)) or not isinstance(images, (list, tuple)):
        return [[images]]
    if len(images) == 0:
        return [[]]
    if isinstance(images[0], (list, tuple)):
        return [list(sample) for sample in images]
    return [list(images)]


def _longest_edge(size):
    if isinstance(size, dict):
        return size.get("longest_edge") or max(size.get("height"), size.get("width"))
    return size


def _resize_to_longest_edge(image: Image.Image, longest_edge: int, resample) -> Image.Image:
    width, height = image.size
    if max(width, height) == longest_edge:
        return image
    scale = longest_edge / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), resample=resample)


def _resize_to_multiple_for_vision(image: Image.Image, max_size: int, resample) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height
    if width >= height:
        new_width = math.ceil(width / max_size) * max_size
        new_height = int(new_width / aspect_ratio)
        new_height = math.ceil(new_height / max_size) * max_size
    else:
        new_height = math.ceil(height / max_size) * max_size
        new_width = int(new_height * aspect_ratio)
        new_width = math.ceil(new_width / max_size) * max_size
    return image.resize((new_width, new_height), resample=resample)


def _normalize(image: Image.Image, image_mean, image_std, do_rescale, do_normalize):
    array = np.asarray(image).astype("float32")
    if do_rescale:
        array = array / 255.0
    if do_normalize:
        array = (array - np.asarray(image_mean, dtype="float32")) / np.asarray(image_std, dtype="float32")
    return array.transpose(2, 0, 1)


def _resize_output_size_rescale_to_max_len(height, width, max_len):
    scale = max_len / max(height, width)
    return int(round(height * scale)), int(round(width * scale))


def _resize_output_size_scale_below_upper_bound(height, width, max_len):
    if max(height, width) <= max_len:
        return height, width
    scale = max_len / max(height, width)
    return int(round(height * scale)), int(round(width * scale))


class Idefics3ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def __init__(
        self,
        size=None,
        max_image_size=None,
        image_mean=None,
        image_std=None,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        do_convert_rgb=True,
        do_image_splitting=True,
        do_pad=True,
        return_row_col_info=False,
        resample=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size or {"longest_edge": 4 * 364}
        self.max_image_size = max_image_size or {"longest_edge": 364}
        self.image_mean = image_mean or IMAGENET_STANDARD_MEAN
        self.image_std = image_std or IMAGENET_STANDARD_STD
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.do_image_splitting = do_image_splitting
        self.do_pad = do_pad
        self.return_row_col_info = return_row_col_info
        self.resample = resample or Image.Resampling.LANCZOS

    def fetch_images(self, images):
        if isinstance(images, str):
            return _to_pil(images)
        if isinstance(images, (list, tuple)):
            return [self.fetch_images(image) for image in images]
        return images

    def split_image(self, image: Image.Image, max_image_size: dict, resample):
        max_size = _longest_edge(max_image_size)
        width, height = image.size
        if height > max_size or width > max_size:
            rows = math.ceil(height / max_size)
            cols = math.ceil(width / max_size)
            padded = Image.new("RGB", (cols * max_size, rows * max_size))
            padded.paste(image, (0, 0))
            frames = []
            for row in range(rows):
                for col in range(cols):
                    left = col * max_size
                    top = row * max_size
                    frames.append(padded.crop((left, top, left + max_size, top + max_size)))
            frames.append(image.resize((max_size, max_size), resample=resample))
            return frames, rows, cols
        return [image], 0, 0

    def preprocess(self, images, return_tensors=None, **kwargs):
        if images is None:
            return BatchFeature(data={}, tensor_type=return_tensors)

        nested_images = _make_nested_images(self.fetch_images(images))
        do_resize = kwargs.get("do_resize", self.do_resize)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        do_image_splitting = kwargs.get("do_image_splitting", self.do_image_splitting)
        do_pad = kwargs.get("do_pad", self.do_pad)
        return_row_col_info = kwargs.get("return_row_col_info", self.return_row_col_info)
        size = kwargs.get("size", self.size)
        max_image_size = kwargs.get("max_image_size", self.max_image_size)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        resample = kwargs.get("resample", self.resample)

        processed_samples = []
        rows, cols = [], []
        for sample in nested_images:
            sample_images = []
            sample_rows, sample_cols = [], []
            for image in sample:
                image = _to_pil(image)
                if do_resize:
                    image = _resize_to_longest_edge(image, _longest_edge(size), resample)
                if do_image_splitting:
                    image = _resize_to_multiple_for_vision(image, _longest_edge(max_image_size), resample)
                    frames, image_rows, image_cols = self.split_image(image, max_image_size, resample)
                else:
                    side = _longest_edge(max_image_size)
                    frames, image_rows, image_cols = [image.resize((side, side), resample=resample)], 0, 0
                sample_images.extend(
                    _normalize(frame, image_mean, image_std, do_rescale, do_normalize) for frame in frames
                )
                sample_rows.append(image_rows)
                sample_cols.append(image_cols)
            processed_samples.append(sample_images)
            rows.append(sample_rows)
            cols.append(sample_cols)

        if do_pad:
            max_num_images = max(max(len(sample), 1) for sample in processed_samples)
            max_height = max((image.shape[-2] for sample in processed_samples for image in sample), default=1)
            max_width = max((image.shape[-1] for sample in processed_samples for image in sample), default=1)
            pixel_values = np.zeros(
                (len(processed_samples), max_num_images, 3, max_height, max_width), dtype="float32"
            )
            pixel_attention_mask = np.zeros(
                (len(processed_samples), max_num_images, max_height, max_width), dtype="int64"
            )
            for sample_idx, sample in enumerate(processed_samples):
                for image_idx, image in enumerate(sample):
                    _, height, width = image.shape
                    pixel_values[sample_idx, image_idx, :, :height, :width] = image
                    pixel_attention_mask[sample_idx, image_idx, :height, :width] = 1
            data = {"pixel_values": pixel_values, "pixel_attention_mask": pixel_attention_mask}
        else:
            data = {"pixel_values": processed_samples}

        if return_row_col_info:
            data["rows"] = rows
            data["cols"] = cols
        return BatchFeature(data=data, tensor_type=return_tensors)

    def __call__(self, images, return_tensors=None, **kwargs):
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.pop("_valid_processor_keys", None)
        output.pop("return_row_col_info", None)
        return output

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict):
        do_image_splitting = images_kwargs.get("do_image_splitting", self.do_image_splitting)
        max_image_size = images_kwargs.get("max_image_size", self.max_image_size)
        size = images_kwargs.get("size", self.size)
        num_patches = num_rows = num_cols = 0
        if do_image_splitting:
            height, width = _resize_output_size_rescale_to_max_len(height, width, max_len=_longest_edge(size))
            height, width = _resize_output_size_scale_below_upper_bound(height, width, max_len=MAX_IMAGE_SIZE)
            aspect_ratio = width / height
            max_size = _longest_edge(max_image_size)
            if width >= height:
                resized_width = math.ceil(width / max_size) * max_size
                resized_height = int(width / aspect_ratio)
                resized_height = math.ceil(height / max_size) * max_size
            else:
                resized_height = math.ceil(height / max_size) * max_size
                resized_width = int(height * aspect_ratio)
                resized_width = math.ceil(width / max_size) * max_size
            if resized_height > max_size or resized_width > max_size:
                num_rows = math.ceil(resized_height / max_size)
                num_cols = math.ceil(resized_width / max_size)
                num_patches = num_rows * num_cols + 1
        return num_patches, num_rows, num_cols


__all__ = ["Idefics3ImageProcessor"]
