# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 OpenGVLab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import List, Optional, Union

import numpy as np
import paddle
from PIL import Image

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor
from ..image_utils import ImageInput, PILImageResampling, is_valid_image, to_numpy_array

__all__ = ["InternVLImageProcessor"]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _to_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = to_numpy_array(image)
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255
        array = array.astype("uint8")
    return Image.fromarray(array).convert("RGB")


class InternVLImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "num_patches_list", "image_flags"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict] = None,
        resample: int = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_patches: int = 1,
        max_patches: int = 12,
        use_thumbnail: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 448, "width": 448}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STD
        self.do_convert_rgb = do_convert_rgb
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.use_thumbnail = use_thumbnail

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

    def dynamic_preprocess(self, image, min_num=None, max_num=None, image_size=None, use_thumbnail=None):
        min_num = min_num if min_num is not None else self.min_patches
        max_num = max_num if max_num is not None else self.max_patches
        image_size = image_size if image_size is not None else self.size["height"]
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail

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
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height), resample=self.resample)
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
            processed_images.append(image.resize((image_size, image_size), resample=self.resample))
        return processed_images

    def _preprocess_tile(self, image):
        image_size = self.size["height"]
        if self.do_resize:
            image = image.resize((image_size, image_size), resample=self.resample)
        array = np.asarray(image).astype("float32")
        if self.do_rescale:
            array = array * self.rescale_factor
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype="float32")
            std = np.asarray(self.image_std, dtype="float32")
            array = (array - mean) / std
        return np.transpose(array, (2, 0, 1))

    def __call__(self, images: ImageInput = None, return_tensors=None, **kwargs):
        if images is None:
            return BatchFeature(data={})
        if is_valid_image(images):
            images = [images]
        if not isinstance(images, (list, tuple)) or not all(is_valid_image(image) for image in images):
            raise ValueError("InternVLImageProcessor expects an image or a list of images.")

        pixel_values = []
        num_patches_list = []
        for image in images:
            pil_image = _to_pil_image(image)
            tiles = self.dynamic_preprocess(
                pil_image,
                min_num=kwargs.pop("min_patches", self.min_patches),
                max_num=kwargs.pop("max_patches", self.max_patches),
                image_size=kwargs.pop("image_size", self.size["height"]),
                use_thumbnail=kwargs.pop("use_thumbnail", self.use_thumbnail),
            )
            num_patches_list.append(len(tiles))
            pixel_values.extend([self._preprocess_tile(tile) for tile in tiles])

        data = {
            "pixel_values": np.stack(pixel_values).astype("float32"),
            "num_patches_list": num_patches_list,
            "image_flags": np.ones([len(pixel_values), 1], dtype="int64"),
        }
        if return_tensors == "pd":
            data["pixel_values"] = paddle.to_tensor(data["pixel_values"])
            data["image_flags"] = paddle.to_tensor(data["image_flags"])
        return BatchFeature(data=data)
