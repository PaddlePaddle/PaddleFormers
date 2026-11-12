# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math

import numpy as np
import paddle
from PIL import Image

from ..feature_extraction_utils import BatchFeature
from ..image_processing_utils import BaseImageProcessor


def _num_image_tokens(image_size: tuple[int, int], patch_size: tuple[int, int]) -> tuple[int, int]:
    height, width = image_size
    patch_height, patch_width = patch_size
    return (height - 1) // patch_height + 1, (width - 1) // patch_width + 1


def get_resize_output_image_size(
    image,
    size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
) -> tuple[int, int]:
    max_height, max_width = size if isinstance(size, (tuple, list)) else (size, size)
    patch_height, patch_width = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    ratio = max(height / max_height, width / max_width)
    if ratio > 1:
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))

    num_height_tokens, num_width_tokens = _num_image_tokens((height, width), (patch_height, patch_width))
    return num_height_tokens * patch_height, num_width_tokens * patch_width


class PixtralImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        do_resize: bool = True,
        size: dict | None = None,
        resample: int = Image.Resampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_convert_rgb: bool = True,
        patch_size: int | dict = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size or {"longest_edge": 1024}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size

    def _flatten_images(self, images):
        if isinstance(images, (Image.Image, np.ndarray, paddle.Tensor)):
            return [images]
        flat = []
        for image in images:
            if isinstance(image, list):
                flat.extend(image)
            else:
                flat.append(image)
        return flat

    def _to_pil(self, image):
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, paddle.Tensor):
            array = image.numpy()
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                array = array.transpose(1, 2, 0)
            pil_image = Image.fromarray(array.astype("uint8"))
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                array = array.transpose(1, 2, 0)
            pil_image = Image.fromarray(array.astype("uint8"))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return pil_image.convert("RGB") if self.do_convert_rgb else pil_image

    def _get_patch_size(self, patch_size=None):
        patch_size = patch_size or self.patch_size
        if isinstance(patch_size, dict):
            return patch_size.get("height", patch_size.get("longest_edge")), patch_size.get(
                "width", patch_size.get("longest_edge")
            )
        return patch_size, patch_size

    def preprocess(
        self,
        images,
        return_tensors: str | None = None,
        size: dict | None = None,
        patch_size: int | dict | None = None,
        **kwargs,
    ):
        images = [self._to_pil(image) for image in self._flatten_images(images)]
        size = size or self.size
        longest_edge = size["longest_edge"] if isinstance(size, dict) else size
        patch_size_tuple = self._get_patch_size(patch_size)

        processed_images = []
        image_sizes = []
        for image in images:
            if self.do_resize:
                height, width = get_resize_output_image_size(image, (longest_edge, longest_edge), patch_size_tuple)
                image = image.resize((width, height), resample=self.resample)
            array = np.asarray(image).astype("float32").transpose(2, 0, 1)
            if self.do_rescale:
                array = array * self.rescale_factor
            if self.do_normalize:
                mean = np.asarray(self.image_mean, dtype="float32")[:, None, None]
                std = np.asarray(self.image_std, dtype="float32")[:, None, None]
                array = (array - mean) / std
            processed_images.append(array)
            image_sizes.append((array.shape[1], array.shape[2]))

        max_height = max(size[0] for size in image_sizes)
        max_width = max(size[1] for size in image_sizes)
        padded_images = []
        for image, (height, width) in zip(processed_images, image_sizes):
            padded = np.zeros((image.shape[0], max_height, max_width), dtype=image.dtype)
            padded[:, :height, :width] = image
            padded_images.append(padded)

        data = {
            "pixel_values": np.stack(padded_images, axis=0),
            "image_sizes": np.asarray(image_sizes, dtype="int64"),
        }
        if return_tensors == "pd":
            data = {key: paddle.to_tensor(value) for key, value in data.items()}
        return BatchFeature(data=data, tensor_type=None)

    __call__ = preprocess


__all__ = ["PixtralImageProcessor", "get_resize_output_image_size"]
