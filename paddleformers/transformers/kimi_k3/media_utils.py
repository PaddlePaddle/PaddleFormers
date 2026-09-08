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
from typing import Literal, TypedDict

import numpy as np
from PIL import Image


class ImageInput(TypedDict):
    type: Literal["image"]
    image: Image.Image


MediaInput = ImageInput


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
):
    # Apply the patch limits.
    s1 = math.sqrt(in_patch_limit / (max(1.0, width // patch_size) * max(1.0, height // patch_size)))
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    # Calculate the padding to make the height and width divisible by the merge kernel size and patch size.
    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    # Calculate new dimensions after padding and patching
    token_height = (new_h + pad_height) // factor
    token_width = (new_w + pad_width) // factor

    assert (
        token_height * merge_kernel_size <= patch_limit_on_one_side
    ), f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > patch_limit_on_one_side {patch_limit_on_one_side}"
    assert (
        token_width * merge_kernel_size <= patch_limit_on_one_side
    ), f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > patch_limit_on_one_side {patch_limit_on_one_side}"

    return {
        "num_tokens": token_height * token_width,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def ensure_media_type(media: MediaInput) -> MediaInput:
    if media["type"] != "image":
        raise ValueError(f"Unsupported media type: {media['type']}")
    image = media["image"]
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    media["image"] = image.convert("RGB")
    return media


def image_to_np(image: Image.Image, resize_to: tuple[int, int]) -> np.ndarray:
    """Bicubic-resize a PIL image to ``resize_to`` and return it as a numpy array."""
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    return np.asarray(image.resize(resize_to, resample=Image.Resampling.BICUBIC))


def navit_patchify(pixel_values: np.ndarray, patch_size: int) -> dict[str, np.ndarray]:
    """Reshape the pixel values to a navit shape.

    Args:
        pixel_values: np.ndarray, shape (t, h, w, c)
        patch_size: int

    Returns:
        dict[str, np.ndarray]
        - patches: np.ndarray, shape (t * h//patch_size * w//patch_size, c, patch_size, patch_size)
        - grid_thw: np.ndarray, (t, h//patch_size, w//patch_size)
    """
    T, H, W, C = pixel_values.shape
    assert C == 3, "pixel_values must have 3 channels"

    patches = pixel_values.reshape(T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    # (T, H//patch_size, W//patch_size, C, patch_size, patch_size)
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = np.array([T, H // patch_size, W // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


def normalize(x: np.ndarray, mean, std_inv, pixels_dtype: np.dtype = np.float32) -> np.ndarray:
    """Normalize the image.

    Args:
        x: The image to normalize. The shape is (..., 3). The dtype is uint8. The range is [0, 255].
        mean: The mean of the image.
        std_inv: The inverse of the std of the image.
        pixels_dtype: The dtype of the image.
    Returns:
        The normalized image. The shape is (..., 3). The dtype is determined by the pixels_dtype.
    """
    x = (x / 255.0).astype(pixels_dtype)
    x -= mean
    x *= std_inv
    return x
