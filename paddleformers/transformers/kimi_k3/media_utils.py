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

import base64
import functools
import io
import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from PIL import Image


class ImageInput(TypedDict):
    type: Literal["image"]
    image: Image.Image


MediaInput = ImageInput


@dataclass
class TransparentBgConfig:
    """The config of the transparent background."""

    pattern: Literal["white", "black", "gray", "chessboard"] = "black"
    """The pattern of the transparent background."""

    chessboard_square_size: int = 16
    """The size of the squares in the chessboard background."""

    chessboard_square_on_top_left: bool = True
    """Whether to start the chessboard with a white square on the top left."""

    chessboard_white_value: int = 255
    """The value of the white pixels in the background."""

    chessboard_gray_value: int = 200
    """The value of the gray pixels in the background."""


@functools.lru_cache(maxsize=256)
def _create_chessboard_background(
    height: int,
    width: int,
    square_size: int,
    square_on_top_left: bool,
    white_value: int,
    gray_value: int,
) -> np.ndarray:
    """Create a chessboard background."""
    bg = np.ones((height, width, 3), dtype=np.uint8) * white_value
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (y // square_size + x // square_size) % 2 == (1 if square_on_top_left else 0):
                bg[y : y + square_size, x : x + square_size] = gray_value
    return bg


def fill_transparent_bg_with(
    image: Image.Image,
    transparent_bg_config: TransparentBgConfig | None = None,
) -> Image.Image:
    """Composite a (possibly) transparent image onto a configured background.

    When ``transparent_bg_config`` is ``None``, the image is simply converted
    to RGB (preserving the historical behavior). Otherwise the alpha channel
    is alpha-composited over a background generated according to the config.
    """
    if transparent_bg_config is None:
        return image.convert("RGB")

    if image.mode == "RGB":
        return image

    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")

    img = np.array(image.convert("RGBA"))
    height, width = img.shape[:2]
    bg_pattern = transparent_bg_config.pattern
    if bg_pattern == "white":
        bg = np.full((height, width, 3), 255, dtype=np.uint8)
    elif bg_pattern == "black":
        bg = np.zeros((height, width, 3), dtype=np.uint8)
    elif bg_pattern == "gray":
        bg = np.full((height, width, 3), 128, dtype=np.uint8)
    elif bg_pattern == "chessboard":
        bg = _create_chessboard_background(
            height,
            width,
            transparent_bg_config.chessboard_square_size,
            transparent_bg_config.chessboard_square_on_top_left,
            transparent_bg_config.chessboard_white_value,
            transparent_bg_config.chessboard_gray_value,
        )
    else:
        raise ValueError(f"Invalid background pattern: {bg_pattern}")

    alpha = img[:, :, 3]
    img_rgb = img[:, :, :3]
    alpha_normalized = alpha.astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha_normalized] * 3, axis=2)
    result = alpha_3d * img_rgb + (1 - alpha_3d) * bg
    result = result.astype(np.uint8)
    return Image.fromarray(result)


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
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

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert (
            token_height * merge_kernel_size <= patch_limit_on_one_side
        ), f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > patch_limit_on_one_side {patch_limit_on_one_side}"
        assert (
            token_width * merge_kernel_size <= patch_limit_on_one_side
        ), f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > patch_limit_on_one_side {patch_limit_on_one_side}"

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def _to_pil(
    data: str | bytes | Image.Image,
    transparent_bg_config: TransparentBgConfig | None = None,
    to_rgb: bool = True,
) -> Image.Image:
    """Load an image and (optionally) composite its transparent background.

    Args:
        data: A PIL Image, a base64 ``data:`` URL, a file path, or raw bytes.
        transparent_bg_config: The config used to fill the transparent
            background. ``None`` keeps the historical behavior of converting
            to RGB without compositing.
        to_rgb: If ``False`` the image is returned as-is (the
            ``transparent_bg_config`` is ignored). The caller is then
            expected to call :func:`fill_transparent_bg_with` later — e.g.
            after a resize.
    """
    if isinstance(data, Image.Image):
        image = data
    elif isinstance(data, str):
        if data.startswith("data:"):
            raw_base64 = data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(raw_base64)))
        else:
            image = Image.open(data)
    elif isinstance(data, bytes):
        image = Image.open(io.BytesIO(data))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    if not to_rgb:
        return image

    return fill_transparent_bg_with(image, transparent_bg_config)


def ensure_media_type(
    media: MediaInput,
    transparent_bg_config: TransparentBgConfig | None = None,
    transparent_bg_fill_stage: Literal["before_resize", "after_resize"] = "before_resize",
) -> MediaInput:
    if media["type"] == "image":
        media["image"] = _to_pil(
            media["image"],
            transparent_bg_config=transparent_bg_config,
            to_rgb=transparent_bg_fill_stage == "before_resize",
        )
        return media
    else:
        raise ValueError(f"Unsupported media type: {media['type']}")


def image_to_np(
    image: Image.Image,
    resize_to: tuple[int, int] | None = None,
    mode: str = "resize",
    raise_error_for_ill_resize: bool = True,
    transparent_bg_config: TransparentBgConfig | None = None,
    transparent_bg_fill_stage: Literal["before_resize", "after_resize"] = "before_resize",
) -> np.ndarray:
    """Convert an image to a numpy array.

    Args:
        content: The image to convert.
        resize_to: The size to resize the image to.
        mode: The mode to resize the image to.
        raise_error_for_ill_resize: Whether to raise an error for ill-sized resize.
        transparent_bg_config: The config of the transparent background. Only
            used when ``transparent_bg_fill_stage == "after_resize"`` (the
            caller is responsible for filling before resize otherwise).
        transparent_bg_fill_stage: When to composite the transparent
            background — before or after the resize step.

    Returns:
        A numpy array.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    if resize_to is not None:
        if mode == "resize":
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
            if transparent_bg_fill_stage == "after_resize":
                image = fill_transparent_bg_with(image, transparent_bg_config)

        elif mode == "rescale_and_pad_to_center":
            scale = min(resize_to[0] / image.width, resize_to[1] / image.height, 1.0)
            new_width = round(image.width * scale)
            new_height = round(image.height * scale)
            if new_width == 0 or new_height == 0:
                if raise_error_for_ill_resize:
                    raise ValueError(f"Invalid resize to: {resize_to}, from image size: {image.size}")
                else:
                    return np.zeros((resize_to[1], resize_to[0], 3), dtype=np.uint8)

            image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
            if transparent_bg_fill_stage == "after_resize":
                image = fill_transparent_bg_with(image, transparent_bg_config)
            padding_left = (resize_to[0] - new_width) // 2
            padding_right = resize_to[0] - new_width - padding_left
            padding_top = (resize_to[1] - new_height) // 2
            padding_bottom = resize_to[1] - new_height - padding_top
            image = np.asarray(image)
            image = np.pad(
                image,
                ((padding_top, padding_bottom), (padding_left, padding_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            assert image.shape == (resize_to[1], resize_to[0], 3)

        elif mode == "rescale_and_pad_to_rightbottom":
            scale = min(resize_to[0] / image.width, resize_to[1] / image.height, 1.0)
            new_width = round(image.width * scale)
            new_height = round(image.height * scale)
            if new_width == 0 or new_height == 0:
                if raise_error_for_ill_resize:
                    raise ValueError(f"Invalid resize to: {resize_to}, from image size: {image.size}")
                else:
                    return np.zeros((resize_to[1], resize_to[0], 3), dtype=np.uint8)

            image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
            if transparent_bg_fill_stage == "after_resize":
                image = fill_transparent_bg_with(image, transparent_bg_config)
            padding_right = resize_to[0] - new_width
            padding_bottom = resize_to[1] - new_height
            image = np.asarray(image)
            image = np.pad(
                image,
                ((0, padding_bottom), (0, padding_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            assert image.shape == (resize_to[1], resize_to[0], 3)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    if isinstance(image, Image.Image):
        return np.asarray(image)
    else:
        return image


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


def _to_tensor(data, **kwargs):
    import paddle

    if isinstance(data, np.ndarray):
        return paddle.to_tensor(data, **kwargs)
    elif isinstance(data, paddle.Tensor):
        return paddle.cast(data, kwargs["dtype"]) if "dtype" in kwargs else data
    elif isinstance(data, list):
        return [_to_tensor(item, **kwargs) for item in data]
    elif isinstance(data, tuple):
        return tuple(_to_tensor(item, **kwargs) for item in data)
    elif isinstance(data, dict):
        return {k: _to_tensor(v, **kwargs) for k, v in data.items()}
    elif data is None:
        return None
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
