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
import io
import math
import os
from datetime import datetime, timezone
from typing import List, Literal, Optional, TypedDict

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

try:
    from mecord import VideoReader
except ImportError:
    VideoReader = None


class VideoSpec(BaseModel):
    media_type: str = Literal["video"]
    height: int = Field(..., gt=0, description="video frame height")
    width: int = Field(..., gt=0, description="video frame width")
    num_frames: int = Field(..., gt=0, description="num frames")
    fps: float = Field(..., gt=0, description="average fps")

    # optional, help to accelerate video reading
    key_indices: list[int] = Field(None, description="key indices")
    frame_time_info: dict = Field(None, description="frame time info")


class ImageInput(TypedDict):
    type: Literal["image"]
    image: Image.Image


class VideoChunkInput(TypedDict):
    type: Literal["video_chunk"]
    video_chunk: List[Image.Image]
    prompt: Optional[str] = None


MediaInput = ImageInput | VideoChunkInput


def get_video_meta(video_src: bytes | str | os.PathLike, accurate: bool = True) -> dict:
    """Get the dimensions of a video."""
    if isinstance(video_src, os.PathLike):
        video_src = str(video_src)
    # if b64 string, decode to bytes
    if isinstance(video_src, str) and video_src.startswith("data:video/mp4;base64,"):
        video_src = base64.b64decode(video_src.split(",")[1])
    video = VideoReader(video_src, auto_init=accurate, num_threads=1)
    assert video.num_frames > 0, "Invalid video format."
    assert video.original_width > 0 and video.original_height > 0, "Invalid video format."
    assert video.avg_fps > 0, "Invalid video format."
    return VideoSpec(
        media_type="video",
        height=video.original_height,
        width=video.original_width,
        num_frames=video.num_frames,
        fps=video.avg_fps,
        key_indices=video.key_indices,
        frame_time_info=video.frame_time_info,
    )


def timestamp_as_str(timestamp: float, timestamp_mode: str = "hh:mm:ss.fff") -> str:
    """Convert a timestamp to a string in the format of HH:MM:SS.mmm."""
    if timestamp_mode == "hh:mm:ss.fff":
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%H:%M:%S")
            + f".{int((timestamp % 1) * 1000):03d}"
        )
    elif timestamp_mode == "mm:ss.fff":
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%M:%S")
            + f".{int((timestamp % 1) * 1000):03d}"
        )
    elif timestamp_mode == "mm:ss":
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%M:%S")
    else:
        raise ValueError(f"Invalid timestamp mode: {timestamp_mode}")


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


def navit_resize_video(
    width: int,
    height: int,
    nframes: int,
    avg_fps: float,
    sample_fps: float,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit_each_frame: int,
    patch_limit_on_one_side: int,
    in_patch_limit_total: int | None,
    max_num_frames_each_video: int | None,
    fixed_output_tokens_each_frame: int | None,
):
    sample_fps = min(sample_fps, avg_fps)
    # Calculate the number of frames to sample based on target FPS
    sampled_nframes = max(round(nframes * sample_fps / avg_fps), 1)
    if max_num_frames_each_video is not None:
        sampled_nframes = min(sampled_nframes, max_num_frames_each_video)

    if in_patch_limit_total is not None:
        in_patch_limit_each_frame = min(round(in_patch_limit_total / sampled_nframes), in_patch_limit_each_frame)

    ret = navit_resize_image(
        width,
        height,
        patch_size,
        merge_kernel_size,
        in_patch_limit_each_frame,
        patch_limit_on_one_side,
        fixed_output_tokens_each_frame,
    )
    ret["sampled_nframes"] = sampled_nframes
    return ret


def real_sample_fps_and_max_num_frames(
    type_name: Literal["video", "video_chunk"],
    sample_fps: float,
    max_num_frames_each_video: int | None,
) -> tuple[int, int | None]:
    if type_name == "video":
        return sample_fps, max_num_frames_each_video
    elif type_name == "video_chunk":
        max_num_frames_each_video = None
        sample_fps = math.inf
        return sample_fps, max_num_frames_each_video
    else:
        return math.inf, None


def _to_pil(data: str | bytes):
    if isinstance(data, Image.Image):

        return data.convert("RGB")
    elif isinstance(data, str):
        if data.startswith("data:"):
            raw_base64 = data.split(",")[1]
            return Image.open(io.BytesIO(base64.b64decode(raw_base64))).convert("RGB")
        else:
            return Image.open(data).convert("RGB")
    elif isinstance(data, bytes):
        return Image.open(io.BytesIO(data)).convert("RGB")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def ensure_media_type(media: MediaInput) -> MediaInput:
    if media["type"] == "image":
        media["image"] = _to_pil(media["image"])
        return media
    elif media["type"] == "video_chunk":
        media["video_chunk"] = [_to_pil(frame) for frame in media["video_chunk"]]
        return media
    else:
        raise ValueError(f"Unsupported media type: {media['type']}")


def image_to_np(
    image: Image.Image,
    resize_to: tuple[int, int] | None = None,
    mode: str = "resize",
    raise_error_for_ill_resize: bool = True,
) -> np.ndarray:
    """Convert an image to a numpy array.
    Args:
        content: The image to convert.
        resize_to: The size to resize the image to.
        mode: The mode to resize the image to.
        raise_error_for_ill_resize: Whether to raise an error for ill-sized resize.
    Returns:
        A numpy array.
    """
    assert isinstance(image, Image.Image), "image must be a PIL Image"
    if resize_to is not None:
        if mode == "resize":
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)

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
        return paddle.from_numpy(data).to(**kwargs)
    elif isinstance(data, paddle.Tensor):
        return data.to(**kwargs)
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
