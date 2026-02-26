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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, TypedDict

import numpy as np
import paddle
from PIL import Image

from ...utils import is_decord_available
from ...utils.log import logger
from ..paddle_vision_utils import pad, resize


@dataclass
class VideoSpec:
    media_type: str
    height: int
    width: int
    num_frames: int
    fps: float
    key_indices: Optional[list[int]] = None
    frame_time_info: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.height <= 0:
            raise ValueError("height must be greater than 0")
        if self.width <= 0:
            raise ValueError("width must be greater than 0")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be greater than 0")
        if self.fps <= 0:
            raise ValueError("fps must be greater than 0")
        if self.media_type != "video":
            raise ValueError("media_type must be 'video'")


class ImageInput(TypedDict):
    type: Literal["image"]
    image: Image.Image


class VideoChunkInput(TypedDict):
    type: Literal["video_chunk"]
    video_chunk: List[Image.Image | np.ndarray | paddle.Tensor]
    prompt: Optional[str] = None


MediaInput = ImageInput | VideoChunkInput


def _read_video_decord(
    video_src: str | bytes | os.PathLike,
    num_threads: int = 0,
    sample_indices: list = None,
    return_video: bool = False,
) -> dict:

    if not is_decord_available():
        raise ImportError(
            "Backend=decord for loading the video but the required library is not found in your environment "
            "Make sure to install 'decord' before loading the video."
        )
    import decord

    logger.info("Loading video with decord backend.")
    st = time.time()
    vr = decord.VideoReader(video_src, num_threads=num_threads)
    total_frames, video_fps = len(vr), vr.get_avg_fps()

    original_height = int(vr[0].shape[0])
    original_width = int(vr[0].shape[1])

    assert total_frames > 0, "Invalid video format."
    assert original_width > 0 and original_height > 0, "Invalid video format."
    assert video_fps > 0, "Invalid video format."

    estimated_frame = max(1, int(video_fps))
    key_indices = list(range(0, total_frames, estimated_frame))

    frame_time_info = {
        "video_start": 0,
        "video_end": total_frames - 1,
        "total_frames": total_frames,
    }
    video = vr.get_batch(indices=sample_indices if sample_indices is not None else key_indices).asnumpy()
    video = paddle.to_tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format

    logger.info(f"decord:  {video_src=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    video_spec = VideoSpec(
        media_type="video",
        height=original_height,
        width=original_width,
        num_frames=total_frames,
        fps=video_fps,
        key_indices=key_indices,
        frame_time_info=frame_time_info,
    )

    return (video, video_spec) if return_video else video_spec


def _read_video_paddlecodec(
    video_src: str | bytes | os.PathLike,
    num_threads: int = 0,
    sample_indices: list = None,
    return_video: bool = False,
) -> dict:
    """read video using torchcodec.decoders.VideoDecoder(via Paddle Proxy)

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        paddle.Tensor: the video tensor with shape (T, C, H, W).
    """
    try:
        import sys

        import paddle

        del sys.modules["torchcodec"]
        paddle.compat.enable_torch_proxy(scope={"torchcodec"})
        from torchcodec.decoders import VideoDecoder

        sys.modules["torchcodec"] = None
    except (ImportError, RuntimeError) as e:
        logger.error(
            f"Failed to load 'torchcodec' backend via Paddle proxy.\n"
            f"  - Common Causes:\n"
            f"    1. Conflict with official 'torch' or 'torchcodec' packages.\n"
            f"    2. Missing FFmpeg libraries or System library mismatch (CXXABI).\n"
            f"  - Recommended Fix Steps:\n"
            f"    1. Install dependencies: `conda install ffmpeg -c conda-forge` or `apt-get update && apt-get install ffmpeg` \n"
            f"    2. Uninstall conflicts: `pip uninstall torchcodec paddlecodec -y`\n"
            f"    3. Reinstall packages: `pip install paddlecodec --force-reinstall`\n"
            f"  - If you encounter 'CXXABI' or 'libstdc++' errors, your system libraries might be outdated.\n"
            f"    Try prioritizing Conda libraries by running: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python your_script.py`\n"
            f"  - Original Error: {e}"
        )
        raise

    logger.info("Loading video with paddlecodec backend.")
    PADDLECODEC_NUM_THREADS = int(os.environ.get("PADDLECODEC_NUM_THREADS", num_threads))
    logger.info(
        f"set PADDLECODEC_NUM_THREADS: {PADDLECODEC_NUM_THREADS if PADDLECODEC_NUM_THREADS != 0 else '0 (Auto)'}"
    )
    st = time.time()
    decoder = VideoDecoder(video_src, num_ffmpeg_threads=PADDLECODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    original_height = decoder.metadata.height
    original_width = decoder.metadata.height

    assert total_frames > 0, "Invalid video format."
    assert original_width > 0 and original_height > 0, "Invalid video format."
    assert video_fps > 0, "Invalid video format."

    estimated_frame = max(1, int(video_fps))
    key_indices = list(range(0, total_frames, estimated_frame))

    video = (
        decoder.get_frames_at(indices=sample_indices if sample_indices is not None else key_indices)
        .data.contiguous()
        .to("cuda")
    )
    logger.info(f"paddlecodec:  {video_src=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    paddle.compat.disable_torch_proxy()

    frame_time_info = {
        "video_start": 0,
        "video_end": total_frames - 1,
        "total_frames": total_frames,
    }

    video_spec = VideoSpec(
        media_type="video",
        height=original_height,
        width=original_width,
        num_frames=total_frames,
        fps=video_fps,
        key_indices=key_indices,
        frame_time_info=frame_time_info,
    )

    return (video, video_spec) if return_video else video_spec


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "paddlecodec": _read_video_paddlecodec,
}


def get_video_meta(video_src: bytes | str | os.PathLike, accurate: bool = True, **kwargs) -> dict:
    """Get the dimensions of a video."""
    if isinstance(video_src, os.PathLike):
        video_src = str(video_src)
    # if b64 string, decode to bytes
    if isinstance(video_src, str) and video_src.startswith("data:video/mp4;base64,"):
        video_src = base64.b64decode(video_src.split(",")[1])

    video_backend = kwargs.get("video_backend", "paddlecodec")

    return VIDEO_READER_BACKENDS[video_backend](video_src, num_threads=1, return_video=False)


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
        if isinstance(media["video_chunk"], np.ndarray):
            video_chunk = media["video_chunk"]
            media["video_chunk"] = paddle.to_tensor(video_chunk).permute(0, 3, 1, 2)

        return media
    else:
        raise ValueError(f"Unsupported media type: {media['type']}")


def image_in_tensor(
    image: paddle.Tensor,
    resize_to: tuple[int, int] | None = None,
    mode: str = "resize",
    raise_error_for_ill_resize: bool = True,
) -> paddle.Tensor:
    """Convert an image to a numpy array.
    Args:
        content: The image to convert.
        resize_to: The size to resize the image to.
        mode: The mode to resize the image to.
        raise_error_for_ill_resize: Whether to raise an error for ill-sized resize.
    Returns:
        A numpy array.
    """
    assert isinstance(image, paddle.Tensor), "image must be a Paddle Tensor"
    if resize_to is not None:
        if mode == "resize":
            image = resize(image, size=resize_to, interpolation="bicubic")

        elif mode == "rescale_and_pad_to_center":
            _, height, width = image.shape
            scale = min(resize_to[0] / width, resize_to[1] / height, 1.0)
            new_width = round(width * scale)
            new_height = round(height * scale)
            if new_width == 0 or new_height == 0:
                if raise_error_for_ill_resize:
                    raise ValueError(
                        f"Invalid resize to: {resize_to}, from image size: {image.shape[1], image.shape[2]}"
                    )
                else:
                    return paddle.zeros((3, resize_to[0], resize_to[1]), dtype="uint8")

            image = resize(image, (new_width, new_height), resample="bicubic")
            padding_left = (resize_to[0] - new_width) // 2
            padding_right = resize_to[0] - new_width - padding_left
            padding_top = (resize_to[1] - new_height) // 2
            padding_bottom = resize_to[1] - new_height - padding_top
            image = pad(
                image,
                padding=[padding_left, padding_top, padding_right, padding_bottom],
                fill=0,
                padding_mode="constant",
            )

        elif mode == "rescale_and_pad_to_rightbottom":
            _, width, height = image.shape
            scale = min(resize_to[0] / width, resize_to[1] / height, 1.0)
            new_width = round(width * scale)
            new_height = round(height * scale)
            if new_width == 0 or new_height == 0:
                if raise_error_for_ill_resize:
                    raise ValueError(
                        f"Invalid resize to: {resize_to}, from image size: {image.shape[1], image.shape[2]}"
                    )
                else:
                    return paddle.zeros((3, resize_to[0], resize_to[1]), dtype="uint8")

            image = resize(image, (new_width, new_height), resample="bicubic")
            padding_right = resize_to[0] - new_width
            padding_bottom = resize_to[1] - new_height
            image = pad(
                image,
                padding=[0, 0, padding_right, padding_bottom],
                fill=0,
                padding_mode="constant",
            )

        else:
            raise ValueError(f"Invalid mode: {mode}")

    return image


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


def navit_patchify(pixel_values: paddle.Tensor, patch_size: int) -> dict[str, paddle.tensor]:
    """Reshape the pixel values to a navit shape.
    Args:
        pixel_values: paddle.Tensor, shape (b, t, h, w, c)
        patch_size: int
    Returns:
        dict[str, paddle.Tensor]
        - patches: paddle.Tensor, shape (b * t * h//patch_size * w//patch_size, c, patch_size, patch_size)
        - grid_thw: paddle.Tensor, (t, h//patch_size, w//patch_size)
    """
    B, T, C, H, W = pixel_values.shape
    assert C == 3, "pixel_values must have 3 channels"

    patches = pixel_values.reshape([B * T, C, H // patch_size, patch_size, W // patch_size, patch_size])
    # (T, H//patch_size, W//patch_size, C, patch_size, patch_size)
    patches = patches.transpose(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    grid_thw = paddle.to_tensor([[T, H // patch_size, W // patch_size]] * B)
    return {"pixel_values": patches, "grid_thw": grid_thw}
