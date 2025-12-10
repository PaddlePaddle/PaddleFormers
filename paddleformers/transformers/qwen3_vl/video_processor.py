# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Video processor class for Qwen3-VL."""

import math
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.nn.functional as F

from ..image_processing_utils import BatchFeature
from ..video_processing_utils import BaseVideoProcessor


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    """
    Calculates the target height and width to fit within pixel limits while maintaining aspect ratio.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3VLVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = [0.5, 0.5, 0.5],
        image_std: Optional[Union[float, List[float]]] = [0.5, 0.5, 0.5],
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_frames: int = 4,
        max_frames: int = 768,
        fps: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.fps = fps
        self.resample = "bicubic"

    def preprocess(
        self,
        videos: List[paddle.Tensor],
        do_resize: bool = None,
        do_rescale: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess the video.
        Args:
            videos: List of tensors, each with shape [T, C, H, W] (channel-first).
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        pixel_values_videos = []
        video_grid_thw = []

        for video in videos:
            # Ensure input is a Paddle Tensor
            if isinstance(video, np.ndarray):
                video = paddle.to_tensor(video)

            # Ensure format [T, C, H, W]
            # If the input is [T, H, W, C], convert to [T, C, H, W]
            if video.shape[-1] == 3:
                video = video.transpose([0, 3, 1, 2])

            T, C, H, W = video.shape

            # 1. Smart Resize
            if do_resize:
                # Calculate target size specific to Qwen3 logic
                resized_height, resized_width = smart_resize(
                    num_frames=T,
                    height=H,
                    width=W,
                    temporal_factor=self.temporal_patch_size,
                    factor=self.patch_size * self.merge_size,
                )

                # Resize frames
                # Treat temporal dimension as batch for 2D interpolation
                video = F.interpolate(
                    video, size=(resized_height, resized_width), mode=self.resample, align_corners=False
                )

            # 2. Rescale
            if do_rescale:
                video = video.astype("float32") * self.rescale_factor

            # 3. Normalize
            if do_normalize:
                mean = paddle.to_tensor(image_mean).reshape([1, 3, 1, 1])
                std = paddle.to_tensor(image_std).reshape([1, 3, 1, 1])
                video = (video - mean) / std

            # 4. Temporal Padding
            # Ensure frame count is divisible by temporal_patch_size
            T_new = video.shape[0]
            if T_new % self.temporal_patch_size != 0:
                pad_len = self.temporal_patch_size - (T_new % self.temporal_patch_size)
                # Repeat the last frame for padding
                last_frame = video[-1:].tile([pad_len, 1, 1, 1])
                video = paddle.concat([video, last_frame], axis=0)

            # 5. Reshape to 3D Tubelets
            # Current shape: [T, C, H, W]
            patches = video
            T, C, H, W = patches.shape

            grid_t = T // self.temporal_patch_size
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size

            # Reshape logic to extract 3D patches
            # Matches HF Qwen3-VL implementation logic
            patches = patches.reshape(
                [
                    grid_t,
                    self.temporal_patch_size,
                    C,
                    grid_h // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                    grid_w // self.merge_size,
                    self.merge_size,
                    self.patch_size,
                ]
            )

            # Permute to organize patches
            # Indices: 0:Gt, 1:t_ps, 2:C, 3:Gh_m, 4:m_h, 5:p_h, 6:Gw_m, 7:m_w, 8:p_w
            # Target: Gt, Gh_m, Gw_m, m_h, m_w, C, t_ps, p_h, p_w
            patches = patches.transpose([0, 3, 6, 4, 7, 2, 1, 5, 8])

            # Flatten
            flatten_patches = patches.reshape(
                [
                    grid_t * grid_h * grid_w,
                    C
                    * self.temporal_patch_size
                    * self.patch_size
                    * self.patch_size
                    * self.merge_size
                    * self.merge_size,
                ]
            )

            pixel_values_videos.append(flatten_patches)
            video_grid_thw.append(paddle.to_tensor([grid_t, grid_h, grid_w], dtype="int64"))

        return BatchFeature({"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw})
