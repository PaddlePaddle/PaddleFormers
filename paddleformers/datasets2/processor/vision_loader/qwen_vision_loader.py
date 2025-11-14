# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""
Vision loader class for Qwen2-VL.
"""

import io
import os
import sys
import copy
import math
import requests
import numpy as np

from PIL import Image
from decord import VideoReader, cpu

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple

from .vision_loader import VisionLoader
from paddleformers.hparams.data_args import DataArguments
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

@dataclass
class Qwen2VLVisionLoader(VisionLoader):
    r"""A loader for Qwen2-VL vision models."""

    def __init__(self, data_args: "DataArguments"):
        super().__init__(data_args)
        self.FPS = data_args.video_fps
        self.FRAME_FACTOR = data_args.temporal_conv_size
        self.FPS_MIN_FRAMES = data_args.video_min_frames
        self.FPS_MAX_FRAMES = data_args.video_max_frames

    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_nframes(
        self,
        total_frames: int,
        video_fps: Union[int, float],
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        min_frames = self.ceil_by_factor(self.FPS_MIN_FRAMES, self.FRAME_FACTOR)
        max_frames = self.floor_by_factor(min(self.FPS_MAX_FRAMES, total_frames), self.FRAME_FACTOR)
        nframes = total_frames / video_fps * self.FPS
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = self.floor_by_factor(nframes, self.FRAME_FACTOR)
        if not (self.FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(f"nframes should in interval [{self.FRAME_FACTOR}, {total_frames}], but got {nframes}.")
        return nframes

    def get_image_info(self, url: str) -> dict:
        bytes_content = self.file_download(url)
        image_obj = Image.open(bytes_content)
        return {"image": image_obj}

    def get_video_info(self, url: str) -> list[dict]:
        bytes_content = self.file_download(url)
        vr = VideoReader(bytes_content, ctx=cpu(0), num_threads=1)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        start_frame, end_frame = 0, total_frames - 1
        nframes = self.smart_nframes(total_frames=total_frames, video_fps=video_fps)
        idx = np.linspace(start_frame, end_frame, nframes).round()
        video = vr.get_batch(idx).asnumpy()

        ret = []
        for frame in video:
            tmp = Image.fromarray(frame, "RGB")
            ret.append({
                "image": tmp,
            })
        return ret