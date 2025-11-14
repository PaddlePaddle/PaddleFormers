# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import copy
import math
import random
import requests
import numpy as np

from PIL import Image
from decord import VideoReader, cpu

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

from paddleformers.hparams.data_args import DataArguments
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

@dataclass
class VisionLoader(ABC):
    r"""A class for vision loaders."""

    def __init__(self, data_args: "DataArguments"):
        """
        init
        """
        self.data_args = data_args

    def file_download(self, url: str) -> bytes:
        if url.startswith("http"):
            response = requests.get(url)
            bytes_data = response.content
        elif os.path.isfile(url):
            bytes_data = open(url, "rb").read()
        else:
            raise ValueError(f"{url} is not a valid url or file path.")
        bytes_content = io.BytesIO(bytes_data)
        return bytes_content

    def get_image_info(self, url: str) -> dict:
        raise NotImplementedError

    def get_video_info(self, url: str) -> list[dict]:
        raise NotImplementedError

    def __call__(self, images: list[str], videos: list[str]) -> Tuple[list[dict], list[list[dict]]]:
        r"""Process vision input."""
        image_inputs = []
        video_inputs = []

        if images and len(images) > 0:
            for url in images:
                image_info = self.get_image_info(url)
                image_inputs.append(image_info)

        if videos and len(videos) > 0:
            for url in videos:
                video_info = self.get_video_info(url)
                video_inputs.append(video_info)

        return image_inputs, video_inputs

@dataclass
class ErnieVisionLoader(VisionLoader):
    r"""A loader for ERNIE vision models."""

    def __init__(self, data_args: "DataArguments"):
        super().__init__(data_args)
        self.video_fps = data_args.video_fps
        self.video_min_frames = data_args.video_min_frames
        self.video_max_frames = data_args.video_max_frames
        self.video_target_frames = data_args.video_target_frames
        self.video_frames_sample = data_args.video_frames_sample
        self.temporal_conv_size = data_args.temporal_conv_size

    def get_target_frames(self, duration: float) -> int:
        video_frame_args = dict()
        video_frame_args["fps"] = self.video_fps
        video_frame_args["min_frames"] = self.video_min_frames
        video_frame_args["max_frames"] = self.video_max_frames
        video_frame_args["target_frames"] = self.video_target_frames

        if video_frame_args["target_frames"] > 0:
            if video_frame_args["fps"] > 0:
                raise ValueError("fps must not be positive if target_frames is given")
            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["target_frames"] < video_frame_args["min_frames"]
            ):
                raise ValueError("target_frames must be larger than min_frames")
            if (
                video_frame_args["max_frames"] > 0
                and video_frame_args["target_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("target_frames must be smaller than max_frames")
        else:
            if video_frame_args["fps"] <= 0:
                raise ValueError(
                    "Must provide either positive target_fps or positive target_frames."
                )
            frames_to_extract = int(duration * video_frame_args["fps"])
            video_frame_args["target_frames"] = frames_to_extract

            if (
                video_frame_args["min_frames"] > 0
                and video_frame_args["max_frames"] > 0
                and video_frame_args["min_frames"] > video_frame_args["max_frames"]
            ):
                raise ValueError("min_frames must be smaller than max_frames")
            if (
                video_frame_args["min_frames"] > 0
                and frames_to_extract < video_frame_args["min_frames"]
            ):
                logger.debug(
                    f"fps={video_frame_args['fps']} too low for min_frames={video_frame_args['min_frames']}, "
                    f"set target_frames={video_frame_args['min_frames']}"
                )
                video_frame_args["target_frames"] = video_frame_args["min_frames"]
            if (
                video_frame_args["max_frames"] > 0
                and frames_to_extract > video_frame_args["max_frames"]
            ):
                logger.debug(
                    f"fps={video_frame_args['fps']} too large for max_frames={video_frame_args['max_frames']},"
                    f" set target_frames={video_frame_args['max_frames']}"
                )
                video_frame_args["target_frames"] = video_frame_args["max_frames"]
        return video_frame_args["target_frames"]

    def read_frames_decord(self, video_reader: "VideoReader") -> Tuple[dict, dict]:
        assert self.video_frames_sample in ["rand", "middle", "leading"]

        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)

        target_frames = self.get_target_frames(duration=duration)

        if target_frames > vlen:
            acc_samples = vlen
            logger.info(
                f"target_frames={target_frames} is larger than video length {vlen}, "
                f"will sample {acc_samples} frames."
            )
        else:
            acc_samples = target_frames
            logger.debug(
                f"sampling at target_frames={target_frames}, frames_sample={self.video_frames_sample}"
            )
        
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if self.video_frames_sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif self.video_frames_sample == "leading":
            frame_indices = [x[0] for x in ranges]
        elif self.video_frames_sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        frames = []
        try:
            frames = video_reader.get_batch(frame_indices).asnumpy()
            video_reader.seek(0)
        except Exception as _:
            logger.info(f"get {frame_indices} frames error")

        assert len(frames) == len(
            frame_indices
        ), f"len(frames): {len(frames)} != len(frame_indices): {len(frame_indices)}"

        ret = []
        for idx, frame in enumerate(frames):
            tmp = Image.fromarray(frame, "RGB")
            ret.append(tmp)

        time_stamps = [
            frame_idx * duration / vlen
            for frame_idx in frame_indices
        ]

        del frame_indices
        assert len(time_stamps) == len(ret)
        return ret, time_stamps

    def get_image_info(self, url: str) -> dict:
        bytes_content = self.file_download(url)
        img = Image.open(bytes_content)

        image_width = img.width
        image_height = img.height
        img_one = {
            "image": img,
            "image_width": image_width,
            "image_height": image_height,
        }
        return img_one

    def get_video_info(self, url: str) -> list[dict]:
        bytes_content = self.file_download(url)
        video_reader = VideoReader(bytes_content, ctx=cpu(0), num_threads=1)

        tmp_frame = Image.fromarray(video_reader[0].asnumpy(), "RGB")
        video_width = tmp_frame.width
        video_height = tmp_frame.height

        frames, time_stamps = self.read_frames_decord(video_reader=video_reader)

        len_frames = len(frames)
        if len_frames % self.temporal_conv_size != 0:
            roundup = (
                math.ceil(len_frames / self.temporal_conv_size)
                * self.temporal_conv_size
            )
            num_padded_images = roundup - len_frames
            tmp_imgs = []
            tmp_stamps = []
            for _ in range(num_padded_images):
                padded_image = copy.deepcopy(frames[-1])
                padded_stamp = copy.deepcopy(time_stamps[-1])
                tmp_imgs.append(padded_image)
                tmp_stamps.append(padded_stamp)
            frames.extend(tmp_imgs)
            time_stamps.extend(tmp_stamps)

        ret = []
        for frame, timestamp in zip(frames, time_stamps):
            ret.append({
                "image": frame,
                "image_width": video_width,
                "image_height": video_height,
                "time_stamp": timestamp,
            })
        return ret
