# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""Image processor class for Kimi-K2.5.
"""

import json
from typing import Any, Dict, Optional, Union

import numpy as np
import paddle
from PIL import Image

from ..image_processing_utils import BaseImageProcessor, BatchFeature
from ..tokenizer_utils import TensorType
from .media_utils import (
    MediaInput,
    VideoChunkInput,
    _to_tensor,
    ensure_media_type,
    get_video_meta,
    image_to_np,
    navit_patchify,
    navit_resize_image,
    navit_resize_video,
    normalize,
    real_sample_fps_and_max_num_frames,
    timestamp_as_str,
)

try:
    from mecord import VideoReader
except ImportError:
    VideoReader = None


def resampling(
    video_bytes: bytes, sample_indices: list[int], key_indices=None, frame_time_info=None, num_threads=4
) -> str:
    video = VideoReader(video_bytes, num_threads=num_threads, frame_time_info=frame_time_info, key_indices=key_indices)
    # extract target frames
    frames = video[sample_indices]
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


class KimiK25VisionProcessor(BaseImageProcessor):
    model_type = "kimi_k25"

    def __init__(
        self,
        media_proc_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.media_proc_cfg = media_proc_cfg
        self.num_frames_per_chunk = media_proc_cfg["temporal_merge_kernel_size"]

    def media_tokens_calculator(self, media: MediaInput):
        media = ensure_media_type(media)
        ret = self.get_resize_config(media)
        return ret["num_tokens"]

    @classmethod
    def make_chunk_prompt(cls, timestamp_text: str) -> str:
        return f"{timestamp_text}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"

    def split_video_chunks(self, video_url: str | bytes) -> list[list[Image.Image]]:
        # video_url should be base64 str or bytes
        video_spec = get_video_meta(video_url)
        sample_fps = min(self.media_proc_cfg["sample_fps"], video_spec.fps)
        sampled_nframes = max(round(video_spec.num_frames * sample_fps / video_spec.fps), 1)
        frame_inds = np.linspace(0, video_spec.num_frames - 1, sampled_nframes).round().astype(int)
        frame_inds = frame_inds.tolist()
        sampled_frame_ids = []
        temporal_merge_kernel_size = self.media_proc_cfg["temporal_merge_kernel_size"]
        num_chunks = 0
        chunk_timestamp = []
        for i in range(0, len(frame_inds), temporal_merge_kernel_size):
            sampled_frame_ids.extend(frame_inds[i : i + temporal_merge_kernel_size])
            start_time = frame_inds[i] / float(video_spec.fps)
            timestamp_text = timestamp_as_str(start_time, self.media_proc_cfg["timestamp_mode"])
            chunk_timestamp.append(timestamp_text)
            num_chunks += 1

        sampled_frames = resampling(video_url, sampled_frame_ids)
        chunks = []
        for chunk_id in range(num_chunks):
            chunk = sampled_frames[chunk_id * temporal_merge_kernel_size : (chunk_id + 1) * temporal_merge_kernel_size]
            chunks.append(
                VideoChunkInput(
                    type="video_chunk", video_chunk=chunk, prompt=self.make_chunk_prompt(chunk_timestamp[chunk_id])
                )
            )
        return chunks

    def get_resize_config(self, media_input: MediaInput) -> dict:
        if media_input["type"] == "image":
            w, h = media_input["image"].size
            ret = navit_resize_image(
                w,
                h,
                self.media_proc_cfg["patch_size"],
                self.media_proc_cfg["merge_kernel_size"],
                self.media_proc_cfg["in_patch_limit"],
                self.media_proc_cfg["patch_limit_on_one_side"],
                self.media_proc_cfg["fixed_output_tokens"],
            )
            return ret
        elif media_input["type"] == "video_chunk":
            frame = media_input["video_chunk"][0]
            width, height = frame.size
            num_frames = len(media_input["video_chunk"])
            fps = 1.0

            sample_fps, max_num_frames_each_video = real_sample_fps_and_max_num_frames(
                media_input["type"],
                self.media_proc_cfg["sample_fps"],
                self.media_proc_cfg["max_num_frames_each_video"],
            )

            in_patch_limit_each_frame = self.media_proc_cfg["in_patch_limit_each_frame"]
            if in_patch_limit_each_frame is None:
                in_patch_limit_each_frame = self.media_proc_cfg["in_patch_limit"]

            ret = navit_resize_video(
                width,
                height,
                num_frames,
                fps,
                sample_fps,
                self.media_proc_cfg["patch_size"],
                self.media_proc_cfg["merge_kernel_size"],
                in_patch_limit_each_frame,
                self.media_proc_cfg["patch_limit_on_one_side"],
                self.media_proc_cfg["in_patch_limit_video"],
                max_num_frames_each_video,
                self.media_proc_cfg["fixed_output_tokens"],
            )
            return ret
        else:
            raise ValueError("Unsupported type: {}".format(media_input["type"]))

    def resize_image(
        self, image: Image.Image, new_width: int, new_height: int, pad_width: int, pad_height: int
    ) -> np.ndarray:
        image_np = image_to_np(image, (new_width, new_height), "resize")
        image_np = np.pad(
            image_np,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return image_np

    def preprocess(
        self,
        medias: list[MediaInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Preprocess a atom vision input (images/video_chunk) into model-ready tensors.

        Args:
            medias: List of MediaInput.
            return_tensors: Desired output format ('pt', 'np', 'tf', or None).

        Returns:
            BatchFeature containing 'pixel_values' and 'grid_thws' tensors.
        """
        if not isinstance(medias, list):
            medias = [medias]
        if medias:
            pixel_values = []
            for item in medias:
                item = ensure_media_type(item)
                resize_config = self.get_resize_config(item)
                new_width, new_height, pad_width, pad_height = (
                    resize_config["new_width"],
                    resize_config["new_height"],
                    resize_config["pad_width"],
                    resize_config["pad_height"],
                )
                if item["type"] == "image":
                    image = item["image"]
                    image_np = self.resize_image(image, new_width, new_height, pad_width, pad_height)
                    pixel_values.append(np.expand_dims(image_np, axis=0))
                elif item["type"] == "video_chunk":
                    pixels = []
                    for frame in item["video_chunk"]:
                        frame_np = self.resize_image(frame, new_width, new_height, pad_width, pad_height)
                        pixels.append(frame_np)
                    pixel_values.append(np.stack(pixels, axis=0))
                else:
                    raise ValueError("Unsupported type: {}".format(item["type"]))
            normalized_pixel_values = []
            image_std_inv = 1.0 / np.array(self.media_proc_cfg["image_std"])
            image_mean = np.array(self.media_proc_cfg["image_mean"])
            for pixels in pixel_values:
                pixels = normalize(pixels, image_mean, image_std_inv)
                pixels_and_thw = navit_patchify(
                    pixels,
                    self.media_proc_cfg["patch_size"],
                )
                normalized_pixel_values.append(pixels_and_thw)

            pixel_values = paddle.cat(
                [_to_tensor(pixel_value["pixel_values"]) for pixel_value in normalized_pixel_values]
            )
            grid_thws = paddle.cat(
                [
                    _to_tensor(pixel_value["grid_thw"], dtype=paddle.int64).unsqueeze(0)
                    for pixel_value in normalized_pixel_values
                ]
            )

            data = {
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }

        else:
            data = {}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def __repr__(self):
        return f"KimiK25VisionProcessor(media_proc_cfg={self.media_proc_cfg})"

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["media_proc_cfg"] = self.media_proc_cfg
        if "media_processor" in output:
            del output["media_processor"]
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        config = config_dict.copy()
        media_proc_cfg = config.pop("media_proc_cfg", {})
        return cls(media_proc_cfg=media_proc_cfg, **config, **kwargs)

    def to_json_string(self):
        dictionary = self.to_dict()
        for key, value in dictionary.items():
            if hasattr(value, "tolist"):
                dictionary[key] = value.tolist()
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"


__all__ = ["KimiK25VisionProcessor"]
