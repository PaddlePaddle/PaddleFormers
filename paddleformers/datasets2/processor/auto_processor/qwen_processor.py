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

import re
import random
import math
import os
import PIL
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import TYPE_CHECKING, Optional, Union, Tuple
from packaging import version

from .auto_processor import AutoProcessor
from paddleformers.transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from paddleformers.transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)

# if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
#     PILImageResampling = PIL.Image.Resampling
# else:
#     PILImageResampling = PIL.Image

from typing_extensions import override

if TYPE_CHECKING:
    from ...hparams import DataArguments

from paddleformers.utils.log import logger

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

@dataclass
class Qwen2VLProcessor(AutoProcessor):
    ignored_index = -100
    image_placeholder = '<image>'
    video_placeholder = '<video>'
    MAX_RATIO = 200
    IMAGE_MIN_TOKEN_NUM = 4
    IMAGE_MAX_TOKEN_NUM = 16384
    VIDEO_MIN_TOKEN_NUM = 128
    VIDEO_MAX_TOKEN_NUM = 768

    def __init__(self, data_args: "DataArguments", **kwargs):
        super().__init__(data_args, **kwargs)
        self.max_seq_len = data_args.get("max_seq_len", 128000)
        self.patch_size = data_args.get("patch_size", 14)
        self.merge_size = data_args.get("merge_size", 2)
        self.spatial_conv_size = data_args.get("spatial_conv_size", 2)
        self.temporal_conv_size = data_args.get("temporal_conv_size", 2)
        self.patch_factor = int(self.patch_size * self.spatial_conv_size)
        self.min_pixels = data_args.get("min_pixels", self.IMAGE_MIN_TOKEN_NUM * self.patch_factor ** 2)
        self.max_pixels = data_args.get("max_pixels", self.IMAGE_MAX_TOKEN_NUM * self.patch_factor ** 2)
        self.video_min_pixels = data_args.get("video_min_pixels", self.VIDEO_MIN_TOKEN_NUM * self.patch_factor ** 2)
        self.video_max_pixels = data_args.get("video_max_pixels", self.VIDEO_MAX_TOKEN_NUM * self.patch_factor ** 2)
        size = data_args.get("size", None)
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        self.size = size
        self.do_resize = data_args.get("do_resize", True)
        self.resample = data_args.get("resample", PILImageResampling.BICUBIC)
        self.do_rescale = data_args.get("do_rescale", True)
        self.rescale_factor = data_args.get("rescale_factor", 1/255)
        self.do_normalize = data_args.get("do_normalize", True)
        self.image_mean = data_args.get("image_mean", OPENAI_CLIP_MEAN)
        self.image_std = data_args.get("image_std", OPENAI_CLIP_STD)
        self.do_convert_rgb = data_args.get("do_convert_rgb", True)

    def get_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        self.image_token = tokenizer.special_tokens_map.get(
            "image_token", "<|image_pad|>"
        )
        self.video_token = tokenizer.special_tokens_map.get(
            "video_token", "<|video_pad|>"
        )
        self.vision_start_token = tokenizer.special_tokens_map.get(
            "vision_start_token", "<|vision_start|>"
        )
        self.vision_end_token = tokenizer.special_tokens_map.get(
            "vision_end_token", "<|vision_end|>"
        )

    def split_by_tags(self, text: str, tags: list[str]=[image_placeholder, video_placeholder]):
        pattern = '|'.join(map(re.escape, tags))
        parts = re.split(f'({pattern})', text)
        return [part for part in parts if part]

    def to_rgb(self, pil_image: Image.Image) -> Image.Image:
        if pil_image.mode == 'RGBA':
            white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
            white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
            return white_background
        else:
            return pil_image.convert("RGB")

    def smart_resize(self, height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor

        assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."
        if max(height, width) / min(height, width) > self.MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def is_scaled_image(self, image: np.ndarray) -> bool:
        """
        Checks to see whether the pixel values have already been rescaled to [0, 1].
        """
        if image.dtype == np.uint8:
            return False

        # It's possible the image has pixel values in [0, 255] but is of floating type
        return np.min(image) >= 0 and np.max(image) <= 1

    def process_images(
        self,
        image_inputs: list[dict],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_conv_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        r"""Process image."""
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_conv_size = temporal_conv_size if temporal_conv_size is not None else self.temporal_conv_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = []
        for image_input in image_inputs:
            image = image_input["image"]
            if do_convert_rgb:
                image = convert_to_rgb(image)

            images.append(to_numpy_array(image))

            if do_rescale and self.is_scaled_image(images[0]):
                logger.warning_once(
                    "It looks like you are trying to rescale already rescaled images. If the input"
                    " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
                )
            if input_data_format is None:
                # We assume that all images have the same channel dimension format.
                input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = self.smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = image.astype("uint8")
                image = Image.fromarray(image)
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    data_format=input_data_format,
                )

            if do_rescale:
                image = rescale(image, scale=rescale_factor, data_format=input_data_format)

            if do_normalize:
                image = normalize(image=image, mean=image_mean, std=image_std, data_format=input_data_format)

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)  # [C, H, W]
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % temporal_conv_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_conv_size - (patches.shape[0] % temporal_conv_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]  # [time, C, H, W]
        grid_t = patches.shape[0] // temporal_conv_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_conv_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_conv_size * patch_size * patch_size
        )

        image_grid_thw = (grid_t, grid_h, grid_w)

        return flatten_patches, image_grid_thw

    def process_vision_info(self, image_inputs: list[dict], video_inputs: list[list[dict]]) -> Tuple[list[dict], list[list[dict]]]:
        r"""Process vision info."""
        for image_input in image_inputs:
            image = image_input["image"]
            image = self.to_rgb(image)
            width, height = image.size
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=self.patch_factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image = image.resize((resized_width, resized_height))
            image_input["image"] = image
        for video in video_inputs:
            image = video[0]["image"]
            width, height = image.size
            min_pixels = self.video_min_pixels
            total_pixels = self.max_seq_len * self.patch_factor * self.patch_factor * 0.9
            max_pixels = max(min(self.video_max_pixels, total_pixels / len(video) * self.temporal_conv_size), int(min_pixels * 1.05))
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=self.patch_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            for i, frame in enumerate(video):
                image = frame["image"]
                image = image.resize((resized_width, resized_height))
                video[i]["image"] = image
        
        final_image_inputs = {
            "pixel_values": [],
            "image_grid_thw": [],
        }
        final_video_inputs = {
            "pixel_values_videos": [],
            "video_grid_thw": [],
        }
        for image_input in image_inputs:
            patches, image_grid_thw = self.process_images(
                [image_input],
            )
            final_image_inputs["pixel_values"].extend(patches)
            final_image_inputs["image_grid_thw"].append(image_grid_thw)
        
        for video_input in video_inputs:
            patches, image_grid_thw = self.process_images(
                video_input,
            )
            final_video_inputs["pixel_values_videos"].extend(patches)
            final_video_inputs["video_grid_thw"].append(image_grid_thw)
        
        final_image_inputs["pixel_values"] = np.array(final_image_inputs["pixel_values"])
        final_image_inputs["image_grid_thw"] = np.array(final_image_inputs["image_grid_thw"])
        final_video_inputs["pixel_values_videos"] = np.array(final_video_inputs["pixel_values_videos"])
        final_video_inputs["video_grid_thw"] = np.array(final_video_inputs["video_grid_thw"])

        return final_image_inputs, final_video_inputs

    def get_rope_index_25(
        self,
        spatial_conv_size = 2,
        input_ids = None,
        image_grid_thw = None,
        video_grid_thw = None,
        second_per_grid_ts = None,
        attention_mask = None,
    ):
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`np.Array` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`np.Array` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`np.Array` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`np.Array` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`np.Array` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`np.Array` of shape `(3, batch_size, sequence_length)`)
        """
        image_token_id = 151655
        video_token_id = 151656
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = np.ones_like(total_input_ids)
            position_ids = np.ones(
                (3, input_ids.shape[0], input_ids.shape[1]),
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = np.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_conv_size,
                        w.item() // spatial_conv_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    )
                    llm_pos_ids_list.append(
                        np.tile(np.arange(text_len).reshape(1, -1), (3, 1)) + st_idx
                    )

                    range_tensor = np.arange(llm_grid_t).reshape(-1, 1)
                    expanded_range = np.tile(range_tensor, (1, llm_grid_h * llm_grid_w))

                    time_tensor = expanded_range * second_per_grid_t * 2

                    time_tensor_long = time_tensor.astype(np.int64)
                    t_index = time_tensor_long.flatten()
     
                    h_index = np.tile(
                        np.arange(llm_grid_h).reshape([1, -1, 1]),
                        ([llm_grid_t, 1, llm_grid_w]),
                    ).flatten()
                    w_index = np.tile(
                        np.arange(llm_grid_w).reshape([1, 1, -1]),
                        ([llm_grid_t, llm_grid_h, 1]),
                    ).flatten()
                    llm_pos_ids_list.append(
                        np.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    )
                    text_len = len(input_tokens) - st
                    arange_tensor = np.arange(text_len).reshape(1, -1)
                    expanded_tensor = np.tile(arange_tensor, (3, 1))
                    llm_pos_ids = expanded_tensor + st_idx 
                    llm_pos_ids_list.append(llm_pos_ids)

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
            return position_ids
        else:
            if attention_mask is not None:
                position_ids = attention_mask.astype(np.int64).cumsum(-1) - 1
                position_ids.masked_fill_(mask=attention_mask == 0, value=1)
                position_ids = np.expand_dims(position_ids, axis=0)
                position_ids = np.tile(position_ids, (3, 1, 1))
            else:
                position_ids = (
                    np.arange(input_ids.shape[1])
                    .reshape(1, 1, -1)
                )
                position_ids = np.tile(position_ids, (3, input_ids.shape[0], 1))

            return position_ids

    @override
    def encode(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], tokenizer: "PreTrainedTokenizer") -> dict:
        history_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        all_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        history_len = len(history_str)
        assert all_str[:history_len] == history_str, f"template(messages[:-1]): {history_str} should be a prefix of template(messages): {all_str}"

        response_str = all_str[history_len:]

        image_inputs, video_inputs = self.process_vision_info(
            image_inputs,
            video_inputs,
        )
        image_grid_thw = image_inputs["image_grid_thw"]
        video_grid_thw = video_inputs["video_grid_thw"]

        input_ids, labels = [], []
        image_id, video_id = 0, 0

        self.get_special_tokens(tokenizer)
        merge_length = self.merge_size ** 2
        for part in self.split_by_tags(history_str):
            if part == self.image_placeholder:
                num_image_tokens = image_grid_thw[image_id].prod() // merge_length
                added_text = self.vision_start_token +  self.image_token * num_image_tokens + self.vision_end_token
                input_id = tokenizer.encode(added_text)
                image_id += 1
            elif part == self.video_placeholder:
                num_video_tokens = video_grid_thw[video_id].prod() // merge_length
                added_text = self.vision_start_token +  self.video_token * num_video_tokens + self.vision_end_token
                input_id = tokenizer.encode(added_text)
                video_id += 1
            else:
                input_id = tokenizer.encode(part)
            input_ids.extend(input_id)
            labels.extend([self.ignored_index] * len(input_id))
    
        response_id = tokenizer.encode(response_str)
        input_ids.extend(response_id)
        labels.extend(response_id)

        position_ids = self.get_rope_index_25(
                spatial_conv_size=self.spatial_conv_size,
                input_ids=np.array([input_ids]),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        model_input = {
            "input_ids": input_ids,
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_inputs["image_grid_thw"],
            "pixel_values_videos": video_inputs["pixel_values_videos"],
            "video_grid_thw": video_inputs["video_grid_thw"],
            "labels": labels,
            "position_ids": position_ids,
        }
        return model_input