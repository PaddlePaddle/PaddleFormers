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
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .auto_processor import AutoProcessor
from paddleformers.transformers.processing_utils import ProcessorMixin

from typing_extensions import override

if TYPE_CHECKING:
    from ...hparams import DataArguments

from paddleformers.utils.log import logger

@dataclass
class Qwen2VLProcessor(AutoProcessor):
    ignored_index = -100
    image_placeholder = '<image>'
    video_placeholder = '<video>'

    def __init__(self, data_args: "DataArguments", **kwargs):
        super().__init__(data_args, **kwargs)
        self.merge_size = data_args.get("merge_size", 2)
        self.spatial_conv_size = data_args.get("spatial_conv_size", 2)

    def get_special_tokens(self, tokenizer) -> None:
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
    def encode(self, messages: list[dict], image_inputs: list[dict], video_inputs: list[list[dict]], processor: "ProcessorMixin") -> dict:
        history_str = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        all_str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        history_len = len(history_str)
        assert all_str[:history_len] == history_str, f"template(messages[:-1]): {history_str} should be a prefix of template(messages): {all_str}"

        response_str = all_str[history_len:]

        inputs = processor(
            text="",
            images=[image_input["image"] for image_input in image_inputs],
            videos=[[frame["image"] for frame in video_input] for video_input in video_inputs],
            return_tensors="pd",
        )
        image_grid_thw = inputs["image_grid_thw"]
        video_grid_thw = inputs["video_grid_thw"]

        input_ids, labels = [], []
        image_id, video_id = 0, 0

        self.get_special_tokens(processor.tokenizer)
        merge_length = self.merge_size ** 2
        for part in self.split_by_tags(history_str):
            if part == self.image_placeholder:
                num_image_tokens = image_grid_thw[image_id].numpy().prod() // merge_length
                added_text = self.vision_start_token +  self.image_token * num_image_tokens + self.vision_end_token
                input_id = processor.tokenizer.encode(added_text)
                image_id += 1
            elif part == self.video_placeholder:
                num_video_tokens = video_grid_thw[video_id].numpy().prod() // merge_length
                added_text = self.vision_start_token +  self.video_token * num_video_tokens + self.vision_end_token
                input_id = processor.tokenizer.encode(added_text)
                video_id += 1
            else:
                input_id = processor.tokenizer.encode(part)
            input_ids.extend(input_id)
            labels.extend([self.ignored_index] * len(input_id))
    
        response_id = processor.tokenizer.encode(response_str)
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
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
            "pixel_values_videos": inputs["pixel_values_videos"],
            "video_grid_thw": inputs["video_grid_thw"],
            "labels": labels,
            "position_ids": position_ids,
        }
        return model_input