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
from __future__ import annotations

import glob
import os

import paddle

paddle.set_printoptions(precision=10)

import random

import numpy as np
import paddle

from paddleformers.transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    ProcessorMixin,
    Qwen2Tokenizer,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)

MODEL_PATH = "/root/paddlejob/workspace/env_run/chenxuran/models/customQwen3-Omni-30B-A3B-Instruct/"

global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return paddle.to_tensor(values, dtype="int64").cuda().view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return paddle.to_tensor(values, dtype="float32").cuda().view(shape).contiguous()


def test_thinker_text_model():
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)
    config.text_config.num_hidden_layers = 12
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    config.audio_config._attn_implementation = "sdpa"

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        config=config,
        load_checkpoint_format="",
    )

    batch_size = 1
    seq_length = 2048
    vocab_size = config.get_text_config().vocab_size
    patch_size = config.vision_config.patch_size
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_row_size = 56
    image_col_size = 56
    num_channels = 3
    temporal_patch_size = config.vision_config.temporal_patch_size
    num_mel_bins = 128
    feat_seq_length = 290

    print("output_ids: ", type(output_ids), output_ids)


def test_thinker_with_dumped_inputs(dumped_input_path=None):
    """Test model with dumped inputs from training"""
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_config(config)

    # Find dumped inputs
    if dumped_input_path is None:
        # Auto-discover the latest dumped input file
        dump_dir = "/root/paddlejob/workspace/env_run/chenxuran/dumped_inputs"
        if not os.path.exists(dump_dir):
            print(f"Warning: Dump directory {dump_dir} does not exist")
            print("Please run training first to generate dumped inputs")
            return

        input_files = glob.glob(os.path.join(dump_dir, "*_inputs.npz"))
        if not input_files:
            print(f"Warning: No dumped input files found in {dump_dir}")
            return

        # Use the latest file
        dumped_input_path = max(input_files, key=os.path.getmtime)

    print(f"Loading dumped inputs from: {dumped_input_path}")

    # Load dumped inputs
    loaded_data = np.load(dumped_input_path)

    # Convert numpy arrays back to paddle tensors
    model_inputs = {}
    for key in loaded_data.files:
        model_inputs[key] = paddle.to_tensor(loaded_data[key])

    print(f"Loaded input keys: {list(model_inputs.keys())}")
    for key, value in model_inputs.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Run model with dumped inputs
    output_ids = model(**model_inputs)

    print("output_ids: ", type(output_ids), output_ids)

    return output_ids

    # calculate image tokens
    num_image_tokens = (image_row_size * image_col_size) // (spatial_merge_size**2)

    # calculate image tokens
    video_temporal = 4
    video_row_size = 28
    video_col_size = 28
    num_video_tokens = (video_temporal * video_row_size * video_col_size) // (spatial_merge_size**2)

    # calculate audio tokens (the same to _get_feat_extract_output_lengths)
    input_lengths_leave = feat_seq_length % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    num_audio_tokens = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (feat_seq_length // 100) * 13

    input_ids = ids_tensor([batch_size, seq_length], vocab_size - 3) + 3
    # set the num_image_tokens position to image_token_id in order to match pixel_values
    input_ids[0, :num_image_tokens] = config.image_token_id
    # set the num_video_tokens position to video_token_id in order to match pixel_values_videos
    input_ids[0, num_image_tokens : num_image_tokens + num_video_tokens] = config.video_token_id
    # set the num_audio_tokens position to audio_token_id in order to match input_features
    input_ids[
        0, num_image_tokens + num_video_tokens : num_image_tokens + num_video_tokens + num_audio_tokens
    ] = config.audio_token_id

    print(f"====== multimodal tokens confirm ======")
    print(f"image_token_id: {config.image_token_id}")
    print(f"video_token_id: {config.video_token_id}")
    print(f"audio_token_id: {config.audio_token_id}")
    print(f"num_image_tokens (expected): {num_image_tokens}")
    print(f"num_video_tokens (expected): {num_video_tokens}")
    print(f"num_audio_tokens (expected): {num_audio_tokens}")
    print(f"image_tokens in input_ids: {(input_ids == config.image_token_id).sum().item()}")
    print(f"video_tokens in input_ids: {(input_ids == config.video_token_id).sum().item()}")
    print(f"audio_tokens in input_ids: {(input_ids == config.audio_token_id).sum().item()}")
    attention_mask = paddle.ones(input_ids.shape, dtype="int64").to(input_ids.place)

    # image data: pixel_values and image_grid_thw
    pixel_values = floats_tensor(
        [
            batch_size * (image_row_size * image_col_size),
            num_channels * (patch_size**2) * temporal_patch_size,
        ]
    ).to(input_ids.place)
    pixel_grid_thw = paddle.to_tensor(
        [[1, image_row_size, image_col_size]] * batch_size, dtype="int64", place=input_ids.place
    )

    # video data: pixel_values_videos and video_grid_thw
    # differ from image with temporal > 1
    pixel_values_videos = floats_tensor(
        [
            batch_size * (video_temporal * video_row_size * video_col_size),
            num_channels * (patch_size**2) * temporal_patch_size,
        ]
    ).to(input_ids.place)
    video_grid_thw = paddle.to_tensor(
        [[video_temporal, video_row_size, video_col_size]] * batch_size, dtype="int64", place=input_ids.place
    )

    # audio data: input_features and feature_attention_mask
    input_features_values = floats_tensor([batch_size, num_mel_bins, feat_seq_length]).to(input_ids.place)
    feature_attention_mask = paddle.ones([batch_size, feat_seq_length], dtype="int64").to(input_ids.place)

    inputs_dict = {
        "input_features": input_features_values,
        "feature_attention_mask": feature_attention_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": pixel_grid_thw,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
    }

    output_ids = model(**inputs_dict)

    print("output_ids: ", type(output_ids), output_ids)


if __name__ == "__main__":
    import sys

    # print("=" * 60)
    # print("Test 1: Random input test")
    # print("=" * 60)
    # test_thinker_text_model()

    print("\n" + "=" * 60)
    print("Test 2: Dumped input test")
    print("=" * 60)

    # Check if a specific dumped input path is provided
    if len(sys.argv) > 1:
        test_thinker_with_dumped_inputs(sys.argv[1])
    else:
        test_thinker_with_dumped_inputs()
