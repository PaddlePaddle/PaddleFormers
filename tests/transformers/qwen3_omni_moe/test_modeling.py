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

import paddle
import numpy as np
import random
from paddleformers.transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerConfig,
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"

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
    config.text_config.num_hidden_layers = 24
    config.text_config._attn_implementation = "flashmask"
    config.vision_config._attn_implementation = "flashmask"
    config.audio_config._attn_implementation = "flashmask"

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
    num_image_tokens = (image_row_size * image_col_size) // (spatial_merge_size ** 2)

    input_ids = ids_tensor([batch_size, seq_length], vocab_size - 3) + 3
    # set the num_image_tokens position to image_token_id in order to match pixel_values
    input_ids[0, :num_image_tokens] = config.image_token_id
    attention_mask = paddle.ones(input_ids.shape, dtype="int64").to(input_ids.place)
    pixel_values = floats_tensor(
        [
            batch_size * (image_row_size * image_col_size),
            num_channels * (patch_size**2) * temporal_patch_size,
        ]
    ).to(input_ids.place)
    pixel_grid_thw = paddle.to_tensor(
        [[1, image_row_size, image_col_size]] * batch_size, 
        dtype="int64", place=input_ids.place
    )
    input_features_values = floats_tensor([batch_size, num_mel_bins, feat_seq_length]).to(input_ids.place)
    feature_attention_mask = paddle.ones([batch_size, feat_seq_length], dtype="int64").to(input_ids.place)

    inputs_dict = {
        "input_features": input_features_values,
        "feature_attention_mask": feature_attention_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": pixel_grid_thw,
        "pixel_values": pixel_values,
    }
    for key, value in inputs_dict.items():
        print(f"{key} shape", value.shape)

    output_ids = model(**inputs_dict)

    print("output_ids: ", type(output_ids), output_ids)

if __name__ == "__main__":
    test_thinker_text_model()
