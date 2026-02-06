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
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerConfig,
)
from paddleformers.transformers import AutoConfig, AutoModel

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

    return paddle.tensor(data=values, dtype=paddle.long).view(shape).contiguous()

def test_config():
    config = Qwen3OmniMoeConfig.from_pretrained(MODEL_PATH)

    model = Qwen3OmniMoeForConditionalGeneration.from_config(config)

    batch_size = 3
    seq_length = 39
    vocab_size = 152064
    input_ids = ids_tensor([batch_size, seq_length], config.get_text_config().vocab_size - 3) + 3
    # visual_token_ids = [config.vision_start_token_id] + [config.image_token_id] * 4
    # input_ids[:, 10 : 10 + len(visual_token_ids)] = visual_token_ids
    # pixel_values = np.random.randn(16, 1536).astype("float32")
    # inputs = {
    #     "input_ids": input_ids,
    #     "pixel_values": pixel_values,
    # }
    # paddle_inputs = {k: paddle.to_tensor(v) for k, v in inputs.items()}
    # print("paddle_inputs, ", type(paddle_inputs))
    print("input_ids, ", type(input_ids), input_ids.shape[0])
    output_id = model.generate(input_ids)


    # print("model init finished...")
    # state_dict_list = []
    # for name, _ in model.state_dict().items():
    #     state_dict_list.append(name)
    # sorted_list = sorted(state_dict_list)
    # for sorted_name in sorted_list:
    #     print(sorted_name)

def test_model():
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="bfloat16",
            _attn_implementation="flash_attention_2",
        )


def test_thinker_text_model():
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_config(config)

    config_dict = model.config.to_dict()
    for key, value in config_dict.items():
        print(f"{key}:{value}")

    input_ids = paddle.to_tensor(np.random.randint(0, 200, [1, 20]).astype("int64"))
    output_ids = model(input_ids=input_ids)

    print("output_ids: ", type(output_ids), output_ids)

if __name__ == "__main__":
    # test_config()

    # test_model()

    # test_thinker_text_model()
    pass