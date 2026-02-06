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
from paddleformers.transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerConfig,
)

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/"

def test_thinker_text_model():
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_config(config)

    config_dict = model.config.to_dict()
    for key, value in config_dict.items():
        print(f"{key}:{value}")

    input_ids = paddle.to_tensor(np.random.randint(0, 200, [1, 20]).astype("int64"))
    output_ids = model(input_ids=input_ids)

    # print("output_ids: ", type(output_ids), output_ids)

if __name__ == "__main__":
    test_thinker_text_model()
    