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

import numpy as np
import paddle

from paddleformers.transformers import (
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

MODEL_PATH = "/root/paddlejob/workspace/env_run/chenxuran/models/customQwen3-Omni-30B-A3B-Instruct/"


def test_thinker_text_model():
    config = Qwen3OmniMoeThinkerConfig.from_pretrained(MODEL_PATH)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_config(config)

    input_ids = paddle.to_tensor(np.random.randint(0, 200, [1, 20]).astype("int64"))
    output_ids = model(input_ids=input_ids)

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
