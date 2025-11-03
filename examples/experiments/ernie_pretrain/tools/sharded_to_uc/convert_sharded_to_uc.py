# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import json
import os
import subprocess

import paddle
from safetensors.numpy import save_file as safe_save_file

from paddleformers.transformers.model_utils import shard_checkpoint
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

USEFUL_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.model",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sharded_path",
        type=str,
        required=True,
        help="The path of checkpoint to be gathered.",
    )
    parser.add_argument(
        "--uc_path",
        type=str,
        required=True,
        help="The output path to gather all checkpoints.",
    )
    args = parser.parse_args()
    return args


def convert_ckpt(args):
    sharded_path = args.sharded_path
    uc_path = args.uc_path
    assert os.path.exists(sharded_path), f"{sharded_path} not exist"
    assert not os.path.exists(uc_path), f"{uc_path} already exist"
    os.makedirs(f"{uc_path}")
    print(f"Convert sharded ckpt {sharded_path} to unified checkpoint {uc_path}")
    for file in USEFUL_FILES:
        assert os.path.exists(f"{sharded_path}/{file}"), f"{sharded_path}/{file} not exist, please check"
        print(f"Copying {sharded_path}/{file}")
        subprocess.run(
            ["cp", f"{sharded_path}/{file}", uc_path],
            capture_output=True,
            text=True,
        )

    assert os.path.exists(
        f"{sharded_path}/model_state.pdparams"
    ), f"{sharded_path}/model_state.pdparams not exist, please check"
    print(f"Loading sharded ckpt {sharded_path}/model_state.pdparams")

    sharded_name_to_uc_name = {
        "ernie.embed.embed_tokens.weight": "ernie.embed_tokens.weight",
        "self_attn.fused_rms_norm_linear.rms_norm_weight": "input_layernorm.weight",
        "self_attn.fused_rms_norm_linear.linear_weight": "self_attn.qkv_proj.weight",
        "embed_share.embed_tokens.weight": "ernie.embed_tokens.weight",
    }

    def parse_name(key):
        for k, v in sharded_name_to_uc_name.items():
            if k in key:
                output_str = key.replace(k, v)
                break
            else:
                output_str = key
        return output_str

    model_state = paddle.load(f"{sharded_path}/model_state.pdparams")
    new_state_dict = {}
    for key, param in model_state.items():
        key = parse_name(key)
        assert key not in new_state_dict
        new_state_dict[key] = param

    for k in list(new_state_dict.keys()):
        if isinstance(new_state_dict[k], paddle.Tensor):
            new_state_dict[k] = new_state_dict.pop(k).cpu().numpy()

    shards, index = shard_checkpoint(
        new_state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )
    for shard_file, shard in shards.items():
        save_file = os.path.join(uc_path, shard_file)
        print(f"Saving {save_file}")
        safe_save_file(shard, save_file, metadata={"format": "np"})

    save_index_file = os.path.join(uc_path, SAFE_WEIGHTS_INDEX_NAME)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2) + "\n"
        print(f"Saving {save_index_file}")
        f.write(content)


if __name__ == "__main__":
    args = parse_args()
    convert_ckpt(args)
