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
import os
import shutil

import paddle

paddle.set_device("cpu")

from safetensors import safe_open

from paddleformers.transformers import AutoModelForCausalLM


def parse_arguments():
    """
    Parse command line arguments for conversion script.

    Returns:
        argparse.Namespace: An object containing all parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paddlenlp_model_path", required=True, type=str, help="Input PaddleNLP model directory path."
    )
    parser.add_argument("--hf_model_path", required=True, type=str, help="Output HF model directory path.")
    parser.add_argument("--max_shard_size", default="4GB", type=str, help="The maximum size of each sub-checkpoint.")
    return parser.parse_args()


def load_safetensors_state_dict(input_dir):
    state_dict = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".safetensors"):
            checkpoint_path = os.path.join(input_dir, filename)
            with safe_open(checkpoint_path, framework="paddle") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if key == "lm_head.weight":
                        tensor = tensor.transpose([-1, -2]).contiguous()
                    elif not key.startswith("model."):
                        prefix, name = key.split(".", 1)
                        key = f"model.{name}"
                    state_dict[key] = tensor
    return state_dict


def trans_paddlenlp2hf():
    args = parse_arguments()

    state_dict = load_safetensors_state_dict(args.paddlenlp_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.paddlenlp_model_path,
        state_dict=state_dict,
        convert_from_hf=False,
    )
    model.save_pretrained(
        args.hf_model_path,
        max_shard_size=args.max_shard_size,
        save_checkpoint_format="flex_checkpoint",
        save_to_hf=True,
    )

    # copy rest files
    for filename in os.listdir(args.paddlenlp_model_path):
        if (
            filename.endswith(".safetensors")
            or filename.startswith("model")
            or filename.startswith(".")
            or filename.endswith(".pdparams")
        ):
            continue
        src_file = os.path.join(args.paddlenlp_model_path, filename)
        dst_file = os.path.join(args.hf_model_path, filename)
        shutil.copy2(src_file, dst_file)


if __name__ == "__main__":
    trans_paddlenlp2hf()
