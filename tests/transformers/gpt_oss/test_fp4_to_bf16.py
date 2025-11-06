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

import json
import os
import unittest

import numpy as np
import paddle
from aistudio_sdk.file_download import model_file_download as aistudio_download
from safetensors.paddle import load_file

from paddleformers.utils.log import logger
from paddleformers.utils.upcast_downcast_triton import downcast_dict, upcast_dict
from tests.testing_utils import slow

PADDLE_DTYPE_MAP = {
    "paddle.float64": 8,
    "paddle.float32": 4,
    "paddle.float16": 2,
    "paddle.uint16": 2,
    "paddle.bfloat16": 2,
    "paddle.uint8": 1,
    "paddle.float8_e4m3fn": 1,
    "paddle.float8_e5m2": 1,
}

POSTFIX_UINT8_LIST = [
    ".down_proj_blocks",
    ".down_proj_scales",
    ".gate_up_proj_blocks",
    ".gate_up_proj_scales",
    "down_proj",
    "gate_up_proj",
]


def find_safetensors_files(directory):
    safetensors_files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.endswith(".safetensors"):
            safetensors_files.append(full_path)
    return safetensors_files


def endswith(key, prefix_list):
    for prefix in prefix_list:
        if key.endswith(prefix):
            return True
    return False


def save_single_safetenors(save_path, state_dict, rank, total_files_size, prefix="model"):
    save_file_name = os.path.join(
        save_path,
        f"{prefix}-{rank + 1:05d}-of-{total_files_size:05d}.safetensors",
    )
    paddle.framework.io._safe_save(
        state_dict,
        save_file_name,
    )


class GptOssWeightChangeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = "./models/gpt-oss"

    def fp4_to_bf16(self):
        load_path = os.path.join(self.tempdir, "tiny-random-gpt-oss-fp4")
        save_path = os.path.join(self.tempdir, "tiny-random-gpt-oss-new-bf16")

        safetensor_prefix = "model"
        save_index_file = os.path.join(save_path, safetensor_prefix + ".safetensors.index.json")
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        file_list = find_safetensors_files(load_path)
        file_num = len(file_list)
        for idx, file_name in enumerate(file_list):
            local_dict = load_file(file_name)

            upcast_dict(local_dict)
            save_single_safetenors(save_path, local_dict, idx, file_num, safetensor_prefix)
            shard_file = f"{safetensor_prefix}-{idx + 1:05d}-of-{file_num:05d}.safetensors"
            for key in list(local_dict.keys()):
                index["weight_map"][key] = shard_file
                shape_ = local_dict[key].shape
                dtype_ = local_dict[key].dtype
                index["metadata"]["total_size"] += int(np.prod(shape_) * PADDLE_DTYPE_MAP[str(dtype_)])

        with open(save_index_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2) + "\n")
        logger.info(f"Model index file saved in {save_index_file}.")

    def bf16_to_fp4(self):
        load_path = os.path.join(self.tempdir, "tiny-random-gpt-oss-bf16")
        save_path = os.path.join(self.tempdir, "tiny-random-gpt-oss-new-fp4")
        safetensor_prefix = "model"
        save_index_file = os.path.join(save_path, safetensor_prefix + ".safetensors.index.json")
        index = {"metadata": {"total_size": 0}, "weight_map": {}}
        file_list = find_safetensors_files(load_path)
        file_num = len(file_list)
        for idx, file_name in enumerate(file_list):
            local_dict = load_file(file_name)

            downcast_dict(local_dict)
            save_single_safetenors(save_path, local_dict, idx, file_num, safetensor_prefix)
            shard_file = f"{safetensor_prefix}-{idx + 1:05d}-of-{file_num:05d}.safetensors"
            for key in list(local_dict.keys()):
                index["weight_map"][key] = shard_file
                shape_ = local_dict[key].shape
                dtype_ = local_dict[key].dtype
                index["metadata"]["total_size"] += int(np.prod(shape_) * PADDLE_DTYPE_MAP[str(dtype_)])

        with open(save_index_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2) + "\n")
        logger.info(f"Model index file saved in {save_index_file}.")

    def check_weight(self, origin_path, new_path, atol):
        origin_file_name = "model.safetensors"
        new_file_name = "model-00001-of-00001.safetensors"

        origin_dict = load_file(os.path.join(origin_path, origin_file_name))
        for key in list(origin_dict.keys()):
            if endswith(key, POSTFIX_UINT8_LIST):
                continue
            else:
                origin_dict.pop(key)

        new_dict = load_file(os.path.join(new_path, new_file_name))
        for key in list(new_dict.keys()):
            if endswith(key, POSTFIX_UINT8_LIST):
                continue
            else:
                new_dict.pop(key)
        assert len(origin_dict) == len(new_dict)
        for key in new_dict.keys():
            assert key in origin_dict.keys()
            assert np.allclose(new_dict[key].numpy(), origin_dict[key].numpy(), atol=atol)

    @slow
    def test_change_weight(self):

        repo_id = "PaddleFormers/tiny-random-gpt-oss-fp4"
        filename = "model.safetensors"
        aistudio_download(repo_id, filename, None, False, os.path.join(self.tempdir, "tiny-random-gpt-oss-fp4/"))

        repo_id = "PaddleFormers/tiny-random-gpt-oss-bf16"
        filename = "model.safetensors"
        aistudio_download(repo_id, filename, None, False, os.path.join(self.tempdir, "tiny-random-gpt-oss-bf16/"))

        self.bf16_to_fp4()
        self.fp4_to_bf16()

        self.check_weight(
            os.path.join(self.tempdir, "tiny-random-gpt-oss-fp4/"),
            os.path.join(self.tempdir, "tiny-random-gpt-oss-new-fp4/"),
            1e-2,
        )
        self.check_weight(
            os.path.join(self.tempdir, "tiny-random-gpt-oss-bf16/"),
            os.path.join(self.tempdir, "tiny-random-gpt-oss-new-bf16/"),
            1e-2,
        )


if __name__ == "__main__":
    unittest.main()
