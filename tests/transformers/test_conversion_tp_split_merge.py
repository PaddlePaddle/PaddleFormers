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

import unittest

import numpy as np
import paddle
from safetensors import safe_open
from safetensors.numpy import save_file

from paddleformers.transformers.conversion_utils import (
    naive_fuse_merge_tp,
    naive_fuse_split_tp,
)


def get_pysafeslice_obj():
    key = "embeddings"
    save_file_path = "test_tp_split_merge_file.safetensors"
    state_dict = {
        key: np.arange(120).reshape(5, 24),
    }
    save_file(state_dict, save_file_path, metadata={"format": "np"})
    with safe_open(save_file_path, framework="np") as f:
        py_safe_slice_ = f.get_slice(key)
        return py_safe_slice_


class TestTPSplitMerge(unittest.TestCase):
    "Test tensor parallel split and merge."

    def test_tp_split_merge(self):
        py_safe_slice_ = get_pysafeslice_obj()
        weight_cases = []
        # test slice
        weight_cases.append(py_safe_slice_)
        # test paddle
        weight_cases.append(paddle.Tensor.__call__(py_safe_slice_[:], zero_copy=True))
        # test numpy
        weight_cases.append(py_safe_slice_[:])

        weight_value = py_safe_slice_[:]
        tensor_model_parallel_size = 2
        is_column = True
        fuse_tensor_parts = 3
        num_kv_groups_list = [1, 2]
        tp_splited_truth_list = [
            [
                np.concatenate([weight_value[:, 0:4], weight_value[:, 8:12], weight_value[:, 16:20]], axis=1),
                np.concatenate([weight_value[:, 4:8], weight_value[:, 12:16], weight_value[:, 20:24]], axis=1),
            ],
            [
                np.concatenate([weight_value[:, 0:6], weight_value[:, 12:15], weight_value[:, 18:21]], axis=1),
                np.concatenate([weight_value[:, 6:12], weight_value[:, 15:18], weight_value[:, 21:24]], axis=1),
            ],
        ]

        for weight in weight_cases:
            for idx in range(len(num_kv_groups_list)):
                num_kv_groups = num_kv_groups_list[idx]
                tp_splited_truth = tp_splited_truth_list[idx]
                splited_list = []
                for tensor_parallel_rank in range(tensor_model_parallel_size):
                    splited_list.append(
                        naive_fuse_split_tp(
                            weight,
                            tensor_model_parallel_size,
                            tensor_parallel_rank,
                            is_column,
                            fuse_tensor_parts,
                            num_kv_groups,
                        )
                    )

                # test tp split
                for tpi in range(len(splited_list)):
                    self.assertTrue((splited_list[tpi] == tp_splited_truth[tpi]).all())

                # test tp merge
                merged_tensor = naive_fuse_merge_tp(splited_list, is_column, fuse_tensor_parts, num_kv_groups)
                self.assertTrue((merged_tensor == weight_value).all())


if __name__ == "__main__":
    unittest.main()
