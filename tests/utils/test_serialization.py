# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import tempfile
from unittest import TestCase

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.utils import load_torch
from tests.testing_utils import require_package


class SerializationTest(TestCase):
    @parameterized.expand(
        [
            "float32",
            "float16",
            "bfloat16",
        ]
    )
    @require_package("torch")
    def test_simple_load(self, dtype: str):
        try:
            sys.modules["torch"] = sys.modules["torch_save"]
        except:
            pass
        try:
            del sys.modules["transformers"]
        except:
            pass
        import torch

        # torch "normal_kernel_cpu" not implemented for 'Char', 'Int', 'Long', so only support float
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,  # test bfloat16
        }
        dtype = dtype_mapping[dtype]

        with tempfile.TemporaryDirectory() as tempdir:
            weight_file_path = os.path.join(tempdir, "pytorch_model.bin")
            torch.save(
                {
                    "a": torch.randn(2, 3, dtype=dtype),
                    "b": torch.randn(3, 4, dtype=dtype),
                    "a_parameter": torch.nn.Parameter(torch.randn(2, 3, dtype=dtype)),  # test torch.nn.Parameter
                    "b_parameter": torch.nn.Parameter(torch.randn(3, 4, dtype=dtype)),
                },
                weight_file_path,
            )
            numpy_data = load_torch(weight_file_path)
            torch_data = torch.load(weight_file_path)

            for key, arr in numpy_data.items():
                assert np.allclose(
                    paddle.to_tensor(arr).cast("float32").cpu().numpy(),
                    torch_data[key].detach().cpu().to(torch.float32).numpy(),
                )
        sys.modules["torch"] = None
        try:
            del sys.modules["transformers"]
        except:
            pass
