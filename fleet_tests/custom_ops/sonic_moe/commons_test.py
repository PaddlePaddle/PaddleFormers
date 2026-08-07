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

import random
from itertools import product
from typing import Any
from unittest import TestCase

import numpy as np
import paddle
from paddle import nn

try:
    from paddle.testing import assert_close as assert_close
except:

    def assert_close(
        actual,
        desired,
        rtol=1e-07,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
        *,
        strict=False,
    ):
        actual = actual.numpy()
        desired = desired.numpy()
        return np.testing.assert_allclose(actual, desired, rtol, atol)


class TestCommons(TestCase):
    @staticmethod
    def get_dtypes() -> list[paddle.dtype]:
        return [paddle.float32, paddle.float16, paddle.bfloat16]

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        paddle.cuda.manual_seed_all(seed)

    def make_args_matrix(*args_lists) -> list[Any]:
        return list(product(*args_lists))

    def assert_equal_tensors(
        self,
        x: paddle.Tensor,
        y: paddle.Tensor,
        exact_match: bool,
        rtol_float32: float | None = None,
        atol_float32: float | None = None,
        rtol_float16: float | None = None,
        atol_float16: float | None = None,
        rtol_bfloat16: float | None = None,
        atol_bfloat16: float | None = None,
        dtype: paddle.dtype = paddle.float32,
    ) -> None:
        if exact_match:
            if x.dtype == paddle.int32 and y.dtype == paddle.int64:
                x = x.to(paddle.int64)
            if x.dtype == paddle.int64 and y.dtype == paddle.int32:
                y = y.to(paddle.int64)
            assert x.equal(y).all()
        else:
            assert x.dtype == y.dtype

            if dtype == paddle.float32:
                assert_close(x, y, rtol=rtol_float32, atol=atol_float32)
            elif dtype == paddle.float16:
                assert_close(x, y, rtol=rtol_float16, atol=atol_float16)
            elif dtype == paddle.bfloat16:
                assert_close(x, y, rtol=rtol_bfloat16, atol=atol_bfloat16)
            else:
                raise ValueError(f"unexpected dtype ({dtype})")

    def get_activation_function(self, is_glu: bool) -> nn.Module:
        return nn.GLU() if is_glu else nn.GELU(approximate="tanh")

    def collect_gradients_from_module_and_zero_grads(
        self, model: nn.Module
    ) -> dict[str, paddle.Tensor]:
        grads = {}
        for weight_name, weight in model.named_parameters():
            grads[weight_name] = weight.grad

        model.zero_grad()

        return grads
