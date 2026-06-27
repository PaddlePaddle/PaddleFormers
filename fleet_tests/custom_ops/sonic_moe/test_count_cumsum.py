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
from collections.abc import Callable

import paddle
from paddlefleet_ops.sonicmoe import count_cumsum
from parameterized import parameterized

from .commons_test import TestCommons


def get_1d_tensor_sizes() -> list[tuple[int]]:
    sizes = set()
    # powers of 2
    for i in range(15):
        start = 2**i
        for j in range(10):
            sizes.add(start + j)
    # not powers of 2
    for _ in range(50):
        sizes.add(3000 + random.randint(-1000, 1000))
    return sizes


class CountCumsumTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [*list(get_1d_tensor_sizes()), 2097152],  # size
            [4, 8, 72, 256, 1920, 2048, 16384, 50000],  # num_experts
            [False, True],  # do_cumsum
            [paddle.CUDAPlace(0)],  # device
            [paddle.long, paddle.int],  # dtype
            [
                count_cumsum
            ],  # , torch.compile(count_cumsum, fullgraph=True)],  # function
        )
    )
    def test_count_cumsum(
        self,
        size: int,
        num_experts: int,
        do_cumsum: bool,
        device: paddle.device,
        dtype: paddle.dtype,
        function: Callable,
    ) -> None:
        # torch._dynamo.config.cache_size_limit = 1024
        # torch._dynamo.config.accumulated_cache_size_limit = 1024

        x = paddle.randint(0, num_experts, (size,), dtype=dtype).to(device)

        z_kernel_cumsum = None
        z_kernel_indices = None

        if do_cumsum:
            z_kernel_count, z_kernel_cumsum = function(
                x=x, E=num_experts, do_cumsum=do_cumsum
            )
        else:
            z_kernel_count = function(x=x, E=num_experts, do_cumsum=do_cumsum)

        z_expected_count = x.view(-1).bincount(minlength=num_experts)
        self.assert_equal_tensors(z_kernel_count, z_expected_count, True)

        if z_kernel_cumsum is not None:
            z_expected_cumsum = z_expected_count.cumsum(-1)
            self.assert_equal_tensors(z_kernel_cumsum, z_expected_cumsum, True)
