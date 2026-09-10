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

import unittest

import paddle

from paddleformers.transformers.moe_layer_auto import combining, dispatching


class TestDispatching(unittest.TestCase):
    """Tests for dispatching function."""

    def test_dispatching_basic(self):
        x = paddle.randn([8, 4], dtype="float32")
        num_experts = 2
        capacity = 4

        dispatch_mask = [
            paddle.ones([8, 1], dtype="float32"),
            paddle.ones([8, 1], dtype="float32"),
        ]
        scatter_index = [
            paddle.randint(0, num_experts * capacity, [8]),
            paddle.randint(0, num_experts * capacity, [8]),
        ]
        result = dispatching(x, dispatch_mask, scatter_index, num_experts, capacity)
        self.assertEqual(result.shape, [num_experts * capacity, 4])

    def test_dispatching_with_tensor_scatter_index(self):
        x = paddle.randn([8, 4], dtype="float32")
        num_experts = 2
        capacity = 4

        dispatch_mask = [
            paddle.ones([8, 1], dtype="float32"),
            paddle.ones([8, 1], dtype="float32"),
        ]
        scatter_index = paddle.stack(
            [
                paddle.randint(0, num_experts * capacity, [8]),
                paddle.randint(0, num_experts * capacity, [8]),
            ],
            axis=1,
        )
        result = dispatching(x, dispatch_mask, scatter_index, num_experts, capacity)
        self.assertEqual(result.shape, [num_experts * capacity, 4])

    def test_dispatching_output_dtype_matches_input(self):
        x = paddle.randn([8, 4], dtype="float32")
        num_experts = 2
        capacity = 4
        dispatch_mask = [
            paddle.ones([8, 1], dtype="float32"),
            paddle.ones([8, 1], dtype="float32"),
        ]
        scatter_index = [
            paddle.randint(0, num_experts * capacity, [8]),
            paddle.randint(0, num_experts * capacity, [8]),
        ]
        result = dispatching(x, dispatch_mask, scatter_index, num_experts, capacity)
        self.assertEqual(result.dtype, paddle.float32)


class TestCombining(unittest.TestCase):
    """Tests for combining function."""

    def test_combining_basic(self):
        num_experts = 2
        capacity = 4
        dim = 8
        x = paddle.randn([num_experts * capacity, dim])

        combine_weights = [
            paddle.randn([6, 1], dtype="float32"),
            paddle.randn([6, 1], dtype="float32"),
        ]
        scatter_index = [
            paddle.randint(0, num_experts * capacity, [6]),
            paddle.randint(0, num_experts * capacity, [6]),
        ]
        result = combining(x, combine_weights, scatter_index)
        self.assertEqual(result.shape[0], 6)
        self.assertEqual(result.shape[-1], dim)

    def test_combining_with_tensor_scatter_index(self):
        num_experts = 2
        capacity = 4
        dim = 8
        x = paddle.randn([num_experts * capacity, dim])

        combine_weights = [
            paddle.randn([6, 1], dtype="float32"),
            paddle.randn([6, 1], dtype="float32"),
        ]
        scatter_index = paddle.stack(
            [
                paddle.randint(0, num_experts * capacity, [6]),
                paddle.randint(0, num_experts * capacity, [6]),
            ],
            axis=1,
        )
        result = combining(x, combine_weights, scatter_index)
        self.assertEqual(result.shape[0], 6)
        self.assertEqual(result.shape[-1], dim)


if __name__ == "__main__":
    unittest.main()
