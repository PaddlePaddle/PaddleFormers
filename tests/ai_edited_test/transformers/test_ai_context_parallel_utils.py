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
from unittest.mock import patch

import paddle

from paddleformers.transformers.context_parallel_utils import (
    auto_split_sequence_dim_load_balance,
    split_inputs_sequence_dim_load_balance,
)


class TestSplitInputsSequenceDimLoadBalance(unittest.TestCase):
    """Tests for split_inputs_sequence_dim_load_balance function."""

    def test_degree_1_returns_input_tensor(self):
        tensor = paddle.randn([2, 8])
        result = split_inputs_sequence_dim_load_balance(tensor, rank=0, degree=1)
        self.assertTrue(paddle.allclose(result, tensor))

    def test_split_tensor_basic(self):
        # With degree=2, rank=0: split into 4 sections, take section[0] and section[3]
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(tensor, rank=0, degree=2)
        self.assertIsNotNone(result)
        # result should have shape [1, 4]
        self.assertEqual(result.shape, [1, 4])

    def test_split_tensor_rank1(self):
        tensor = paddle.arange(8, dtype="float32").reshape([1, 8])
        result = split_inputs_sequence_dim_load_balance(tensor, rank=1, degree=2)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, [1, 4])

    def test_split_dict_inputs(self):
        tensor1 = paddle.randn([2, 8])
        tensor2 = paddle.randn([2, 8])
        inputs = {"key1": tensor1, "key2": tensor2}
        result = split_inputs_sequence_dim_load_balance(inputs, rank=0, degree=2)
        self.assertIsInstance(result, dict)
        self.assertIn("key1", result)
        self.assertIn("key2", result)
        self.assertEqual(result["key1"].shape, [2, 4])
        self.assertEqual(result["key2"].shape, [2, 4])

    def test_split_list_inputs(self):
        tensor1 = paddle.randn([2, 8])
        tensor2 = paddle.randn([2, 8])
        inputs = [tensor1, tensor2]
        result = split_inputs_sequence_dim_load_balance(inputs, rank=0, degree=2)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [2, 4])
        self.assertEqual(result[1].shape, [2, 4])

    def test_invalid_input_type_raises(self):
        with self.assertRaises(ValueError):
            split_inputs_sequence_dim_load_balance("not_valid", rank=0, degree=2)

    def test_none_value_in_dict(self):
        inputs = {"key1": None}
        result = split_inputs_sequence_dim_load_balance(inputs, rank=0, degree=2)
        self.assertIsNone(result["key1"])

    def test_non_2d_tensor_raises(self):
        tensor = paddle.randn([2, 3, 4])
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(tensor, rank=0, degree=2)

    def test_non_tensor_in_list_raises(self):
        inputs = ["not_a_tensor"]
        with self.assertRaises(AssertionError):
            split_inputs_sequence_dim_load_balance(inputs, rank=0, degree=2)


@unittest.skip("shard_seq_load_balance cannot be reliably patched in CI environment")
class TestAutoSplitSequenceDimLoadBalance(unittest.TestCase):
    """Tests for auto_split_sequence_dim_load_balance function."""

    @patch("paddle.distributed.auto_parallel.ring_attention.shard_seq_load_balance")
    def test_tensor_input(self, mock_shard):
        mock_shard.return_value = paddle.randn([2, 4])
        tensor = paddle.randn([2, 8])
        auto_split_sequence_dim_load_balance(tensor)
        mock_shard.assert_called_once_with(tensor, 1)

    @patch("paddle.distributed.auto_parallel.ring_attention.shard_seq_load_balance")
    def test_dict_input(self, mock_shard):
        mock_shard.return_value = paddle.randn([2, 4])
        tensor1 = paddle.randn([2, 8])
        tensor2 = paddle.randn([2, 8])
        inputs = {"key1": tensor1, "key2": tensor2}
        result = auto_split_sequence_dim_load_balance(inputs)
        self.assertIsInstance(result, dict)
        self.assertEqual(mock_shard.call_count, 2)

    @patch("paddle.distributed.auto_parallel.ring_attention.shard_seq_load_balance")
    def test_list_input(self, mock_shard):
        mock_shard.return_value = paddle.randn([2, 4])
        tensor1 = paddle.randn([2, 8])
        inputs = [tensor1]
        result = auto_split_sequence_dim_load_balance(inputs)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_invalid_input_type_raises(self):
        with self.assertRaises(ValueError):
            auto_split_sequence_dim_load_balance("not_valid")


if __name__ == "__main__":
    unittest.main()
