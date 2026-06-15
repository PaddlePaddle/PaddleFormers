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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    AddAuxiliaryLoss,
    RandomSTE,
    apply_random_logits,
    permute,
    unpermute,
)


class TestRandomSTEDetailed(unittest.TestCase):
    """Detailed tests for RandomSTE."""

    def test_forward_basic(self):
        """Test RandomSTE forward produces output of same shape and dtype."""
        x = paddle.randn([4, 8], dtype="float32")
        with patch("paddle.distributed.get_world_size", return_value=1):
            out = RandomSTE.apply(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_backward_returns_zeros(self):
        """Test RandomSTE backward returns zero gradients."""
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        with patch("paddle.distributed.get_world_size", return_value=1):
            out = RandomSTE.apply(x)
        out_sum = out.sum()
        out_sum.backward()
        self.assertTrue(paddle.allclose(x.grad, paddle.zeros_like(x)).item())


class TestApplyRandomLogits(unittest.TestCase):
    """Tests for apply_random_logits."""

    def test_basic_apply(self):
        """Test apply_random_logits returns same shape."""
        logits = paddle.randn([4, 8], dtype="float32")
        with patch("paddle.distributed.get_world_size", return_value=1):
            out = apply_random_logits(logits)
        self.assertEqual(out.shape, logits.shape)


class TestPermuteDetailed(unittest.TestCase):
    """Detailed tests for permute function."""

    def test_permute_multiple_experts(self):
        """Test permute with multiple experts."""
        tokens = paddle.randn([6, 8], dtype="float32")
        routing_map = paddle.to_tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype="float32",
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        self.assertEqual(permuted.shape[0], 6)
        self.assertEqual(permuted.shape[1], 8)
        self.assertEqual(sorted_indices.shape[0], 6)

    def test_permute_preserves_values(self):
        """Test that permute preserves token values."""
        tokens = paddle.randn([4, 8], dtype="float32")
        routing_map = paddle.to_tensor([[1, 0], [1, 0], [1, 0], [1, 0]], dtype="float32")
        permuted, sorted_indices = permute(tokens, routing_map)
        # All tokens go to expert 0, should be in original order
        self.assertTrue(paddle.allclose(tokens, permuted).item())


class TestUnpermuteDetailed(unittest.TestCase):
    """Detailed tests for unpermute function."""

    def test_unpermute_without_probs(self):
        """Test unpermute without probs."""
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        restore_shape = [4, 8]
        output = unpermute(permuted_tokens, sorted_indices, restore_shape)
        self.assertEqual(output.shape, [4, 8])

    def test_unpermute_with_probs_and_routing_map(self):
        """Test unpermute with probs and routing_map."""
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        restore_shape = [4, 8]
        probs = paddle.to_tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]], dtype="float32")
        routing_map = paddle.to_tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype="float32")
        output = unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
        )
        self.assertEqual(output.shape, [4, 8])

    def test_unpermute_drop_and_pad_raises(self):
        """Test unpermute with drop_and_pad raises."""
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        restore_shape = [4, 8]
        with self.assertRaises(AssertionError):
            unpermute(
                permuted_tokens,
                sorted_indices,
                restore_shape,
                drop_and_pad=True,
            )


class TestAddAuxiliaryLossBackward(unittest.TestCase):
    """Tests for AddAuxiliaryLoss backward pass."""

    def test_backward_with_required_loss(self):
        """Test backward when aux loss gradient is required."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        out_sum = out.sum()
        out_sum.backward()
        # x should have gradient
        self.assertIsNotNone(x.grad)

    def test_backward_with_non_required_loss(self):
        """Test backward when aux loss gradient is not required."""
        x = paddle.randn([4, 8])
        x.stop_gradient = False
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = True
        out = AddAuxiliaryLoss.apply(x, loss)
        out_sum = out.sum()
        out_sum.backward()
        # x should still have gradient from normal backward
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
