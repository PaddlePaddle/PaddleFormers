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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    AddAuxiliaryLoss,
    RandomSTE,
    _AllToAll,
    apply_random_logits,
    detach_and_requires_grad_,
    is_tensor,
    permute,
    unpermute,
)


class TestPermute(unittest.TestCase):
    """Test permute function."""

    def test_basic_permute(self):
        tokens = paddle.randn([4, 8], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
            dtype="float32",
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        # Should have 4 tokens, 3 go to experts 0, 1, 2
        self.assertEqual(permuted.shape[0], 4)
        self.assertEqual(permuted.shape[1], 8)

    def test_permute_single_expert(self):
        tokens = paddle.randn([3, 8], dtype="float32")
        routing_map = paddle.to_tensor(
            [[1, 0], [1, 0], [1, 0]],
            dtype="float32",
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        self.assertEqual(permuted.shape[0], 3)

    def test_permute_drop_and_pad_raises(self):
        tokens = paddle.randn([4, 8], dtype="float32")
        routing_map = paddle.ones([4, 2], dtype="float32")
        with self.assertRaises(AssertionError):
            permute(tokens, routing_map, drop_and_pad=True)


class TestUnpermute(unittest.TestCase):
    """Test unpermute function."""

    def test_basic_unpermute(self):
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        restore_shape = [4, 8]
        output = unpermute(permuted_tokens, sorted_indices, restore_shape)
        self.assertEqual(output.shape, [4, 8])

    def test_unpermute_with_probs(self):
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        restore_shape = [4, 8]
        # probs shape [num_tokens, num_experts], routing_map shape [num_tokens, num_experts]
        probs = paddle.ones([4, 2], dtype="float32")
        # Each token goes to exactly one expert so masked_select produces 4 elements
        routing_map = paddle.to_tensor(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype="float32"
        )
        output = unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
        )
        self.assertEqual(output.shape, [4, 8])

    def test_unpermute_probs_requires_mask(self):
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        restore_shape = [4, 8]
        probs = paddle.ones([4, 2], dtype="float32")
        with self.assertRaises(AssertionError):
            unpermute(
                permuted_tokens, sorted_indices, restore_shape, probs=probs
            )

    def test_unpermute_drop_and_pad_raises(self):
        permuted_tokens = paddle.randn([4, 8], dtype="float32")
        sorted_indices = paddle.to_tensor([0, 2, 1, 3], dtype="int64")
        restore_shape = [4, 8]
        with self.assertRaises(AssertionError):
            unpermute(
                permuted_tokens,
                sorted_indices,
                restore_shape,
                drop_and_pad=True,
            )


class TestAddAuxiliaryLoss(unittest.TestCase):
    """Test AddAuxiliaryLoss PyLayer."""

    def test_forward_clones_input(self):
        x = paddle.randn([2, 4], dtype="float32")
        x.stop_gradient = False
        loss = paddle.randn([1], dtype="float32")
        loss.stop_gradient = False
        result = AddAuxiliaryLoss.apply(x, loss)
        self.assertEqual(result.shape, [2, 4])
        # Result should be a clone (different object)
        self.assertFalse(x is result)

    def test_forward_single_element_loss_assertion(self):
        x = paddle.randn([2, 4], dtype="float32")
        loss = paddle.randn([2], dtype="float32")
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.apply(x, loss)


class TestRandomSTE(unittest.TestCase):
    """Test RandomSTE PyLayer."""

    def test_forward_shape_preserved(self):
        x = paddle.randn([2, 4], dtype="float32")
        result = RandomSTE.apply(x)
        self.assertEqual(result.shape, [2, 4])
        self.assertEqual(result.dtype, paddle.float32)


class TestApplyRandomLogits(unittest.TestCase):
    """Test apply_random_logits function."""

    def test_returns_tensor(self):
        logits = paddle.randn([4, 8], dtype="float32")
        result = apply_random_logits(logits)
        self.assertEqual(result.shape, [4, 8])


class TestIsTensor(unittest.TestCase):
    """Test is_tensor helper."""

    def test_paddle_tensor(self):
        x = paddle.randn([2, 3])
        self.assertTrue(is_tensor(x))

    def test_non_tensor(self):
        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor("hello"))


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Test detach_and_requires_grad_ helper."""

    def test_detach_tensor(self):
        x = paddle.randn([2, 3])
        result = detach_and_requires_grad_(x)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, [2, 3])

    def test_detach_mixed(self):
        x = paddle.randn([2, 3])
        result = detach_and_requires_grad_(x, "string", 42)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1], "string")
        self.assertEqual(result[2], 42)

    def test_preserves_stop_gradient(self):
        x = paddle.randn([2, 3])
        x.stop_gradient = True
        result = detach_and_requires_grad_(x)
        self.assertTrue(result[0].stop_gradient)

    def test_preserves_requires_grad(self):
        x = paddle.randn([2, 3])
        x.stop_gradient = False
        result = detach_and_requires_grad_(x)
        self.assertFalse(result[0].stop_gradient)


class TestAllToAllLayer(unittest.TestCase):
    """Test _AllToAll PyLayer exists."""

    def test_has_forward_and_backward(self):
        self.assertTrue(hasattr(_AllToAll, "forward"))
        self.assertTrue(hasattr(_AllToAll, "backward"))


if __name__ == "__main__":
    unittest.main()
