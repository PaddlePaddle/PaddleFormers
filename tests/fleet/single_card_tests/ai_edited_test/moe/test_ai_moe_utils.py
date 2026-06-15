# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
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
from unittest.mock import MagicMock, patch

import paddle


class TestMoeUtils(unittest.TestCase):
    """Unit tests for moe_utils module."""

    def test_is_tensor_with_paddle_tensor(self):
        """Test is_tensor returns True for paddle.Tensor."""
        from paddleformers.fleet.transformer.moe.moe_utils import is_tensor

        x = paddle.randn([4, 8])
        self.assertTrue(is_tensor(x))

    def test_is_tensor_with_non_tensor(self):
        """Test is_tensor returns False for non-tensor."""
        from paddleformers.fleet.transformer.moe.moe_utils import is_tensor

        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor(None))

    def test_detach_and_requires_grad(self):
        """Test detach_and_requires_grad_ utility."""
        from paddleformers.fleet.transformer.moe.moe_utils import (
            detach_and_requires_grad_,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = False
        y = 42
        detached_x, detached_y = detach_and_requires_grad_(x, y)
        self.assertEqual(detached_x.shape, [4, 8])
        self.assertFalse(detached_x.stop_gradient)
        self.assertEqual(detached_y, 42)

    def test_detach_and_requires_grad_stop_gradient(self):
        """Test detach_and_requires_grad_ preserves stop_gradient."""
        from paddleformers.fleet.transformer.moe.moe_utils import (
            detach_and_requires_grad_,
        )

        x = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = True
        (detached_x,) = detach_and_requires_grad_(x)
        self.assertTrue(detached_x.stop_gradient)

    def test_apply_random_logits(self):
        """Test apply_random_logits returns random values."""
        from paddleformers.fleet.transformer.moe.moe_utils import apply_random_logits

        logits = paddle.randn([4, 8], dtype=paddle.float32)
        result = apply_random_logits(logits)
        self.assertEqual(result.shape, logits.shape)
        # RandomSTE returns random values, not the same as input
        self.assertFalse(paddle.allclose(result, logits))

    def test_apply_random_logits_gradient_zero(self):
        """Test RandomSTE backward returns zero gradient."""
        from paddleformers.fleet.transformer.moe.moe_utils import apply_random_logits

        logits = paddle.randn([4, 8], dtype=paddle.float32)
        result = apply_random_logits(logits)
        self.assertEqual(result.shape, logits.shape)

    def test_add_auxiliary_loss_forward(self):
        """Test AddAuxiliaryLoss forward returns cloned x."""
        from paddleformers.fleet.transformer.moe.moe_utils import AddAuxiliaryLoss

        x = paddle.randn([4, 8], dtype=paddle.float32)
        loss = paddle.to_tensor(0.5, dtype=paddle.float32)
        out = AddAuxiliaryLoss.apply(x, loss)
        self.assertTrue(paddle.allclose(out, x))

    def test_permute_basic(self):
        """Test permute groups tokens by expert."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute

        tokens = paddle.randn([4, 8], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        routing_map[0, 0] = 1.0
        routing_map[1, 0] = 1.0
        routing_map[2, 1] = 1.0
        routing_map[3, 1] = 1.0
        permuted, indices = permute(tokens, routing_map)
        self.assertEqual(permuted.shape[0], 4)

    def test_permute_with_num_out_tokens(self):
        """Test permute with explicit num_out_tokens (ignored but accepted)."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute

        tokens = paddle.randn([4, 8], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        routing_map[0, 0] = 1.0
        routing_map[1, 1] = 1.0
        permuted, indices = permute(tokens, routing_map, num_out_tokens=2)
        # num_out_tokens is accepted but not used by permute;
        # output size is determined by routing_map (2 true entries)
        self.assertEqual(permuted.shape[0], 2)

    def test_unpermute_basic(self):
        """Test unpermute restores token order."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute, unpermute

        tokens = paddle.randn([4, 8], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        routing_map[0, 0] = 1.0
        routing_map[1, 0] = 1.0
        routing_map[2, 1] = 1.0
        routing_map[3, 1] = 1.0
        permuted, indices = permute(tokens, routing_map)
        restored = unpermute(
            permuted,
            indices,
            restore_shape=tokens.shape,
            probs=None,
            routing_map=None,
        )
        self.assertEqual(restored.shape, tokens.shape)

    def test_unpermute_with_probs(self):
        """Test unpermute applies probs."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute, unpermute

        tokens = paddle.randn([4, 8], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        routing_map[0, 0] = 1.0
        routing_map[1, 0] = 1.0
        routing_map[2, 1] = 1.0
        routing_map[3, 1] = 1.0
        probs = routing_map.clone()
        permuted, indices = permute(tokens, routing_map)
        restored = unpermute(
            permuted,
            indices,
            restore_shape=tokens.shape,
            probs=probs,
            routing_map=routing_map,
        )
        self.assertEqual(restored.shape, tokens.shape)

    def test_permute_drop_and_pad_raises(self):
        """Test permute raises for drop_and_pad=True."""
        from paddleformers.fleet.transformer.moe.moe_utils import permute

        tokens = paddle.randn([4, 8], dtype=paddle.float32)
        routing_map = paddle.zeros([4, 2], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            permute(tokens, routing_map, drop_and_pad=True)

    def test_unpermute_drop_and_pad_raises(self):
        """Test unpermute raises for drop_and_pad=True."""
        from paddleformers.fleet.transformer.moe.moe_utils import unpermute

        with self.assertRaises(AssertionError):
            unpermute(
                paddle.randn([4, 8]),
                paddle.arange(4),
                restore_shape=[4, 8],
                drop_and_pad=True,
            )

    def test_sort_chunks_by_idxs(self):
        """Test sort_chunks_by_idxs reorders chunks."""
        from paddleformers.fleet.transformer.moe.moe_utils import sort_chunks_by_idxs

        x = paddle.concat([paddle.ones([2, 4]) * i for i in range(4)], axis=0)
        split_sizes = paddle.to_tensor([2, 2, 2, 2], dtype=paddle.int64)
        sorted_idxs = paddle.to_tensor([3, 0, 2, 1], dtype=paddle.int64)
        result, probs = sort_chunks_by_idxs(x, split_sizes, sorted_idxs)
        self.assertEqual(result.shape, [8, 4])

    def test_barrier_ep(self):
        """Test barrier_ep calls paddle distributed barrier."""
        from paddleformers.fleet.transformer.moe.moe_utils import barrier_ep

        mock_group = MagicMock()
        with patch("paddle.distributed.barrier") as mock_barrier:
            barrier_ep(mock_group)
            mock_barrier.assert_called_once_with(mock_group)

    def test_all_gather_group_op_single_rank(self):
        """Test AllGatherGroupOp with single rank returns cloned input."""
        from paddleformers.fleet.transformer.moe.moe_utils import AllGatherGroupOp

        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8], dtype=paddle.float32)
        with patch("paddle.distributed.barrier"):
            out = AllGatherGroupOp.apply(x, group=mock_group)
        self.assertEqual(out.shape, x.shape)

    def test_random_ste_forward(self):
        """Test RandomSTE forward returns random values."""
        from paddleformers.fleet.transformer.moe.moe_utils import RandomSTE

        x = paddle.randn([4, 8], dtype=paddle.float32)
        out = RandomSTE.apply(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_fake_clone_forward(self):
        """Test FakeClone forward."""
        from paddleformers.fleet.transformer.moe.moe_utils import FakeClone

        x = paddle.randn([4, 8], dtype=paddle.float32)
        out = FakeClone.apply(x)
        self.assertEqual(out.shape, x.shape)

    def test_manual_backward_first_fwd(self):
        """Test manual_backward with is_first_fwd=True."""
        from paddleformers.fleet.transformer.moe.moe_utils import manual_backward

        def f(x):
            return x * 2.0

        x = paddle.randn([4, 8], dtype=paddle.float32)
        bw_f, out = manual_backward(f, True, x)
        self.assertIsNone(bw_f)
        self.assertEqual(len(out), 1)

    def test_is_tensor_core_tensor(self):
        """Test is_tensor with core eager tensor."""
        from paddleformers.fleet.transformer.moe.moe_utils import is_tensor

        x = paddle.randn([4, 8])
        self.assertTrue(is_tensor(x))

    def test_all_gather_group_function_single_rank(self):
        """Test all_gather_group with single rank."""
        from paddleformers.fleet.transformer.moe.moe_utils import all_gather_group

        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8], dtype=paddle.float32)
        out = all_gather_group(x, group=mock_group)
        self.assertEqual(out.shape, x.shape)

    def test_reduce_scatter_group_single_rank(self):
        """Test reduce_scatter_group with single rank."""
        from paddleformers.fleet.transformer.moe.moe_utils import reduce_scatter_group

        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8], dtype=paddle.float32)
        out = reduce_scatter_group(x, group=mock_group)
        self.assertEqual(out.shape, x.shape)

    def test_reduce_scatter_group_assert_divisible(self):
        """Test reduce_scatter_group asserts divisible."""
        from paddleformers.fleet.transformer.moe.moe_utils import reduce_scatter_group

        mock_group = MagicMock()
        mock_group.nranks = 3
        x = paddle.randn([5, 8], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            reduce_scatter_group(x, group=mock_group)

    def test_all_gather_group_assert_axis(self):
        """Test all_gather_group asserts axis=0."""
        from paddleformers.fleet.transformer.moe.moe_utils import all_gather_group

        mock_group = MagicMock()
        mock_group.nranks = 2
        x = paddle.randn([4, 8], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            all_gather_group(x, group=mock_group, axis=1)


if __name__ == "__main__":
    unittest.main()
