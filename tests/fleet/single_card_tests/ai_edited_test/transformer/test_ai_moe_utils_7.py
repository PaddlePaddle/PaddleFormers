# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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

from paddleformers.fleet.transformer.moe.moe_utils import (
    AllGatherGroupOp,
    RandomSTE,
    apply_random_logits,
)


class TestRandomSTE(unittest.TestCase):
    """Tests for RandomSTE PyLayer."""

    def test_forward_returns_correct_shape(self):
        """RandomSTE forward should return a tensor with the same shape as input."""
        paddle.disable_static()
        x = paddle.randn([3, 4])
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.dist.get_world_size",
            return_value=1,
        ):
            result = RandomSTE.forward(MagicMock(), x)
            self.assertEqual(list(result.shape), [3, 4])

    def test_backward_returns_zeros(self):
        """RandomSTE backward should return zeros with same shape as input."""
        ctx = MagicMock()
        ctx.x_shape = [3, 4]
        ctx.x_dtype = paddle.float32
        grad = paddle.randn([3, 4])
        result = RandomSTE.backward(ctx, grad)
        self.assertTrue(paddle.all(result == 0))

    def test_backward_preserves_shape_and_dtype(self):
        """RandomSTE backward should match input shape and dtype."""
        ctx = MagicMock()
        ctx.x_shape = [2, 3, 4]
        ctx.x_dtype = paddle.float32
        grad = paddle.randn([2, 3, 4])
        result = RandomSTE.backward(ctx, grad)
        self.assertEqual(list(result.shape), [2, 3, 4])
        self.assertEqual(result.dtype, paddle.float32)


class TestApplyRandomLogits(unittest.TestCase):
    """Tests for apply_random_logits function."""

    def test_returns_tensor(self):
        """apply_random_logits should return a tensor."""
        paddle.disable_static()
        logits = paddle.randn([4, 8])
        with patch(
            "paddleformers.fleet.transformer.moe.moe_utils.dist.get_world_size",
            return_value=1,
        ):
            result = apply_random_logits(logits)
            self.assertTrue(paddle.is_tensor(result))
            self.assertEqual(list(result.shape), [4, 8])


class TestAllGatherGroupOpForward(unittest.TestCase):
    """Tests for AllGatherGroupOp forward."""

    def test_forward_with_single_process(self):
        """AllGatherGroupOp forward should clone input when group has 1 rank."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        paddle.disable_static()
        x = paddle.randn([4, 8])
        with patch("paddleformers.fleet.transformer.moe.moe_utils.paddle.distributed.barrier"):
            result = AllGatherGroupOp.forward(MagicMock(), x, mock_group)
            self.assertEqual(result.shape, x.shape)
            self.assertTrue(paddle.allclose(result, x))


class TestAllGatherGroupOpBackward(unittest.TestCase):
    """Tests for AllGatherGroupOp backward."""

    def test_backward_with_single_process(self):
        """AllGatherGroupOp backward with 1-rank group should return cloned grad."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        ctx = MagicMock()
        ctx.group = mock_group
        grad = paddle.randn([4, 8])
        with patch("paddleformers.fleet.transformer.moe.moe_utils.paddle.distributed.barrier"):
            result = AllGatherGroupOp.backward(ctx, grad)
            self.assertEqual(result.shape, grad.shape)
            self.assertTrue(paddle.allclose(result, grad))


if __name__ == "__main__":
    unittest.main()
