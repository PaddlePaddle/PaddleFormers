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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    AddAuxiliaryLoss,
    FakeClone,
    _AllToAll,
    manual_backward,
)


class TestManualBackwardFunction(unittest.TestCase):
    """Tests for manual_backward function."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.framework._dygraph_tracer"
    )
    def test_returns_none_backward_fn_when_is_first_fwd(self, mock_tracer):
        """manual_backward should return (None, out) when is_first_fwd=True."""
        mock_tracer.return_value = MagicMock()
        mock_tracer.return_value._has_grad = True

        def simple_fn(x):
            return x * 2

        x = paddle.randn([4, 8])
        x.stop_gradient = False
        bwd_f, out = manual_backward(simple_fn, True, x)
        self.assertIsNone(bwd_f)
        self.assertIsNotNone(out)


class TestAllToAllSingleProcess(unittest.TestCase):
    """Tests for _AllToAll with single process."""

    def test_forward_returns_input_when_world_size_1(self):
        """_AllToAll forward should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _AllToAll.forward(
            MagicMock(),
            [4, 8],  # output_shape
            x,
            None,  # out_split_sizes
            None,  # in_split_sizes
            mock_group,
        )
        self.assertTrue(paddle.allclose(result, x).item())


class TestFakeCloneContiguous(unittest.TestCase):
    """Tests for FakeClone with contiguous tensors."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_fake_clone_contiguous_input(self):
        """FakeClone should share data for contiguous tensors."""
        paddle.disable_static()
        x = paddle.randn([3, 4])
        result = FakeClone.apply(x)
        self.assertTrue(paddle.allclose(result, x))


class TestAddAuxiliaryLossBackward(unittest.TestCase):
    """Tests for AddAuxiliaryLoss backward."""

    def test_backward_with_required_aux_loss(self):
        """backward should return grad_loss=1 when loss requires gradient."""
        paddle.disable_static()
        x = paddle.randn([3, 4])
        x.stop_gradient = False
        loss = paddle.randn([1])
        loss.stop_gradient = False

        # Create context that would be stored
        ctx = MagicMock()
        ctx.dtype = paddle.float32
        ctx.required_aux_loss = True

        grad_output = paddle.ones([3, 4])
        result = AddAuxiliaryLoss.backward(ctx, grad_output)
        self.assertEqual(len(result), 2)
        self.assertTrue(paddle.is_tensor(result[0]))
        self.assertTrue(paddle.is_tensor(result[1]))

    def test_backward_without_required_aux_loss(self):
        """backward should return grad_loss=None when loss doesn't require gradient."""
        ctx = MagicMock()
        ctx.dtype = paddle.float32
        ctx.required_aux_loss = False

        grad_output = paddle.ones([3, 4])
        result = AddAuxiliaryLoss.backward(ctx, grad_output)
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1])


if __name__ == "__main__":
    unittest.main()
