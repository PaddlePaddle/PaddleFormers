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
    AddAuxiliaryLoss,
    FakeClone,
    detach_and_requires_grad_,
    is_tensor,
    log_moe_losses,
    permute,
    unpermute,
)


class TestIsTensor(unittest.TestCase):
    """Tests for is_tensor function."""

    def test_is_tensor_with_paddle_tensor(self):
        """is_tensor should return True for paddle.Tensor."""
        self.assertTrue(is_tensor(paddle.randn([2, 3])))

    def test_is_tensor_with_non_tensor(self):
        """is_tensor should return False for non-tensor values."""
        self.assertFalse(is_tensor(42))
        self.assertFalse(is_tensor([1, 2, 3]))
        self.assertFalse(is_tensor("hello"))

    def test_is_tensor_with_numpy_array(self):
        """is_tensor should return False for numpy arrays."""
        import numpy as np

        self.assertFalse(is_tensor(np.array([1, 2, 3])))


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad_ function."""

    def test_detaches_tensors(self):
        """detach_and_requires_grad_ should detach all tensors and return a list."""
        t = paddle.randn([2, 3])
        t.stop_gradient = False
        result = detach_and_requires_grad_(t)
        self.assertIsInstance(result, list)
        self.assertTrue(paddle.is_tensor(result[0]))
        self.assertTrue(result[0].stop_gradient == t.stop_gradient)

    def test_preserves_non_tensor_values(self):
        """detach_and_requires_grad_ should preserve non-tensor values."""
        result = detach_and_requires_grad_(42, "hello")
        self.assertEqual(result[0], 42)
        self.assertEqual(result[1], "hello")

    def test_preserves_stop_gradient(self):
        """detach_and_requires_grad_ should preserve stop_gradient settings."""
        t1 = paddle.randn([2, 3])
        t1.stop_gradient = True
        t2 = paddle.randn([2, 3])
        t2.stop_gradient = False
        result = detach_and_requires_grad_(t1, t2)
        self.assertEqual(result[0].stop_gradient, True)
        self.assertEqual(result[1].stop_gradient, False)


class TestAddAuxiliaryLoss(unittest.TestCase):
    """Tests for AddAuxiliaryLoss PyLayer."""

    def test_forward_returns_clone(self):
        """AddAuxiliaryLoss.forward should return a clone of input."""
        x = paddle.randn([3, 4])
        loss = paddle.randn([1])
        result = AddAuxiliaryLoss.apply(x, loss)
        self.assertTrue(paddle.allclose(result, x))

    def test_forward_with_stop_gradient_loss(self):
        """AddAuxiliaryLoss.forward with stop_gradient loss should not require aux loss."""
        x = paddle.randn([3, 4])
        loss = paddle.randn([1])
        loss.stop_gradient = True
        result = AddAuxiliaryLoss.apply(x, loss)
        self.assertTrue(paddle.allclose(result, x))


class TestPermuteUnpermute(unittest.TestCase):
    """Tests for permute and unpermute functions."""

    def test_permute_groups_tokens_by_expert(self):
        """permute should group tokens by their assigned expert."""
        paddle.disable_static()
        tokens = paddle.randn([4, 8])
        routing_map = paddle.to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype="float32")
        permuted, indices = permute(tokens, routing_map)
        # 4 non-zero entries in routing_map (rows 0 and 3 both select expert 0)
        self.assertEqual(permuted.shape[0], 4)
        self.assertEqual(permuted.shape[1], 8)

    def test_unpermute_restores_shape(self):
        """unpermute should restore tokens to original shape."""
        paddle.disable_static()
        tokens = paddle.randn([4, 8])
        routing_map = paddle.to_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype="float32")
        permuted, indices = permute(tokens, routing_map)
        restored = unpermute(permuted, indices, tokens.shape)
        self.assertEqual(list(restored.shape), [4, 8])

    def test_permute_rejects_drop_and_pad(self):
        """permute should reject drop_and_pad=True."""
        with self.assertRaises(AssertionError):
            permute(paddle.randn([4, 8]), paddle.randn([4, 3]), drop_and_pad=True)

    def test_unpermute_rejects_drop_and_pad(self):
        """unpermute should reject drop_and_pad=True."""
        with self.assertRaises(AssertionError):
            unpermute(
                paddle.randn([3, 8]),
                paddle.to_tensor([0, 1, 2]),
                [4, 8],
                drop_and_pad=True,
            )


class TestLogMoeLosses(unittest.TestCase):
    """Tests for log_moe_losses function."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
        return_value=False,
    )
    def test_returns_early_when_not_enabled(self, mock_enabled):
        """log_moe_losses should return early when global logs not enabled."""
        # Should not raise any errors
        log_moe_losses(layer_number=0, aux_loss=paddle.randn([1]))

    @patch(
        "paddleformers.fleet.transformer.moe.moe_utils.global_moe_balance_training_logs_enabled",
        return_value=True,
    )
    @patch("paddleformers.fleet.transformer.moe.moe_utils.get_global_training_logs")
    def test_logs_aux_and_z_loss(self, mock_get_logs, mock_enabled):
        """log_moe_losses should log both aux_loss and z_loss when provided."""
        mock_logs = MagicMock()
        mock_get_logs.return_value = mock_logs
        log_moe_losses(
            layer_number=2,
            aux_loss=paddle.randn([1]),
            z_loss=paddle.randn([1]),
        )
        self.assertTrue(mock_logs.update.called)


class TestFakeClone(unittest.TestCase):
    """Tests for FakeClone PyLayer."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_fake_clone_preserves_values(self):
        """FakeClone should produce output with same values as input."""
        paddle.disable_static()
        x = paddle.randn([3, 4])
        result = FakeClone.apply(x)
        self.assertTrue(paddle.allclose(result, x))


if __name__ == "__main__":
    unittest.main()
