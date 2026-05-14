# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddleformers.cli.train.ernie_pretrain.models.utils import (
    FakeClone,
    FakeGather,
    detach_and_requires_grad_,
    get_global_training_logs,
    global_training_logs_enabled,
    manual_backward,
)


class TestGetGlobalTrainingLogs(unittest.TestCase):
    """Tests for get_global_training_logs function."""

    def test_returns_something(self):
        """Test that function returns something (dict or TrainingLogs)."""
        result = get_global_training_logs()
        # It might return a dict, TrainingLogs instance, or other
        self.assertIsNotNone(result)

    @patch("paddleformers.cli.train.ernie_pretrain.models.utils.get_global_training_logs")
    def test_global_training_logs_enabled_with_dict(self, mock_get_logs):
        """Test global_training_logs_enabled with a dict."""
        mock_get_logs.return_value = {}
        self.assertTrue(global_training_logs_enabled())

    @patch("paddleformers.cli.train.ernie_pretrain.models.utils.get_global_training_logs")
    def test_global_training_logs_enabled_with_enabled_object(self, mock_get_logs):
        """Test global_training_logs_enabled with an enabled object."""
        mock_obj = MagicMock()
        mock_obj.is_enabled.return_value = True
        mock_get_logs.return_value = mock_obj
        self.assertTrue(global_training_logs_enabled())


class TestDetachAndRequiresGrad(unittest.TestCase):
    """Tests for detach_and_requires_grad_ function."""

    def test_basic_detach(self):
        """Test that tensors are properly detached."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y = paddle.to_tensor([4.0, 5.0])
        y.stop_gradient = True

        result = detach_and_requires_grad_(x, y)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].stop_gradient == x.stop_gradient)
        self.assertTrue(result[1].stop_gradient == y.stop_gradient)

    def test_none_handling(self):
        """Test that None values are preserved."""
        x = paddle.to_tensor([1.0])
        result = detach_and_requires_grad_(x, None)
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[1])

    def test_all_none(self):
        """Test with all None values."""
        result = detach_and_requires_grad_(None, None)
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])


class TestFakeClone(unittest.TestCase):
    """Tests for FakeClone PyLayer."""

    def test_forward_produces_output(self):
        """Test that FakeClone forward produces output."""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = FakeClone.apply(x)
        np.testing.assert_allclose(result.numpy(), x.numpy(), rtol=1e-5)

    def test_backward_passes_grad_through(self):
        """Test that FakeClone backward passes gradient through."""
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = FakeClone.apply(x)
        loss = y.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], rtol=1e-5)


class TestFakeGather(unittest.TestCase):
    """Tests for FakeGather PyLayer."""

    def test_forward_basic(self):
        """Test FakeGather forward with basic indices."""
        input_tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = paddle.to_tensor([0, 2], dtype=paddle.int64)
        result = FakeGather.apply(input_tensor, indices)
        expected = paddle.to_tensor([[1.0, 2.0], [5.0, 6.0]])
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-5)

    def test_forward_empty_indices(self):
        """Test FakeGather forward with empty indices."""
        input_tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        indices = paddle.to_tensor([], dtype=paddle.int64)
        result = FakeGather.apply(input_tensor, indices)
        self.assertEqual(result.shape[0], 0)

    def test_backward(self):
        """Test FakeGather backward."""
        input_tensor = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], stop_gradient=False)
        indices = paddle.to_tensor([0, 2], dtype=paddle.int64)
        result = FakeGather.apply(input_tensor, indices)
        loss = result.sum()
        loss.backward()
        # Gradients should be scattered back
        self.assertIsNotNone(input_tensor.grad)


class TestManualBackward(unittest.TestCase):
    """Tests for manual_backward function."""

    def test_first_fwd_returns_none_grad_fn(self):
        """Test that manual_backward with is_first_fwd=True returns None for grad fn."""

        def simple_fn(x):
            return x * 2

        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        bwd_f, out = manual_backward(simple_fn, True, x)
        self.assertIsNone(bwd_f)
        self.assertIsInstance(out, tuple)


if __name__ == "__main__":
    unittest.main()
