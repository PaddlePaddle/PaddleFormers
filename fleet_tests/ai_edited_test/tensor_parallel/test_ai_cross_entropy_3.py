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


# Extra tests for paddlefleet/tensor_parallel/cross_entropy.py
# Focus on: VocabParallelCrossEntropy static methods, label_smoothing,
# _VocabParallelCrossEntropy backward with label_smoothing

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.tensor_parallel.cross_entropy import (
    VocabParallelCrossEntropy,
    _VocabParallelCrossEntropy,
    vocab_parallel_cross_entropy,
)


class TestCalculateLogitsMax(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_logits_max."""

    def test_returns_float_tensor(self):
        """Test that logits are cast to float."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype=paddle.float16)
        result_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(logits)
        )
        self.assertEqual(result_logits.dtype, paddle.float32)

    def test_logits_max_correct(self):
        """Test that logits_max is the max along last dim."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0]], dtype=paddle.float32)
        result_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(logits)
        )
        self.assertAlmostEqual(logits_max.item(), 5.0)

    def test_logits_max_2d(self):
        """Test logits_max with 2D input."""
        logits = paddle.to_tensor(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=paddle.float32
        )
        result_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(logits)
        )
        self.assertAlmostEqual(logits_max[0].item(), 5.0)
        self.assertAlmostEqual(logits_max[1].item(), 6.0)


class TestCalculatePredictedLogits(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_predicted_logits."""

    def test_target_in_range(self):
        """Test when target is within the vocab range."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0]], dtype=paddle.float32)
        logits_max = paddle.to_tensor([5.0])
        target = paddle.to_tensor([1])

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start_index=0, vocab_end_index=3
        )

        self.assertFalse(target_mask[0].item())
        self.assertAlmostEqual(predicted_logits[0].item(), 0.0)  # 5.0 - 5.0 = 0

    def test_target_out_of_range(self):
        """Test when target is outside the vocab range."""
        logits = paddle.to_tensor([[1.0, 5.0, 3.0]], dtype=paddle.float32)
        logits_max = paddle.to_tensor([5.0])
        target = paddle.to_tensor([5])

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start_index=0, vocab_end_index=3
        )

        self.assertTrue(target_mask[0].item())
        self.assertAlmostEqual(predicted_logits[0].item(), 0.0)


class TestCalculateCrossEntropyLoss(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_cross_entropy_loss."""

    def test_loss_computation(self):
        """Test loss = log(sum_exp) - predicted_logit."""
        import math

        predicted_logits = paddle.to_tensor([0.0])
        exp_logits = paddle.to_tensor([[math.exp(-4.0), 1.0, math.exp(-2.0)]])
        sum_exp_logits = paddle.to_tensor([exp_logits.sum().item()])

        result_exp, loss = (
            VocabParallelCrossEntropy.calculate_cross_entropy_loss(
                exp_logits, predicted_logits, sum_exp_logits
            )
        )

        self.assertAlmostEqual(
            loss.item(), math.log(sum_exp_logits.item()), places=4
        )

    def test_exp_logits_normalized(self):
        """Test that exp_logits are normalized (softmax probabilities)."""

        predicted_logits = paddle.to_tensor([0.0])
        exp_logits = paddle.to_tensor([[1.0, 2.0, 3.0]])
        sum_exp_logits = paddle.to_tensor([6.0])

        result_exp, loss = (
            VocabParallelCrossEntropy.calculate_cross_entropy_loss(
                exp_logits, predicted_logits, sum_exp_logits
            )
        )

        # After normalization, each exp_logit should be divided by sum
        expected = paddle.to_tensor([[1.0 / 6, 2.0 / 6, 3.0 / 6]])
        self.assertTrue(paddle.allclose(result_exp, expected, atol=1e-5))


class TestPrepareGradientCalculationOperands(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.prepare_gradient_calculation_operands."""

    def test_returns_correct_shapes(self):
        """Test that returned tensors have correct shapes."""
        softmax = paddle.to_tensor([[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]])
        target_mask = paddle.to_tensor([False, True])

        grad_2d, arange_1d, softmax_update, grad_input = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        self.assertEqual(grad_2d.shape, [2, 3])
        self.assertEqual(arange_1d.shape, [2])
        self.assertEqual(softmax_update.shape, [2])

    def test_softmax_update_values(self):
        """Test softmax_update is 1.0 for valid, 0.0 for masked targets."""
        softmax = paddle.to_tensor([[0.1, 0.5, 0.4], [0.2, 0.3, 0.5]])
        target_mask = paddle.to_tensor([False, True])

        _, _, softmax_update, _ = (
            VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
                softmax, target_mask
            )
        )

        self.assertAlmostEqual(softmax_update[0].item(), 1.0)
        self.assertAlmostEqual(softmax_update[1].item(), 0.0)


class TestCalculateGradients(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_gradients."""

    def test_gradient_computation(self):
        """Test gradient calculation."""
        grad_2d = paddle.to_tensor([[0.1, 0.5, 0.4]])
        arange_1d = paddle.to_tensor([0])
        masked_target_1d = paddle.to_tensor([1])
        softmax_update = paddle.to_tensor([1.0])
        grad_input = paddle.to_tensor([[0.1, 0.5, 0.4]])
        grad_output = paddle.to_tensor([1.0])

        result = VocabParallelCrossEntropy.calculate_gradients(
            grad_2d,
            arange_1d,
            masked_target_1d,
            softmax_update,
            grad_input,
            grad_output,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, [1, 3])


class TestVocabParallelCrossEntropyForward(unittest.TestCase):
    """Tests for _VocabParallelCrossEntropy forward."""

    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    def test_forward_no_tp_group(self, mock_ws, mock_rank, mock_group):
        """Test forward without tensor parallel group."""
        logits = paddle.to_tensor(
            [[1.0, 2.0, 3.0]], dtype=paddle.float32, stop_gradient=False
        )
        target = paddle.to_tensor([2])

        loss = _VocabParallelCrossEntropy.apply(logits, target, 0.0)
        self.assertIsNotNone(loss)
        self.assertFalse(paddle.isnan(loss))


class TestVocabParallelCrossEntropyBackward(unittest.TestCase):
    """Tests for _VocabParallelCrossEntropy backward."""

    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    def test_backward_no_label_smoothing(self, mock_ws, mock_rank, mock_group):
        """Test backward without label smoothing."""
        logits = paddle.to_tensor(
            [[1.0, 2.0, 3.0]], dtype=paddle.float32, stop_gradient=False
        )
        target = paddle.to_tensor([2])

        loss = _VocabParallelCrossEntropy.apply(logits, target, 0.0)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestVocabParallelCrossEntropyWrapper(unittest.TestCase):
    """Tests for vocab_parallel_cross_entropy wrapper function."""

    def test_wrapper_is_callable(self):
        """Test that vocab_parallel_cross_entropy is callable."""
        self.assertTrue(callable(vocab_parallel_cross_entropy))

    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    def test_wrapper_returns_loss(self, mock_ws, mock_rank, mock_group):
        """Test that wrapper returns a loss tensor."""
        logits = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype=paddle.float32)
        target = paddle.to_tensor([2])
        loss = vocab_parallel_cross_entropy(logits, target)
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()
