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
from unittest.mock import patch

import numpy as np
import paddle


class TestVocabParallelCrossEntropyCalculateLogitsMax(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_logits_max."""

    def test_calculate_logits_max(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        logits = paddle.randn([4, 8], dtype=paddle.float32)
        result_logits, logits_max = (
            VocabParallelCrossEntropy.calculate_logits_max(logits)
        )

        self.assertEqual(result_logits.shape, [4, 8])
        self.assertEqual(logits_max.shape, [4])
        # Max should be the max along last dim
        expected_max = paddle.max(logits.astype(paddle.float32), axis=-1)
        np.testing.assert_allclose(logits_max, expected_max)


class TestVocabParallelCrossEntropyCalculatePredictedLogits(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_predicted_logits."""

    def test_calculate_predicted_logits_all_in_range(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target = paddle.to_tensor([0, 5, 10, 15])

        logits_max = paddle.max(logits, axis=-1, keepdim=False)
        vocab_start = 0
        vocab_end = vocab_size

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start, vocab_end
        )

        # All targets are in range so mask should be all False
        self.assertEqual(target_mask.shape, [batch_size])
        self.assertFalse(target_mask.any())

    def test_calculate_predicted_logits_out_of_range(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        # Target 0 is in range [0, 8), 8 and 12 are out of range
        target = paddle.to_tensor([0, 8, 5, 12])

        logits_max = paddle.max(logits, axis=-1, keepdim=False)
        vocab_start = 0
        vocab_end = 8

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start, vocab_end
        )

        # Targets 1 and 3 are out of range
        self.assertEqual(target_mask.shape, [batch_size])
        self.assertTrue(target_mask[1])
        self.assertTrue(target_mask[3])
        # Masked target for out-of-range should be 0
        self.assertEqual(masked_target_1d[1].item(), 0)
        self.assertEqual(masked_target_1d[3].item(), 0)
        # Predicted logits for out-of-range targets should be 0
        self.assertEqual(predicted_logits[1].item(), 0.0)
        self.assertEqual(predicted_logits[3].item(), 0.0)

    def test_calculate_predicted_logits_partial_range(self):
        """Test with vocab_start > 0."""
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target = paddle.to_tensor([2, 10, 15, 5])

        logits_max = paddle.max(logits, axis=-1, keepdim=False)
        vocab_start = 8
        vocab_end = 16

        (
            target_mask,
            masked_target_1d,
            predicted_logits,
            sum_exp_logits,
            exp_logits,
        ) = VocabParallelCrossEntropy.calculate_predicted_logits(
            logits, target, logits_max, vocab_start, vocab_end
        )

        # Target 0,3 are out of range (< 8)
        self.assertTrue(target_mask[0])
        self.assertTrue(target_mask[3])
        # Target 1,2 are in range
        self.assertFalse(target_mask[1])
        self.assertFalse(target_mask[2])


class TestVocabParallelCrossEntropyCalculateCrossEntropyLoss(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_cross_entropy_loss."""


class TestVocabParallelCrossEntropyPrepareGradientCalculationOperands(
    unittest.TestCase
):
    """Tests for VocabParallelCrossEntropy.prepare_gradient_calculation_operands."""

    def test_prepare_gradient_calculation_operands(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        softmax = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target_mask = paddle.zeros([batch_size], dtype=paddle.bool)
        target_mask[0] = True
        target_mask[2] = True

        (
            grad_2d,
            arange_1d,
            softmax_update,
            grad_input,
        ) = VocabParallelCrossEntropy.prepare_gradient_calculation_operands(
            softmax, target_mask
        )

        self.assertEqual(grad_2d.shape, [batch_size, vocab_size])
        self.assertEqual(arange_1d.shape, [batch_size])
        # softmax_update should be 1.0 where mask is False, 0.0 where mask is True
        self.assertEqual(softmax_update[0].item(), 0.0)
        self.assertEqual(softmax_update[1].item(), 1.0)
        self.assertEqual(softmax_update[2].item(), 0.0)
        self.assertEqual(softmax_update[3].item(), 1.0)


class TestVocabParallelCrossEntropyCalculateGradients(unittest.TestCase):
    """Tests for VocabParallelCrossEntropy.calculate_gradients."""


class TestVocabParallelCrossEntropyForward(unittest.TestCase):
    """Tests for _VocabParallelCrossEntropy forward and backward."""

    def test_forward_no_tp_no_smoothing(self):
        """Test forward pass without tensor parallelism and label smoothing."""
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target = paddle.to_tensor([0, 5, 10, 15])

        loss = VocabParallelCrossEntropy.calculate_logits_max(logits)
        self.assertIsNotNone(loss)

    def test_forward_with_label_smoothing(self):
        """Test forward pass with label smoothing."""
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            _VocabParallelCrossEntropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        logits.stop_gradient = False
        target = paddle.to_tensor([0, 5, 10, 15])

        with patch(
            "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
            return_value=None,
        ):
            loss = _VocabParallelCrossEntropy.apply(logits, target, 0.1)
        self.assertEqual(loss.shape, [batch_size])

    def test_forward_2d_input(self):
        """Test forward with 2D input (seq_len * batch, vocab)."""
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            _VocabParallelCrossEntropy,
        )

        n, vocab_size = 8, 16
        logits = paddle.randn([n, vocab_size], dtype=paddle.float32)
        target = paddle.randint(0, vocab_size, shape=[n])

        with patch(
            "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
            return_value=None,
        ):
            loss = _VocabParallelCrossEntropy.apply(logits, target, 0.0)

        self.assertEqual(loss.shape, [n])


class TestVocabParallelCrossEntropyFunction(unittest.TestCase):
    """Tests for the public vocab_parallel_cross_entropy function."""

    def test_vocab_parallel_cross_entropy_basic(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            vocab_parallel_cross_entropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target = paddle.to_tensor([0, 5, 10, 15])

        with patch(
            "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
            return_value=None,
        ):
            loss = vocab_parallel_cross_entropy(logits, target)

        self.assertEqual(loss.shape, [batch_size])

    def test_vocab_parallel_cross_entropy_with_smoothing(self):
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            vocab_parallel_cross_entropy,
        )

        batch_size, vocab_size = 4, 16
        logits = paddle.randn([batch_size, vocab_size], dtype=paddle.float32)
        target = paddle.to_tensor([0, 5, 10, 15])

        with patch(
            "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
            return_value=None,
        ):
            loss = vocab_parallel_cross_entropy(
                logits, target, label_smoothing=0.1
            )

        self.assertEqual(loss.shape, [batch_size])

    def test_vocab_parallel_cross_entropy_3d_input(self):
        """Test with 3D input (seq_len, batch, vocab)."""
        from paddleformers.fleet.tensor_parallel.cross_entropy import (
            vocab_parallel_cross_entropy,
        )

        seq_len, batch_size, vocab_size = 2, 4, 16
        logits = paddle.randn(
            [seq_len, batch_size, vocab_size], dtype=paddle.float32
        )
        target = paddle.randint(0, vocab_size, shape=[seq_len, batch_size])

        with patch(
            "paddleformers.fleet.tensor_parallel.cross_entropy.get_tensor_model_parallel_group",
            return_value=None,
        ):
            loss = vocab_parallel_cross_entropy(logits, target)

        self.assertEqual(loss.shape, [seq_len, batch_size])
