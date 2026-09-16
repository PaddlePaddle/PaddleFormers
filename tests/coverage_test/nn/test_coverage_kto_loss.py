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
import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestKtoPreprocessInputs(unittest.TestCase):
    """Tests for kto_preprocess_inputs function."""

    def test_import(self):
        """Test that kto_preprocess_inputs can be imported."""
        from paddleformers.nn.criterion.kto_loss import kto_preprocess_inputs

        self.assertTrue(callable(kto_preprocess_inputs))

    def test_plain_logits(self):
        """Test that plain tensor logits pass through unchanged."""
        from paddleformers.nn.criterion.kto_loss import kto_preprocess_inputs

        mock_self = MagicMock()
        logits = paddle.randn([2, 4, 8])
        labels = paddle.randint(0, 8, [2, 4])
        result_logits, result_labels, hidden_states, lm_head_weight, lm_head_bias, transpose_y = kto_preprocess_inputs(
            mock_self, logits, labels
        )
        self.assertTrue(paddle.allclose(result_logits, logits).item())
        self.assertTrue(paddle.allclose(result_labels, labels).item())
        self.assertIsNone(hidden_states)
        self.assertIsNone(lm_head_weight)
        self.assertIsNone(lm_head_bias)
        self.assertIsNone(transpose_y)

    def test_tuple_logits_length_4(self):
        """Test that tuple logits with length 4 are unpacked correctly."""
        from paddleformers.nn.criterion.kto_loss import kto_preprocess_inputs

        mock_self = MagicMock()
        hidden = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 16])
        bias = paddle.randn([16])
        ty = True
        logits_tuple = (hidden, weight, bias, ty)
        labels = paddle.randint(0, 16, [2, 4])
        result_logits, result_labels, hs, w, b, t = kto_preprocess_inputs(mock_self, logits_tuple, labels)
        self.assertIsNone(result_logits)
        self.assertTrue(paddle.allclose(hs, hidden).item())
        self.assertTrue(paddle.allclose(w, weight).item())

    def test_tuple_logits_length_1(self):
        """Test that single-element tuple is recursively unpacked."""
        from paddleformers.nn.criterion.kto_loss import kto_preprocess_inputs

        mock_self = MagicMock()
        inner = paddle.randn([2, 4, 8])
        logits_tuple = (inner,)
        labels = paddle.randint(0, 8, [2, 4])
        result_logits, _, _, _, _, _ = kto_preprocess_inputs(mock_self, logits_tuple, labels)
        self.assertTrue(paddle.allclose(result_logits, inner).item())


class TestNestedGather(unittest.TestCase):
    """Tests for _nested_gather function."""

    def test_import(self):
        """Test that _nested_gather can be imported."""
        from paddleformers.nn.criterion.kto_loss import _nested_gather

        self.assertTrue(callable(_nested_gather))

    def test_none_tensors(self):
        """Test _nested_gather returns None when tensors is None."""
        from paddleformers.nn.criterion.kto_loss import _nested_gather

        mock_self = MagicMock()
        result = _nested_gather(mock_self, None)
        self.assertIsNone(result)

    @patch.dict(os.environ, {"PADDLE_RANK_IN_NODE": "-1"})
    def test_single_rank_returns_tensors(self):
        """Test _nested_gather returns tensors on single rank."""
        from paddleformers.nn.criterion.kto_loss import _nested_gather

        mock_self = MagicMock()
        mock_self.comm_group = None
        tensors = paddle.randn([2, 4])
        # local_rank will be -1, so tensors should be returned unchanged
        result = _nested_gather(mock_self, tensors)
        self.assertIsNotNone(result)


class TestKtoLoss(unittest.TestCase):
    """Tests for kto_loss function."""

    def test_import(self):
        """Test that kto_loss can be imported."""
        from paddleformers.nn.criterion.kto_loss import kto_loss

        self.assertTrue(callable(kto_loss))

    def test_kto_loss_basic(self):
        """Test kto_loss returns a tuple of (loss, kl)."""
        from paddleformers.nn.criterion.kto_loss import kto_loss

        mock_self = MagicMock()
        mock_self.config.kto_config.beta = 0.1
        mock_self.config.kto_config.desirable_weight = 1.0
        mock_self.config.kto_config.undesirable_weight = 1.0

        with patch("paddleformers.nn.criterion.kto_loss.dist.get_world_size", return_value=1):
            loss, kl = kto_loss(
                mock_self,
                policy_chosen_logps=paddle.randn([4]),
                policy_rejected_logps=paddle.randn([4]),
                policy_kl_logps=paddle.randn([4]),
                reference_chosen_logps=paddle.randn([4]),
                reference_rejected_logps=paddle.randn([4]),
                reference_kl_logps=paddle.randn([4]),
            )
        self.assertEqual(loss.shape, [])
        self.assertIsNotNone(kl)

    def test_kto_loss_empty_chosen(self):
        """Test kto_loss with empty chosen logps."""
        from paddleformers.nn.criterion.kto_loss import kto_loss

        mock_self = MagicMock()
        mock_self.config.kto_config.beta = 0.1
        mock_self.config.kto_config.desirable_weight = 1.0
        mock_self.config.kto_config.undesirable_weight = 1.0

        with patch("paddleformers.nn.criterion.kto_loss.dist.get_world_size", return_value=1):
            loss, kl = kto_loss(
                mock_self,
                policy_chosen_logps=paddle.zeros([0]),
                policy_rejected_logps=paddle.randn([4]),
                policy_kl_logps=paddle.randn([4]),
                reference_chosen_logps=paddle.zeros([0]),
                reference_rejected_logps=paddle.randn([4]),
                reference_kl_logps=paddle.randn([4]),
            )
        self.assertEqual(loss.shape, [])

    def test_kto_loss_empty_rejected(self):
        """Test kto_loss with empty rejected logps."""
        from paddleformers.nn.criterion.kto_loss import kto_loss

        mock_self = MagicMock()
        mock_self.config.kto_config.beta = 0.1
        mock_self.config.kto_config.desirable_weight = 1.0
        mock_self.config.kto_config.undesirable_weight = 1.0

        with patch("paddleformers.nn.criterion.kto_loss.dist.get_world_size", return_value=1):
            loss, kl = kto_loss(
                mock_self,
                policy_chosen_logps=paddle.randn([4]),
                policy_rejected_logps=paddle.zeros([0]),
                policy_kl_logps=paddle.randn([4]),
                reference_chosen_logps=paddle.randn([4]),
                reference_rejected_logps=paddle.zeros([0]),
                reference_kl_logps=paddle.randn([4]),
            )
        self.assertEqual(loss.shape, [])

    def test_kto_loss_weights_applied(self):
        """Test that desirable_weight and undesirable_weight are applied."""
        from paddleformers.nn.criterion.kto_loss import kto_loss

        mock_self = MagicMock()
        mock_self.config.kto_config.beta = 0.1
        mock_self.config.kto_config.desirable_weight = 2.0
        mock_self.config.kto_config.undesirable_weight = 0.5

        with patch("paddleformers.nn.criterion.kto_loss.dist.get_world_size", return_value=1):
            loss, kl = kto_loss(
                mock_self,
                policy_chosen_logps=paddle.randn([4]),
                policy_rejected_logps=paddle.randn([4]),
                policy_kl_logps=paddle.randn([4]),
                reference_chosen_logps=paddle.randn([4]),
                reference_rejected_logps=paddle.randn([4]),
                reference_kl_logps=paddle.randn([4]),
            )
        self.assertEqual(loss.shape, [])


class TestKtoLossForward(unittest.TestCase):
    """Tests for kto_loss_forward function."""

    def test_import(self):
        """Test that kto_loss_forward can be imported."""
        from paddleformers.nn.criterion.kto_loss import kto_loss_forward

        self.assertTrue(callable(kto_loss_forward))


if __name__ == "__main__":
    unittest.main()
