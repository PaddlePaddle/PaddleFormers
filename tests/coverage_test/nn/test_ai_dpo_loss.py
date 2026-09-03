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
from unittest.mock import MagicMock

import numpy as np
import paddle


def _make_mock_self(
    loss_type="sigmoid",
    beta=0.1,
    label_smoothing=0.0,
    offset_alpha=0.0,
    simpo_gamma=0.5,
    dpop_lambda=1.0,
    pref_loss_ratio=1.0,
    sft_loss_ratio=0.1,
    normalize_logps=False,
    ignore_eos_token=False,
):
    """Create a mock self object with dpo_config and other required attributes."""
    mock_self = MagicMock()
    mock_dpo_config = MagicMock()
    mock_dpo_config.loss_type = loss_type
    mock_dpo_config.beta = beta
    mock_dpo_config.label_smoothing = label_smoothing
    mock_dpo_config.offset_alpha = offset_alpha
    mock_dpo_config.simpo_gamma = simpo_gamma
    mock_dpo_config.dpop_lambda = dpop_lambda
    mock_dpo_config.pref_loss_ratio = pref_loss_ratio
    mock_dpo_config.sft_loss_ratio = sft_loss_ratio
    mock_dpo_config.normalize_logps = normalize_logps
    mock_dpo_config.ignore_eos_token = ignore_eos_token
    mock_self.dpo_config = mock_dpo_config
    return mock_self


class TestDpoPreprocessInputs(unittest.TestCase):
    """Tests for dpo_preprocess_inputs function."""

    def test_import(self):
        """Test that dpo_preprocess_inputs can be imported."""
        from paddleformers.nn.criterion.dpo_loss import dpo_preprocess_inputs

        self.assertTrue(callable(dpo_preprocess_inputs))

    def test_plain_logits(self):
        """Test that plain tensor logits pass through unchanged."""
        from paddleformers.nn.criterion.dpo_loss import dpo_preprocess_inputs

        mock_self = MagicMock()
        logits = paddle.randn([2, 4, 8])
        labels = paddle.randint(0, 8, [2, 4])
        result_logits, result_labels, hidden_states, lm_head_weight, lm_head_bias, transpose_y = dpo_preprocess_inputs(
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
        from paddleformers.nn.criterion.dpo_loss import dpo_preprocess_inputs

        mock_self = MagicMock()
        hidden = paddle.randn([2, 4, 8])
        weight = paddle.randn([8, 16])
        bias = paddle.randn([16])
        ty = True
        logits_tuple = (hidden, weight, bias, ty)
        labels = paddle.randint(0, 16, [2, 4])
        result_logits, result_labels, hs, w, b, t = dpo_preprocess_inputs(mock_self, logits_tuple, labels)
        self.assertIsNone(result_logits)
        self.assertTrue(paddle.allclose(hs, hidden).item())
        self.assertTrue(paddle.allclose(w, weight).item())

    def test_tuple_logits_length_1(self):
        """Test that single-element tuple is recursively unpacked."""
        from paddleformers.nn.criterion.dpo_loss import dpo_preprocess_inputs

        mock_self = MagicMock()
        inner = paddle.randn([2, 4, 8])
        logits_tuple = (inner,)
        labels = paddle.randint(0, 8, [2, 4])
        result_logits, _, _, _, _, _ = dpo_preprocess_inputs(mock_self, logits_tuple, labels)
        self.assertTrue(paddle.allclose(result_logits, inner).item())


class TestLossImpl(unittest.TestCase):
    """Tests for loss_impl function."""

    def test_import(self):
        """Test that loss_impl can be imported."""
        from paddleformers.nn.criterion.dpo_loss import loss_impl

        self.assertTrue(callable(loss_impl))

    def test_loss_impl_calls_loss_func(self):
        """Test that loss_impl calls self.loss_func with float32 logits."""
        from paddleformers.nn.criterion.dpo_loss import loss_impl

        mock_self = MagicMock()
        expected_loss = paddle.to_tensor(1.5)
        mock_self.loss_func.return_value = expected_loss

        logits = paddle.randn([2, 4, 8], dtype="float16")
        labels = paddle.randint(0, 8, [2, 4])
        loss_impl(mock_self, logits, labels)

        mock_self.loss_func.assert_called_once()
        # Check logits were cast to float32
        call_args = mock_self.loss_func.call_args[0]
        self.assertEqual(call_args[0].dtype, paddle.float32)


class TestCalDpoLoss(unittest.TestCase):
    """Tests for cal_dpo_loss function."""

    def test_import(self):
        """Test that cal_dpo_loss can be imported."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        self.assertTrue(callable(cal_dpo_loss))

    def test_sigmoid_loss(self):
        """Test cal_dpo_loss with sigmoid loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="sigmoid")
        policy_chosen = paddle.randn([4])
        policy_rejected = paddle.randn([4])
        ref_chosen = paddle.randn([4])
        ref_rejected = paddle.randn([4])

        loss = cal_dpo_loss(mock_self, policy_chosen, policy_rejected, ref_chosen, ref_rejected, None)
        self.assertEqual(loss.shape, [])
        self.assertFalse(paddle.isnan(loss).item())

    def test_hinge_loss(self):
        """Test cal_dpo_loss with hinge loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="hinge")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(loss.numpy().item() >= 0)

    def test_simpo_loss(self):
        """Test cal_dpo_loss with simpo loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="simpo")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_ipo_loss(self):
        """Test cal_dpo_loss with ipo loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="ipo")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_dpop_loss(self):
        """Test cal_dpo_loss with dpop loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="dpop")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_kto_pair_loss(self):
        """Test cal_dpo_loss with kto_pair loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="kto_pair")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_sppo_hard_loss(self):
        """Test cal_dpo_loss with sppo_hard loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="sppo_hard")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_nca_pair_loss(self):
        """Test cal_dpo_loss with nca_pair loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="nca_pair")
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_or_loss(self):
        """Test cal_dpo_loss with or loss type."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="or")
        # or loss uses log1p(-exp()), needs negative logprobs
        policy_chosen = -paddle.abs(paddle.randn([4]))
        policy_rejected = -paddle.abs(paddle.randn([4]))
        ref_chosen = -paddle.abs(paddle.randn([4]))
        ref_rejected = -paddle.abs(paddle.randn([4]))
        loss = cal_dpo_loss(mock_self, policy_chosen, policy_rejected, ref_chosen, ref_rejected, None)
        self.assertEqual(loss.shape, [])

    def test_invalid_loss_type_raises(self):
        """Test cal_dpo_loss with invalid loss type raises ValueError."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="invalid_type")
        with self.assertRaises(ValueError):
            cal_dpo_loss(
                mock_self,
                paddle.randn([4]),
                paddle.randn([4]),
                paddle.randn([4]),
                paddle.randn([4]),
                None,
            )

    def test_sigmoid_with_offset_alpha(self):
        """Test sigmoid loss with offset_alpha > 0 and score_deltas provided."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="sigmoid", offset_alpha=0.5)
        score_deltas = paddle.abs(paddle.randn([4]))
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            score_deltas,
        )
        self.assertEqual(loss.shape, [])

    def test_pref_loss_ratio_applied(self):
        """Test that pref_loss_ratio scales the loss."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self_1 = _make_mock_self(loss_type="sigmoid", pref_loss_ratio=1.0)
        mock_self_2 = _make_mock_self(loss_type="sigmoid", pref_loss_ratio=2.0)

        pc = paddle.randn([4])
        pr = paddle.randn([4])
        rc = paddle.randn([4])
        rr = paddle.randn([4])

        loss_1 = cal_dpo_loss(mock_self_1, pc, pr, rc, rr, None)
        loss_2 = cal_dpo_loss(mock_self_2, pc, pr, rc, rr, None)

        np.testing.assert_allclose(loss_2.numpy(), 2.0 * loss_1.numpy(), atol=1e-5)

    def test_label_smoothing_applied(self):
        """Test sigmoid loss with label_smoothing > 0."""
        from paddleformers.nn.criterion.dpo_loss import cal_dpo_loss

        mock_self = _make_mock_self(loss_type="sigmoid", label_smoothing=0.1)
        loss = cal_dpo_loss(
            mock_self,
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            paddle.randn([4]),
            None,
        )
        self.assertEqual(loss.shape, [])


class TestDpoLossForward(unittest.TestCase):
    """Tests for dpo_loss_forward function."""

    def test_import(self):
        """Test that dpo_loss_forward can be imported."""
        from paddleformers.nn.criterion.dpo_loss import dpo_loss_forward

        self.assertTrue(callable(dpo_loss_forward))


if __name__ == "__main__":
    unittest.main()
