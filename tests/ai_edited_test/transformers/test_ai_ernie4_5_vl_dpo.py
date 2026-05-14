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

import paddle
import paddle.nn as nn

from paddleformers.transformers.ernie4_5_moe_vl.model.loss.dpo import ErnieDPOCriterion


class TestErnieDPOCriterion(unittest.TestCase):
    """Tests for ErnieDPOCriterion."""

    def _make_criterion(
        self,
        loss_type="sigmoid",
        offset_alpha=0.0,
        beta=0.1,
        label_smoothing=0.0,
        use_infohub=False,
        simpo_gamma=0.5,
        dpop_lambda=1.0,
    ):
        # Create mock config objects
        dpo_config = MagicMock()
        dpo_config.loss_type = loss_type
        dpo_config.offset_alpha = offset_alpha
        dpo_config.beta = beta
        dpo_config.label_smoothing = label_smoothing
        dpo_config.sft_loss_ratio = 1.0
        dpo_config.pref_loss_ratio = 1.0
        dpo_config.normalize_logps = False
        dpo_config.simpo_gamma = simpo_gamma
        dpo_config.dpop_lambda = dpop_lambda

        config = MagicMock()
        config.use_filtered_label_loss = False
        config.use_fused_head_and_loss_fn = False
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.tensor_parallel_output = False
        config.text_config = MagicMock()
        config.text_config.vocab_size = 32000
        config.text_config.tie_word_embeddings = True

        # Use __new__ to skip ErnieDPOCriterion.__init__, then call nn.Layer.__init__
        # to properly initialize Paddle internals
        criterion = ErnieDPOCriterion.__new__(ErnieDPOCriterion)
        nn.Layer.__init__(criterion)
        criterion.dpo_config = dpo_config
        criterion.config = config
        criterion.use_infohub = use_infohub
        return criterion

    def test_dpo_loss_sigmoid(self):
        criterion = self._make_criterion(loss_type="sigmoid", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])  # loss.mean() returns scalar

    def test_dpo_loss_hinge(self):
        criterion = self._make_criterion(loss_type="hinge", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_ipo(self):
        criterion = self._make_criterion(loss_type="ipo", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_simpo(self):
        criterion = self._make_criterion(loss_type="simpo", beta=0.1, simpo_gamma=0.5)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_kto_pair(self):
        criterion = self._make_criterion(loss_type="kto_pair", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        # KTO applies loss.mean() which returns a scalar
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_sppo_hard(self):
        criterion = self._make_criterion(loss_type="sppo_hard", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_nca_pair(self):
        criterion = self._make_criterion(loss_type="nca_pair", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_or(self):
        criterion = self._make_criterion(loss_type="or", beta=0.1)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_dpop(self):
        criterion = self._make_criterion(loss_type="dpop", beta=0.1, dpop_lambda=1.0)
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        loss = criterion.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        self.assertEqual(loss.shape, [])

    def test_dpo_loss_unknown_type(self):
        criterion = self._make_criterion(loss_type="unknown_type")
        policy_chosen_logps = paddle.randn([4])
        policy_rejected_logps = paddle.randn([4])
        reference_chosen_logps = paddle.randn([4])
        reference_rejected_logps = paddle.randn([4])
        score_deltas = paddle.ones([4])

        with self.assertRaises(ValueError):
            criterion.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                score_deltas,
            )

    def test_forward_with_no_reference(self):
        """Test forward when reference logprobs are None (should compute them)."""
        criterion = self._make_criterion()

        # Mock dpo_logps to return dummy values
        chosen_logps = paddle.randn([4])
        rejected_logps = paddle.randn([4])
        sft_loss = paddle.randn([])

        with patch.object(criterion, "dpo_logps", return_value=(chosen_logps, rejected_logps, sft_loss)):
            logits = (paddle.randn([4, 10]), paddle.randn([10, 5]), None, False)
            response_labels = paddle.randint(0, 10, [4, 5])
            response_indexs = paddle.tensor([[0, 1, 3, 5]])
            # offset_alpha=0, so labels unpack to 4 elements (no score_deltas)
            labels = (response_labels, response_indexs, None, None)

            result = criterion.forward(logits, labels)
            # Should return (reference_chosen_logps, reference_rejected_logps) since no reference
            self.assertEqual(len(result), 2)

    def test_forward_with_reference(self):
        """Test forward with reference logprobs."""
        criterion = self._make_criterion()

        chosen_logps = paddle.randn([4])
        rejected_logps = paddle.randn([4])
        sft_loss = paddle.randn([])
        dpo_loss_val = paddle.randn([])

        with patch.object(criterion, "dpo_logps", return_value=(chosen_logps, rejected_logps, sft_loss)):
            with patch.object(criterion, "dpo_loss", return_value=dpo_loss_val):
                logits = (paddle.randn([4, 10]), paddle.randn([10, 5]), None, False)
                response_labels = paddle.randint(0, 10, [4, 5])
                response_indexs = paddle.tensor([[0, 1, 3, 5]])
                ref_chosen = paddle.randn([4])
                ref_rejected = paddle.randn([4])
                # offset_alpha=0, so labels unpack to 4 elements (no score_deltas)
                labels = (response_labels, response_indexs, ref_chosen, ref_rejected)

                result = criterion.forward(logits, labels)
                # Should return (policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss)
                self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
