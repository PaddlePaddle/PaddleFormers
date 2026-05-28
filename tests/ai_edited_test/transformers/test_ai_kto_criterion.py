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

import unittest
from unittest.mock import patch

import paddle
import paddle.nn as nn


class _MockKTOConfig:
    """Mock KTO config object."""

    def __init__(self, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight


class _MockConfig:
    """Mock model config for KTO criterion."""

    def __init__(self, **kwargs):
        self.tensor_parallel_output = kwargs.get("tensor_parallel_output", False)
        self.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
        self.vocab_size = kwargs.get("vocab_size", 1000)
        self.fused_linear = kwargs.get("fused_linear", False)
        self.use_fused_head_and_loss_fn = kwargs.get("use_fused_head_and_loss_fn", False)
        self.use_filtered_label_loss = kwargs.get("use_filtered_label_loss", False)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.chunk_size = kwargs.get("chunk_size", 1024)
        self.kto_config = kwargs.get("kto_config", None)


class TestKTOCriterionInit(unittest.TestCase):
    """Tests for KTOCriterion initialization."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_with_config_kto_config(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)
        self.assertEqual(criterion.kto_config.beta, kto_config.beta)
        self.assertEqual(criterion.kto_config.desirable_weight, kto_config.desirable_weight)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_with_explicit_kto_config(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig(beta=0.2)
        config = _MockConfig()
        criterion = KTOCriterion(config, kto_config=kto_config)
        self.assertEqual(criterion.kto_config.beta, 0.2)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_missing_kto_config_raises(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        config = _MockConfig(kto_config=None)
        with self.assertRaises(ValueError):
            KTOCriterion(config)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_with_infohub(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config, use_infohub=True)
        self.assertTrue(criterion.use_infohub)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_with_ignore_label(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config, ignore_label=-100)
        self.assertEqual(criterion.ignore_label, -100)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_uses_cross_entropy_loss(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)
        self.assertIsInstance(criterion.logprobs, nn.CrossEntropyLoss)


class TestKTOCriterionNestedGather(unittest.TestCase):
    """Tests for KTOCriterion._nested_gather."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_nested_gather_none(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)
        result = criterion._nested_gather(None)
        self.assertIsNone(result)

    @patch("paddle.distributed.get_world_size", return_value=1)
    @patch.dict("os.environ", {"PADDLE_RANK_IN_NODE": "-1"})
    def test_nested_gather_single_gpu(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)
        tensor = paddle.randn([2, 4], dtype="float32")
        result = criterion._nested_gather(tensor)
        # With single GPU and no rank, should return tensor as-is
        self.assertIsNotNone(result)


class TestKTOCriterionKtoLoss(unittest.TestCase):
    """Tests for KTOCriterion.kto_loss."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_loss_basic(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig(beta=0.1)
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)

        policy_chosen_logps = paddle.randn([4], dtype="float32")
        policy_rejected_logps = paddle.randn([4], dtype="float32")
        policy_kl_logps = paddle.randn([4], dtype="float32")
        reference_chosen_logps = paddle.randn([4], dtype="float32")
        reference_rejected_logps = paddle.randn([4], dtype="float32")
        reference_kl_logps = paddle.randn([4], dtype="float32")

        loss, kl = criterion.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        self.assertIsNotNone(loss)
        self.assertIsNotNone(kl)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_loss_empty_chosen(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig(beta=0.1)
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)

        policy_chosen_logps = paddle.zeros([0], dtype="float32")
        policy_rejected_logps = paddle.randn([4], dtype="float32")
        policy_kl_logps = paddle.randn([4], dtype="float32")
        reference_chosen_logps = paddle.zeros([0], dtype="float32")
        reference_rejected_logps = paddle.randn([4], dtype="float32")
        reference_kl_logps = paddle.randn([4], dtype="float32")

        loss, kl = criterion.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        self.assertIsNotNone(loss)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_loss_empty_rejected(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig(beta=0.1)
        config = _MockConfig(kto_config=kto_config)
        criterion = KTOCriterion(config)

        policy_chosen_logps = paddle.randn([4], dtype="float32")
        policy_rejected_logps = paddle.zeros([0], dtype="float32")
        policy_kl_logps = paddle.randn([4], dtype="float32")
        reference_chosen_logps = paddle.randn([4], dtype="float32")
        reference_rejected_logps = paddle.zeros([0], dtype="float32")
        reference_kl_logps = paddle.randn([4], dtype="float32")

        loss, kl = criterion.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        self.assertIsNotNone(loss)


class TestKTOCriterionKtoLogps(unittest.TestCase):
    """Tests for KTOCriterion.kto_logps."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_logps_basic(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        batch_size, seq_len = 2, 10
        logits = paddle.randn([batch_size, seq_len, 100], dtype="float32")
        # In the default (non-fused, non-filtered) path, labels = response_labels + response_kl_labels
        # uses element-wise addition for paddle tensors. Shape must match logits.shape[:-1].
        response_labels = paddle.randint(0, 100, [batch_size, seq_len], dtype="int64")
        response_kl_labels = paddle.zeros([batch_size, seq_len], dtype="int64")
        # response_indexs: [batch_idx, start, chosen_end, kl_end, is_chosen(1) or rejected(0)]
        response_indexs = paddle.to_tensor([[0, 0, 5, 8, 1], [1, 0, 5, 8, 1]], dtype="int64")

        chosen_logps, rejected_logps, kl_logps = criterion.kto_logps(
            logits, response_labels, response_kl_labels, response_indexs
        )
        self.assertIsNotNone(chosen_logps)
        self.assertIsNotNone(rejected_logps)
        self.assertIsNotNone(kl_logps)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_logps_mixed_chosen_rejected(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        batch_size, seq_len = 2, 10
        logits = paddle.randn([batch_size, seq_len, 100], dtype="float32")
        response_labels = paddle.randint(0, 100, [batch_size, seq_len], dtype="int64")
        response_kl_labels = paddle.zeros([batch_size, seq_len], dtype="int64")
        # One chosen, one rejected
        response_indexs = paddle.to_tensor([[0, 0, 5, 8, 1], [1, 0, 5, 8, 0]], dtype="int64")

        chosen_logps, rejected_logps, kl_logps = criterion.kto_logps(
            logits, response_labels, response_kl_labels, response_indexs
        )
        # Should have one chosen and one rejected entry
        self.assertGreater(chosen_logps.shape[0], 0)
        self.assertGreater(rejected_logps.shape[0], 0)


class TestKTOCriterionForward(unittest.TestCase):
    """Tests for KTOCriterion.forward."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_forward_reference_phase(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        batch_size, seq_len = 2, 10
        logits = paddle.randn([batch_size, seq_len, 100], dtype="float32")
        response_labels = paddle.randint(0, 100, [batch_size, seq_len], dtype="int64")
        response_kl_labels = paddle.zeros([batch_size, seq_len], dtype="int64")
        response_indexs = paddle.to_tensor([[0, 0, 5, 8, 1], [1, 0, 5, 8, 0]], dtype="int64")

        labels = (response_labels, response_kl_labels, response_indexs, None, None, None)
        result = criterion.forward(logits, labels)
        # Should return reference logps when reference is None
        self.assertEqual(len(result), 3)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_forward_policy_phase(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        batch_size, seq_len = 2, 10
        logits = paddle.randn([batch_size, seq_len, 100], dtype="float32")
        response_labels = paddle.randint(0, 100, [batch_size, seq_len], dtype="int64")
        response_kl_labels = paddle.zeros([batch_size, seq_len], dtype="int64")
        response_indexs = paddle.to_tensor([[0, 0, 5, 8, 1], [1, 0, 5, 8, 0]], dtype="int64")

        ref_chosen = paddle.randn([1], dtype="float32")
        ref_rejected = paddle.randn([1], dtype="float32")
        ref_kl = paddle.randn([2], dtype="float32")

        labels = (response_labels, response_kl_labels, response_indexs, ref_chosen, ref_rejected, ref_kl)
        result = criterion.forward(logits, labels)
        # Should return policy logps, loss, and kl when reference is provided
        self.assertEqual(len(result), 5)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_forward_logits_tuple(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        batch_size, seq_len = 2, 10
        logits = (paddle.randn([batch_size, seq_len, 100], dtype="float32"),)
        response_labels = paddle.randint(0, 100, [batch_size, seq_len], dtype="int64")
        response_kl_labels = paddle.zeros([batch_size, seq_len], dtype="int64")
        response_indexs = paddle.to_tensor([[0, 0, 5, 8, 1], [1, 0, 5, 8, 0]], dtype="int64")

        labels = (response_labels, response_kl_labels, response_indexs, None, None, None)
        result = criterion.forward(logits, labels)
        self.assertEqual(len(result), 3)


class TestKTOCriterionLogpsShapeMismatch(unittest.TestCase):
    """Tests for shape mismatch errors in kto_logps."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_kto_logps_shape_mismatch_raises(self, mock_world_size):
        from paddleformers.transformers.kto_criterion import KTOCriterion

        kto_config = _MockKTOConfig()
        config = _MockConfig(kto_config=kto_config, vocab_size=100)
        criterion = KTOCriterion(config)

        logits = paddle.randn([2, 10, 100], dtype="float32")
        # response_labels shape [2, 8] + response_kl_labels shape [2, 8] = [2, 8]
        # but logits.shape[:-1] = [2, 10], so shape mismatch triggers ValueError
        response_labels = paddle.randint(0, 100, [2, 8], dtype="int64")
        response_kl_labels = paddle.randint(0, 100, [2, 8], dtype="int64")
        response_indexs = paddle.to_tensor([[0, 0, 3, 5, 1]], dtype="int64")

        with self.assertRaises(ValueError):
            criterion.kto_logps(logits, response_labels, response_kl_labels, response_indexs)


if __name__ == "__main__":
    unittest.main()
