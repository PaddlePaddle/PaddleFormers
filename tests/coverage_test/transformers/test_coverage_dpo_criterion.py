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

import paddle
import paddle.nn as nn

from paddleformers.transformers.dpo_criterion import AutoDPOCriterion, DPOCriterion


class _MockDPOConfig:
    """Mock DPO config object."""

    def __init__(self, **kwargs):
        self.loss_type = kwargs.get("loss_type", "sigmoid")
        self.beta = kwargs.get("beta", 0.1)
        self.label_smoothing = kwargs.get("label_smoothing", 0.0)
        self.pref_loss_ratio = kwargs.get("pref_loss_ratio", 1.0)
        self.sft_loss_ratio = kwargs.get("sft_loss_ratio", 1.0)
        self.simpo_gamma = kwargs.get("simpo_gamma", 0.5)
        self.dpop_lambda = kwargs.get("dpop_lambda", 1.0)
        self.normalize_logps = kwargs.get("normalize_logps", False)


class _MockModelConfig:
    """Mock model config object."""

    def __init__(self, **kwargs):
        self.tensor_parallel_output = kwargs.get("tensor_parallel_output", False)
        self.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
        self.dpo_config = kwargs.get("dpo_config", None)
        self.use_fused_head_and_loss_fn = kwargs.get("use_fused_head_and_loss_fn", False)
        self.use_filtered_label_loss = kwargs.get("use_filtered_label_loss", False)
        self.chunk_size = kwargs.get("chunk_size", 1024)
        self.vocab_size = kwargs.get("vocab_size", 1024)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.max_sequence_length = kwargs.get("max_sequence_length", 512)


class TestDPOCriterionInit(unittest.TestCase):
    """Tests for DPOCriterion initialization."""

    def test_init_with_dpo_config(self):
        model_config = _MockModelConfig()
        dpo_config = _MockDPOConfig()
        criterion = DPOCriterion(model_config, dpo_config=dpo_config)
        self.assertEqual(criterion.dpo_config.loss_type, "sigmoid")
        self.assertEqual(criterion.dpo_config.beta, 0.1)
        self.assertFalse(criterion.use_infohub)
        self.assertFalse(criterion.ignore_eos_token)

    def test_init_with_config_dpo_config(self):
        dpo_config = _MockDPOConfig(loss_type="hinge")
        model_config = _MockModelConfig(dpo_config=dpo_config)
        criterion = DPOCriterion(model_config)
        self.assertEqual(criterion.dpo_config.loss_type, "hinge")

    def test_init_without_dpo_config_raises(self):
        model_config = _MockModelConfig(dpo_config=None)
        with self.assertRaises(ValueError):
            DPOCriterion(model_config)

    def test_init_with_use_infohub(self):
        model_config = _MockModelConfig()
        dpo_config = _MockDPOConfig()
        criterion = DPOCriterion(model_config, dpo_config=dpo_config, use_infohub=True)
        self.assertTrue(criterion.use_infohub)

    def test_init_with_ignore_eos_token(self):
        model_config = _MockModelConfig()
        dpo_config = _MockDPOConfig()
        criterion = DPOCriterion(model_config, dpo_config=dpo_config, ignore_eos_token=True)
        self.assertTrue(criterion.ignore_eos_token)


class TestDPOLoss(unittest.TestCase):
    """Tests for DPOCriterion.dpo_loss method."""

    def setUp(self):
        self.model_config = _MockModelConfig()
        self.dpo_config = _MockDPOConfig()
        self.criterion = DPOCriterion(self.model_config, dpo_config=self.dpo_config)

    def test_sigmoid_loss(self):
        policy_chosen = paddle.to_tensor([0.5])
        policy_rejected = paddle.to_tensor([0.3])
        ref_chosen = paddle.to_tensor([0.4])
        ref_rejected = paddle.to_tensor([0.2])
        loss = self.criterion.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        self.assertEqual(loss.shape, [])
        self.assertTrue(float(loss) >= 0)

    def test_hinge_loss(self):
        dpo_config = _MockDPOConfig(loss_type="hinge")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_ipo_loss(self):
        dpo_config = _MockDPOConfig(loss_type="ipo")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_simpo_loss(self):
        dpo_config = _MockDPOConfig(loss_type="simpo")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_dpop_loss(self):
        dpo_config = _MockDPOConfig(loss_type="dpop")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_kto_pair_loss(self):
        dpo_config = _MockDPOConfig(loss_type="kto_pair")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_sppo_hard_loss(self):
        dpo_config = _MockDPOConfig(loss_type="sppo_hard")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_nca_pair_loss(self):
        dpo_config = _MockDPOConfig(loss_type="nca_pair")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([0.5]),
            paddle.to_tensor([0.3]),
            paddle.to_tensor([0.4]),
            paddle.to_tensor([0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_or_loss(self):
        dpo_config = _MockDPOConfig(loss_type="or")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        loss = criterion.dpo_loss(
            paddle.to_tensor([-0.5]),
            paddle.to_tensor([-0.3]),
            paddle.to_tensor([-0.4]),
            paddle.to_tensor([-0.2]),
        )
        self.assertEqual(loss.shape, [])

    def test_unknown_loss_type_raises(self):
        dpo_config = _MockDPOConfig(loss_type="unknown_type")
        criterion = DPOCriterion(self.model_config, dpo_config=dpo_config)
        with self.assertRaises(ValueError):
            criterion.dpo_loss(
                paddle.to_tensor([0.5]),
                paddle.to_tensor([0.3]),
                paddle.to_tensor([0.4]),
                paddle.to_tensor([0.2]),
            )


class TestAutoDPOCriterionInit(unittest.TestCase):
    """Tests for AutoDPOCriterion initialization."""

    def test_init(self):
        model_config = _MockModelConfig()
        dpo_config = _MockDPOConfig()
        criterion = AutoDPOCriterion(model_config, dpo_config=dpo_config)
        # Should use nn.CrossEntropyLoss instead of ParallelCrossEntropy
        self.assertIsInstance(criterion.logprobs, nn.CrossEntropyLoss)


if __name__ == "__main__":
    unittest.main()
