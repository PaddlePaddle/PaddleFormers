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
import paddle.nn.functional as F


class _MockConfig:
    """Mock config for PretrainedMoEGate."""

    def __init__(self, **kwargs):
        self.scoring_func = kwargs.get("scoring_func", None)
        self.seq_length = kwargs.get("seq_length", 128)
        self.moe_subbatch_token_num_before_dispatch = kwargs.get("moe_subbatch_token_num_before_dispatch", 0)
        self.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.seq_aux = kwargs.get("seq_aux", False)


class TestMoEGateMixinGateScoreFunc(unittest.TestCase):
    """Tests for MoEGateMixin.gate_score_func."""

    def _make_gate(self, scoring_func=None):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("TestGate", (MoEGateMixin,), {})()
        gate.scoring_func = scoring_func
        return gate

    def test_softmax(self):
        gate = self._make_gate("softmax")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4])))

    def test_sigmoid(self):
        gate = self._make_gate("sigmoid")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

    def test_tanh(self):
        gate = self._make_gate("tanh")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue((scores >= -1).all() and (scores <= 1).all())

    def test_relu(self):
        gate = self._make_gate("relu")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue((scores >= 0).all())

    def test_gelu(self):
        gate = self._make_gate("gelu")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 8])

    def test_leaky_relu(self):
        gate = self._make_gate("leaky_relu")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 8])

    def test_none_defaults_to_softmax(self):
        gate = self._make_gate(None)
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4])))

    def test_unknown_defaults_to_softmax(self):
        gate = self._make_gate("unknown_func")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4])))


class TestMoEGateMixinSampleFunctions(unittest.TestCase):
    """Tests for MoEGateMixin sampling functions."""

    def test_gumbel_rsample(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        logits = paddle.randn([4, 8], dtype="float32")
        samples = gate.gumbel_rsample(logits)
        self.assertEqual(samples.shape, logits.shape)

    def test_uniform_sample(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        logits = paddle.randn([4, 8], dtype="float32")
        samples = gate.uniform_sample(logits)
        self.assertEqual(samples.shape, logits.shape)

    def test_one_hot_to_float(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        x = paddle.to_tensor([0, 1, 2, 0], dtype="int64")
        result = gate._one_hot_to_float(x, num_classes=3)
        self.assertEqual(result.shape, [4, 3])
        # Each row should have exactly one 1.0
        self.assertTrue(paddle.allclose(result.sum(axis=-1), paddle.ones([4])))

    def test_one_hot_to_int64(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        x = paddle.to_tensor([0, 1, 2], dtype="int64")
        result = gate._one_hot_to_int64(x, num_classes=3)
        self.assertEqual(result.dtype, paddle.int64)
        self.assertEqual(result.shape, [3, 3])


class TestMoEGateMixinCapacity(unittest.TestCase):
    """Tests for MoEGateMixin._capacity."""

    def test_capacity_basic(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gates = paddle.randn([100, 8], dtype="float32")
        capacity = gate._capacity(gates, capacity_factor=1.0)
        self.assertEqual(capacity, 12)  # 100 // 8 * 1.0 = 12

    def test_capacity_factor(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gates = paddle.randn([100, 10], dtype="float32")
        capacity = gate._capacity(gates, capacity_factor=2.0)
        self.assertEqual(capacity, 20)  # 100 // 10 * 2.0 = 20

    def test_capacity_assertion_2d(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gates = paddle.randn([100], dtype="float32")  # 1D
        with self.assertRaises(AssertionError):
            gate._capacity(gates, 1.0)

    def test_capacity_assertion_positive(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gates = paddle.randn([100, 8], dtype="float32")
        with self.assertRaises(AssertionError):
            gate._capacity(gates, 0.0)


class TestMoEGateMixinAuxLoss(unittest.TestCase):
    """Tests for MoEGateMixin._cal_aux_loss."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_aux_loss_local(self, mock_world_size):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gate.global_aux_loss = False
        gate.num_experts = 8

        gates = F.softmax(paddle.randn([10, 8], dtype="float32"), axis=-1)
        mask = F.one_hot(paddle.argmax(gates, axis=-1), num_classes=8).cast("float32")
        aux_loss = gate._cal_aux_loss(gates, mask)
        self.assertIsNotNone(aux_loss)
        self.assertTrue(paddle.is_tensor(aux_loss))


class TestMoEGateMixinZLoss(unittest.TestCase):
    """Tests for MoEGateMixin._cal_z_loss."""

    def test_z_loss_basic(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        logits = paddle.randn([10, 8], dtype="float32")
        z_loss = gate._cal_z_loss(logits)
        self.assertIsNotNone(z_loss)
        self.assertTrue(z_loss >= 0)


class TestMoEGateMixinOrthogonalLoss(unittest.TestCase):
    """Tests for MoEGateMixin._cal_orthogonal_loss."""

    def test_orthogonal_loss(self):
        from paddleformers.transformers.moe_gate import MoEGateMixin

        gate = type("G", (MoEGateMixin,), {})()
        gate.num_experts = 4
        gate.weight = paddle.nn.Parameter(paddle.randn([8, 4], dtype="float32"))
        result = gate._cal_orthogonal_loss()
        self.assertIsNotNone(result)
        self.assertTrue(result >= 0)


class TestPretrainedMoEGateInit(unittest.TestCase):
    """Tests for PretrainedMoEGate initialization."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_default_init(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64)
        self.assertEqual(gate.num_experts, 8)
        self.assertEqual(gate.expert_hidden_size, 64)
        self.assertEqual(gate.top_k, 2)
        self.assertEqual(gate.n_group, 1)
        self.assertEqual(gate.topk_group, 1)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_custom_init(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig(scoring_func="sigmoid")
        gate = PretrainedMoEGate(
            config,
            num_experts=16,
            expert_hidden_size=128,
            top_k=4,
            n_group=4,
            topk_group=2,
            moe_expert_capacity_factor=1.0,
            norm_topk_prob=True,
            routed_scaling_factor=2.0,
        )
        self.assertEqual(gate.num_experts, 16)
        self.assertEqual(gate.top_k, 4)
        self.assertEqual(gate.n_group, 4)
        self.assertEqual(gate.topk_group, 2)
        self.assertTrue(gate.drop_tokens)
        self.assertTrue(gate.norm_topk_prob)
        self.assertEqual(gate.routed_scaling_factor, 2.0)
        self.assertEqual(gate.scoring_func, "sigmoid")

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_init_no_drop_tokens(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64, moe_expert_capacity_factor=0.0)
        self.assertFalse(gate.drop_tokens)


class TestPretrainedMoEGateTopKGreedy(unittest.TestCase):
    """Tests for PretrainedMoEGate._topk_greedy."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_topk_greedy_basic(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64, top_k=2)
        gate.eval()
        scores = paddle.randn([10, 8], dtype="float32")
        topk_weight, topk_idx = gate._topk_greedy(scores, k=2)
        self.assertEqual(topk_weight.shape, [10, 2])
        self.assertEqual(topk_idx.shape, [10, 2])


class TestPretrainedMoEGateTopKGroupLimitedGreedy(unittest.TestCase):
    """Tests for PretrainedMoEGate._topk_group_limited_greedy."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_group_limited_greedy(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64)
        gate.eval()
        scores = paddle.randn([10, 8], dtype="float32")
        topk_weight, topk_idx = gate._topk_group_limited_greedy(scores, k=2, n_group=2, topk_group=1)
        self.assertEqual(topk_weight.shape, [10, 2])
        self.assertEqual(topk_idx.shape, [10, 2])

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_group_limited_greedy_invalid_n_group(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=7, expert_hidden_size=64)  # 7 not divisible by 2
        gate.eval()
        scores = paddle.randn([10, 7], dtype="float32")
        with self.assertRaises(AssertionError):
            gate._topk_group_limited_greedy(scores, k=2, n_group=2, topk_group=1)


class TestPretrainedMoEGatePriority(unittest.TestCase):
    """Tests for PretrainedMoEGate._priority."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_priority(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=4, expert_hidden_size=64, top_k=2)
        topk_idx = paddle.to_tensor([[0, 1], [0, 2], [1, 3]], dtype="int64")
        priority = gate._priority(topk_idx, capacity=2)
        # _priority returns shape [batch, num_experts] after summing over k dimension
        self.assertEqual(priority.shape, [3, 4])
        # Values should be 0 or 1
        self.assertTrue((priority >= 0).all() and (priority <= 1).all())


class TestPretrainedMoEGateTopKGating(unittest.TestCase):
    """Tests for PretrainedMoEGate.topkgating."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_topkgating_greedy(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(
            config,
            num_experts=8,
            expert_hidden_size=64,
            top_k=2,
            topk_method="greedy",
            moe_expert_capacity_factor=0.0,
        )
        gate.eval()
        # gates shape must be [N, num_experts] so topk indices match num_experts
        gates = paddle.randn([4, 8], dtype="float32")
        result = gate.topkgating(gates)
        self.assertEqual(len(result), 6)  # capacity, gates, mask, exp_counts, aux_loss, z_loss

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_topkgating_with_norm(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(
            config,
            num_experts=8,
            expert_hidden_size=64,
            top_k=2,
            topk_method="greedy",
            norm_topk_prob=True,
            moe_expert_capacity_factor=0.0,
        )
        gate.eval()
        gates = paddle.randn([4, 8], dtype="float32")
        result = gate.topkgating(gates)
        self.assertEqual(len(result), 6)


class TestPretrainedMoEGateTopKGatingNoDrop(unittest.TestCase):
    """Tests for PretrainedMoEGate.topkgating_nodrop."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_topkgating_nodrop(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64, top_k=2)
        gate.eval()
        gates = paddle.randn([4, 16], dtype="float32")
        result = gate.topkgating_nodrop(gates)
        self.assertEqual(len(result), 5)  # gates_masked, mask, exp_counts, aux_loss, z_loss

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_topkgating_nodrop_sigmoid(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig(scoring_func="sigmoid")
        gate = PretrainedMoEGate(config, num_experts=8, expert_hidden_size=64, top_k=2)
        gate.eval()
        gates = paddle.randn([4, 16], dtype="float32")
        result = gate.topkgating_nodrop(gates)
        self.assertEqual(len(result), 5)


class TestPretrainedMoEGateTop1Gating(unittest.TestCase):
    """Tests for PretrainedMoEGate.top1gating."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_top1gating(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(
            config,
            num_experts=4,
            expert_hidden_size=64,
            moe_expert_capacity_factor=0.0,
        )
        gate.eval()
        logits = paddle.randn([10, 4], dtype="float32")
        result = gate.top1gating(logits)
        self.assertEqual(len(result), 6)

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_top1gating_with_used_token(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(
            config,
            num_experts=4,
            expert_hidden_size=64,
            moe_expert_capacity_factor=0.0,
        )
        gate.eval()
        logits = paddle.randn([10, 4], dtype="float32")
        used_token = paddle.ones([10], dtype="float32")
        result = gate.top1gating(logits, used_token=used_token)
        self.assertEqual(len(result), 6)


class TestPretrainedMoEGateTop2Gating(unittest.TestCase):
    """Tests for PretrainedMoEGate.top2gating."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    def test_top2gating(self, mock_world_size):
        from paddleformers.transformers.moe_gate import PretrainedMoEGate

        config = _MockConfig()
        gate = PretrainedMoEGate(
            config,
            num_experts=4,
            expert_hidden_size=64,
            moe_expert_capacity_factor=0.0,
        )
        gate.eval()
        logits = paddle.randn([10, 4], dtype="float32")
        result = gate.top2gating(logits)
        self.assertEqual(len(result), 6)


if __name__ == "__main__":
    unittest.main()
