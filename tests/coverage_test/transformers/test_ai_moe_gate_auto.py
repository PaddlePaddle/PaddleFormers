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
import paddle.nn.functional as F

from paddleformers.transformers.moe_gate_auto import MoEGateMixin, PretrainedMoEGate


class _MockConfig:
    """Mock config for PretrainedMoEGate auto."""

    def __init__(self, **kwargs):
        self.scoring_func = kwargs.get("scoring_func", None)
        self.seq_length = kwargs.get("seq_length", 128)
        self.moe_subbatch_token_num_before_dispatch = kwargs.get("moe_subbatch_token_num_before_dispatch", 0)
        self.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size", 1)
        self.sequence_parallel = kwargs.get("sequence_parallel", False)
        self.seq_aux = kwargs.get("seq_aux", False)


class TestMoEGateMixinAutoGateScoreFunc(unittest.TestCase):
    """Tests for MoEGateMixin gate_score_func (auto version)."""

    def _make_gate(self, scoring_func=None):
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
        self.assertTrue(paddle.all(scores >= 0))
        self.assertTrue(paddle.all(scores <= 1))

    def test_tanh(self):
        gate = self._make_gate("tanh")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.all(scores >= -1))
        self.assertTrue(paddle.all(scores <= 1))

    def test_relu(self):
        gate = self._make_gate("relu")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.all(scores >= 0))

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

    def test_unknown_scoring_func_defaults_to_softmax(self):
        gate = self._make_gate("unknown")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4]), atol=1e-5))


class TestMoEGateMixinAutoHelpers(unittest.TestCase):
    """Tests for MoEGateMixin helper methods (auto version)."""

    def _make_gate(self, num_experts=8, **kwargs):
        config = _MockConfig(**kwargs)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=num_experts,
            expert_hidden_size=32,
            **kwargs,
        )
        return gate

    def test_one_hot_to_float(self):
        gate = self._make_gate()
        x = paddle.to_tensor([0, 1, 2], dtype="int64")
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        # Result dtype should be float
        self.assertIn(str(result.dtype), ["float32", "paddle.float32"])

    def test_one_hot_to_int64(self):
        gate = self._make_gate()
        x = paddle.to_tensor([0, 1, 2], dtype="int64")
        result = gate._one_hot_to_int64(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        self.assertEqual(result.dtype, paddle.int64)

    def test_capacity_calculation(self):
        gate = self._make_gate()
        gates = paddle.randn([32, 8])
        capacity = gate._capacity(gates, capacity_factor=1.0)
        self.assertEqual(capacity, 4)

    def test_cal_z_loss(self):
        gate = self._make_gate()
        logits = paddle.randn([4, 8])
        z_loss = gate._cal_z_loss(logits)
        self.assertEqual(z_loss.shape, [])
        self.assertTrue(float(z_loss) >= 0)

    def test_cal_aux_loss(self):
        gate = self._make_gate(global_aux_loss=False)
        gates = F.softmax(paddle.randn([4, 8]), axis=-1)
        mask = paddle.zeros([4, 8])
        mask[0, 0] = 1
        mask[1, 1] = 1
        aux_loss = gate._cal_aux_loss(gates, mask)
        self.assertEqual(aux_loss.shape, [])

    def test_cal_orthogonal_loss(self):
        gate = self._make_gate(num_experts=4)
        # Create a weight parameter for the gate
        gate.weight = paddle.create_parameter(
            shape=[32, 4], dtype="float32", default_initializer=paddle.nn.initializer.Uniform()
        )
        loss = gate._cal_orthogonal_loss()
        self.assertEqual(loss.shape, [])
        self.assertTrue(float(loss) >= 0)


class TestPretrainedMoEGateAutoInit(unittest.TestCase):
    """Tests for PretrainedMoEGate initialization (auto version)."""

    def test_default_init(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
        )
        self.assertEqual(gate.num_experts, 8)
        self.assertEqual(gate.expert_hidden_size, 32)
        self.assertFalse(gate.drop_tokens)
        self.assertEqual(gate.top_k, 2)
        self.assertEqual(gate.topk_method, "greedy")
        self.assertFalse(gate.norm_topk_prob)
        self.assertAlmostEqual(gate.routed_scaling_factor, 1.0)

    def test_drop_tokens_with_capacity_factor(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            moe_expert_capacity_factor=1.5,
        )
        self.assertTrue(gate.drop_tokens)


class TestPretrainedMoEGateAutoTopkGreedy(unittest.TestCase):
    """Tests for _topk_greedy method (auto version)."""

    def test_topk_greedy_basic(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
        )
        scores = paddle.randn([4, 8])
        topk_weight, topk_idx = gate._topk_greedy(scores, k=2)
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])


class TestPretrainedMoEGateAutoTopkGroupLimitedGreedy(unittest.TestCase):
    """Tests for _topk_group_limited_greedy method (auto version)."""

    def test_group_limited_greedy(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            n_group=4,
            topk_group=2,
        )
        scores = paddle.randn([4, 8])
        topk_weight, topk_idx = gate._topk_group_limited_greedy(scores, k=2, n_group=4, topk_group=2)
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])


class TestPretrainedMoEGateAutoTop1Gating(unittest.TestCase):
    """Tests for top1gating method (auto version)."""

    def test_top1gating_basic(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            use_rts=False,
        )
        logits = paddle.randn([16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.top1gating(logits)
        self.assertIsInstance(capacity, int)
        self.assertTrue(capacity > 0)
        self.assertEqual(combine_weights.shape, [16, 8, capacity])


class TestPretrainedMoEGateAutoTop2Gating(unittest.TestCase):
    """Tests for top2gating method (auto version)."""

    def test_top2gating_basic(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top2_2nd_expert_sampling=False,
        )
        logits = paddle.randn([16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.top2gating(logits)
        self.assertIsInstance(capacity, int)
        self.assertTrue(capacity > 0)
        self.assertEqual(combine_weights.shape, [16, 8, capacity])


class TestPretrainedMoEGateAutoTopkGating(unittest.TestCase):
    """Tests for topkgating method (auto version)."""

    def test_topkgating_basic(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="greedy",
        )
        gates = paddle.randn([2, 16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.topkgating(gates)
        self.assertIsInstance(capacity, int)

    def test_topkgating_with_norm_topk_prob(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="greedy",
            norm_topk_prob=True,
        )
        gates = paddle.randn([2, 16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.topkgating(gates)
        self.assertIsInstance(capacity, int)

    def test_topkgating_group_limited_greedy(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="group_limited_greedy",
            n_group=4,
            topk_group=2,
        )
        gates = paddle.randn([2, 16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.topkgating(gates)
        self.assertIsInstance(capacity, int)


class TestPretrainedMoEGateAutoTopkgatingPart1Part2(unittest.TestCase):
    """Tests for topkgating_part1 and topkgating_part2 methods."""

    def test_topkgating_part1_basic(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="greedy",
        )
        scores = paddle.randn([2, 16, 8])
        exp_counts, l_aux, l_zloss = gate.topkgating_part1(scores, None)
        self.assertEqual(exp_counts.shape, [8])
        self.assertEqual(l_aux.shape, [])
        self.assertEqual(l_zloss.shape, [])

    def test_topkgating_part1_sets_internal_state(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="greedy",
        )
        scores = paddle.randn([2, 16, 8])
        gate.topkgating_part1(scores, None)
        # Check that internal state is set
        self.assertIsNotNone(gate.mask)
        # When drop_tokens=False, capacity and token_priority are None
        self.assertIsNone(gate.capacity)
        self.assertIsNone(gate.token_priority)


if __name__ == "__main__":
    unittest.main()
