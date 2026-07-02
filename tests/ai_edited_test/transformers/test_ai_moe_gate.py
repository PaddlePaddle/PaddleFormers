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

from paddleformers.transformers.moe_gate import MoEGateMixin, PretrainedMoEGate


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
        gate = self._make_gate("unknown_func")
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        # Should fall back to softmax
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4]), atol=1e-5))

    def test_none_scoring_func_defaults_to_softmax(self):
        gate = self._make_gate(None)
        logits = paddle.randn([4, 8], dtype="float32")
        scores = gate.gate_score_func(logits)
        self.assertTrue(paddle.allclose(scores.sum(axis=-1), paddle.ones([4]), atol=1e-5))


class TestMoEGateMixinHelpers(unittest.TestCase):
    """Tests for MoEGateMixin helper methods."""

    def _make_gate(self, num_experts=8, **kwargs):
        config = _MockConfig(**kwargs)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=num_experts,
            expert_hidden_size=32,
            **kwargs,
        )
        return gate

    def test_one_hot_to_float_int_input(self):
        gate = self._make_gate()
        x = paddle.to_tensor([0, 1, 2], dtype="int64")
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        # Result dtype should be float (the default dtype)
        self.assertIn(str(result.dtype), ["float32", "paddle.float32"])

    def test_one_hot_to_float_non_int_input(self):
        gate = self._make_gate()
        x = paddle.to_tensor([0.0, 1.0], dtype="float32")
        result = gate._one_hot_to_float(x, num_classes=4)
        self.assertEqual(result.shape, [2, 4])

    def test_one_hot_to_int64_int_input(self):
        gate = self._make_gate()
        x = paddle.to_tensor([0, 1, 2], dtype="int64")
        result = gate._one_hot_to_int64(x, num_classes=4)
        self.assertEqual(result.shape, [3, 4])
        self.assertEqual(result.dtype, paddle.int64)

    def test_capacity_calculation(self):
        gate = self._make_gate()
        gates = paddle.randn([32, 8])
        capacity = gate._capacity(gates, capacity_factor=1.0)
        # (32 // 8) * 1.0 = 4
        self.assertEqual(capacity, 4)

    def test_capacity_raises_on_zero(self):
        gate = self._make_gate()
        gates = paddle.randn([4, 8])
        # (4 // 8) * 1.0 = 0, which should raise
        with self.assertRaises(AssertionError):
            gate._capacity(gates, capacity_factor=1.0)

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


class TestPretrainedMoEGateInit(unittest.TestCase):
    """Tests for PretrainedMoEGate initialization."""

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
        self.assertEqual(gate.norm_topk_prob, False)
        self.assertAlmostEqual(gate.routed_scaling_factor, 1.0)

    def test_drop_tokens_true(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            moe_expert_capacity_factor=1.5,
        )
        self.assertTrue(gate.drop_tokens)

    def test_custom_topk(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=4,
        )
        self.assertEqual(gate.top_k, 4)


class TestPretrainedMoEGateTopkGreedy(unittest.TestCase):
    """Tests for _topk_greedy method."""

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

    def test_topk_greedy_sorted(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
        )
        scores = paddle.randn([4, 8])
        topk_weight, topk_idx = gate._topk_greedy(scores, k=2)
        # Weights should be in descending order
        for i in range(4):
            self.assertTrue(float(topk_weight[i, 0]) >= float(topk_weight[i, 1]))


class TestPretrainedMoEGateTopkGroupLimitedGreedy(unittest.TestCase):
    """Tests for _topk_group_limited_greedy method."""

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


class TestPretrainedMoEGateTop1Gating(unittest.TestCase):
    """Tests for top1gating method."""

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
        self.assertEqual(dispatch_mask.shape, [16, 8, capacity])
        self.assertEqual(exp_counts.shape, [8])

    def test_top1gating_with_used_token(self):
        config = _MockConfig()
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            use_rts=False,
        )
        logits = paddle.randn([16, 8])
        used_token = paddle.ones([16])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.top1gating(
            logits, used_token=used_token
        )
        self.assertIsInstance(capacity, int)


class TestPretrainedMoEGateTop2Gating(unittest.TestCase):
    """Tests for top2gating method."""

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
        self.assertEqual(dispatch_mask.shape, [16, 8, capacity])


class TestPretrainedMoEGateTopkGating(unittest.TestCase):
    """Tests for topkgating method."""

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
        self.assertEqual(combine_weights.shape[0], 2 * 16)

    def test_topkgating_with_drop_tokens(self):
        config = _MockConfig(seq_aux=False)
        gate = PretrainedMoEGate(
            config=config,
            num_experts=8,
            expert_hidden_size=32,
            top_k=2,
            topk_method="greedy",
            moe_expert_capacity_factor=1.5,
            moe_token_drop_policy="position",
        )
        gates = paddle.randn([2, 16, 8])
        capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss = gate.topkgating(gates)
        self.assertIsInstance(capacity, int)


if __name__ == "__main__":
    unittest.main()
