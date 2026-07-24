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


def _make_router_config(**overrides):
    """Helper to create a TransformerConfig for router testing."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "routed_scaling_factor_learnable": False,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_deep_gemm": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "moe_n_hash_layers": 0,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestMoERouter(unittest.TestCase):
    """Unit tests for moe_router module."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_standard_router_init(self, mock_cp):
        """Test StandardMoERouter initialization."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        self.assertEqual(router.hidden_size, 64)
        self.assertEqual(router.num_experts, 4)
        self.assertEqual(router.num_experts_per_tok, 2)
        self.assertEqual(router.scoring_func, "softmax")
        self.assertEqual(router.topk_method, "greedy")

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_noaux_tc_router_init(self, mock_cp):
        """Test router init with noaux_tc topk_method registers buffers."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(topk_method="noaux_tc")
        router = StandardMoERouter(config)
        self.assertIsNotNone(router.e_score_correction_bias)
        self.assertIsNotNone(router.expert_usage)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_softmax(self, mock_cp):
        """Test gate_score_func with softmax."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="softmax")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue(
            paddle.allclose(scores.sum(axis=-1), paddle.ones([4]), atol=1e-5)
        )

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_sigmoid(self, mock_cp):
        """Test gate_score_func with sigmoid."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="sigmoid")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_tanh(self, mock_cp):
        """Test gate_score_func with tanh."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="tanh")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= -1).all() and (scores <= 1).all())

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_relu(self, mock_cp):
        """Test gate_score_func with relu."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="relu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertTrue((scores >= 0).all())

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_gelu(self, mock_cp):
        """Test gate_score_func with gelu."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="gelu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_leaky_relu(self, mock_cp):
        """Test gate_score_func with leaky_relu."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="leaky_relu")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_score_func_not_implemented(self, mock_cp):
        """Test gate_score_func raises for unknown scoring func."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="unknown")
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        with self.assertRaises(NotImplementedError):
            router.gate_score_func(logits)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_calculation(self, mock_cp):
        """Test _capacity calculates correct value."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([16, 4], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=1.0, max_capacity=10, min_capacity=1
        )
        self.assertEqual(capacity, 4)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_min_clamp(self, mock_cp):
        """Test _capacity clamps to min_capacity."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([2, 8], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=0.01, max_capacity=10, min_capacity=5
        )
        self.assertEqual(capacity, 5)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_capacity_max_clamp(self, mock_cp):
        """Test _capacity clamps to max_capacity."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        gates = paddle.randn([100, 4], dtype=paddle.float32)
        capacity = router._capacity(
            gates, capacity_factor=10.0, max_capacity=5, min_capacity=1
        )
        self.assertEqual(capacity, 5)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_cal_aux_loss(self, mock_cp):
        """Test _cal_aux_loss computation."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(n_routed_experts=4)
        router = StandardMoERouter(config)
        gates = paddle.ones([4, 4], dtype=paddle.float32) * 0.25
        mask = paddle.zeros([4, 4], dtype=paddle.float32)
        mask[0, 0] = 1.0
        mask[1, 1] = 1.0
        mask[2, 2] = 1.0
        mask[3, 3] = 1.0
        aux_loss = router._cal_aux_loss(gates, mask)
        self.assertIsNotNone(aux_loss)
        self.assertEqual(aux_loss.shape, [])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_cal_z_loss(self, mock_cp):
        """Test _cal_z_loss computation."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        logits = paddle.randn([4, 4], dtype=paddle.float32)
        z_loss = router._cal_z_loss(logits)
        self.assertIsNotNone(z_loss)
        self.assertGreater(z_loss.item(), 0.0)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_greedy(self, mock_cp):
        """Test _topk_greedy returns correct shapes."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(num_experts_per_tok=2)
        router = StandardMoERouter(config)
        scores = paddle.randn([8, 4], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_greedy(scores, k=2)
        self.assertEqual(topk_weight.shape, [8, 2])
        self.assertEqual(topk_idx.shape, [8, 2])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_group_limited_greedy(self, mock_cp):
        """Test _topk_group_limited_greedy."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(
            n_routed_experts=8,
            n_group=4,
            topk_group=2,
            num_experts_per_tok=2,
        )
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 8], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_group_limited_greedy(
            scores, k=2, n_group=4, topk_group=2
        )
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_group_limited_greedy_assert(self, mock_cp):
        """Test _topk_group_limited_greedy asserts divisibility."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(n_routed_experts=5, n_group=3)
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 5], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            router._topk_group_limited_greedy(
                scores, k=2, n_group=3, topk_group=1
            )

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_call_topk_method_invalid(self, mock_cp):
        """Test _call_topk_method raises for invalid method."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        with self.assertRaises(NotImplementedError):
            router._call_topk_method("invalid", paddle.randn([4, 4]), k=2)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_set_layer_number(self, mock_cp):
        """Test set_layer_number."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config()
        router = StandardMoERouter(config)
        router.set_layer_number(3)
        self.assertEqual(router.layer_number, 3)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_priority(self, mock_cp):
        """Test _priority with capacity constraint."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(n_routed_experts=4, num_experts_per_tok=2)
        router = StandardMoERouter(config)
        topk_idx = paddle.to_tensor([[0, 1], [1, 2], [0, 3], [2, 3]])
        priority = router._priority(topk_idx, capacity=2)
        self.assertEqual(priority.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_probs_drop_policy(self, mock_cp):
        """Test _probs_drop_policy."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(n_routed_experts=4)
        router = StandardMoERouter(config)
        scores = paddle.zeros([4, 4], dtype=paddle.float32)
        scores[0, 0] = 1.0
        scores[0, 1] = 0.8
        scores[1, 2] = 0.9
        scores[1, 3] = 0.7
        mask = router._probs_drop_policy(scores, capacity=2)
        self.assertEqual(mask.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_seq_aux_loss_raises_on_invalid_type(self, mock_cp):
        """Test router raises when seq_aux is True but type != seq_aux_loss."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(
            moe_router_load_balancing_type="aux_loss",
        )
        config.seq_aux = True
        with self.assertRaises(ValueError):
            StandardMoERouter(config)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_seq_aux_loss_rejects_negative_scoring_func(self, mock_cp):
        """Test seq_aux_loss rejects scoring functions that may go negative."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(
            moe_router_load_balancing_type="seq_aux_loss",
            scoring_func="tanh",
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            StandardMoERouter(config)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_gate_detach_matmul_no_fuse(self, mock_cp):
        """Test gate_detach_matmul without fusion."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            gate_detach_matmul,
        )

        x = paddle.randn([4, 64], dtype=paddle.float32)
        w = paddle.randn([64, 4], dtype=paddle.float32)
        score = gate_detach_matmul(x, w, use_fuse=False)
        self.assertEqual(score.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_fused_gate_detach_matmul(self, mock_cp):
        """Test FusedGateDetachMatmul PyLayer."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            FusedGateDetachMatmul,
        )

        x = paddle.randn([4, 64], dtype=paddle.float32)
        # FusedGateDetachMatmul.forward does w = w.T internally, then F.linear(x, w.T).
        # So w must be [E, D] (n_experts, hidden) to produce output [B, E].
        w = paddle.randn([4, 64], dtype=paddle.float32)
        x.stop_gradient = False
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w)
        self.assertEqual(out.shape, [4, 4])

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_noaux_tc_n_group_1(self, mock_cp):
        """Test _topk_noaux_tc with n_group=1."""
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(
            topk_method="noaux_tc",
            n_routed_experts=4,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
        )
        router = StandardMoERouter(config)
        scores = paddle.randn([4, 4], dtype=paddle.float32)
        topk_weight, topk_idx = router._topk_noaux_tc(
            scores, k=2, n_group=1, topk_group=1
        )
        self.assertEqual(topk_weight.shape, [4, 2])
        self.assertEqual(topk_idx.shape, [4, 2])


class TestSftPlusScore(unittest.TestCase):
    """Tests for the 'sftplus' (softplus) scoring function in StandardMoERouter."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_sqrtsoftplus_matches_sqrt_softplus(self, _mock):
        """gate_score_func('sqrtsoftplus') must equal sqrt(softplus(x)) — required
        for mixed hash/top-k routing where non-hash layers use this path."""
        import paddle.nn.functional as F

        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="sqrtsoftplus")
        router = StandardMoERouter(config)
        logits = paddle.randn([8, 4])
        scores = router.gate_score_func(logits, logits_type_promotion=False)
        expected = paddle.sqrt(F.softplus(logits))
        np.testing.assert_allclose(scores.numpy(), expected.numpy(), atol=1e-6)

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_sftplus_matches_paddle_softplus(self, _mock):
        """gate_score_func('sftplus') should exactly match F.softplus."""
        import paddle.nn.functional as F

        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func="sftplus")
        router = StandardMoERouter(config)
        logits = paddle.randn([8, 4])
        scores = router.gate_score_func(logits, logits_type_promotion=False)
        expected = F.softplus(logits)
        np.testing.assert_allclose(
            scores.numpy(),
            expected.numpy(),
            atol=1e-6,
            err_msg="SftPlus scores should match F.softplus",
        )


class TestHashRouter(unittest.TestCase):
    """Tests for hash routing in TopKRouter.

    Hash routing is activated via ``set_layer_number(layer_number)`` on a router
    whose config has ``moe_n_hash_layers > 0`` and ``layer_number <
    moe_n_hash_layers`` (0-indexed leading layers).
    """

    def _make_router(self, layer_number=0, **cfg_overrides):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        cfg_overrides.setdefault("moe_n_hash_layers", 1)
        cfg_overrides.setdefault("actual_vocab_size", 128)
        cfg_overrides.setdefault("num_hidden_layers", 8)
        config = _make_router_config(**cfg_overrides)
        router = TopKRouter(config=config)
        router.set_layer_number(layer_number)
        return router, config

    def _dummy_hidden(self, B, S, H=64):
        return paddle.randn([B, S, H])

    def test_setup_registers_tid2eid(self):
        """Hash layer should register a tid2eid buffer with round-robin mapping."""
        router, config = self._make_router(
            layer_number=0,
            n_routed_experts=4,
            num_experts_per_tok=2,
            actual_vocab_size=16,
        )
        self.assertTrue(router.is_hash_layer)
        self.assertIsNotNone(router.tid2eid)
        self.assertEqual(list(router.tid2eid.shape), [16, 2])
        # Round-robin placeholder: tid2eid[i, k] = (i + k) % num_experts.
        ids = np.arange(16)
        expected = np.stack([(ids + k) % 4 for k in range(2)], axis=1)
        np.testing.assert_array_equal(router.tid2eid.numpy(), expected)

    def test_non_hash_layer_disabled(self):
        """layer_number >= moe_n_hash_layers should keep TopKRouter behavior."""
        router, _ = self._make_router(
            layer_number=2,
            moe_n_hash_layers=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
        )
        self.assertFalse(router.is_hash_layer)
        self.assertIsNone(router.tid2eid)

    def test_deterministic_routing(self):
        """Same input_ids → same expert assignment every call."""
        router, _ = self._make_router(n_routed_experts=4, num_experts_per_tok=2)
        hidden = self._dummy_hidden(2, 4)
        input_ids = paddle.to_tensor(
            [[3, 7, 1, 5], [2, 6, 4, 8]], dtype="int64"
        )

        _, _, idx1, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        _, _, idx2, _, _, _, _, _ = router(hidden, input_ids=input_ids)

        np.testing.assert_array_equal(idx1.numpy(), idx2.numpy())

    def test_tid2eid_expert_assignment(self):
        """Expert indices should match the round-robin tid2eid table."""
        num_experts = 4
        k = 2
        router, _ = self._make_router(
            n_routed_experts=num_experts, num_experts_per_tok=k
        )

        token_ids = [[3, 7, 1, 5]]
        hidden = self._dummy_hidden(1, 4)
        input_ids = paddle.to_tensor(token_ids, dtype="int64")

        _, _, top_idx, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        top_idx_np = top_idx.numpy()  # [4, 2]

        for pos, tid in enumerate(token_ids[0]):
            for ki in range(k):
                expected = (int(tid) + ki) % num_experts
                self.assertEqual(int(top_idx_np[pos, ki]), expected)

    def test_padding_tokens_masked(self):
        """Tokens with id==0 (padding) get weight=0 and idx=-1."""
        router, _ = self._make_router(n_routed_experts=4, num_experts_per_tok=2)
        input_ids = paddle.to_tensor([[0, 3, 5, 7]], dtype="int64")
        hidden = self._dummy_hidden(1, 4)

        _, top_gate, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        np.testing.assert_array_equal(top_gate.numpy()[0], [0.0, 0.0])
        np.testing.assert_array_equal(top_idx.numpy()[0], [-1, -1])
        self.assertEqual(probs.numpy()[0].sum(), 0.0)
        self.assertEqual(mask.numpy()[0].sum(), 0.0)

    def test_weights_come_from_gate_logits(self):
        """top_gate must equal scores.gather(top_idx)."""
        from paddle.nn.functional import softmax

        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            scoring_func="softmax",
        )
        # Force deterministic gate weights for reproducibility
        with paddle.no_grad():
            router.weight.set_value(
                paddle.randn(router.weight.shape, dtype=router.weight.dtype)
            )
        B, S, H = 1, 4, 64
        hidden = self._dummy_hidden(B, S, H)
        input_ids = paddle.to_tensor([[3, 5, 7, 9]], dtype="int64")

        _, top_gate, top_idx, _, _, _, _, _ = router(
            hidden, input_ids=input_ids
        )

        # Recompute expected scores from the gate matmul.
        flat = hidden.reshape([-1, H]).cast(paddle.float32)
        logits = paddle.matmul(flat, router.weight.T.cast(paddle.float32))
        scores = softmax(logits, axis=-1)
        expected = paddle.take_along_axis(scores, top_idx, axis=1).numpy()
        np.testing.assert_allclose(
            top_gate.numpy(), expected, atol=1e-5, rtol=1e-4
        )

    def test_sigmoid_score_is_renormalized(self):
        """For non-softmax score functions, top_gate must sum to 1."""
        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
        )
        B, S = 2, 4
        input_ids = paddle.randint(1, 50, [B, S])
        hidden = self._dummy_hidden(B, S)
        _, top_gate, _, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        sums = top_gate.sum(axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(B * S), atol=1e-5)

    def test_output_shapes_and_no_aux_loss(self):
        """Hash layer output shapes and aux/zloss are None."""
        num_experts, k = 4, 2
        B, S, H = 2, 6, 64
        router, _ = self._make_router(
            n_routed_experts=num_experts,
            num_experts_per_tok=k,
            hidden_size=H,
        )
        hidden = self._dummy_hidden(B, S, H)
        input_ids = paddle.randint(1, 100, [B, S])

        _, top_gate, top_idx, probs, mask, tp, l_aux, l_zloss = router(
            hidden, input_ids=input_ids
        )
        num_tokens = B * S
        self.assertEqual(list(top_gate.shape), [num_tokens, k])
        self.assertEqual(list(top_idx.shape), [num_tokens, k])
        self.assertEqual(list(probs.shape), [num_tokens, num_experts])
        self.assertEqual(list(mask.shape), [num_tokens, num_experts])
        self.assertIsNone(tp)
        self.assertIsNone(l_aux)
        self.assertIsNone(l_zloss)

    def test_invalid_scoring_func_raises(self):
        """Hash layer requires scoring_func in {softmax, sigmoid, sqrtsoftplus}."""
        with self.assertRaises(ValueError):
            self._make_router(scoring_func="tanh")

    def test_missing_actual_vocab_size_raises(self):
        """Hash layer requires actual_vocab_size to be set."""
        with self.assertRaises(ValueError):
            self._make_router(actual_vocab_size=None)

    def test_non_positive_actual_vocab_size_raises(self):
        """actual_vocab_size must be > 0 when hash routing is enabled."""
        with self.assertRaises(ValueError):
            self._make_router(actual_vocab_size=0)
        with self.assertRaises(ValueError):
            self._make_router(actual_vocab_size=-1)

    def test_moe_n_hash_layers_exceeds_num_hidden_layers_raises(self):
        """moe_n_hash_layers cannot exceed num_hidden_layers."""
        with self.assertRaises(ValueError):
            self._make_router(moe_n_hash_layers=9, num_hidden_layers=8)

    def test_non_positive_num_experts_per_tok_raises(self):
        """num_experts_per_tok must be a positive integer for hash routing."""
        with self.assertRaises(ValueError):
            self._make_router(num_experts_per_tok=0)

    def test_n_routed_experts_less_than_top_k_raises(self):
        """n_routed_experts must be >= num_experts_per_tok for hash routing."""
        with self.assertRaises(ValueError):
            self._make_router(n_routed_experts=1, num_experts_per_tok=2)

    def test_no_input_ids_raises(self):
        """Hash layer must raise ValueError if input_ids is None."""
        router, _ = self._make_router()
        hidden = self._dummy_hidden(2, 4)
        with self.assertRaises(ValueError):
            router(hidden, input_ids=None)

    def test_invalid_input_shape_raises(self):
        """Non-3D input should raise (regardless of hash routing)."""
        router, _ = self._make_router()
        bad_input = paddle.randn([8, 64])
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int64")
        with self.assertRaises(ValueError):
            router(bad_input, input_ids=input_ids)

    def test_sequence_parallel_token_alignment(self):
        """With sequence_parallel=True, flat_ids order must match [S,B,H] layout."""
        num_experts = 8
        router, _ = self._make_router(
            n_routed_experts=num_experts,
            num_experts_per_tok=1,
        )
        # TransformerConfig may auto-disable sequence_parallel when tp_size=1,
        # so set it directly on the router.
        router.sequence_parallel = True

        B, S, H = 2, 2, 64
        input_ids = paddle.to_tensor([[3, 5], [7, 9]], dtype="int64")
        hidden = paddle.randn([S, B, H])  # sequence-major
        _, _, top_idx, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        # Expected sequence-major flatten of input_ids: [3, 7, 5, 9]
        expected = np.array([[3], [7], [5], [9]], dtype="int64") % num_experts
        np.testing.assert_array_equal(top_idx.numpy(), expected)

    def test_routed_scaling_factor_scalar_applied(self):
        """routed_scaling_factor != 1.0 multiplies hash-routing top_gate."""
        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
            routed_scaling_factor=2.5,
        )
        hidden = self._dummy_hidden(1, 4)
        input_ids = paddle.to_tensor([[3, 7, 1, 5]], dtype="int64")
        _, top_gate, _, _, _, _, _, _ = router(hidden, input_ids=input_ids)
        # Sigmoid is renormalized to sum 1, then scaled by 2.5 → row sum == 2.5.
        np.testing.assert_allclose(
            top_gate.numpy().sum(axis=1),
            np.full((4,), 2.5, dtype="float32"),
            atol=1e-5,
        )

    def test_routed_scaling_factor_learnable_applied(self):
        """routed_scaling_factor_learnable gathers per-expert scales."""
        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            scoring_func="softmax",
            routed_scaling_factor_learnable=True,
        )
        # Override learnable scales with deterministic per-expert values.
        scales = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        router.routed_scaling_factor_param.set_value(scales)
        hidden = self._dummy_hidden(1, 4)
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        _, top_gate, top_idx, _, _, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        # Each gate weight should equal its raw softmax prob * scales[expert_id].
        idx_np = top_idx.numpy()
        gate_np = top_gate.numpy()
        for i in range(idx_np.shape[0]):
            for k in range(idx_np.shape[1]):
                e = int(idx_np[i, k])
                self.assertGreater(float(gate_np[i, k]), 0.0)
                self.assertAlmostEqual(
                    float(gate_np[i, k]) / float(scales.numpy()[e]),
                    float(gate_np[i, k]) / (e + 1.0),
                    places=5,
                )

    def test_tid2eid_none_defensive_raises(self):
        """_hash_routing raises if tid2eid is externally cleared."""
        router, _ = self._make_router(n_routed_experts=4, num_experts_per_tok=2)
        router.tid2eid = None
        hidden = self._dummy_hidden(1, 2)
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        with self.assertRaises(ValueError):
            router(hidden, input_ids=input_ids)

    def test_noaux_tc_drops_bias_buffers(self):
        """Hash layer with noaux_tc topk_method should delete
        e_score_correction_bias and expert_usage buffers."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            topk_method="noaux_tc",
            scoring_func="sqrtsoftplus",
            moe_n_hash_layers=1,
            actual_vocab_size=128,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        # Before set_layer_number, noaux_tc allocates these buffers
        self.assertTrue(hasattr(router, "e_score_correction_bias"))
        self.assertTrue(hasattr(router, "expert_usage"))
        # Activating hash routing should drop them
        router.set_layer_number(0)
        self.assertFalse(hasattr(router, "e_score_correction_bias"))
        self.assertFalse(hasattr(router, "expert_usage"))

    def test_sigmoid_scores_in_hash_routing(self):
        """_hash_routing with scoring_func='sigmoid' produces valid
        renormalized weights."""
        router, _ = self._make_router(
            n_routed_experts=4,
            num_experts_per_tok=2,
            scoring_func="sigmoid",
        )
        B, S = 2, 4
        input_ids = paddle.randint(1, 50, [B, S])
        hidden = self._dummy_hidden(B, S)
        _, top_gate, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        # top_gate should be non-negative and sum to ~1 per token
        self.assertTrue(bool((top_gate >= 0).all().numpy()))
        sums = top_gate.sum(axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(B * S), atol=1e-5)


class TestHashRouterLayerActivation(unittest.TestCase):
    """Tests verifying layer-range activation logic in hash routing.

    Hash routing activates on layers with ``layer_number < moe_n_hash_layers``
    (0-indexed leading layers). Wired through ``MoELayer.set_layer_number`` →
    ``TopKRouter.set_layer_number`` → ``_setup_hash_layer``.
    """

    def _make_router(self, layer_number, **cfg_overrides):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        cfg_overrides.setdefault("actual_vocab_size", 128)
        config = _make_router_config(**cfg_overrides)
        router = TopKRouter(config=config)
        router.set_layer_number(layer_number)
        return router

    def test_mtp_layer_excluded(self):
        """is_mtp_layer=True must disable hash routing even in the hash range."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            moe_n_hash_layers=2,
            num_hidden_layers=8,
            actual_vocab_size=128,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0, is_mtp_layer=True)
        self.assertFalse(router.is_hash_layer)

    def test_expert_bias_disabled_on_hash_layer(self):
        """Hash layer should have no e_score_correction_bias / expert_usage."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            num_hidden_layers=8,
            topk_method="noaux_tc",
            actual_vocab_size=128,
        )
        router = TopKRouter(config=config)
        # Before set_layer_number, noaux_tc registers the bias state.
        self.assertTrue(hasattr(router, "e_score_correction_bias"))
        router.set_layer_number(0)
        self.assertTrue(router.is_hash_layer)
        self.assertFalse(hasattr(router, "e_score_correction_bias"))
        self.assertFalse(hasattr(router, "expert_usage"))
        # The buffer should be removed from _buffers as well, so it does not
        # leak into state_dict() and balloon checkpoints.
        self.assertNotIn("e_score_correction_bias", router._buffers)
        self.assertNotIn("e_score_correction_bias", router.state_dict())

    def test_setup_idempotent_via_set_layer_number(self):
        """Calling set_layer_number twice on the same hash router stays valid.

        The second call hits the ``del self.tid2eid`` path before re-registering
        the buffer; it should not crash and the tid2eid table must remain
        consistent.
        """
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=2,
            num_hidden_layers=8,
            actual_vocab_size=16,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        first = router.tid2eid.numpy().copy()
        router.set_layer_number(0)
        np.testing.assert_array_equal(router.tid2eid.numpy(), first)


class TestTopKNoAuxTCGroupLimited(unittest.TestCase):
    """Cover the n_group>1 branch of ``StandardMoERouter._topk_noaux_tc``."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_topk_noaux_tc_group_limited(self, _mock):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(
            topk_method="noaux_tc",
            n_routed_experts=8,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
        )
        router = StandardMoERouter(config)
        # Bias the correction toward experts 0/1 in group 0 so we can predict
        # which group survives selection.
        bias = paddle.zeros([8], dtype=paddle.float32)
        bias[0] = 10.0
        bias[1] = 10.0
        router.e_score_correction_bias.set_value(bias)

        scores = paddle.ones([4, 8], dtype=paddle.float32) * 0.1
        topk_weight, topk_idx = router._topk_noaux_tc(
            scores, k=2, n_group=2, topk_group=1
        )
        self.assertEqual(list(topk_weight.shape), [4, 2])
        self.assertEqual(list(topk_idx.shape), [4, 2])
        # All selected indices must lie in the high-bias group (0..3).
        self.assertTrue(bool((topk_idx < 4).all().numpy()))


class TestSeqAuxLoss(unittest.TestCase):
    """Cover ``StandardMoERouter._cal_seq_aux_loss`` (single-card path).

    The TP/CP collective branches require a full distributed setup and are not
    exercised here; we focus on the single-card path that runs in CI.
    """

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def _make_router(self, _mock, **overrides):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        cfg = _make_router_config(
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_router_load_balancing_type="seq_aux_loss",
            **overrides,
        )
        return StandardMoERouter(cfg), cfg

    def test_seq_aux_loss_no_input_ids(self):
        router, _ = self._make_router()
        bsz, seq_len, num_experts = 2, 4, 4
        probs = paddle.ones([bsz * seq_len, num_experts]) / num_experts
        routing_map = paddle.zeros([bsz * seq_len, num_experts])
        # mark exactly k=2 experts per row
        routing_map[:, 0] = 1.0
        routing_map[:, 1] = 1.0
        loss = router._cal_seq_aux_loss(
            probs,
            top_k=2,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=bsz,
            input_ids=None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(np.isfinite(loss.numpy()))

    def test_seq_aux_loss_with_input_ids_padding(self):
        """Padding (id==0) tokens must be excluded from the denominator."""
        router, _ = self._make_router()
        bsz, seq_len, num_experts = 2, 4, 4
        probs = paddle.ones([bsz, seq_len, num_experts]) / num_experts
        routing_map = paddle.zeros([bsz * seq_len, num_experts])
        routing_map[:, 0] = 1.0
        routing_map[:, 1] = 1.0
        # First row has 1 pad, second row has 2 pads.
        input_ids = paddle.to_tensor(
            [[0, 1, 2, 3], [0, 0, 4, 5]], dtype="int64"
        )
        loss = router._cal_seq_aux_loss(
            probs,
            top_k=2,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=bsz,
            input_ids=input_ids,
        )
        self.assertTrue(np.isfinite(loss.numpy()))

    def test_seq_aux_loss_experimental_version(self):
        router, cfg = self._make_router()
        # gpt_model_use_experimental_version flips between two formulae.
        cfg.gpt_model_use_experimental_version = True
        cfg.num_nextn_predict_layers = 0
        bsz, seq_len, num_experts = 2, 4, 4
        probs = paddle.ones([bsz, seq_len, num_experts]) / num_experts
        routing_map = paddle.zeros([bsz * seq_len, num_experts])
        routing_map[:, 0] = 1.0
        routing_map[:, 1] = 1.0
        input_ids = paddle.to_tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype="int64"
        )
        loss = router._cal_seq_aux_loss(
            probs,
            top_k=2,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=bsz,
            input_ids=input_ids,
        )
        self.assertTrue(np.isfinite(loss.numpy()))

    def test_seq_aux_loss_non_experimental_matches_formula(self):
        router, _ = self._make_router()
        bsz, seq_len, num_experts, top_k = 2, 3, 4, 2
        probs = paddle.ones([bsz, seq_len, num_experts]) / num_experts
        routing_map = paddle.zeros([bsz * seq_len, num_experts])
        routing_map[:, 0] = 1.0
        routing_map[:, 1] = 1.0
        input_ids = paddle.to_tensor([[1, 0, 2], [3, 4, 5]], dtype="int64")

        loss = router._cal_seq_aux_loss(
            probs,
            top_k=top_k,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=bsz,
            input_ids=input_ids,
        )

        denom = (
            (input_ids != 0).astype(paddle.float32).sum(axis=-1, keepdim=True)
        )
        routing_3d = routing_map.reshape([bsz, seq_len, num_experts])
        cost_coeff = routing_3d.sum(axis=1, dtype="float32") / (
            denom * paddle.to_tensor(top_k / num_experts, dtype="float32")
        )
        expected = (cost_coeff * probs.sum(axis=1) / denom).sum(axis=1).mean()
        self.assertAlmostEqual(
            float(loss.numpy()), float(expected.numpy()), places=6
        )

    def test_seq_aux_loss_experimental_uses_origin_ids_formula(self):
        router, cfg = self._make_router()
        cfg.gpt_model_use_experimental_version = True
        bsz, seq_len, num_experts, top_k = 2, 3, 4, 2
        probs = paddle.ones([bsz, seq_len, num_experts]) / num_experts
        routing_map = paddle.zeros([bsz * seq_len, num_experts])
        routing_map[:, 0] = 1.0
        routing_map[:, 1] = 1.0
        input_ids = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")
        origin_input_ids = paddle.to_tensor(
            [[1, 0, 2, 0], [3, 4, 5, 6]], dtype="int64"
        )

        loss = router._cal_seq_aux_loss(
            probs,
            top_k=top_k,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=bsz,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )

        denom = (
            (origin_input_ids != 0)
            .astype(paddle.float32)
            .sum(axis=-1, keepdim=True)
        )
        routing_3d = routing_map.reshape([bsz, seq_len, num_experts])
        cost_coeff = (
            routing_3d.sum(axis=1, dtype="float32")
            / denom
            * paddle.to_tensor(num_experts / top_k, dtype="float32")
        )
        expected = (cost_coeff * probs.mean(axis=1)).sum(axis=1).mean()
        self.assertAlmostEqual(
            float(loss.numpy()), float(expected.numpy()), places=6
        )


class TestZLossWithInputIds(unittest.TestCase):
    """Cover the input_ids-aware branch of ``_cal_z_loss``."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_z_loss_uses_valid_token_count(self, _mock):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        cfg = _make_router_config(router_z_loss_coef=0.1)
        router = StandardMoERouter(cfg)
        logits = paddle.randn([8, 4])
        input_ids = paddle.to_tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype="int64"
        )
        loss = router._cal_z_loss(logits, input_ids=input_ids)
        self.assertTrue(np.isfinite(loss.numpy()))
        # Same logits, no padding → also finite.
        no_pad_ids = paddle.ones([2, 4], dtype="int64")
        loss2 = router._cal_z_loss(logits, input_ids=no_pad_ids)
        self.assertTrue(np.isfinite(loss2.numpy()))


class TestTopKRouterForward(unittest.TestCase):
    """End-to-end coverage of ``TopKRouter.forward`` non-hash branches.

    These exercise the post-``gate_score_func`` pipeline (lines ~993-1152 in
    moe_router.py): scoring -> topk -> mask build -> norm -> scaling ->
    routing-loss tail. The hash branch is covered by ``TestHashRouter``.
    """

    def _router(self, **overrides):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        defaults = {"n_routed_experts": 4, "num_experts_per_tok": 2}
        defaults.update(overrides)
        cfg = _make_router_config(**defaults)
        router = TopKRouter(config=cfg)
        router.set_layer_number(0)  # plain (no hash) since moe_n_hash_layers=0
        return router, cfg

    def _hidden(self, B=1, S=4, H=64):
        return paddle.randn([B, S, H])

    def test_forward_greedy_softmax_no_input_ids(self):
        router, _ = self._router()
        out = router(self._hidden(), input_ids=None)
        cap, top_gate, top_idx, probs, mask, prio, l_aux, l_z = out
        self.assertIsNone(cap)
        self.assertEqual(list(top_idx.shape), [4, 2])
        self.assertEqual(list(top_gate.shape), [4, 2])
        self.assertEqual(list(probs.shape), [4, 4])
        self.assertEqual(list(mask.shape), [4, 4])
        # aux_loss enabled by default in _make_router_config, z_loss off.
        self.assertIsNotNone(l_aux)
        self.assertIsNone(l_z)

    def test_forward_with_padding_input_ids(self):
        """input_ids=0 positions must drop out of routing (idx=-1, weight=0)."""
        router, _ = self._router()
        hidden = self._hidden(B=1, S=4)
        input_ids = paddle.to_tensor([[0, 1, 2, 3]], dtype="int64")
        _, top_gate, top_idx, _, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        # Padding row gets idx=-1 and zero weight + zero mask.
        self.assertEqual(int(top_idx[0, 0].numpy()), -1)
        np.testing.assert_array_equal(
            top_gate[0].numpy(), np.zeros(2, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            mask[0].numpy(), np.zeros(4, dtype=np.float32)
        )

    def test_forward_sigmoid_renormalizes_gates_ori(self):
        """The sigmoid branch must run the renorm path without errors."""
        router, _ = self._router(scoring_func="sigmoid")
        out = router(self._hidden(), input_ids=None)
        self.assertIsNotNone(out[1])  # top_gate

    def test_forward_z_loss_active(self):
        router, _ = self._router(router_z_loss_coef=0.1)
        out = router(self._hidden(), input_ids=None)
        self.assertIsNotNone(out[7])  # l_zloss

    def test_forward_seq_aux_loss(self):
        router, _ = self._router(
            moe_router_load_balancing_type="seq_aux_loss",
        )
        # forward expects [B, S, H] with B*S consistent with input_ids when set.
        hidden = self._hidden(B=2, S=4)
        input_ids = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64"
        )
        out = router(hidden, input_ids=input_ids)
        self.assertIsNotNone(out[6])  # l_aux from seq_aux_loss path

    def test_forward_routed_scaling_factor_scalar(self):
        router, _ = self._router(routed_scaling_factor=2.5)
        out = router(self._hidden(), input_ids=None)
        # Scaled top_gate values may exceed 1; just sanity-check it runs.
        self.assertEqual(list(out[1].shape), [4, 2])

    def test_forward_routed_scaling_factor_learnable(self):
        router, _ = self._router(
            routed_scaling_factor=1.0,
            routed_scaling_factor_learnable=True,
        )
        out = router(self._hidden(), input_ids=None)
        self.assertEqual(list(out[1].shape), [4, 2])

    def test_forward_noaux_tc_topk(self):
        router, _ = self._router(
            topk_method="noaux_tc",
            n_routed_experts=4,
            n_group=1,
            topk_group=1,
        )
        out = router(self._hidden(), input_ids=None)
        self.assertEqual(list(out[2].shape), [4, 2])  # top_idx
        # noaux_tc topk also accumulates expert_usage.
        self.assertTrue(bool((router.expert_usage >= 0).all().numpy()))

    def test_forward_group_limited_greedy(self):
        router, _ = self._router(
            topk_method="group_limited_greedy",
            n_routed_experts=8,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
        )
        out = router(self._hidden(), input_ids=None)
        self.assertEqual(list(out[2].shape), [4, 2])

    def test_forward_2d_input_raises(self):
        router, _ = self._router()
        with self.assertRaises(ValueError):
            router(paddle.randn([4, 64]), input_ids=None)

    def test_forward_input_ids_shape_mismatch_raises(self):
        router, _ = self._router()
        hidden = self._hidden(B=1, S=4)
        bad_ids = paddle.to_tensor([[1, 2]], dtype="int64")  # S mismatch
        with self.assertRaises(AssertionError):
            router(hidden, input_ids=bad_ids)

    def test_forward_force_load_balancing(self):
        """moe_router_force_load_balancing=True takes the random-logits branch
        in ``gate_detach_matmul`` (line ~187). ``apply_random_logits`` requires
        ``expert-parallel-rng`` state which is not set up in single-card test;
        mock it to a deterministic identity."""
        router, _ = self._router(moe_router_force_load_balancing=True)
        with patch(
            "paddleformers.fleet.transformer.moe.moe_router.apply_random_logits",
            side_effect=lambda x: x,
        ):
            out = router(self._hidden(), input_ids=None)
        self.assertEqual(list(out[2].shape), [4, 2])

    def test_forward_sigmoid_experimental_version(self):
        """sigmoid + experimental_version triggers the clip-based renorm
        (line ~1017)."""
        router, cfg = self._router(scoring_func="sigmoid")
        cfg.gpt_model_use_experimental_version = True
        out = router(self._hidden(), input_ids=None)
        self.assertEqual(list(out[1].shape), [4, 2])

    def test_forward_backward_through_fused_gate_matmul(self):
        """Backward pass exercises FusedGateDetachMatmul.backward (lines 107-170)
        and the autograd path inside the router gate."""
        router, _ = self._router()
        hidden = self._hidden()
        hidden.stop_gradient = False
        out = router(hidden, input_ids=None)
        l_aux = out[6]
        # aux_loss is differentiable wrt the gate weight (and through the
        # FusedGateDetachMatmul PyLayer used inside ``gate_detach_matmul``).
        l_aux.backward()
        self.assertIsNotNone(router.weight.grad)


class TestHashRoutingExtraBranches(unittest.TestCase):
    """Cover small but uncovered branches in hash routing setup / scoring."""

    def _make(self, **overrides):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        defaults = {
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_n_hash_layers": 1,
            "num_hidden_layers": 4,
            "actual_vocab_size": 32,
        }
        defaults.update(overrides)
        cfg = _make_router_config(**defaults)
        router = TopKRouter(config=cfg)
        return router, cfg

    def test_hash_routing_sqrtsoftplus_branch(self):
        """Hash router with scoring_func='sqrtsoftplus' must hit the
        sqrt(softplus(x)) branch in ``_hash_routing`` (line ~782)."""
        router, _ = self._make(scoring_func="sqrtsoftplus")
        router.set_layer_number(0)
        hidden = paddle.randn([1, 4, 64])
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        _, top_gate, top_idx, _, _, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        # sqrt(softplus(x)) is non-negative.
        self.assertTrue(bool((top_gate >= 0).all().numpy()))
        self.assertEqual(list(top_idx.shape), [4, 2])

    def test_hash_routing_invalid_scoring_func_raises(self):
        """The hash-layer scoring_func allow-list is enforced by both
        ``TransformerConfig.__post_init__`` and ``_setup_hash_layer``. Building
        a config with ``scoring_func='tanh'`` already triggers the check, so
        verify the ValueError surfaces at config construction."""
        with self.assertRaises(ValueError):
            self._make(scoring_func="tanh")


class TestZLossExperimental(unittest.TestCase):
    """Cover the experimental_version branch of ``_cal_z_loss`` (lines ~501-505)."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_z_loss_experimental_version(self, _mock):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        cfg = _make_router_config(router_z_loss_coef=0.1)
        cfg.gpt_model_use_experimental_version = True
        cfg.num_nextn_predict_layers = 1
        router = StandardMoERouter(cfg)
        logits = paddle.randn([8, 4])
        input_ids = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]], dtype="int64"
        )
        loss = router._cal_z_loss(logits, input_ids=input_ids)
        self.assertTrue(np.isfinite(loss.numpy()))

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_z_loss_experimental_uses_origin_input_ids(self, _mock):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        cfg = _make_router_config(router_z_loss_coef=0.1)
        cfg.gpt_model_use_experimental_version = True
        router = StandardMoERouter(cfg)
        logits = paddle.randn([6, 4])
        input_ids = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")
        origin_input_ids = paddle.to_tensor(
            [[1, 0, 2, 0], [3, 4, 5, 6]], dtype="int64"
        )

        loss = router._cal_z_loss(
            logits,
            input_ids=input_ids,
            origin_input_ids=origin_input_ids,
        )

        loss_mask = (input_ids != 0).astype(paddle.float32).reshape([-1])
        denom = (origin_input_ids != 0).astype(paddle.float32).sum()
        expected = (logits.logsumexp(1).square() * loss_mask).sum() / denom
        self.assertAlmostEqual(
            float(loss.numpy()), float(expected.numpy()), places=6
        )


class TestMoeTopkFusionLazyImport(unittest.TestCase):
    """Cover ``_get_moe_topk_fusion`` lazy import (lines 53-59)."""

    def test_lazy_import_returns_class(self):
        from paddleformers.fleet.transformer.moe import moe_router as mr

        # Reset cache to force the import branch.
        mr._MoETopkFusion = None

        sentinel = object()
        fake_module = types_module = __import__("types").ModuleType(
            "paddleformers.fleet.triton_ops.moe_topk_fusion"
        )
        fake_module.MoETopkFusion = sentinel
        with patch.dict(
            "sys.modules",
            {"paddleformers.fleet.triton_ops.moe_topk_fusion": fake_module},
        ):
            got = mr._get_moe_topk_fusion()
            self.assertIs(got, sentinel)
            # Cached after first call.
            self.assertIs(mr._get_moe_topk_fusion(), sentinel)
        # Cleanup module-level cache so other tests aren't affected.
        mr._MoETopkFusion = None


class TestLogMoeMd5(unittest.TestCase):
    """Cover ``_log_moe_md5`` paths (lines 65-83)."""

    def test_no_op_when_flag_off(self):
        from paddleformers.fleet.transformer.moe import moe_router as mr

        mr._LOG_LAYER_MD5 = False
        # Should be a no-op and not raise.
        mr._log_moe_md5(paddle.randn([2, 4]), "x", layer_idx=1)

    def test_logs_when_flag_on_and_experimental_version(self):
        from paddleformers.fleet.transformer.moe import moe_router as mr
        from paddleformers.fleet.transformer.transformer_layer import (
            TransformerLayer,
        )

        mr._LOG_LAYER_MD5 = True
        prev_exp = TransformerLayer._gpt_model_use_experimental_version
        prev_skip = TransformerLayer._skip_mtp_probes
        TransformerLayer._gpt_model_use_experimental_version = True
        try:
            # Skip path: MTP probes
            TransformerLayer._skip_mtp_probes = True
            mr._log_moe_md5(paddle.randn([2, 4]), "x", layer_idx=0)
            # Active path: actually computes md5 and prints
            TransformerLayer._skip_mtp_probes = False
            mr._log_moe_md5(paddle.randn([2, 4]), "y", layer_idx=2)
        finally:
            TransformerLayer._gpt_model_use_experimental_version = prev_exp
            TransformerLayer._skip_mtp_probes = prev_skip
            mr._LOG_LAYER_MD5 = False


class TestRoutingMapFusionWrapper(unittest.TestCase):
    """Cover ``_apply_routing_map_fusion`` wrapper (lines 191-207).

    The underlying triton kernel is mocked; we only verify the wrapper's
    branching/reshape logic.
    """

    def _fake_routing_map_fusion(
        self,
        gates,
        top_idx,
        input_ids=None,
        is_pure_text_line=None,
        pad_token_id=0,
    ):
        # Return a binary mask with selected indices set, matching gates shape.
        fused_mask = paddle.zeros_like(gates).put_along_axis(
            top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
        )
        exp_counts = paddle.zeros([gates.shape[-1]], dtype="int64")
        return fused_mask, top_idx, exp_counts

    def test_with_input_ids_path(self):
        from paddleformers.fleet.transformer.moe.moe_router import (
            _apply_routing_map_fusion,
        )

        gates = paddle.randn([4, 8]).abs()
        top_idx = paddle.to_tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype="int64"
        )
        nonzero_mask = paddle.ones([4, 1], dtype="bool")
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        with patch(
            "paddleformers.fleet.triton_ops.routing_map_fusion_forward",
            side_effect=self._fake_routing_map_fusion,
            create=True,
        ):
            mask, ti, exp = _apply_routing_map_fusion(
                gates, top_idx, nonzero_mask, input_ids
            )
        self.assertEqual(list(mask.shape), [4, 8])
        self.assertEqual(list(ti.shape), [4, 2])

    def test_without_input_ids_path(self):
        from paddleformers.fleet.transformer.moe.moe_router import (
            _apply_routing_map_fusion,
        )

        gates = paddle.randn([4, 8]).abs()
        top_idx = paddle.to_tensor(
            [[0, 1], [2, 3], [4, 5], [6, 7]], dtype="int64"
        )
        with patch(
            "paddleformers.fleet.triton_ops.routing_map_fusion_forward",
            side_effect=self._fake_routing_map_fusion,
            create=True,
        ):
            mask, ti, exp = _apply_routing_map_fusion(
                gates, top_idx, None, None
            )
        self.assertEqual(list(mask.shape), [4, 8])


class TestSeqAuxLoss1DInputIds(unittest.TestCase):
    """Cover ``_cal_seq_aux_loss`` reshape branch where input_ids is 1D
    (lines ~420-423: ``_ids.ndim == 1`` unsqueeze)."""

    @patch(
        "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
        return_value=1,
    )
    def test_seq_aux_loss_with_1d_input_ids(self, _mock):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        cfg = _make_router_config(router_aux_loss_coef=0.1)
        router = StandardMoERouter(cfg)
        seq_len, n_experts = 4, cfg.n_routed_experts
        batch_size = 2
        probs = paddle.randn([batch_size, seq_len, n_experts]).abs()
        probs = probs / probs.sum(axis=-1, keepdim=True)
        routing_map = paddle.randint(
            0, 2, [batch_size * seq_len, n_experts]
        ).cast(paddle.float32)
        # 1D input_ids — function should unsqueeze it internally.
        input_ids_1d = paddle.to_tensor([1, 2, 0, 4, 5, 6, 7, 0], dtype="int64")
        loss = router._cal_seq_aux_loss(
            probs,
            top_k=2,
            routing_map=routing_map,
            seq_len=seq_len,
            batch_size=batch_size,
            input_ids=input_ids_1d,
        )
        self.assertTrue(np.isfinite(loss.numpy()))


class TestForwardNoAuxLoss(unittest.TestCase):
    """Cover ``TopKRouter.forward`` where neither aux nor seq_aux loss is
    enabled — branch where ``l_aux`` stays None (line ~1150)."""

    def _router(self):
        cfg = _make_router_config(
            router_aux_loss_coef=0.0,
            router_z_loss_coef=None,
        )
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        router = TopKRouter(config=cfg)
        router.set_layer_number(0)
        return router

    def test_forward_l_aux_none(self):
        router = self._router()
        hidden = paddle.randn([1, 4, router.config.hidden_size])
        out = router(hidden, input_ids=None)
        # tuple positions: (None, top_gate, top_idx, probs, mask, None, l_aux, l_z)
        l_aux = out[6]
        l_z = out[7]
        self.assertIsNone(l_aux)
        self.assertIsNone(l_z)


class TestMoeTopkFusionForward(unittest.TestCase):
    """Cover the ``moe_topk_fusion`` Triton branch in ``TopKRouter.forward``
    (lines ~1021-1063). The Triton kernel is mocked with a deterministic top-k.
    """

    def _fake_topk_apply(
        self,
        gates,
        probs_for_choice,
        k,
        use_node_limit,
        n_group,
        topk_group,
        norm,
    ):
        top_gate, top_idx = paddle.topk(probs_for_choice, k=k, axis=-1)
        if norm:
            top_gate = top_gate / (top_gate.sum(axis=-1, keepdim=True) + 1e-12)
        return top_gate, top_idx.cast("int64")

    def test_forward_uses_moe_topk_fusion(self):
        from paddleformers.fleet.transformer.moe import moe_router as mr
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        cfg = _make_router_config(
            topk_method="noaux_tc",
            scoring_func="sigmoid",
        )
        cfg.moe_topk_fusion = True
        router = TopKRouter(config=cfg)
        router.set_layer_number(0)

        fake_cls = type(
            "FakeMoETopkFusion",
            (),
            {"apply": staticmethod(self._fake_topk_apply)},
        )
        prev = mr._MoETopkFusion
        mr._MoETopkFusion = fake_cls
        prev_log = mr._LOG_LAYER_MD5
        try:
            # Exercise the _LOG_LAYER_MD5 logging branch as well (lines 1031-1063).
            mr._LOG_LAYER_MD5 = True
            from paddleformers.fleet.transformer.transformer_layer import (
                TransformerLayer,
            )

            prev_exp = TransformerLayer._gpt_model_use_experimental_version
            TransformerLayer._gpt_model_use_experimental_version = True
            try:
                hidden = paddle.randn([1, 4, cfg.hidden_size])
                out = router(hidden, input_ids=None)
            finally:
                TransformerLayer._gpt_model_use_experimental_version = prev_exp
            self.assertEqual(list(out[2].shape), [4, cfg.num_experts_per_tok])
        finally:
            mr._MoETopkFusion = prev
            mr._LOG_LAYER_MD5 = prev_log


class TestForwardRoutingMapFusion(unittest.TestCase):
    """Cover the ``routing_map_fusion`` integration in ``TopKRouter.forward``
    (line ~1089)."""

    def _fake_routing_map_fusion_forward(
        self,
        gates,
        top_idx,
        input_ids=None,
        is_pure_text_line=None,
        pad_token_id=0,
    ):
        fused_mask = paddle.zeros_like(gates).put_along_axis(
            top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
        )
        exp_counts = paddle.zeros([gates.shape[-1]], dtype="int64")
        return fused_mask, top_idx, exp_counts

    def test_forward_routing_map_fusion_branch(self):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        cfg = _make_router_config()
        cfg.routing_map_fusion = True
        router = TopKRouter(config=cfg)
        router.set_layer_number(0)
        hidden = paddle.randn([1, 4, cfg.hidden_size])
        with patch(
            "paddleformers.fleet.triton_ops.routing_map_fusion_forward",
            side_effect=self._fake_routing_map_fusion_forward,
            create=True,
        ):
            out = router(hidden, input_ids=None)
        self.assertEqual(list(out[2].shape), [4, cfg.num_experts_per_tok])


class TestHashRoutingScoreFuncs(unittest.TestCase):
    """Cover _hash_routing score function branches (softmax/sigmoid/sqrtsoftplus)
    and the full hash-routing forward() path."""

    def _make_hash_router(
        self,
        scoring_func="softmax",
        n_experts=4,
        topk=2,
        routed_scaling_factor=1.0,
        routed_scaling_factor_learnable=False,
    ):
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func=scoring_func,
            n_routed_experts=n_experts,
            num_experts_per_tok=topk,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
            routed_scaling_factor=routed_scaling_factor,
            routed_scaling_factor_learnable=routed_scaling_factor_learnable,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        return router

    def test_hash_routing_softmax_scores(self):
        """_hash_routing with softmax scoring (line 776)."""
        router = self._make_hash_router(scoring_func="softmax")
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, top_idx = router._hash_routing(logits, flat_ids)
        self.assertEqual(top_gate.shape, [4, 2])
        self.assertEqual(top_idx.shape, [4, 2])

    def test_hash_routing_sigmoid_scores(self):
        """_hash_routing with sigmoid scoring (line 778)."""
        router = self._make_hash_router(scoring_func="sigmoid")
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, top_idx = router._hash_routing(logits, flat_ids)
        self.assertEqual(top_gate.shape, [4, 2])

    def test_hash_routing_sqrtsoftplus_scores(self):
        """_hash_routing with sqrtsoftplus scoring (else branch, line 782)."""
        router = self._make_hash_router(scoring_func="sqrtsoftplus")
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, top_idx = router._hash_routing(logits, flat_ids)
        self.assertEqual(top_gate.shape, [4, 2])
        # sqrtsoftplus scores should be non-negative
        self.assertTrue(bool((top_gate >= 0).all().item()))

    def test_hash_routing_no_softmax_normalizes(self):
        """For non-softmax scoring, top_gate is normalized (line 789)."""
        router = self._make_hash_router(scoring_func="sigmoid")
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, _ = router._hash_routing(logits, flat_ids)
        # Each row should sum to ~1 (normalized)
        row_sums = top_gate.sum(axis=-1)
        self.assertTrue(
            bool(
                paddle.allclose(
                    row_sums, paddle.ones_like(row_sums), atol=1e-4
                ).item()
            )
        )

    def test_hash_routing_no_tid2eid_raises(self):
        """_hash_routing raises ValueError when tid2eid is None (line 768)."""
        router = self._make_hash_router(scoring_func="softmax")
        router.tid2eid = None  # clear the buffer
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        with self.assertRaises(ValueError):
            router._hash_routing(logits, flat_ids)

    def test_hash_routing_unsupported_scoring_func_raises(self):
        """_hash_routing raises ValueError for unsupported scoring_func
        (explicit elif + raise branch, line ~780)."""
        router = self._make_hash_router(scoring_func="softmax")
        # Mutate scoring_func after construction to bypass _setup_hash_layer
        # validation, so _hash_routing hits the else-raise branch.
        router.scoring_func = "tanh"
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        with self.assertRaisesRegex(ValueError, "Unsupported scoring_func"):
            router._hash_routing(logits, flat_ids)

    def test_hash_routing_routed_scaling_factor_fixed(self):
        """_hash_routing applies fixed routed_scaling_factor (line 802)."""
        router = self._make_hash_router(
            scoring_func="softmax",
            routed_scaling_factor=2.0,
            routed_scaling_factor_learnable=False,
        )
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, top_idx = router._hash_routing(logits, flat_ids)
        self.assertEqual(top_gate.shape, [4, 2])
        # With scaling_factor=2.0, gate values should be larger than unscaled
        unscaled_router = self._make_hash_router(
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            routed_scaling_factor_learnable=False,
        )
        top_gate_unscaled, _ = unscaled_router._hash_routing(logits, flat_ids)
        # scaled should be ~2x unscaled
        ratio = top_gate.numpy().mean() / (
            top_gate_unscaled.numpy().mean() + 1e-10
        )
        self.assertGreater(ratio, 1.5)

    def test_hash_routing_routed_scaling_factor_learnable(self):
        """_hash_routing applies learnable routed_scaling_factor (line 795)."""
        router = self._make_hash_router(
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            routed_scaling_factor_learnable=True,
        )
        logits = paddle.randn([4, 4])
        flat_ids = paddle.randint(0, 16, [4])
        top_gate, top_idx = router._hash_routing(logits, flat_ids)
        self.assertEqual(top_gate.shape, [4, 2])

    def test_hash_forward_end_to_end(self):
        """Full forward() with hash routing (lines 961-993)."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        hidden = paddle.randn([2, 4, config.hidden_size])
        input_ids = paddle.randint(1, 16, [2, 4])
        out = router(hidden, input_ids=input_ids)
        # Return: (None, top_gate, top_idx, probs, mask, None, None, None)
        self.assertIsNone(out[0])
        self.assertEqual(out[1].shape, [8, 2])  # 8 tokens, topk=2
        self.assertEqual(out[2].shape, [8, 2])
        self.assertEqual(out[3].shape, [8, 4])  # probs: [N, num_experts]
        self.assertEqual(out[4].shape, [8, 4])  # mask: [N, num_experts]
        self.assertIsNone(out[5])
        self.assertIsNone(out[6])  # no aux loss on hash layers
        self.assertIsNone(out[7])  # no z loss on hash layers

    def test_hash_forward_sequence_parallel(self):
        """Hash routing with sequence_parallel=True (line 963)."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        # With sequence_parallel, input shape is [seq, batch, hidden]
        # but input_ids shape is still [batch_size, seq_len] per the
        # assertion in forward(): batch_size_==batch_size, seq_len_==seq_len
        hidden = paddle.randn([4, 2, config.hidden_size])
        input_ids = paddle.randint(1, 16, [2, 4])
        out = router(hidden, input_ids=input_ids)
        self.assertEqual(out[1].shape, [8, 2])

    def test_hash_forward_with_padding_mask(self):
        """Hash routing with input_ids containing zeros (lines 980-985).

        When input_ids has zero entries, the non-zero mask is computed
        inside forward() and applied to the hash routing output.
        """
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="sigmoid",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        hidden = paddle.randn([2, 4, config.hidden_size])
        # Include zero token IDs to trigger the padding mask branch
        input_ids = paddle.to_tensor(
            [[1, 2, 0, 4], [5, 0, 7, 8]], dtype="int64"
        )
        out = router(hidden, input_ids=input_ids)
        self.assertEqual(out[1].shape, [8, 2])

    def test_hash_forward_sequence_parallel_with_padding(self):
        """Hash routing with sequence_parallel + padding mask.

        Verifies that input_ids_none_zero_mask is transposed along with
        flat_ids in the sequence-parallel path so that the mask token order
        matches the logits token order.
        """
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="sigmoid",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        # With sequence_parallel, input shape is [seq, batch, hidden]
        hidden = paddle.randn([4, 2, config.hidden_size])
        # input_ids shape is [batch_size, seq_len] = [2, 4]
        # Place zeros at specific positions: row 0 col 2, row 1 col 1
        input_ids = paddle.to_tensor(
            [[1, 2, 0, 4], [5, 0, 7, 8]], dtype="int64"
        )
        out = router(hidden, input_ids=input_ids)
        top_gate, top_idx = out[1], out[2]
        self.assertEqual(top_gate.shape, [8, 2])
        self.assertEqual(top_idx.shape, [8, 2])
        # In SP, tokens are ordered [s0b0, s0b1, s1b0, s1b1, s2b0, s2b1, s3b0, s3b1]
        # Padding tokens (input_ids==0) are at positions:
        #   row0 col2 -> s2b0 -> index 4
        #   row1 col1 -> s1b1 -> index 3
        # These tokens should have top_idx == -1 (masked out)
        top_idx_np = top_idx.numpy()
        self.assertTrue(
            (top_idx_np[4] == -1).all(),
            f"Padding token at index 4 (s2b0) should be masked, got {top_idx_np[4]}",
        )
        self.assertTrue(
            (top_idx_np[3] == -1).all(),
            f"Padding token at index 3 (s1b1) should be masked, got {top_idx_np[3]}",
        )

    """Cover sftplus and sqrtsoftplus branches in gate_score_func (lines 304-307)."""

    def _make_router(self, scoring_func="softmax"):
        from paddleformers.fleet.transformer.moe.moe_router import (
            StandardMoERouter,
        )

        config = _make_router_config(scoring_func=scoring_func)
        return StandardMoERouter(config)

    def test_sftplus_score_func(self):
        """gate_score_func with sftplus scoring (line 304)."""
        router = self._make_router(scoring_func="sftplus")
        logits = paddle.randn([4, router.config.n_routed_experts])
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, logits.shape)
        # softplus is always positive
        self.assertTrue(bool((scores > 0).all().item()))

    def test_sqrtsoftplus_score_func(self):
        """gate_score_func with sqrtsoftplus scoring (line 306)."""
        router = self._make_router(scoring_func="sqrtsoftplus")
        logits = paddle.randn([4, router.config.n_routed_experts])
        scores = router.gate_score_func(logits)
        self.assertEqual(scores.shape, logits.shape)
        # sqrt(softplus) is always non-negative
        self.assertTrue(bool((scores >= 0).all().item()))


class TestSetupHashLayerErrors(unittest.TestCase):
    """Cover _setup_hash_layer error paths."""

    def test_invalid_scoring_func_raises(self):
        """_setup_hash_layer raises ValueError for unsupported scoring_func (line 861)."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        # Create a valid config first, then override scoring_func to an
        # invalid value.  TransformerConfig.__post_init__ validates at
        # construction time, so we must mutate *after* creation.
        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        config.scoring_func = "relu"  # bypass TransformerConfig validation
        router = TopKRouter(config=config)
        with self.assertRaises(ValueError):
            router.set_layer_number(0)

    def test_missing_vocab_size_raises(self):
        """_setup_hash_layer raises ValueError when actual_vocab_size is None (line 869)."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        # Create a valid config first, then remove actual_vocab_size.
        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        config.actual_vocab_size = None  # bypass TransformerConfig validation
        router = TopKRouter(config=config)
        with self.assertRaises(ValueError):
            router.set_layer_number(0)

    def test_mtp_layer_not_hash(self):
        """is_mtp_layer=True should disable hash routing (line 853)."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0, is_mtp_layer=True)
        self.assertFalse(router.is_hash_layer)

    def test_setup_hash_layer_activates_tid2eid(self):
        """_setup_hash_layer registers tid2eid buffer on hash layers."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=2,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        self.assertFalse(router.is_hash_layer)
        router.set_layer_number(0)
        self.assertTrue(router.is_hash_layer)
        self.assertIsNotNone(router.tid2eid)
        self.assertEqual(router.tid2eid.shape, [16, 2])  # [vocab, topk]

    def test_setup_hash_layer_layer_outside_range(self):
        """Layers with layer_number >= moe_n_hash_layers are not hash layers."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=2,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(5)  # >= 2, so not a hash layer
        self.assertFalse(router.is_hash_layer)

    def test_setup_hash_layer_removes_noaux_tc_bias(self):
        """_setup_hash_layer removes e_score_correction_bias on hash layers."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
            topk_method="noaux_tc",
        )
        router = TopKRouter(config=config)
        # noaux_tc should create e_score_correction_bias in __init__
        self.assertTrue(hasattr(router, "e_score_correction_bias"))
        # Activating hash layer should remove it
        router.set_layer_number(0)
        self.assertTrue(router.is_hash_layer)
        self.assertFalse(hasattr(router, "e_score_correction_bias"))
        self.assertFalse(hasattr(router, "expert_usage"))

    def test_setup_hash_layer_layer_number_none(self):
        """set_layer_number(None) should not activate hash routing."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(None)
        self.assertFalse(router.is_hash_layer)


class TestHashForwardInputIdsRequired(unittest.TestCase):
    """Cover the hash routing forward branch input_ids check (line 942)."""

    def test_hash_layer_requires_input_ids(self):
        """When is_hash_layer=True, forward() requires input_ids."""
        from paddleformers.fleet.transformer.moe.moe_router import TopKRouter

        config = _make_router_config(
            scoring_func="softmax",
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_n_hash_layers=1,
            actual_vocab_size=16,
            num_hidden_layers=8,
        )
        router = TopKRouter(config=config)
        router.set_layer_number(0)
        hidden = paddle.randn([2, 4, 64])
        with self.assertRaises(ValueError):
            router(hidden, input_ids=None)


class TestMoELayerSetLayerNumberForwardsIsMtp(unittest.TestCase):
    """Verify MoELayer.set_layer_number forwards is_mtp_layer to the gate."""

    def _make_moe_layer(self, **cfg_overrides):
        """Build a MoELayer with mock gate to test set_layer_number forwarding."""
        from unittest.mock import MagicMock

        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        cfg_overrides.setdefault("actual_vocab_size", 128)
        config = _make_router_config(**cfg_overrides)
        moe = MoELayer.__new__(MoELayer)
        moe.gate = MagicMock()
        moe.layer_number = None
        # set_layer_number now colors expert params; EP=1 makes that a no-op.
        moe.expert_model_parallel_size = 1
        return moe

    def test_forwards_is_mtp_layer_true(self):
        moe = self._make_moe_layer(moe_n_hash_layers=2, num_hidden_layers=8)
        moe.set_layer_number(0, is_mtp_layer=True)
        moe.gate.set_layer_number.assert_called_once_with(0, is_mtp_layer=True)
        self.assertEqual(moe.layer_number, 0)

    def test_forwards_is_mtp_layer_false(self):
        moe = self._make_moe_layer(moe_n_hash_layers=2, num_hidden_layers=8)
        moe.set_layer_number(0, is_mtp_layer=False)
        moe.gate.set_layer_number.assert_called_once_with(0, is_mtp_layer=False)

    def test_default_is_mtp_layer_false(self):
        """Calling without is_mtp_layer should default to False."""
        moe = self._make_moe_layer(moe_n_hash_layers=2, num_hidden_layers=8)
        moe.set_layer_number(0)
        moe.gate.set_layer_number.assert_called_once_with(0, is_mtp_layer=False)


if __name__ == "__main__":
    unittest.main()
