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
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

# Adjust the import path according to your actual project structure
from paddleformers.fleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    TopKRouter,
)


# ================= Mock Dependency Environment =================
class MockTransformerConfig:
    """
    Mock configuration object to simulate
    paddleformers.fleet.transformer.transformer_config.TransformerConfig
    """

    def __init__(self):
        # Basic Model Parameters
        self.hidden_size = 64
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1

        # Router Specific Parameters
        self.init_method = paddle.nn.initializer.Normal(mean=0.0, std=0.02)
        self.topk_method = "noaux_tc"
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self.routed_scaling_factor_learnable = False
        self.scoring_func = "softmax"
        self.moe_router_load_balancing_type = "aux_loss"
        self.moe_router_force_load_balancing = False
        self.moe_router_fusion = True

        # Loss Coefficients
        self.router_z_loss_coef = 0.01
        self.router_aux_loss_coef = 0.01

        # Parallelism Parameters
        self.tensor_model_parallel_size = 1
        self.context_parallel_size = 1
        self.sequence_parallel = False

        # Experimental version flag
        self.gpt_model_use_experimental_version = False

        # Internal storage to simulate the .get() method behavior
        self._extra_conf = {"seq_aux": False}

    def get(self, key, default=None):
        """
        Simulate the dictionary-like get behavior of the config object.
        It prioritizes _extra_conf, then falls back to object attributes.
        """
        return self._extra_conf.get(key, getattr(self, key, default))


# ================= Test Class Definition =================


class TestRouterComponents(unittest.TestCase):
    def setUp(self):
        paddle.seed(2025)
        np.random.seed(2025)

    def test_fused_op_gradient(self):
        """
        Test the correctness of forward and backward gradients for the
        custom FusedGateDetachMatmul operator.
        """
        B, D_in, D_out = 4, 16, 8
        x = paddle.randn([B, D_in], dtype="float32")
        # FusedGateDetachMatmul.forward does w = w.T internally, so pass w as [D_out, D_in].
        # The op is equivalent to F.linear(x, w.T) = x @ w, i.e. x @ [D_in, D_out].
        w = paddle.randn([D_out, D_in], dtype="float32")

        x.stop_gradient = False
        w.stop_gradient = False

        # 1. Custom Operator Path
        y_custom = FusedGateDetachMatmul.apply(x, w)
        loss_custom = y_custom.sum()
        loss_custom.backward()
        x_grad_custom = x.grad.clone()
        w_grad_custom = w.grad.clone()

        x.clear_grad()
        w.clear_grad()

        # 2. Paddle Native Operator Path (Baseline)
        # FusedGateDetachMatmul(x, w) == F.linear(x, w.T) == x @ w (Paddle matmul convention)
        y_ref = F.linear(x, w.T)
        loss_ref = y_ref.sum()
        loss_ref.backward()

        # Verify numerical consistency
        np.testing.assert_allclose(
            y_custom.numpy(),
            y_ref.numpy(),
            rtol=1e-5,
            err_msg="Forward output mismatch",
        )
        np.testing.assert_allclose(
            x_grad_custom.numpy(),
            x.grad.numpy(),
            rtol=1e-5,
            err_msg="Input gradient mismatch",
        )
        np.testing.assert_allclose(
            w_grad_custom.numpy(),
            w.grad.numpy(),
            rtol=1e-5,
            err_msg="Weight gradient mismatch",
        )


class TestTopKRouter(unittest.TestCase):
    def setUp(self):
        self.config = MockTransformerConfig()

        # Mock the parallel state function to prevent errors during single-card testing.
        # Note: Adjust the patch path if get_context_parallel_world_size is imported differently.
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def test_initialization_modes(self):
        """
        Test that initialization behavior differs based on `topk_method`.
        """
        # Case 1: noaux_tc (Common mode for DeepEP)
        self.config.topk_method = "noaux_tc"
        router_tc = TopKRouter(self.config)
        # Verify that score correction bias is registered
        self.assertTrue(hasattr(router_tc, "e_score_correction_bias"))
        # Verify that expert_usage is initialized
        self.assertTrue(hasattr(router_tc, "expert_usage"))

        # Case 2: greedy (Standard mode)
        self.config.topk_method = "greedy"
        router_greedy = TopKRouter(self.config)
        # Verify that these attributes do NOT exist in greedy mode
        self.assertFalse(hasattr(router_greedy, "e_score_correction_bias"))
        self.assertFalse(hasattr(router_greedy, "expert_usage"))

    def test_call_topk_method_directly(self):
        """
        Directly test `_call_topk_method` to ensure it returns a tuple (gate, idx).
        This isolates the routing logic from the forward pass preprocessing
        and prevents unpacking errors.
        """
        router = TopKRouter(self.config)
        batch_size = 2
        seq_len = 5
        # Simulate Gates [B*S, E]
        gates = paddle.rand(
            [batch_size * seq_len, self.config.n_routed_experts]
        )

        # Test 1: noaux_tc
        res = router._call_topk_method(
            "noaux_tc", gates, k=2, n_group=1, topk_group=1
        )
        self.assertIsNotNone(
            res, "_call_topk_method returned None for 'noaux_tc'"
        )
        self.assertIsInstance(res, tuple, "Should return a tuple")
        self.assertEqual(len(res), 2, "Should return (top_gate, top_idx)")

        # Test 2: greedy
        res_greedy = router._call_topk_method("greedy", gates, k=2)
        self.assertIsNotNone(
            res_greedy, "_call_topk_method returned None for 'greedy'"
        )
        self.assertEqual(len(res_greedy), 2)

    def test_forward_shape_and_logic(self):
        """
        Test the input/output shapes of the forward pass and verify
        TopKRouter-specific return values (e.g., None for capacity).
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        batch_size = 2
        seq_len = 10
        hidden_size = self.config.hidden_size

        # Input must be 3D [B, S, H]
        hidden_states = paddle.randn([batch_size, seq_len, hidden_size])

        # Execute Forward
        outputs = router(hidden_states)

        # Ensure output is not None
        self.assertIsNotNone(outputs, "Forward returned None")

        (
            capacity,  # Should be None
            top_gate,
            top_idx,
            gates_masked,
            mask,
            token_priority,  # Should be None
            l_aux,
            l_zloss,
        ) = outputs

        # 1. Verify DeepEP/TopKRouter specific None return values
        self.assertIsNone(capacity, "Capacity should be None for TopKRouter")
        self.assertIsNone(
            token_priority, "Token priority should be None for TopKRouter"
        )

        # 2. Verify Shapes
        expected_tokens = batch_size * seq_len
        k = self.config.num_experts_per_tok
        n_experts = self.config.n_routed_experts

        self.assertEqual(top_gate.shape, [expected_tokens, k])
        self.assertEqual(top_idx.shape, [expected_tokens, k])
        self.assertEqual(mask.shape, [expected_tokens, n_experts])

        # 3. Verify Loss Calculation
        if self.config.router_aux_loss_coef > 0:
            self.assertIsNotNone(l_aux)
            self.assertEqual(l_aux.shape, [])  # Expecting a scalar

        if self.config.router_z_loss_coef > 0:
            self.assertIsNotNone(l_zloss)
            self.assertEqual(l_zloss.shape, [])  # Expecting a scalar

    def test_input_dimension_assertion(self):
        """
        Ensure the router raises ValueError for incorrect input dimensions
        (e.g., 2D tensors instead of 3D).
        """
        router = TopKRouter(self.config)
        # Input 2D [B*S, H] -> Should raise ValueError as TopKRouter strictly checks len(shape)==2
        hidden_states = paddle.randn([20, self.config.hidden_size])
        with self.assertRaises(ValueError):
            router(hidden_states)

    def test_expert_usage_update(self):
        """
        Verify that `expert_usage` is correctly updated when running in 'noaux_tc' mode.
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        # Initial state should be all zeros
        initial_usage = router.expert_usage.numpy().copy()
        self.assertEqual(initial_usage.sum(), 0)

        hidden_states = paddle.randn([2, 5, self.config.hidden_size])
        router(hidden_states)

        new_usage = router.expert_usage.numpy()

        # Usage sum should equal Total Tokens * K
        expected_hits = 2 * 5 * self.config.num_experts_per_tok
        self.assertEqual(new_usage.sum(), expected_hits)
        self.assertGreater(new_usage.sum(), initial_usage.sum())

    def test_greedy_no_usage_update(self):
        """
        Verify that `expert_usage` logic is ignored (attribute does not exist)
        when running in 'greedy' mode.
        """
        self.config.topk_method = "greedy"
        router = TopKRouter(self.config)

        hidden_states = paddle.randn([2, 5, self.config.hidden_size])

        # Run forward to ensure no errors occur due to accessing missing attributes
        outputs = router(hidden_states)
        self.assertIsNotNone(outputs)

        # Double check that the attribute still does not exist
        self.assertFalse(hasattr(router, "expert_usage"))

    def test_forward_with_input_ids(self):
        """Cover input_ids masking branches: mask zeroing and top_idx fill -1."""
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        batch_size, seq_len = 2, 4
        paddle.seed(42)
        hidden_states = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size]
        )
        # positions where input_ids==0 are padding tokens
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        _, top_gate, top_idx, gates_masked, mask, _, l_aux, _ = router(
            hidden_states, input_ids=input_ids
        )

        total = batch_size * seq_len
        k = self.config.num_experts_per_tok
        self.assertEqual(top_idx.shape, [total, k])

        # Padding positions (flat idx 2,3,7) must have top_idx==-1 and zero mask
        for p in [2, 3, 7]:
            self.assertTrue((top_idx[p] == -1).all().item())
            self.assertAlmostEqual(mask[p].sum().item(), 0.0)
            self.assertAlmostEqual(gates_masked[p].sum().item(), 0.0)

        # Valid positions must have non-negative expert indices
        for p in [0, 1, 4, 5, 6]:
            self.assertTrue((top_idx[p] >= 0).all().item())

    def test_forward_with_input_ids_seq_aux_loss(self):
        """Cover _cal_seq_aux_loss with input_ids (per-line valid token denom)."""
        self.config.topk_method = "noaux_tc"
        self.config.moe_router_load_balancing_type = "seq_aux_loss"
        router = TopKRouter(self.config)

        paddle.seed(42)
        hidden_states = paddle.randn([2, 4, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        _, _, _, _, _, _, l_aux, _ = router(hidden_states, input_ids=input_ids)
        self.assertIsNotNone(l_aux)
        self.assertEqual(l_aux.shape, [])

    def test_cal_seq_aux_loss_1d_input_ids(self):
        """Cover _cal_seq_aux_loss ndim==1 unsqueeze branch."""
        self.config.topk_method = "greedy"
        router = TopKRouter(self.config)

        seq_len = 4
        n_e = self.config.n_routed_experts
        k = self.config.num_experts_per_tok
        probs = paddle.rand([seq_len, n_e])
        routing_map = paddle.zeros([seq_len, n_e])
        for i in range(seq_len):
            for j in range(k):
                routing_map[i, j] = 1.0

        loss = router._cal_seq_aux_loss(
            probs,
            k,
            routing_map,
            seq_len,
            batch_size=1,
            input_ids=paddle.to_tensor([1, 2, 0, 0]),
        )
        self.assertEqual(loss.shape, [])

    def test_cal_seq_aux_loss_experimental_version(self):
        """Cover gpt_model_use_experimental_version=True branch in _cal_seq_aux_loss."""
        self.config.topk_method = "greedy"
        self.config.gpt_model_use_experimental_version = True
        self.config.num_nextn_predict_layers = 0
        router = TopKRouter(self.config)

        batch_size, seq_len = 2, 4
        n_e = self.config.n_routed_experts
        k = self.config.num_experts_per_tok
        paddle.seed(42)
        probs = paddle.rand([batch_size, seq_len, n_e])
        routing_map = paddle.zeros([batch_size * seq_len, n_e])
        for i in range(batch_size * seq_len):
            for j in range(k):
                routing_map[i, j] = 1.0

        loss = router._cal_seq_aux_loss(
            probs,
            k,
            routing_map,
            seq_len,
            batch_size=batch_size,
            input_ids=paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]]),
        )
        self.assertEqual(loss.shape, [])
        self.assertGreater(loss.item(), 0.0)


class TestScalingFactorInit(unittest.TestCase):
    """Test initialization behavior for routed_scaling_factor and routed_scaling_factor_learnable."""

    def setUp(self):
        self.config = MockTransformerConfig()
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    # ---- routed_scaling_factor (scalar) ----

    def test_routed_scaling_factor_default(self):
        """routed_scaling_factor=1.0 (default): stored as float, no learnable param."""
        router = TopKRouter(self.config)
        self.assertIsInstance(router.routed_scaling_factor, float)
        self.assertAlmostEqual(router.routed_scaling_factor, 1.0)
        self.assertFalse(hasattr(router, "routed_scaling_factor_param"))

    def test_routed_scaling_factor_float(self):
        """routed_scaling_factor=2.5 (float): stored as float."""
        self.config.routed_scaling_factor = 2.5
        router = TopKRouter(self.config)
        self.assertIsInstance(router.routed_scaling_factor, float)
        self.assertAlmostEqual(router.routed_scaling_factor, 2.5)
        self.assertFalse(hasattr(router, "routed_scaling_factor_param"))

    # ---- routed_scaling_factor_learnable ----

    def test_routed_scaling_factor_learnable_default_init(self):
        """routed_scaling_factor_learnable=True with default 1.0: creates Parameter of shape [num_experts], init 1.0."""
        self.config.routed_scaling_factor_learnable = True
        router = TopKRouter(self.config)
        self.assertTrue(hasattr(router, "routed_scaling_factor_param"))
        param = router.routed_scaling_factor_param
        self.assertIsInstance(param, paddle.Tensor)
        self.assertEqual(list(param.shape), [self.config.n_routed_experts])
        np.testing.assert_allclose(
            param.numpy(),
            np.ones(self.config.n_routed_experts, dtype="float32"),
        )
        self.assertFalse(param.stop_gradient)

    def test_routed_scaling_factor_learnable_custom_init(self):
        """routed_scaling_factor_learnable=True with routed_scaling_factor=2.5: Parameter initialized to 2.5."""
        self.config.routed_scaling_factor = 2.5
        self.config.routed_scaling_factor_learnable = True
        router = TopKRouter(self.config)
        param = router.routed_scaling_factor_param
        np.testing.assert_allclose(
            param.numpy(),
            np.full(self.config.n_routed_experts, 2.5, dtype="float32"),
            rtol=1e-5,
        )


class TestRoutedScalingFactorForward(unittest.TestCase):
    """Test that routed_scaling_factor correctly scales top_gate after top-k selection."""

    def setUp(self):
        self.config = MockTransformerConfig()
        self.config.topk_method = "greedy"
        self.config.norm_topk_prob = False
        self.config.router_aux_loss_coef = 0.0
        self.config.router_z_loss_coef = 0.0
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)
        paddle.seed(99)
        self.hidden = paddle.randn([2, 4, self.config.hidden_size])

    def _make_router(self, rsf, learnable=False):
        self.config.routed_scaling_factor = rsf
        self.config.routed_scaling_factor_learnable = learnable
        return TopKRouter(self.config)

    def test_scalar_one_is_noop(self):
        """routed_scaling_factor=1.0 (default): top_gate should be unchanged."""
        router = self._make_router(1.0)
        _, top_gate_1, _, _, _, _, _, _ = router(self.hidden)

        self.config.routed_scaling_factor = 1.0
        self.config.routed_scaling_factor_learnable = False
        router2 = TopKRouter(self.config)
        router2.weight.set_value(router.weight.clone())
        _, top_gate_2, _, _, _, _, _, _ = router2(self.hidden)

        np.testing.assert_allclose(
            top_gate_1.numpy(),
            top_gate_2.numpy(),
            atol=1e-5,
            err_msg="routed_scaling_factor=1.0 should be a no-op",
        )

    def test_float_scaling(self):
        """routed_scaling_factor=2.5: top_gate should be multiplied by 2.5."""
        router_1 = self._make_router(1.0)
        router_25 = self._make_router(2.5)
        router_25.weight.set_value(router_1.weight.clone())

        _, top_gate_1, _, _, _, _, _, _ = router_1(self.hidden)
        _, top_gate_25, _, _, _, _, _, _ = router_25(self.hidden)

        np.testing.assert_allclose(
            top_gate_25.numpy(),
            top_gate_1.numpy() * 2.5,
            rtol=1e-5,
            err_msg="routed_scaling_factor=2.5 should multiply top_gate by 2.5",
        )

    def test_learnable_scaling_init_equal_to_scalar(self):
        """routed_scaling_factor_learnable=True, init=2.5: at init equals scalar 2.5."""
        router_scalar = self._make_router(2.5, learnable=False)
        router_learn = self._make_router(2.5, learnable=True)
        router_learn.weight.set_value(router_scalar.weight.clone())

        _, top_gate_scalar, _, _, _, _, _, _ = router_scalar(self.hidden)
        _, top_gate_learn, _, _, _, _, _, _ = router_learn(self.hidden)

        np.testing.assert_allclose(
            top_gate_learn.numpy(),
            top_gate_scalar.numpy(),
            atol=1e-5,
            err_msg="learnable scales init=2.5 should give same result as scalar 2.5",
        )

    def test_probs_sparse_layout_consistency(self):
        """probs[S, E] should be 0 for non-selected experts and equal to top_gate for selected ones."""
        router = self._make_router(2.5)
        hidden = paddle.randn([1, 6, self.config.hidden_size])
        _, top_gate, top_idx, probs, mask, _, _, _ = router(hidden)

        num_tokens = 6
        k = self.config.num_experts_per_tok

        for t in range(num_tokens):
            for ki in range(k):
                expert_id = top_idx[t, ki].item()
                self.assertGreaterEqual(
                    expert_id, 0, "Expert index should be non-negative"
                )
                self.assertAlmostEqual(
                    probs[t, expert_id].item(),
                    top_gate[t, ki].item(),
                    places=5,
                    msg=f"probs[{t},{expert_id}] should equal top_gate[{t},{ki}]",
                )


class TestForwardOutputRenaming(unittest.TestCase):
    """Test that the return tuple at position 3 is 'probs' (formerly 'gates_masked')."""

    def setUp(self):
        self.config = MockTransformerConfig()
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def test_probs_output_shape_and_sparsity(self):
        """
        Return value at index 3 (probs) should have shape [S, E].
        Non-selected expert positions should be exactly 0.
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)
        batch_size, seq_len = 2, 5
        hidden = paddle.randn([batch_size, seq_len, self.config.hidden_size])

        (_, top_gate, top_idx, probs, mask, _, _, _) = router(hidden)

        num_tokens = batch_size * seq_len
        k = self.config.num_experts_per_tok
        n_e = self.config.n_routed_experts

        self.assertEqual(probs.shape, [num_tokens, n_e])

        # Count non-zero entries per token; should equal k (num_experts_per_tok)
        probs_np = probs.numpy()
        for t in range(num_tokens):
            nnz = np.count_nonzero(probs_np[t])
            self.assertEqual(
                nnz,
                k,
                f"Token {t}: expected {k} non-zero probs, got {nnz}",
            )

    def test_probs_consistent_with_mask(self):
        """
        probs non-zero positions should exactly match mask==1 positions.
        """
        self.config.topk_method = "greedy"
        self.config.norm_topk_prob = False
        self.config.router_aux_loss_coef = 0.0
        self.config.router_z_loss_coef = 0.0
        router = TopKRouter(self.config)

        hidden = paddle.randn([2, 4, self.config.hidden_size])
        (_, _, _, probs, mask, _, _, _) = router(hidden)

        probs_np = probs.numpy()
        mask_np = mask.numpy()

        # Where mask==0, probs must be 0
        np.testing.assert_array_equal(
            (probs_np != 0).astype(int),
            mask_np.astype(int),
            err_msg="probs non-zero pattern must match mask",
        )

    def test_forward_with_input_ids_probs_name(self):
        """
        With input_ids masking, padding tokens' probs row should be all zeros.
        This replaces the old gates_masked variable name check.
        """
        self.config.topk_method = "noaux_tc"
        router = TopKRouter(self.config)

        batch_size, seq_len = 2, 4
        paddle.seed(42)
        hidden = paddle.randn([batch_size, seq_len, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        _, top_gate, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )

        # Padding flat positions: 2, 3, 7
        for p in [2, 3, 7]:
            self.assertAlmostEqual(
                probs[p].sum().item(),
                0.0,
                places=5,
                msg=f"probs at padding position {p} should be 0",
            )

        # Valid flat positions must have non-zero probs rows
        for p in [0, 1, 4, 5, 6]:
            self.assertGreater(
                probs[p].sum().item(),
                0.0,
                msg=f"probs at valid position {p} should be non-zero",
            )


class TestCalZLoss(unittest.TestCase):
    """Unit tests for StandardMoERouter._cal_z_loss (new input_ids branch)."""

    def setUp(self):
        self.config = MockTransformerConfig()
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)
        self.config.topk_method = "greedy"
        self.config.router_aux_loss_coef = 0.0
        self.config.router_z_loss_coef = 0.01
        self.router = TopKRouter(self.config)

    def test_no_input_ids_mean_reduction(self):
        """Without input_ids, z_loss = logsumexp(logits,1).square().mean()."""
        paddle.seed(0)
        logits = paddle.randn([6, self.config.n_routed_experts])
        loss = self.router._cal_z_loss(logits)
        expected = paddle.logsumexp(logits, axis=1).square().mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_input_ids_excludes_padding(self):
        """With input_ids, padding tokens (id==0) contribute 0 to z_loss."""
        n_tokens = 6
        paddle.seed(2)
        logits = paddle.randn([n_tokens, self.config.n_routed_experts])
        # tokens 0-3 valid, tokens 4-5 padding → flat input_ids
        input_ids = paddle.to_tensor([[1, 2, 3, 4, 0, 0]])

        loss = self.router._cal_z_loss(logits, input_ids)
        self.assertEqual(loss.shape, [])
        # Full-mean (no masking) should differ
        loss_no_mask = self.router._cal_z_loss(logits)
        self.assertNotAlmostEqual(loss.item(), loss_no_mask.item(), places=4)

    def test_input_ids_all_valid_equals_masked_sum_over_count(self):
        """When all tokens are valid, z_loss with input_ids == sum / n_tokens."""
        n_tokens = 4
        paddle.seed(3)
        logits = paddle.randn([n_tokens, self.config.n_routed_experts])
        input_ids = paddle.to_tensor([[1, 2, 3, 4]])

        loss_ids = self.router._cal_z_loss(logits, input_ids)
        # Expected: sum / 4
        expected = logits.logsumexp(1).square().sum() / n_tokens
        self.assertAlmostEqual(loss_ids.item(), expected.item(), places=5)

    def test_experimental_version_adds_mtp_denom(self):
        """gpt_model_use_experimental_version=True uses denom + num_nextn_predict_layers * batch."""
        self.config.gpt_model_use_experimental_version = True
        self.config.num_nextn_predict_layers = 2
        router = TopKRouter(self.config)

        batch_size, seq_len = 2, 4
        paddle.seed(4)
        logits = paddle.randn(
            [batch_size * seq_len, self.config.n_routed_experts]
        )
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        loss = router._cal_z_loss(logits, input_ids)
        self.assertEqual(loss.shape, [])

        # Manually compute expected
        origin_mask = (input_ids != 0).astype(paddle.float32)
        loss_mask = origin_mask.reshape([-1])
        denom = origin_mask.sum() + origin_mask.shape[0] * 2
        expected = (
            logits.logsumexp(1).square() * loss_mask
        ).sum() / paddle.clip(denom, min=1e-6)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_experimental_version_without_mtp(self):
        """gpt_model_use_experimental_version=True, num_nextn_predict_layers=0: denom equals valid count."""
        self.config.gpt_model_use_experimental_version = True
        self.config.num_nextn_predict_layers = 0
        router = TopKRouter(self.config)

        batch_size, seq_len = 2, 4
        paddle.seed(5)
        logits = paddle.randn(
            [batch_size * seq_len, self.config.n_routed_experts]
        )
        input_ids = paddle.to_tensor([[1, 2, 3, 0], [5, 6, 7, 0]])

        loss = router._cal_z_loss(logits, input_ids)
        # With num_nextn_predict_layers=0, denom == valid token count == 6
        origin_mask = (input_ids != 0).astype(paddle.float32)
        loss_mask = origin_mask.reshape([-1])
        denom = origin_mask.sum()
        expected = (
            logits.logsumexp(1).square() * loss_mask
        ).sum() / paddle.clip(denom, min=1e-6)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_forward_z_loss_with_input_ids(self):
        """TopKRouter forward with z_loss coef > 0 and input_ids passes through correctly."""
        self.config.router_aux_loss_coef = 0.0
        self.config.router_z_loss_coef = 0.01
        router = TopKRouter(self.config)

        paddle.seed(42)
        hidden = paddle.randn([2, 4, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        _, _, _, _, _, _, l_aux, l_zloss = router(hidden, input_ids=input_ids)
        self.assertIsNone(l_aux)
        self.assertIsNotNone(l_zloss)
        self.assertEqual(l_zloss.shape, [])
        self.assertGreater(l_zloss.item(), 0.0)


class TestPadTokenId(unittest.TestCase):
    """Tests covering config.pad_token_id usage in TopKRouter / loss helpers."""

    def setUp(self):
        self.config = MockTransformerConfig()
        self.config.topk_method = "noaux_tc"
        patcher = patch(
            "paddleformers.fleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        self.mock_cp = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_router(self):
        return TopKRouter(self.config)

    def test_default_pad_token_id_zero_masks_zero_ids(self):
        """When pad_token_id defaults to 0, input_ids==0 are treated as padding."""
        self.config.pad_token_id = 0
        router = self._make_router()
        paddle.seed(7)
        hidden = paddle.randn([2, 4, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        _, _, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        for p in [2, 3, 7]:
            self.assertTrue((top_idx[p] == -1).all().item())
            self.assertAlmostEqual(mask[p].sum().item(), 0.0)
            self.assertAlmostEqual(probs[p].sum().item(), 0.0)
        for p in [0, 1, 4, 5, 6]:
            self.assertGreater(probs[p].sum().item(), 0.0)

    def test_custom_pad_token_id_masks_only_that_id(self):
        """Setting pad_token_id=99 masks tokens with id 99, not id 0."""
        self.config.pad_token_id = 99
        router = self._make_router()
        paddle.seed(7)
        hidden = paddle.randn([2, 4, self.config.hidden_size])
        # Token id 0 must NOT be masked; id 99 must be masked.
        input_ids = paddle.to_tensor([[1, 0, 99, 99], [2, 99, 3, 0]])

        _, _, top_idx, probs, mask, _, _, _ = router(
            hidden, input_ids=input_ids
        )
        # Padding flat positions where id == 99: 2, 3, 5
        for p in [2, 3, 5]:
            self.assertTrue((top_idx[p] == -1).all().item())
            self.assertAlmostEqual(probs[p].sum().item(), 0.0)
        # id == 0 positions (1, 7) are valid now.
        for p in [0, 1, 4, 6, 7]:
            self.assertGreater(probs[p].sum().item(), 0.0)

    def test_missing_pad_token_id_attr_falls_back_to_zero(self):
        """If config has no pad_token_id attribute, getattr fallback uses 0."""
        # MockTransformerConfig defines no pad_token_id by default.
        self.assertFalse(hasattr(self.config, "pad_token_id"))
        router = self._make_router()
        paddle.seed(7)
        hidden = paddle.randn([2, 4, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        # Should not raise, and id==0 should still be masked.
        _, _, top_idx, _, mask, _, _, _ = router(hidden, input_ids=input_ids)
        for p in [2, 3, 7]:
            self.assertTrue((top_idx[p] == -1).all().item())
            self.assertAlmostEqual(mask[p].sum().item(), 0.0)

    def test_none_pad_token_id_falls_back_to_zero(self):
        """If config.pad_token_id is None at runtime, fallback uses 0 instead of erroring."""
        self.config.pad_token_id = None
        router = self._make_router()
        paddle.seed(7)
        hidden = paddle.randn([2, 4, self.config.hidden_size])
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 4, 5, 0]])

        # `input_ids == None` would otherwise return Python bool and crash;
        # the defensive fallback must convert None to 0.
        _, _, top_idx, _, mask, _, _, _ = router(hidden, input_ids=input_ids)
        for p in [2, 3, 7]:
            self.assertTrue((top_idx[p] == -1).all().item())
            self.assertAlmostEqual(mask[p].sum().item(), 0.0)

    def test_cal_seq_aux_loss_uses_pad_token_id(self):
        """_cal_seq_aux_loss masks tokens equal to config.pad_token_id."""
        self.config.pad_token_id = 7
        self.config.topk_method = "greedy"
        router = self._make_router()

        seq_len = 4
        n_e = self.config.n_routed_experts
        k = self.config.num_experts_per_tok
        probs = paddle.rand([seq_len, n_e])
        routing_map = paddle.zeros([seq_len, n_e])
        for i in range(seq_len):
            for j in range(k):
                routing_map[i, j] = 1.0

        # All tokens valid (none equal to 7) → loss should be finite and > 0.
        loss_no_pad = router._cal_seq_aux_loss(
            probs,
            k,
            routing_map,
            seq_len,
            batch_size=1,
            input_ids=paddle.to_tensor([1, 2, 3, 4]),
        )
        # With token id 7 marked as padding, denom shrinks.
        loss_with_pad = router._cal_seq_aux_loss(
            probs,
            k,
            routing_map,
            seq_len,
            batch_size=1,
            input_ids=paddle.to_tensor([1, 2, 7, 7]),
        )
        self.assertEqual(loss_no_pad.shape, [])
        self.assertEqual(loss_with_pad.shape, [])
        # The two losses must differ (different effective denominator).
        self.assertNotAlmostEqual(
            loss_no_pad.item(), loss_with_pad.item(), places=4
        )

    def test_cal_z_loss_uses_pad_token_id(self):
        """_cal_z_loss masks tokens equal to config.pad_token_id."""
        self.config.pad_token_id = 9
        self.config.topk_method = "greedy"
        self.config.router_aux_loss_coef = 0.0
        self.config.router_z_loss_coef = 0.01
        router = self._make_router()

        paddle.seed(11)
        logits = paddle.randn([4, self.config.n_routed_experts])
        # 2 tokens valid, 2 tokens are padding (id == 9).
        input_ids = paddle.to_tensor([[1, 2, 9, 9]])
        loss = router._cal_z_loss(logits, input_ids)

        loss_mask = (input_ids != 9).astype(paddle.float32).reshape([-1])
        denom = (input_ids != 9).astype(paddle.float32).sum()
        expected = (
            logits.logsumexp(1).square() * loss_mask
        ).sum() / paddle.clip(denom, min=1e-6)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


if __name__ == "__main__":
    unittest.main()
