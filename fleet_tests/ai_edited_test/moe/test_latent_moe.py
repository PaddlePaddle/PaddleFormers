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

"""
Unit tests for Latent MoE feature.

Covers new code in:
  - src/paddlefleet/transformer/transformer_config.py  (moe_latent_size)
  - src/paddlefleet/transformer/moe/moe_layer.py       (__init__, dispatch_preprocess,
                                                         aux_loss_compute, forward)

Target: >90% line coverage of all newly added lines.
"""

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
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_moe_config(**overrides):
    """Helper: create a TransformerConfig with sensible MoE defaults."""
    from paddleformers.fleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "gated_linear_unit": True,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "moe_token_dispatcher_type": "alltoall",
        "moe_use_fusion_node": False,
        "moe_expert_fusion": False,
        "moe_deep_gemm": False,
        "moe_ep_barrier": True,
        "fp8": None,
        "fp8_wgrad": True,
        "using_sonic_moe": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "n_shared_experts": 0,
        "moe_shared_expert_overlap": False,
        "recompute_granularity": None,
        "recompute_modules": [],
        "use_bias": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_pg_collection():
    pg = MagicMock()
    pg.ep = MagicMock()
    pg.ep.world_size = 1
    pg.expt_dp = MagicMock()
    pg.tp = MagicMock()
    pg.tp.size.return_value = 1
    pg.cp = MagicMock()
    pg.cp.rank.return_value = 0
    pg.cp.size.return_value = 1
    return pg


def _make_moe_sublayers():
    from paddleformers.fleet.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
    from paddleformers.fleet.transformer.moe.moe_layer import MoESublayers

    return MoESublayers(
        mlp_spec=MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            hidden_act=None,
            down_proj=RowParallelLinear,
        )
    )


def _make_moe_layer(config, pg_collection=None):
    """Instantiate a real MoELayer with standard mocks for heavy deps."""
    from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

    if pg_collection is None:
        pg_collection = _make_pg_collection()
    sublayers = _make_moe_sublayers()
    return MoELayer(config, sublayers=sublayers, pg_collection=pg_collection)


# ---------------------------------------------------------------------------
# 1. TransformerConfig – new fields
# ---------------------------------------------------------------------------


class TestLatentMoEConfig(unittest.TestCase):
    """Verify that the latent MoE config field has the correct defaults."""

    def test_moe_latent_size_defaults_none(self):
        config = _make_moe_config()
        self.assertIsNone(config.moe_latent_size)

    def test_set_positive_moe_latent_size(self):
        config = _make_moe_config(moe_latent_size=32)
        self.assertEqual(config.moe_latent_size, 32)

    def test_set_non_positive_moe_latent_size(self):
        config = _make_moe_config(moe_latent_size=0)
        self.assertEqual(config.moe_latent_size, 0)


# ---------------------------------------------------------------------------
# 2. MoELayer.__init__ – latent projection initialization
# ---------------------------------------------------------------------------


def _rng_tracker_ctx():
    """Return a context manager factory that mocks the CUDA RNG tracker fork."""
    import contextlib

    mock_tracker = MagicMock()
    mock_tracker.fork.return_value = contextlib.nullcontext()
    return mock_tracker


class TestLatentMoEInit(unittest.TestCase):
    """Test MoELayer.__init__ branches for latent MoE setup."""

    def _run_init(self, config):
        """Instantiate MoELayer with full GPU-level mocks."""
        with (
            patch(
                "paddleformers.fleet.tensor_parallel.random.get_cuda_rng_tracker",
                return_value=_rng_tracker_ctx(),
            ),
            patch(
                "paddleformers.fleet.tensor_parallel.layers.get_cuda_rng_tracker",
                return_value=_rng_tracker_ctx(),
            ),
        ):
            return _make_moe_layer(config)

    def test_latent_moe_disabled_by_default(self):
        """Default config has no projection layers."""
        layer = self._run_init(_make_moe_config())
        self.assertFalse(layer.use_latent_moe)
        self.assertFalse(hasattr(layer, "fc1_latent_proj"))
        self.assertFalse(hasattr(layer, "fc2_latent_proj"))

    def test_latent_moe_disabled_when_size_is_none(self):
        """moe_latent_size=None disables latent MoE."""
        config = _make_moe_config(moe_latent_size=None)
        layer = self._run_init(config)
        self.assertFalse(layer.use_latent_moe)
        self.assertFalse(hasattr(layer, "fc1_latent_proj"))

    def test_latent_moe_disabled_when_size_is_zero(self):
        """moe_latent_size=0 disables latent MoE."""
        config = _make_moe_config(moe_latent_size=0)
        layer = self._run_init(config)
        self.assertFalse(layer.use_latent_moe)
        self.assertFalse(hasattr(layer, "fc1_latent_proj"))

    def test_latent_moe_disabled_when_size_is_negative(self):
        """moe_latent_size<0 disables latent MoE."""
        config = _make_moe_config(moe_latent_size=-1)
        layer = self._run_init(config)
        self.assertFalse(layer.use_latent_moe)
        self.assertFalse(hasattr(layer, "fc1_latent_proj"))

    def test_latent_moe_enabled(self):
        """moe_latent_size=32 creates latent projection layers."""
        config = _make_moe_config(moe_latent_size=32)
        layer = self._run_init(config)
        self.assertTrue(layer.use_latent_moe)
        # fc1: hidden_size → latent_size
        self.assertIsInstance(layer.fc1_latent_proj, nn.Linear)
        self.assertEqual(
            layer.fc1_latent_proj.weight.shape[0], 64
        )  # in_features
        self.assertEqual(
            layer.fc1_latent_proj.weight.shape[1], 32
        )  # out_features
        # fc2: latent_size → hidden_size
        self.assertIsInstance(layer.fc2_latent_proj, nn.Linear)
        self.assertEqual(layer.fc2_latent_proj.weight.shape[0], 32)
        self.assertEqual(layer.fc2_latent_proj.weight.shape[1], 64)

    def test_latent_moe_sets_expert_hidden_size(self):
        """Enabling latent MoE must reduce the expert input hidden_size."""
        config = _make_moe_config(moe_latent_size=32)
        layer = self._run_init(config)
        # Experts must operate on latent_size, not hidden_size
        for expert in layer.experts:
            if expert is not None:
                # The first linear in each expert receives latent_size features
                first_linear = None
                for sub in expert.sublayers():
                    if isinstance(sub, (nn.Linear, paddle.nn.Linear)):
                        first_linear = sub
                        break
                if first_linear is not None:
                    self.assertEqual(
                        first_linear.weight.shape[0],
                        32,
                        "Expert input must equal moe_latent_size=32",
                    )
                break  # Only check the first present expert

    def test_latent_moe_shared_experts_use_original_hidden_size(self):
        """shared_experts must use original hidden_size, not moe_latent_size."""
        config = _make_moe_config(
            moe_latent_size=32,
            n_shared_experts=2,
        )
        layer = self._run_init(config)
        # shared_experts should NOT be affected by latent MoE
        # It must still use hidden_size=64, not moe_latent_size=32
        self.assertIsNotNone(layer.shared_experts)
        # Check that shared_experts operates on hidden_size=64
        for sub in layer.shared_experts.sublayers():
            if isinstance(sub, (nn.Linear, paddle.nn.Linear)):
                self.assertEqual(
                    sub.weight.shape[1],
                    64,
                    "shared_experts input must equal hidden_size=64, not moe_latent_size=32",
                )
                break


# ---------------------------------------------------------------------------
# 3. dispatch_preprocess – latent projection before token dispatch
# ---------------------------------------------------------------------------


class TestDispatchPreprocessLatent(unittest.TestCase):
    """Test the latent MoE branch in dispatch_preprocess."""

    def _make_stub(self, use_latent_moe, hidden_size=64, latent_size=32):
        """Build a minimal layer-like object for unbound method testing."""
        stub = MagicMock()
        stub.use_latent_moe = use_latent_moe
        if use_latent_moe:
            stub.fc1_latent_proj = nn.Linear(hidden_size, latent_size)
        # Make token_dispatcher pass isinstance check by patching the class to `object`
        stub.token_dispatcher = MagicMock()
        stub.token_dispatcher.dispatch_preprocess_overlap.return_value = (
            paddle.randn([4, latent_size if use_latent_moe else hidden_size])
        )
        stub.token_dispatcher._comm_manager.token_probs = paddle.ones([4, 2])
        stub.token_dispatcher._comm_manager.token_indices = paddle.zeros(
            [4, 2], dtype="int64"
        )
        return stub

    def test_dispatch_preprocess_applies_fc1_when_latent(self):
        """fc1_latent_proj must compress hidden_states before dispatch."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub = self._make_stub(use_latent_moe=True)
        hidden = paddle.randn([4, 64])
        probs = paddle.ones([4, 2])
        indices = paddle.zeros([4, 2], dtype="int64")

        with patch(
            "paddleformers.fleet.transformer.moe.moe_layer.MoEFlexTokenDispatcher",
            new=object,
        ):
            result = MoELayer.dispatch_preprocess(
                stub, (hidden, probs, indices)
            )

        # dispatch_preprocess_overlap receives tensor of latent_size=32
        call_args = stub.token_dispatcher.dispatch_preprocess_overlap.call_args
        dispatched_hidden = call_args[0][0]
        self.assertEqual(dispatched_hidden.shape[-1], 32)

    def test_dispatch_preprocess_skips_fc1_when_not_latent(self):
        """Without latent MoE, hidden_states must pass unchanged to dispatch."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub = self._make_stub(use_latent_moe=False)
        hidden = paddle.randn([4, 64])
        probs = paddle.ones([4, 2])
        indices = paddle.zeros([4, 2], dtype="int64")

        with patch(
            "paddleformers.fleet.transformer.moe.moe_layer.MoEFlexTokenDispatcher",
            new=object,
        ):
            MoELayer.dispatch_preprocess(stub, (hidden, probs, indices))

        call_args = stub.token_dispatcher.dispatch_preprocess_overlap.call_args
        dispatched_hidden = call_args[0][0]
        # Shape must still be hidden_size=64
        self.assertEqual(dispatched_hidden.shape[-1], 64)


# ---------------------------------------------------------------------------
# 4. aux_loss_compute – latent projection after combine
# ---------------------------------------------------------------------------


class TestAuxLossComputeLatent(unittest.TestCase):
    """Test the latent MoE branch in aux_loss_compute."""

    def _make_stub(self, use_latent_moe, hidden_size=64, latent_size=32):
        stub = MagicMock()
        stub.use_latent_moe = use_latent_moe
        if use_latent_moe:
            stub.fc2_latent_proj = nn.Linear(latent_size, hidden_size)
        stub.training = False
        stub.router_aux_loss_coef = 0.0
        stub.shared_experts = None
        stub.expert_model_parallel_size = 1
        stub.sequence_parallel = False
        return stub

    def test_aux_loss_compute_applies_fc2_when_latent(self):
        """fc2_latent_proj must expand hidden_states back to hidden_size."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub = self._make_stub(use_latent_moe=True)
        # Input to aux_loss_compute is in latent space (32-dim)
        hidden = paddle.randn([4, 32])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.zeros([1])
        z_loss = None

        output = MoELayer.aux_loss_compute(
            stub, (hidden, aux_loss, z_loss, residuals)
        )

        # Must be expanded back to hidden_size=64
        self.assertEqual(output.shape[-1], 64)

    def test_aux_loss_compute_skips_fc2_when_not_latent(self):
        """Without latent MoE, hidden_states must remain unchanged."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub = self._make_stub(use_latent_moe=False)
        hidden = paddle.randn([4, 64])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.zeros([1])
        z_loss = None

        output = MoELayer.aux_loss_compute(
            stub, (hidden, aux_loss, z_loss, residuals)
        )

        self.assertEqual(output.shape[-1], 64)


# ---------------------------------------------------------------------------
# 4b. z_loss handling in aux_loss_compute
# ---------------------------------------------------------------------------


class TestAuxLossComputeZLoss(unittest.TestCase):
    """Test z_loss branches in aux_loss_compute added by commit b0bbb0a."""

    def _make_stub(self, training, router_aux_loss_coef=0.0, hidden_size=64):
        stub = MagicMock()
        stub.use_latent_moe = False
        stub.training = training
        stub.router_aux_loss_coef = router_aux_loss_coef
        stub.shared_experts = None
        stub.expert_model_parallel_size = 1
        stub.sequence_parallel = False
        return stub

    def test_z_loss_applied_when_training_and_not_none(self):
        """z_loss should be added via AddAuxiliaryLoss when training=True and z_loss is not None."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        stub = self._make_stub(training=True)
        hidden = paddle.randn([4, 64])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.zeros([1])
        z_loss = paddle.to_tensor([0.5], dtype="float32")

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.aux_loss_compute(
                stub, (hidden, aux_loss, z_loss, residuals)
            )
            # AddAuxiliaryLoss.apply should be called once for z_loss
            # (aux_loss path is skipped because router_aux_loss_coef=0.0)
            mock_apply.assert_called_once()
            _, call_loss = mock_apply.call_args[0]
            self.assertTrue(
                paddle.equal_all(call_loss, z_loss),
                "AddAuxiliaryLoss must be called with z_loss",
            )

    def test_z_loss_skipped_when_not_training(self):
        """z_loss should NOT be applied when training=False, even if z_loss is not None."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        stub = self._make_stub(training=False)
        hidden = paddle.randn([4, 64])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.zeros([1])
        z_loss = paddle.to_tensor([0.5], dtype="float32")

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.aux_loss_compute(
                stub, (hidden, aux_loss, z_loss, residuals)
            )
            mock_apply.assert_not_called()

    def test_z_loss_skipped_when_none(self):
        """z_loss=None should not trigger AddAuxiliaryLoss even in training mode."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        stub = self._make_stub(training=True)
        hidden = paddle.randn([4, 64])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.zeros([1])
        z_loss = None

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.aux_loss_compute(
                stub, (hidden, aux_loss, z_loss, residuals)
            )
            mock_apply.assert_not_called()

    def test_z_loss_and_aux_loss_both_applied(self):
        """When both aux_loss and z_loss are active, AddAuxiliaryLoss is called twice."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        stub = self._make_stub(training=True, router_aux_loss_coef=0.01)
        hidden = paddle.randn([4, 64])
        residuals = paddle.randn([4, 64])
        aux_loss = paddle.to_tensor([1.0], dtype="float32")
        z_loss = paddle.to_tensor([0.5], dtype="float32")

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.aux_loss_compute(
                stub, (hidden, aux_loss, z_loss, residuals)
            )
            self.assertEqual(
                mock_apply.call_count,
                2,
                "AddAuxiliaryLoss must be called for both aux_loss and z_loss",
            )
            # Second call should be for z_loss
            _, second_call_loss = mock_apply.call_args_list[1][0]
            self.assertTrue(
                paddle.equal_all(second_call_loss, z_loss),
                "Second AddAuxiliaryLoss call must use z_loss",
            )


# ---------------------------------------------------------------------------
# 5. forward – latent projections in the non-overlap single-card path
# ---------------------------------------------------------------------------


class TestForwardLatent(unittest.TestCase):
    """Test latent projections inside MoELayer.forward (single-card path)."""

    def _make_forward_stub(
        self, use_latent_moe, hidden_size=64, latent_size=32
    ):
        """
        Minimal stub for forward().  Mocks gate + expert computation so we can
        observe whether fc1/fc2 projections are applied.
        """
        num_experts = 4
        topk = 2
        bs, seq = 2, 6
        expert_out_size = latent_size if use_latent_moe else hidden_size

        stub = MagicMock()
        stub.use_latent_moe = use_latent_moe
        if use_latent_moe:
            stub.fc1_latent_proj = nn.Linear(hidden_size, latent_size)
            stub.fc2_latent_proj = nn.Linear(latent_size, hidden_size)

        # Parallel / sequence flags → single-card non-sequence path
        stub.expert_model_parallel_size = 1
        stub.sequence_parallel = False
        stub.moe_expert_fusion = False
        stub.shared_experts = None
        stub.moe_shared_expert_overlap = False
        stub.moe_use_fusion_node = False
        stub.training = False
        stub.router_aux_loss_coef = 0.0

        # gate returns: (capacity, topk_weights, topk_indices, gates_masked, mask,
        #                priorities, aux_loss, z_loss)
        topk_weights = paddle.ones([bs * seq, topk]) / topk
        topk_indices = paddle.zeros([bs * seq, topk], dtype="int64")
        gates_masked = paddle.ones([bs * seq, num_experts]) / num_experts
        mask = paddle.ones([bs * seq, num_experts], dtype="bool")
        aux_loss = paddle.zeros([1])
        z_loss = paddle.zeros([1])
        stub.gate.return_value = (
            None,
            topk_weights,
            topk_indices,
            gates_masked,
            mask,
            None,
            aux_loss,
            z_loss,
        )

        # _forward_single_card_moe returns a tensor in latent or hidden space
        stub._forward_single_card_moe.return_value = paddle.randn(
            [bs * seq, expert_out_size]
        )

        return stub, bs, seq

    def test_forward_with_latent_moe_applies_both_projections(self):
        """
        With latent MoE enabled:
          - hidden_states must be projected to latent_size before expert computation
          - output must be projected back to hidden_size after expert computation
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub, bs, seq = self._make_forward_stub(use_latent_moe=True)
        hidden = paddle.randn([bs, seq, 64])

        output, _ = MoELayer.forward(stub, hidden)

        # fc1 was called: _forward_single_card_moe received latent-size input
        call_args = stub._forward_single_card_moe.call_args[0]
        dispatched_input = call_args[0]
        self.assertEqual(
            dispatched_input.shape[-1],
            32,
            "Expert input must be latent_size=32",
        )

        # fc2 was called: final output must be hidden_size=64
        self.assertEqual(
            output.shape,
            [bs, seq, 64],
            "Output must be restored to hidden_size=64",
        )

    def test_forward_without_latent_moe_skips_projections(self):
        """
        Without latent MoE, hidden_states must flow through at full hidden_size.
        """
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub, bs, seq = self._make_forward_stub(use_latent_moe=False)
        hidden = paddle.randn([bs, seq, 64])

        output, _ = MoELayer.forward(stub, hidden)

        # Expert received full hidden_size input
        call_args = stub._forward_single_card_moe.call_args[0]
        dispatched_input = call_args[0]
        self.assertEqual(dispatched_input.shape[-1], 64)
        self.assertEqual(output.shape, [bs, seq, 64])

    def _make_forward_stub_for_zloss(
        self, training, z_loss_val, router_aux_loss_coef=0.0
    ):
        """Build a stub for forward() focused on z_loss testing."""
        num_experts = 4
        topk = 2
        bs, seq = 2, 6
        hidden_size = 64

        stub = MagicMock()
        stub.use_latent_moe = False
        stub.expert_model_parallel_size = 1
        stub.sequence_parallel = False
        stub.moe_expert_fusion = False
        stub.shared_experts = None
        stub.moe_shared_expert_overlap = False
        stub.moe_use_fusion_node = False
        stub.training = training
        stub.router_aux_loss_coef = router_aux_loss_coef

        topk_weights = paddle.ones([bs * seq, topk]) / topk
        topk_indices = paddle.zeros([bs * seq, topk], dtype="int64")
        gates_masked = paddle.ones([bs * seq, num_experts]) / num_experts
        mask = paddle.ones([bs * seq, num_experts], dtype="bool")
        aux_loss = paddle.zeros([1])
        stub.gate.return_value = (
            None,
            topk_weights,
            topk_indices,
            gates_masked,
            mask,
            None,
            aux_loss,
            z_loss_val,
        )
        stub._forward_single_card_moe.return_value = paddle.randn(
            [bs * seq, hidden_size]
        )
        return stub, bs, seq

    def test_forward_applies_z_loss_when_training(self):
        """In forward(), z_loss must be applied via AddAuxiliaryLoss when training=True."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        z_loss = paddle.to_tensor([0.5], dtype="float32")
        stub, bs, seq = self._make_forward_stub_for_zloss(
            training=True, z_loss_val=z_loss
        )
        hidden = paddle.randn([bs, seq, 64])

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.forward(stub, hidden)
            # aux_loss path skipped (coef=0.0), only z_loss path fires
            mock_apply.assert_called_once()
            _, call_loss = mock_apply.call_args[0]
            self.assertTrue(
                paddle.equal_all(call_loss, z_loss),
                "forward() must call AddAuxiliaryLoss with z_loss",
            )

    def test_forward_skips_z_loss_when_not_training(self):
        """In forward(), z_loss must NOT be applied when training=False."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        z_loss = paddle.to_tensor([0.5], dtype="float32")
        stub, bs, seq = self._make_forward_stub_for_zloss(
            training=False, z_loss_val=z_loss
        )
        hidden = paddle.randn([bs, seq, 64])

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.forward(stub, hidden)
            mock_apply.assert_not_called()

    def test_forward_skips_z_loss_when_none(self):
        """In forward(), z_loss=None must not trigger AddAuxiliaryLoss even in training."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        stub, bs, seq = self._make_forward_stub_for_zloss(
            training=True, z_loss_val=None
        )
        hidden = paddle.randn([bs, seq, 64])

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.forward(stub, hidden)
            mock_apply.assert_not_called()

    def test_forward_applies_both_aux_loss_and_z_loss(self):
        """In forward(), both aux_loss and z_loss should be applied when training=True."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            AddAuxiliaryLoss,
            MoELayer,
        )

        z_loss = paddle.to_tensor([0.5], dtype="float32")
        stub, bs, seq = self._make_forward_stub_for_zloss(
            training=True, z_loss_val=z_loss, router_aux_loss_coef=0.01
        )
        hidden = paddle.randn([bs, seq, 64])

        with patch.object(
            AddAuxiliaryLoss, "apply", side_effect=lambda x, _loss: x
        ) as mock_apply:
            MoELayer.forward(stub, hidden)
            self.assertEqual(
                mock_apply.call_count,
                2,
                "forward() must call AddAuxiliaryLoss for both aux_loss and z_loss",
            )
            _, second_call_loss = mock_apply.call_args_list[1][0]
            self.assertTrue(
                paddle.equal_all(second_call_loss, z_loss),
                "Second call must use z_loss",
            )


# ---------------------------------------------------------------------------
# 6. custom_forward – latent projections in the non-overlap multi-EP path
# ---------------------------------------------------------------------------


class TestCustomForwardLatent(unittest.TestCase):
    """Test the latent MoE branches in custom_forward."""

    def _make_stub(self, use_latent_moe, hidden_size=64, latent_size=32):
        bs_seq = 8
        expert_out_size = latent_size if use_latent_moe else hidden_size
        stub = MagicMock()
        stub.use_latent_moe = use_latent_moe
        if use_latent_moe:
            stub.fc1_latent_proj = nn.Linear(hidden_size, latent_size)
            stub.fc2_latent_proj = nn.Linear(latent_size, hidden_size)
        stub.dispatch.return_value = (
            paddle.randn([bs_seq, expert_out_size]),
            None,
        )
        stub.routed_experts_compute.return_value = paddle.randn(
            [bs_seq, expert_out_size]
        )
        stub.combine.return_value = paddle.randn([bs_seq, expert_out_size])
        return stub, bs_seq

    def test_custom_forward_applies_fc1_fc2_when_latent(self):
        """fc1 compresses input before dispatch; fc2 expands output after combine."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub, bs_seq = self._make_stub(use_latent_moe=True)
        hidden = paddle.randn([bs_seq, 64])
        output = MoELayer.custom_forward(stub, hidden, MagicMock(), MagicMock())
        dispatch_input = stub.dispatch.call_args[0][0]
        self.assertEqual(
            dispatch_input.shape[-1], 32, "dispatch must receive latent_size=32"
        )
        self.assertEqual(
            output.shape[-1], 64, "output must be restored to hidden_size=64"
        )

    def test_custom_forward_skips_projections_when_not_latent(self):
        """Without latent MoE, hidden_states flow unchanged at full hidden_size."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        stub, bs_seq = self._make_stub(use_latent_moe=False)
        hidden = paddle.randn([bs_seq, 64])
        output = MoELayer.custom_forward(stub, hidden, MagicMock(), MagicMock())
        dispatch_input = stub.dispatch.call_args[0][0]
        self.assertEqual(dispatch_input.shape[-1], 64)
        self.assertEqual(output.shape[-1], 64)


# ---------------------------------------------------------------------------
# 7. fusion_moe_forward – latent projections in the fusion multi-EP path
# ---------------------------------------------------------------------------


class TestFusionMoeForwardLatent(unittest.TestCase):
    """Test the latent MoE branches in fusion_moe_forward."""

    def _make_stub(self, use_latent_moe, hidden_size=64, latent_size=32):
        bs_seq = 8
        expert_out_size = latent_size if use_latent_moe else hidden_size
        stub = MagicMock()
        stub.use_latent_moe = use_latent_moe
        if use_latent_moe:
            stub.fc1_latent_proj = nn.Linear(hidden_size, latent_size)
            stub.fc2_latent_proj = nn.Linear(latent_size, hidden_size)
        stub.using_sonic_moe = False
        stub.fp8 = False
        stub.moe_deep_gemm = False
        stub.moe_expert_fusion = False
        stub.recompute_moe_gate_up = False
        stub.recompute_moe_premute = False
        stub.fp8_wgrad = True
        stub.dispatch.return_value = (
            paddle.randn([bs_seq, expert_out_size]),
            None,
        )
        stub.token_dispatcher._comm_manager.combine.return_value = paddle.randn(
            [bs_seq, expert_out_size]
        )
        return stub, bs_seq

    def test_fusion_moe_forward_applies_fc1_fc2_when_latent(self):
        """fc1 compresses input before dispatch; fc2 expands output after combine."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            FusionMoePyLayer,
            MoELayer,
        )

        stub, bs_seq = self._make_stub(use_latent_moe=True)
        hidden = paddle.randn([bs_seq, 64])
        with patch.object(
            FusionMoePyLayer, "apply", return_value=paddle.randn([bs_seq, 32])
        ):
            output = MoELayer.fusion_moe_forward(
                stub, hidden, MagicMock(), MagicMock(), None
            )
        dispatch_input = stub.dispatch.call_args[0][0]
        self.assertEqual(
            dispatch_input.shape[-1], 32, "dispatch must receive latent_size=32"
        )
        self.assertEqual(
            output.shape[-1], 64, "output must be restored to hidden_size=64"
        )

    def test_fusion_moe_forward_skips_projections_when_not_latent(self):
        """Without latent MoE, hidden_states flow unchanged at full hidden_size."""
        from paddleformers.fleet.transformer.moe.moe_layer import (
            FusionMoePyLayer,
            MoELayer,
        )

        stub, bs_seq = self._make_stub(use_latent_moe=False)
        hidden = paddle.randn([bs_seq, 64])
        with patch.object(
            FusionMoePyLayer, "apply", return_value=paddle.randn([bs_seq, 64])
        ):
            output = MoELayer.fusion_moe_forward(
                stub, hidden, MagicMock(), MagicMock(), None
            )
        dispatch_input = stub.dispatch.call_args[0][0]
        self.assertEqual(dispatch_input.shape[-1], 64)
        self.assertEqual(output.shape[-1], 64)


if __name__ == "__main__":
    unittest.main()
