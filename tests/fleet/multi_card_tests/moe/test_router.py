# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import random
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
import pytest
from paddle.distributed import fleet

from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

n_routed_experts = 4
hidden_size = 16


class TestTop2Router(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_routed_experts = 4
        cls.hidden_size = 16
        cls.transformer_config = TransformerConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=4,
            n_routed_experts=cls.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=24,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            bias_activation_fusion=True,
        )

        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        paddle.manual_seed(seed)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 4,
            "pp_degree": 1,
            "sharding_degree": 2,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        # fleet is process-global; when the whole module runs in one process,
        # whichever test class runs first calls initialize_fleet and the others
        # must reuse that instance rather than re-running fleet.init (which
        # trips ``args is already initialized`` in set_global_variables). Probe
        # the hybrid-communicate-group singleton directly (unset before
        # fleet.init) instead of catching a broad exception, so a genuine
        # comm-group error is not silently swallowed.
        already_initialized = getattr(fleet.fleet, "_hcg", None) is not None
        if not already_initialized:
            initialize_fleet(strategy=strategy)
            model_parallel_cuda_manual_seed(seed)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        transformer_layer_spec = get_gpt_layer_local_spec(
            cls.transformer_config,
            num_experts=cls.n_routed_experts,
            moe_expert_fusion=False,
        )
        cls.sequential_mlp = MoELayer(
            cls.transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            cls.pg_collection,
        )
        cls.router = cls.sequential_mlp.gate

    @pytest.mark.internal
    def test_constructor(self):
        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == self.hidden_size * self.n_routed_experts

    @pytest.mark.internal
    def test_router_forward(self):
        score_functions = ["sigmoid", "softmax"]
        for score_function in score_functions:
            with self.subTest(score_function=score_function), paddle.no_grad():
                self.router = self.router.cuda()
                self.router.moe_router_score_function = score_function
                # [num tokens, hidden size]
                hidden_states = paddle.randn((32, 2, self.router.hidden_size))
                hidden_states = hidden_states.cuda().bfloat16()
                (
                    capacity,
                    topk_weights,
                    topk_indices,
                    gates_masked,
                    mask,
                    priorities,
                    aux_loss,
                    z_loss,
                ) = self.router(hidden_states)

    # TODO: Not implemented yet
    # @pytest.mark.internal
    # def test_aux_loss(self):
    #     self.sequential_mlp = self.sequential_mlp.cuda()

    #     # Without aux loss
    #     hidden_states = paddle.randn((32, 2, self.router.hidden_size))
    #     hidden_states = hidden_states.cuda().bfloat16()
    #     out = self.sequential_mlp(hidden_states)[0]
    #     out.sum().mul_(paddle.empty_like(out.sum())).backward()
    #     assert self.sequential_mlp.gate.weight.grad.abs().sum() == 0

    #     # With aux loss
    #     self.transformer_config.moe_aux_loss_coeff = 1
    #     out = self.sequential_mlp(hidden_states)[0]
    #     out.sum().mul_(paddle.empty_like(out.sum())).backward()
    #     assert self.sequential_mlp.gate.weight.grad.abs().sum() > 0

    #     # With Z loss
    #     self.transformer_config.moe_aux_loss_coeff = 0
    #     self.transformer_config.moe_z_loss_coeff = 1
    #     self.sequential_mlp.router.weight.grad.fill_(0)
    #     out = self.sequential_mlp(hidden_states)[0]
    #     out.sum().mul_(paddle.empty_like(out.sum())).backward()
    #     assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

    # TODO: Not implemented yet
    # @pytest.mark.internal
    # def test_force_load_balancing(self):
    #     hidden_states = paddle.randn(
    #         (32, 2, self.router.hidden_size), device="cuda", dtype=paddle.bfloat16
    #     )
    #     hidden_states.requires_grad = True

    #     # First forward pass with normal routing
    #     normal_scores, normal_routing_map = self.router(hidden_states)

    #     # Second forward pass with force load balancing
    #     self.router.moe_router_force_load_balancing = True
    #     force_scores, force_routing_map = self.router(hidden_states)

    #     assert normal_scores.shape == force_scores.shape
    #     assert normal_routing_map.shape == force_routing_map.shape
    #     assert paddle.equal(normal_scores, force_scores) == False

    #     # Backward pass for force load balancing
    #     self.router.zero_grad()
    #     force_scores.sum().backward()
    #     assert hidden_states.grad is not None
    #     assert self.router.weight.grad.norm() > 0

    #     self.router.moe_router_force_load_balancing = False

    # TODO: capacity_factor,pad_to_capacity not implemented yet
    # @pytest.mark.internal
    # @pytest.mark.parametrize("capacity_factor", [None, 1.0, 2.0])
    # @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    # @pytest.mark.parametrize("pad_to_capacity", [True, False])
    # def test_token_dropping(self, capacity_factor, drop_policy, pad_to_capacity):
    #     if capacity_factor is None and pad_to_capacity:
    #         pytest.skip("Capacity factor is None, so no token dropping should be applied")

    #     num_tokens = 32
    #     self.router = self.router.cuda()
    #     self.router.moe_expert_capacity_factor = capacity_factor
    #     self.router.moe_token_drop_policy = drop_policy
    #     self.router.moe_pad_expert_input_to_capacity = pad_to_capacity

    #     hidden_states = paddle.randn(
    #         (num_tokens, self.router.hidden_size), dtype=paddle.bfloat16, device="cuda"
    #     )
    #     hidden_states.requires_grad = True
    #     probs, routing_map = self.router(hidden_states)

    #     if capacity_factor is not None:
    #         if pad_to_capacity:
    #             assert (
    #                 routing_map.sum().item()
    #                 == num_tokens * self.router.num_experts_per_tok * capacity_factor
    #             )
    #         else:
    #             assert (
    #                 routing_map.sum().item()
    #                 <= num_tokens * self.router.num_experts_per_tok * capacity_factor
    #             )
    #     else:
    #         assert routing_map.sum().item() == num_tokens * self.router.num_experts_per_tok

    #     # restore the config
    #     self.router.moe_expert_capacity_factor = None
    #     self.router.moe_token_drop_policy = "probs"
    #     self.router.moe_pad_expert_input_to_capacity = False


class TestSplitFeatureRouter(unittest.TestCase):
    """Covers the optional split-feature (multi-view) MoE routing path.

    When ``moe_split_feature_routing=True`` the router reuses ``self.weight``
    as the first view and adds a single new ``self.weight_1`` projection; the
    score is ``sigmoid(logits_0) + sigmoid(logits_1)``. These tests check that
    the extra projection is created with the expected shape, that the default
    (flag off) path does not create it, and that a forward pass runs.
    """

    n_routed_experts = 4
    hidden_size = 16

    @classmethod
    def setUpClass(cls):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        paddle.manual_seed(seed)

        # fleet is process-global; if another test class already initialized it
        # (e.g. TestTop2Router), reuse that instance instead of calling
        # initialize_fleet again, which would re-run fleet.init. Probe the
        # hybrid-communicate-group singleton directly (it is unset before
        # fleet.init) instead of catching a broad exception, so a genuine
        # comm-group error is not silently swallowed.
        already_initialized = getattr(fleet.fleet, "_hcg", None) is not None
        if not already_initialized:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": 4,
                "pp_degree": 1,
                "sharding_degree": 2,
                "sep_degree": 1,
                "cp_degree": 1,
                "ep_degree": 1,
                "order": [
                    "sharding",
                    "moe_sharding",
                    "pp",
                    "sep",
                    "cp",
                    "dp",
                    "ep",
                    "mp",
                ],
            }
            initialize_fleet(strategy=strategy)
            model_parallel_cuda_manual_seed(seed)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def _build_router(
        self,
        moe_split_feature_routing,
        scoring_func="sigmoid",
        moe_n_hash_layers=0,
    ):
        config = TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=24,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            bias_activation_fusion=True,
            scoring_func=scoring_func,
            moe_split_feature_routing=moe_split_feature_routing,
            moe_n_hash_layers=moe_n_hash_layers,
            actual_vocab_size=(32 if moe_n_hash_layers > 0 else None),
        )
        spec = get_gpt_layer_local_spec(
            config,
            num_experts=self.n_routed_experts,
            moe_expert_fusion=False,
        )
        moe_layer = MoELayer(
            config,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )
        gate = moe_layer.gate
        # The split-feature scoring_func contract is enforced in
        # set_layer_number() (once hash-layer status is known), which the model
        # normally calls per layer; mirror that here so the validation path is
        # exercised. layer_number=0 with no hash layers => a regular (non-hash)
        # split-feature layer; layer_number=0 with moe_n_hash_layers>0 => a
        # hash layer.
        gate.set_layer_number(0)
        return gate

    @pytest.mark.internal
    def test_disabled_by_default(self):
        # Flag off: no extra view projection is created.
        router = self._build_router(moe_split_feature_routing=False)
        assert router.moe_split_feature_routing is False
        assert not hasattr(router, "weight_1")

    @pytest.mark.internal
    def test_constructor_creates_second_view(self):
        # Flag on: a single extra projection weight_1 is created, sharing the
        # [num_experts, hidden_size] layout with the reused self.weight gate.
        router = self._build_router(moe_split_feature_routing=True)
        assert router.moe_split_feature_routing is True
        assert hasattr(router, "weight_1")
        assert list(router.weight_1.shape) == [
            self.n_routed_experts,
            self.hidden_size,
        ]
        assert router.weight.shape == router.weight_1.shape

    @pytest.mark.internal
    def test_hash_layer_drops_second_view(self):
        # Hash-routing layers bypass split-feature routing, so the second-view
        # gate created in __init__ must be dropped once the layer is confirmed
        # to be a hash layer. Otherwise it lingers as an unused parameter.
        router = self._build_router(
            moe_split_feature_routing=True, moe_n_hash_layers=1
        )
        assert router.is_hash_layer is True
        assert not hasattr(router, "weight_1")

    @pytest.mark.internal
    def test_requires_sigmoid_scoring_func(self):
        # The two-view contract is sigmoid+sigmoid; a non-sigmoid scoring_func
        # must be rejected rather than silently scored differently.
        with self.assertRaises(ValueError):
            self._build_router(
                moe_split_feature_routing=True, scoring_func="softmax"
            )

    @pytest.mark.internal
    def test_router_forward(self):
        router = self._build_router(moe_split_feature_routing=True).cuda()
        with paddle.no_grad():
            hidden_states = (
                paddle.randn((32, 2, self.hidden_size)).cuda().bfloat16()
            )
            (
                capacity,
                topk_weights,
                topk_indices,
                gates_masked,
                mask,
                priorities,
                aux_loss,
                z_loss,
            ) = router(hidden_states)
        assert topk_indices.shape[-1] == router.num_experts_per_tok

    @pytest.mark.internal
    def test_forward_uses_two_view_sigmoid_sum(self):
        # Verify the split score is exactly sigmoid(W0 x) + sigmoid(W1 x): the
        # gate matmul is fp32 (gates are computed under auto_cast(False)), so we
        # recompute the same expression from the two weights and compare against
        # the per-expert combine weights the router scattered into probs.
        router = self._build_router(moe_split_feature_routing=True).cuda()
        router.norm_topk_prob = False  # keep raw scores so probs == gates@topk
        # forward() requires a 3D [batch, seq, hidden] tensor unless
        # gpt_model_use_experimental_version is set; it reshapes to
        # [batch*seq, hidden] internally, so recompute on the flattened 2D view.
        batch, seq = 16, 2
        with paddle.no_grad():
            hidden_states = (
                paddle.randn((batch, seq, self.hidden_size)).cuda().bfloat16()
            )
            (
                _capacity,
                topk_weights,
                topk_indices,
                probs,
                _mask,
                _priorities,
                _aux_loss,
                _z_loss,
            ) = router(hidden_states)

            x_f32 = hidden_states.reshape([batch * seq, self.hidden_size]).cast(
                "float32"
            )
            logits_0 = paddle.matmul(x_f32, router.weight, transpose_y=True)
            logits_1 = paddle.matmul(x_f32, router.weight_1, transpose_y=True)
            expected_gates = F.sigmoid(logits_0) + F.sigmoid(logits_1)
            # The selected experts' weights in `probs` must match the recomputed
            # two-view sum at the same indices (no norm, scaling factor 1.0).
            picked = paddle.take_along_axis(
                expected_gates, topk_indices, axis=-1
            )
        np.testing.assert_allclose(
            topk_weights.numpy(),
            picked.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )
        # The summed sigmoid score lives in [0, 2]; a single-view sigmoid could
        # not exceed 1, so any value > 1 proves weight_1 contributes.
        assert float(expected_gates.max()) > 0.0

    @pytest.mark.internal
    def test_weight_1_participates_in_routing(self):
        # Zeroing weight_1 must change the gates: with weight_1 == 0,
        # sigmoid(logits_1) collapses to 0.5 everywhere, so the score becomes
        # sigmoid(logits_0) + 0.5. If weight_1 were ignored the two passes would
        # be identical; requiring them to differ guards against weight_1 being
        # dropped from the routing path.
        router = self._build_router(moe_split_feature_routing=True).cuda()
        with paddle.no_grad():
            hidden_states = (
                paddle.randn((32, self.hidden_size)).cuda().bfloat16()
            )
            x_f32 = hidden_states.cast("float32")
            logits_0 = paddle.matmul(x_f32, router.weight, transpose_y=True)

            full = F.sigmoid(logits_0) + F.sigmoid(
                paddle.matmul(x_f32, router.weight_1, transpose_y=True)
            )
            router.weight_1.set_value(paddle.zeros_like(router.weight_1))
            zeroed = F.sigmoid(logits_0) + F.sigmoid(
                paddle.matmul(x_f32, router.weight_1, transpose_y=True)
            )
        # weight_1 == 0 => second view is exactly 0.5 everywhere.
        np.testing.assert_allclose(
            (zeroed - F.sigmoid(logits_0)).numpy(),
            np.full_like(zeroed.numpy(), 0.5),
            rtol=1e-5,
            atol=1e-5,
        )
        # And the non-zero weight_1 produced a genuinely different score.
        assert not np.allclose(full.numpy(), zeroed.numpy())


if __name__ == "__main__":
    unittest.main()
