# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import random
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
import pytest
from paddle.distributed import fleet

from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
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


if __name__ == "__main__":
    unittest.main()
