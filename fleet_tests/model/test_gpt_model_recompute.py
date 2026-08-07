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


import copy
import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

# from tests.unit_tests.test_utilities import Utils
import paddleformers.fleet.parallel_state as ps

# from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig


class TestGPTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment, only need to execute once"""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
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
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)

    def setUp(self):
        """Reset random seed before each test case"""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        fleet_env = fleet.fleet
        self.strategy = fleet_env._user_defined_strategy

    def _create_base_config(self):
        """Create base transformer configuration for testing"""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            first_k_dense_replace=1,
            attention_dropout=0.0,
            n_routed_experts=8,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            recompute_granularity=None,
            recompute_modules=[],
        )
        return config

    def _create_recompute_config(self, base_config):
        """Create recompute-enabled configuration based on base config"""
        config = copy.deepcopy(base_config)
        config.recompute_granularity = "selective"
        config.recompute_modules = [
            "core_attn",
            "norm",
            "mlp",
            "lm_head",
            "embedding",
            "loss_fn",
        ]
        return config

    def _create_gpt_model(self, config):
        """Create GPT model based on given configuration"""
        return gpt_builder(config, num_stages=1)

    def _prepare_input_data(self, config):
        """Prepare input data for model forward/backward"""
        sequence_length = config.max_sequence_length
        micro_batch_size = 1

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, sequence_length, sequence_length),
            dtype=bool,
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        return (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
                "attention_mask": [attention_mask],
            },
            [labels],
        )

    def _run_model_and_get_results(self, config):
        """Run model forward/backward and return loss and gradients"""
        gpt_model = self._create_gpt_model(config)
        data = self._prepare_input_data(config)

        gpt_pipe_model = NoPipelineParallel(gpt_model, self.strategy)
        loss = gpt_pipe_model.forward_backward_pipeline(data)

        grad_dict = {}
        for name, param in gpt_model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.detach().clone()

        return loss.item(), grad_dict

    def test_recompute_precision_alignment(self):
        """Test that recompute on/off produces aligned precision results"""
        # Create base config (recompute disabled)
        base_config = self._create_base_config()

        # Create recompute config (recompute enabled)
        recompute_config = self._create_recompute_config(base_config)

        # Run model without recompute
        print("\n=== Running model WITHOUT recompute ===")
        self.setUp()
        loss_no_recompute, grads_no_recompute = self._run_model_and_get_results(
            base_config
        )
        print(f"Loss (no recompute): {loss_no_recompute}")

        # Run model with recompute
        print("\n=== Running model WITH recompute ===")
        self.setUp()
        loss_with_recompute, grads_with_recompute = (
            self._run_model_and_get_results(recompute_config)
        )
        print(f"Loss (with recompute): {loss_with_recompute}")

        # Compare loss
        print("\n=== Comparing results ===")
        print(f"Loss diff: {abs(loss_no_recompute - loss_with_recompute)}")

        self.assertAlmostEqual(
            loss_no_recompute,
            loss_with_recompute,
            places=5,
            msg=f"Loss mismatch: no_recompute={loss_no_recompute}, with_recompute={loss_with_recompute}",
        )

        # Compare gradients
        for name in grads_no_recompute:
            if name in grads_with_recompute:
                grad_no_recompute = grads_no_recompute[name]
                grad_with_recompute = grads_with_recompute[name]

                grad_norm_no_recompute = grad_no_recompute.norm().item()
                grad_norm_with_recompute = grad_with_recompute.norm().item()

                max_diff = (
                    (grad_no_recompute - grad_with_recompute).abs().max().item()
                )

                print(
                    f"{name}: norm_diff={abs(grad_norm_no_recompute - grad_norm_with_recompute):.8f}, max_diff={max_diff:.8f}"
                )

                self.assertAlmostEqual(
                    grad_norm_no_recompute,
                    grad_norm_with_recompute,
                    places=5,
                    msg=f"Gradient norm mismatch for {name}: no_recompute={grad_norm_no_recompute}, with_recompute={grad_norm_with_recompute}",
                )

        print("\n=== Recompute precision alignment test PASSED ===")


if __name__ == "__main__":
    unittest.main()
