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

import random
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
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


class TestFusionBF16ExpertParallel(unittest.TestCase):
    def setUp(self):
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
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def test_moe_fusion(self):
        n_routed_experts = 8
        hidden_size = 64
        transformer_config_moe = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=4,
            n_routed_experts=n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=128,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            bias_activation_fusion=True,
        )

        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config_moe, num_experts=n_routed_experts
        )

        moe_layer = MoELayer(
            transformer_config_moe,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

        gate_weight_shape = moe_layer.gate.weight.shape

        moe_layer.gate.weight.set_value(
            paddle.zeros(gate_weight_shape).astype("float32")
        )

        input_data = paddle.ones(16, 4, hidden_size, dtype=paddle.bfloat16)

        output = moe_layer(input_data)[0]

        assert output.shape == (16, 4, hidden_size)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
