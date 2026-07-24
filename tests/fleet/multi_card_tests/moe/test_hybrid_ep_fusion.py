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

import random
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet_ops import is_hybrid_ep_available
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
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerWithOverlap,
)


class TestHybridEPFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_hybrid_ep_available():
            raise unittest.SkipTest("HybridEP is not available")

        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        paddle.manual_seed(seed)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 8,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 8,
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
            "pp_configs": {
                "forward_backward_overlap_scheduler": True,
                "overlap_p2p_comm": True,
                "enable_dynamic_shape": True,
            },
        }
        initialize_fleet(strategy=strategy)
        model_parallel_cuda_manual_seed(seed)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def _build_config(self):
        hidden_size = 256
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=8,
            n_routed_experts=64,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=64,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            moe_token_dispatcher_type="hybridep",
            moe_shared_expert_overlap=True,
            norm_topk_prob=False,
            bias_activation_fusion=True,
        )
        config.hybridep_buffer_configs = {
            "num_sms_dispatch_api": 8,
            "num_sms_combine_api": 8,
            "num_sms_preprocessing_api": 8,
        }
        return config

    def _build_moe_layer(self, config):
        transformer_layer_spec = get_gpt_layer_local_spec(
            config, num_experts=config.n_routed_experts
        )
        return MoELayer(
            config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            self.pg_collection,
        )

    def test_hybrid_ep_moe_fusion_and_overlap_segments(self):
        config = self._build_config()
        moe_layer = self._build_moe_layer(config)
        self.assertTrue(moe_layer.use_hybrid_ep_backend)
        self.assertFalse(moe_layer.moe_shared_expert_overlap)
        self.assertEqual(
            moe_layer.token_dispatcher._comm_manager.hybridep_buffer_configs,
            config.hybridep_buffer_configs,
        )

        input_data = paddle.randn(
            4, 64, config.hidden_size, dtype=paddle.bfloat16
        )
        output = moe_layer(input_data)[0]
        self.assertEqual(output.shape, input_data.shape)

        (
            _,
            topk_weights,
            topk_indices,
            _,
            _,
            _,
            _,
            _,
        ) = moe_layer.gate(input_data)
        dispatch_args = moe_layer.dispatch_preprocess(
            (input_data, topk_weights, topk_indices)
        )
        dispatched_args = moe_layer.compute_dispatch(
            dispatch_args, async_finish=True
        )
        expert_output = moe_layer.compute_experts(
            dispatched_args, is_first_fwd=True
        )
        overlapped_output = moe_layer.compute_combine(
            expert_output, async_finish=True
        )
        self.assertEqual(
            overlapped_output.shape,
            [input_data.shape[0] * input_data.shape[1], input_data.shape[2]],
        )

    def test_hybrid_ep_dispatcher_is_allowed_for_overlap_layer(self):
        config = self._build_config()
        transformer_layer_spec = get_gpt_layer_local_spec(
            config, num_experts=config.n_routed_experts
        )
        self.assertIs(transformer_layer_spec.layer, TransformerLayerWithOverlap)

        transformer_layer = build_spec_layer(
            transformer_layer_spec, pg_collection=self.pg_collection
        )

        self.assertEqual(
            transformer_layer.mlp.moe_token_dispatcher_type, "hybridep"
        )


if __name__ == "__main__":
    unittest.main()
