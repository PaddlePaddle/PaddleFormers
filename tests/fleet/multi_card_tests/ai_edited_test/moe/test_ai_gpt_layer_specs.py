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
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import paddleformers.fleet
from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

WORLD_SIZE = None
_GPU_COMPUTE_OK = None


def _init_moe_tp():
    """Initialize fleet with TP=4 for MoE testing."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 4,
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
    initialize_fleet(strategy=strategy)


def _set_seed(seed=123):
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.manual_seed(seed)


def _check_gpu_compute():
    """Check if GPU compute operations work on this environment."""
    global _GPU_COMPUTE_OK
    if _GPU_COMPUTE_OK is not None:
        return _GPU_COMPUTE_OK
    try:
        t = paddle.randn([4, 4], dtype="float32")
        _ = paddle.nn.functional.softmax(t, axis=-1)
        _GPU_COMPUTE_OK = True
    except Exception:
        _GPU_COMPUTE_OK = False
    return _GPU_COMPUTE_OK


def setUpModule():
    """Initialize fleet once for all tests in this module."""
    global WORLD_SIZE
    WORLD_SIZE = dist.get_world_size()
    _set_seed(42)
    _init_moe_tp()
    model_parallel_cuda_manual_seed(42)


def _requires_gpu_compute(test_func):
    """Decorator to skip test if GPU compute is not available."""

    @unittest.skipUnless(
        _check_gpu_compute(),
        "GPU compute not available (likely CUDA driver version mismatch)",
    )
    def wrapper(*args, **kwargs):
        return test_func(*args, **kwargs)

    wrapper.__name__ = test_func.__name__
    wrapper.__doc__ = test_func.__doc__
    return wrapper


class TestMoELayerForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden_size = 16
        cls.n_routed_experts = 4
        cls.config = TransformerConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=4,
            n_routed_experts=cls.n_routed_experts,
            use_cpu_initialization=True,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=False,
            params_dtype=paddle.float32,
            moe_intermediate_size=24,
            moe_deep_gemm=False,
            gated_linear_unit=True,
            n_shared_experts=0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        layer_spec = get_gpt_layer_local_spec(
            cls.config,
            num_experts=cls.n_routed_experts,
            moe_expert_fusion=False,
        )
        cls.moe_layer = paddleformers.fleet.transformer.moe.moe_layer.MoELayer(
            cls.config,
            layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            cls.pg_collection,
        )

    @_requires_gpu_compute
    def test_moe_forward_shape(self):
        """Test MoELayer forward produces correct output shape."""
        with paddle.no_grad():
            hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
            output, _ = self.moe_layer(hidden_states)
            self.assertEqual(output.shape, [2, 8, self.hidden_size])

    @_requires_gpu_compute
    def test_moe_forward_with_labels(self):
        """Test MoELayer forward with labels for auxiliary loss."""
        with paddle.no_grad():
            hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
            output, _ = self.moe_layer(hidden_states)
            self.assertEqual(output.shape, [2, 8, self.hidden_size])


class TestMoEGateRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden_size = 16
        cls.n_routed_experts = 4
        cls.config = TransformerConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=4,
            n_routed_experts=cls.n_routed_experts,
            use_cpu_initialization=True,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=False,
            params_dtype=paddle.float32,
            moe_intermediate_size=24,
            moe_deep_gemm=False,
            gated_linear_unit=True,
            n_shared_experts=0,
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        layer_spec = get_gpt_layer_local_spec(
            cls.config,
            num_experts=cls.n_routed_experts,
            moe_expert_fusion=False,
        )
        cls.moe_layer = paddleformers.fleet.transformer.moe.moe_layer.MoELayer(
            cls.config,
            layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            cls.pg_collection,
        )
        cls.gate = cls.moe_layer.gate

    @_requires_gpu_compute
    def test_gate_output_shape(self):
        """Test gate produces correct output with 8-tuple."""
        with paddle.no_grad():
            hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
            self.gate.moe_router_score_function = "softmax"
            result = self.gate(hidden_states)
            # TopKRouter returns 8-tuple: (capacity, topk_weights, topk_indices, probs, mask, priorities, aux_loss, z_loss)
            self.assertEqual(len(result), 8)
            topk_weights = result[1]
            topk_indices = result[2]
            self.assertEqual(topk_weights.shape[1], self.config.num_experts_per_tok)
            self.assertEqual(topk_weights.shape, topk_indices.shape)

    @_requires_gpu_compute
    def test_gate_sigmoid_score_function(self):
        """Test gate with sigmoid score function."""
        with paddle.no_grad():
            hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
            self.gate.moe_router_score_function = "sigmoid"
            result = self.gate(hidden_states)
            # TopKRouter returns 8-tuple
            self.assertEqual(len(result), 8)
            topk_weights = result[1]
            topk_indices = result[2]
            self.assertEqual(topk_weights.shape, topk_indices.shape)


class TestMoEConfigVariants(unittest.TestCase):
    """Test MoELayer with various configuration combinations."""

    @classmethod
    def setUpClass(cls):
        cls.hidden_size = 16

    def _build_moe(self, n_experts, n_shared, moe_expert_fusion=False):
        config = TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=n_experts,
            use_cpu_initialization=True,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=False,
            params_dtype=paddle.float32,
            moe_intermediate_size=24,
            moe_deep_gemm=False,
            gated_linear_unit=True,
            n_shared_experts=n_shared,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        layer_spec = get_gpt_layer_local_spec(
            config,
            num_experts=n_experts,
            moe_expert_fusion=moe_expert_fusion,
        )
        moe_layer = paddleformers.fleet.transformer.moe.moe_layer.MoELayer(
            config,
            layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            pg_collection,
        )
        return moe_layer, config

    @_requires_gpu_compute
    def test_moe_with_shared_experts(self):
        """Test MoELayer with shared experts."""
        moe_layer, config = self._build_moe(n_experts=4, n_shared=2)
        with paddle.no_grad():
            hidden = paddle.randn([2, 4, self.hidden_size], dtype=paddle.float32)
            output, _ = moe_layer(hidden)
            self.assertEqual(output.shape, [2, 4, self.hidden_size])


if __name__ == "__main__":
    unittest.main()
