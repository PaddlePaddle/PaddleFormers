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

import functools
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

from paddleformers.fleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

WORLD_SIZE = None
_GPU_COMPUTE_OK = None


def _init_tp_for_transformer():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 4,
        "pp_degree": 1,
        "sharding_degree": 2,
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


def _set_seed(seed=42):
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
    _init_tp_for_transformer()
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


class TestTransformerLayerBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden_size = 64
        cls.config = TransformerConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=4,
            intermediate_size=128,
            use_cpu_initialization=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            bf16=False,
            params_dtype=paddle.float32,
            rms_norm_eps=1e-5,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        layer_spec = get_gpt_layer_local_spec(cls.config)
        cls.transformer_layer = layer_spec.layer(
            cls.config,
            layer_spec.sublayers_spec,
            layer_number=1,
            pg_collection=cls.pg_collection,
        )

    @_requires_gpu_compute
    def test_forward_shape(self):
        """Test transformer layer forward produces correct output shape."""
        hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
        result = self.transformer_layer({"hidden_states": hidden_states})
        self.assertEqual(result["hidden_states"].shape, [2, 8, self.hidden_size])

    @_requires_gpu_compute
    def test_forward_with_attention_mask(self):
        """Test transformer layer forward with attention mask."""
        hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
        attention_mask = paddle.triu(paddle.full([8, 8], float("-inf"), dtype=paddle.float32), diagonal=1)
        result = self.transformer_layer({"hidden_states": hidden_states, "attention_mask": attention_mask})
        self.assertEqual(result["hidden_states"].shape, [2, 8, self.hidden_size])


class TestTransformerLayerWithMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hidden_size = 64
        cls.n_experts = 4
        cls.config = TransformerConfig(
            hidden_size=cls.hidden_size,
            num_attention_heads=4,
            n_routed_experts=cls.n_experts,
            use_cpu_initialization=True,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=False,
            params_dtype=paddle.float32,
            moe_intermediate_size=64,
            moe_deep_gemm=False,
            gated_linear_unit=True,
            n_shared_experts=0,
            rms_norm_eps=1e-5,
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
            output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        layer_spec = get_gpt_layer_local_spec(cls.config, num_experts=cls.n_experts, moe_expert_fusion=False)
        cls.transformer_layer = layer_spec.layer(
            cls.config,
            layer_spec.sublayers_spec,
            layer_number=1,
            pg_collection=cls.pg_collection,
        )

    @_requires_gpu_compute
    def test_forward_moe_shape(self):
        """Test transformer layer with MoE produces correct output shape."""
        hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
        result = self.transformer_layer({"hidden_states": hidden_states})
        self.assertEqual(result["hidden_states"].shape, [2, 8, self.hidden_size])

    @_requires_gpu_compute
    def test_forward_moe_with_labels(self):
        """Test transformer layer with MoE and input_ids for aux loss."""
        hidden_states = paddle.randn([2, 8, self.hidden_size], dtype=paddle.float32)
        input_ids = paddle.randint(0, 100, [2, 8])
        result = self.transformer_layer({"hidden_states": hidden_states, "input_ids": input_ids})
        self.assertEqual(result["hidden_states"].shape, [2, 8, self.hidden_size])


if __name__ == "__main__":
    unittest.main()
