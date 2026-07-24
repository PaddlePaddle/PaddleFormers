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
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    StandardMoERouter,
    TopKRouter,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

MP_DEGREE = 4
SEED = 99
WORLD_SIZE = None
_GPU_COMPUTE_OK = None


def _set_seed(seed_):
    np.random.seed(seed_)
    paddle.seed(seed_)
    paddle.manual_seed(seed_)


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


def _build_router_config(**overrides):
    defaults = dict(  # noqa: C408
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        use_cpu_initialization=True,
        num_experts_per_tok=2,
        tensor_model_parallel_size=MP_DEGREE,
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
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def setUpModule():
    """Initialize fleet once for all tests in this module (MP=4)."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
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
    model_parallel_cuda_manual_seed(SEED)


class TestTopKRouter(unittest.TestCase):
    """Test TopKRouter creation."""

    @classmethod
    def setUpClass(cls):
        _set_seed(SEED)
        cls.config = _build_router_config()
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @_requires_gpu_compute
    def test_topk_router_creation(self):
        """Test TopKRouter creation with config."""
        router = TopKRouter(self.config, pg_collection=self.pg_collection)
        self.assertIsNotNone(router)
        self.assertEqual(router.num_experts, self.config.n_routed_experts)
        self.assertEqual(
            router.num_experts_per_tok, self.config.num_experts_per_tok
        )


class TestStandardMoERouter(unittest.TestCase):
    """Test StandardMoERouter creation."""

    @classmethod
    def setUpClass(cls):
        _set_seed(SEED + 1)
        cls.config = _build_router_config()
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    @_requires_gpu_compute
    def test_standard_moe_router_creation(self):
        """Test StandardMoERouter creation with config."""
        router = StandardMoERouter(
            self.config, pg_collection=self.pg_collection
        )
        self.assertIsNotNone(router)
        self.assertEqual(router.num_experts, self.config.n_routed_experts)


class TestFusedGateDetachMatmul(unittest.TestCase):
    """Test FusedGateDetachMatmul layer."""

    @_requires_gpu_compute
    def test_fused_gate_detach_matmul_creation(self):
        """Test FusedGateDetachMatmul can be used as a fused op."""
        x = paddle.randn([4, 8], dtype="float32")
        # FusedGateDetachMatmul expects weight in [out_features, in_features] format
        w = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w)
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
