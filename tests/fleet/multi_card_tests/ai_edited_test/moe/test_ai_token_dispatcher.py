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
from paddleformers.fleet.transformer.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    _DeepEPManager,
    _HybridEPManager,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

EP_DEGREE = 4
SEED = 88
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


def _init_fleet():
    """Initialize fleet with EP=4, MP=1, DP=1, PP=1, sharding=1."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": EP_DEGREE,
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
    _set_seed(SEED)
    model_parallel_cuda_manual_seed(SEED)


def _build_moe_config(**overrides):
    defaults = dict(  # noqa: C408
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        use_cpu_initialization=True,
        num_experts_per_tok=2,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=EP_DEGREE,
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
        output_layer_init_method=functools.partial(
            paddle.nn.init.xavier_uniform_, gain=1.0
        ),
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def setUpModule():
    """Initialize fleet once for all tests in this module."""
    global WORLD_SIZE
    WORLD_SIZE = dist.get_world_size()
    try:
        _init_fleet()
    except Exception:
        pass


class TestTokenDispatcher(unittest.TestCase):
    """Test token dispatcher creation and comm manager type."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.config = _build_moe_config()
            cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            cls.ep_group = cls.pg_collection.ep
            if cls.ep_group is None:
                raise unittest.SkipTest("EP group not available")
        except unittest.SkipTest:
            raise
        except Exception:
            raise unittest.SkipTest("Fleet initialization failed")

    @_requires_gpu_compute
    def test_deep_ep_manager_creation(self):
        """Test _DeepEPManager creates successfully with config."""
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available")
        try:
            manager = _DeepEPManager(
                group=self.ep_group,
                router_topk=self.config.num_experts_per_tok,
                num_experts=self.config.n_routed_experts,
                num_local_experts=(self.config.n_routed_experts // EP_DEGREE),
                moe_ep_barrier=True,
            )
            self.assertIsNotNone(manager)
            self.assertEqual(
                manager.router_topk, self.config.num_experts_per_tok
            )
        except ImportError:
            self.skipTest("DeepEP runtime not available in this environment")

    @_requires_gpu_compute
    def test_hybrid_ep_manager_creation(self):
        """Test _HybridEPManager creates successfully with config."""
        try:
            manager = _HybridEPManager(
                group=self.ep_group,
                router_topk=self.config.num_experts_per_tok,
                num_experts=self.config.n_routed_experts,
                num_local_experts=(self.config.n_routed_experts // EP_DEGREE),
                moe_ep_barrier=True,
            )
            self.assertIsNotNone(manager)
            self.assertEqual(
                manager.router_topk, self.config.num_experts_per_tok
            )
        except ImportError:
            self.skipTest("HybridEP runtime not available in this environment")

    @_requires_gpu_compute
    def test_moe_flex_token_dispatcher_creation(self):
        """Test MoEFlexTokenDispatcher creation with config."""
        num_local_experts = self.config.n_routed_experts // EP_DEGREE
        dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            n_routed_experts=self.config.n_routed_experts,
            ep_group=self.ep_group,
            moe_ep_barrier=True,
            dispatcher_type="deepep",
        )
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.num_local_experts, num_local_experts)

    @_requires_gpu_compute
    def test_alltoall_token_dispatcher_creation(self):
        """Test AllToAllTokenDispatcher creation with config."""
        num_experts_per_device = self.config.n_routed_experts // EP_DEGREE
        local_expert_indices = list(range(num_experts_per_device))
        dispatcher = AllToAllTokenDispatcher(
            moe_group=self.ep_group,
            expert_model_parallel_size=EP_DEGREE,
            num_experts_per_device=num_experts_per_device,
            local_expert_indices=local_expert_indices,
        )
        self.assertIsNotNone(dispatcher)
        self.assertEqual(dispatcher.num_local_experts, num_experts_per_device)

    @_requires_gpu_compute
    def test_dispatcher_comm_manager_type(self):
        """Verify the comm_manager is of correct type based on config."""
        num_local_experts = self.config.n_routed_experts // EP_DEGREE
        dispatcher_deepep = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            n_routed_experts=self.config.n_routed_experts,
            ep_group=self.ep_group,
            moe_ep_barrier=True,
            dispatcher_type="deepep",
        )
        self.assertIsInstance(dispatcher_deepep._comm_manager, _DeepEPManager)


if __name__ == "__main__":
    unittest.main()
