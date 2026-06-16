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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

TP_SIZE = None
_NCCL_COLLECTIVE_OK = None


def _init_fleet_custom(mp=4, pp=1, dp=1, sharding=2, cp=1, ep=1):
    strategy = fleet.DistributedStrategy()
    moe_sharding = 1
    if ep > 1:
        world_size = pp * mp * dp * sharding
        moe_sharding = world_size // (pp * ep)
    strategy.hybrid_configs = {
        "dp_degree": dp,
        "mp_degree": mp,
        "pp_degree": pp,
        "sharding_degree": sharding,
        "sep_degree": 1,
        "cp_degree": cp,
        "ep_degree": ep,
        "moe_sharding_degree": moe_sharding,
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


def _check_nccl_collective():
    """Check if NCCL all-reduce works on this environment."""
    global _NCCL_COLLECTIVE_OK
    if _NCCL_COLLECTIVE_OK is not None:
        return _NCCL_COLLECTIVE_OK
    try:
        t = paddle.ones([4], dtype="float32")
        dist.all_reduce(t)
        _NCCL_COLLECTIVE_OK = True
    except Exception:
        _NCCL_COLLECTIVE_OK = False
    return _NCCL_COLLECTIVE_OK


def setUpModule():
    """Initialize fleet once for all tests in this module (TP=4, sharding=2)."""
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    _init_fleet_custom(
        mp=TP_SIZE, pp=1, sharding=TP_SIZE // 4 if TP_SIZE >= 4 else 1
    )
    np.random.seed(42)
    paddle.seed(42)
    model_parallel_cuda_manual_seed(42)


def _requires_nccl(test_func):
    """Decorator to skip test if NCCL collectives are not available."""

    @unittest.skipUnless(
        _check_nccl_collective(),
        "NCCL collective not available (likely CUDA driver version mismatch)",
    )
    def wrapper(*args, **kwargs):
        return test_func(*args, **kwargs)

    wrapper.__name__ = test_func.__name__
    wrapper.__doc__ = test_func.__doc__
    return wrapper


class TestColumnParallelLinear(unittest.TestCase):
    """Test ColumnParallelLinear layer."""

    @_requires_nccl
    def test_forward_shape(self):
        """Test ColumnParallelLinear splits weight columns across TP ranks."""
        hidden_size = 16
        output_size = 32
        layer = ColumnParallelLinear(
            hidden_size,
            output_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            config=TransformerConfig(
                tensor_model_parallel_size=TP_SIZE,
                num_attention_heads=4,
                num_key_value_heads=4,
                use_cpu_initialization=True,
            ),
        )
        input_tensor = paddle.randn([4, hidden_size], dtype="float32")
        input_tensor.stop_gradient = False
        output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        expected_output_size = output_size // TP_SIZE
        self.assertEqual(output.shape[-1], expected_output_size)
        self.assertFalse(paddle.isnan(output).any())


class TestRowParallelLinear(unittest.TestCase):
    """Test RowParallelLinear layer."""

    @_requires_nccl
    def test_forward_shape(self):
        """Test RowParallelLinear splits weight rows across TP ranks."""
        input_size = 16
        output_size = 8
        layer = RowParallelLinear(
            input_size,
            output_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            config=TransformerConfig(
                tensor_model_parallel_size=TP_SIZE,
                num_attention_heads=4,
                num_key_value_heads=4,
                use_cpu_initialization=True,
            ),
        )
        local_input_size = input_size // TP_SIZE
        input_tensor = paddle.randn([4, local_input_size], dtype="float32")
        input_tensor.stop_gradient = False
        output = layer(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        self.assertEqual(output.shape[-1], output_size)
        self.assertFalse(paddle.isnan(output).any())


class TestScatterToTPRegion(unittest.TestCase):
    """Test scatter_to_tensor_model_parallel_region."""

    def test_scatter_shape(self):
        """Test scatter splits input along last dimension across TP ranks."""
        input_tensor = paddle.randn([4, 16], dtype="float32")
        input_tensor.stop_gradient = False
        output = scatter_to_tensor_model_parallel_region(input_tensor)
        local_size = 16 // TP_SIZE
        self.assertEqual(output.shape[-1], local_size)


class TestGatherFromSPRegion(unittest.TestCase):
    """Test gather_from_sequence_parallel_region."""

    @_requires_nccl
    def test_gather_shape(self):
        """Test gather collects input along sequence dimension."""
        seq_len = 8
        hidden = 16
        input_tensor = paddle.randn(
            [seq_len // TP_SIZE, hidden], dtype="float32"
        )
        input_tensor.stop_gradient = False
        output = gather_from_sequence_parallel_region(input_tensor)
        self.assertEqual(output.shape[0], seq_len)


class TestScatterToSPRegion(unittest.TestCase):
    """Test scatter_to_sequence_parallel_region."""

    def test_scatter_shape(self):
        """Test scatter splits input along sequence dimension."""
        seq_len = 8
        hidden = 16
        input_tensor = paddle.randn([seq_len, hidden], dtype="float32")
        input_tensor.stop_gradient = False
        output = scatter_to_sequence_parallel_region(input_tensor)
        local_seq = seq_len // TP_SIZE
        self.assertEqual(output.shape[0], local_seq)


class TestReduceFromTPRegion(unittest.TestCase):
    """Test reduce_from_tensor_model_parallel_region."""

    @_requires_nccl
    def test_reduce_shape(self):
        """Test reduce performs all-reduce across TP ranks."""
        input_tensor = paddle.randn([4, 16], dtype="float32")
        input_tensor.stop_gradient = False
        output = reduce_from_tensor_model_parallel_region(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)


if __name__ == "__main__":
    unittest.main()
