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

from paddleformers.fleet.tensor_parallel.mappings import (
    _AllGatherFromTensorParallelRegion,
    _CopyToModelParallelRegion,
    _GatherFromModelParallelRegion,
    _GatherFromSequenceParallelRegion,
    _ReduceFromModelParallelRegion,
    _ReduceScatterToSequenceParallelRegion,
    _ScatterToModelParallelRegion,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

TP_SIZE = 4


def _init_fleet_tp():
    """Initialize fleet with TP=4, DP=1, PP=1."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": TP_SIZE,
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


def setUpModule():
    """Initialize fleet once for all tests in this module (TP=4)."""
    _init_fleet_tp()
    np.random.seed(42)
    paddle.seed(42)
    model_parallel_cuda_manual_seed(42)


class TestGatherFromModelParallelRegionBasic(unittest.TestCase):
    """Test _GatherFromModelParallelRegion gathers split tensor."""

    def test_gather_from_model_parallel_region_basic(self):
        """Gather should collect partial tensors into full tensor along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        input_data = paddle.ones([8]).cuda() * dist.get_rank()
        actual_output = _GatherFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        parts = [paddle.ones([8]).cuda() * r for r in range(TP_SIZE)]
        expected = paddle.concat(parts)
        self.assertTrue(paddle.equal_all(actual_output, expected))


class TestScatterToModelParallelRegionBasic(unittest.TestCase):
    """Test _ScatterToModelParallelRegion scatters full tensor."""

    def test_scatter_to_model_parallel_region_basic(self):
        """Scatter should split full tensor across ranks along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.rand((8, 16)).cuda()

        output_data = _ScatterToModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        rank = dist.get_rank()
        expected = input_data[:, rank * 4 : (rank + 1) * 4]
        self.assertTrue(paddle.equal_all(output_data, expected))


class TestCopyToModelParallelRegionIdentity(unittest.TestCase):
    """Test _CopyToModelParallelRegion passes through in forward."""

    def test_copy_to_model_parallel_region_identity(self):
        """Forward should be identity."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        input_data = paddle.ones([1]).cuda() * dist.get_rank()
        output = _CopyToModelParallelRegion.symbolic(None, input_data, tp_group)
        self.assertTrue(paddle.equal_all(input_data, output))


class TestReduceFromModelParallelRegionBasic(unittest.TestCase):
    """Test _ReduceFromModelParallelRegion reduces across ranks."""

    def test_reduce_from_model_parallel_region_basic(self):
        """Forward should all-reduce."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.ones([1]).cuda() * dist.get_rank()

        output = _ReduceFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        )
        expected = paddle.ones([1]).cuda() * sum(range(TP_SIZE))
        self.assertTrue(paddle.equal_all(output, expected))


class TestGatherFromSequenceParallelRegion(unittest.TestCase):
    """Test _GatherFromSequenceParallelRegion gathers along sequence dim."""

    def test_gather_from_sequence_parallel_region(self):
        """Gather should collect partial sequence tensors into full tensor."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        input_data = paddle.ones([4]).cuda() * dist.get_rank()

        output_data = _GatherFromSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        )
        parts = [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        expected = paddle.concat(parts)
        self.assertTrue(paddle.equal_all(output_data, expected))


class TestReduceScatterToSequenceParallelRegion(unittest.TestCase):
    """Test _ReduceScatterToSequenceParallelRegion reduce-scatters along seq dim."""

    def test_reduce_scatter_to_sequence_parallel_region(self):
        """Reduce-scatter should reduce then scatter along sequence dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        full_input = paddle.concat(
            [paddle.ones([4]).cuda() * r for r in range(TP_SIZE)]
        )

        output_data = _ReduceScatterToSequenceParallelRegion.symbolic(
            None, full_input, tp_group
        )
        # Verify output shape
        self.assertIsNotNone(output_data)
        self.assertEqual(output_data.shape[-1], 4)


class TestAllGatherFromTensorParallelRegion(unittest.TestCase):
    """Test _AllGatherFromTensorParallelRegion gathers along last dim."""

    def test_all_gather_from_tensor_parallel_region(self):
        """Forward should gather along last dim."""
        tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

        input_data = paddle.ones([4, 4]).cuda() * dist.get_rank()
        output = _AllGatherFromTensorParallelRegion.symbolic(
            None, input_data, tp_group
        )
        # Output should have TP_SIZE times the last dim
        self.assertEqual(output.shape[-1], 4 * TP_SIZE)


if __name__ == "__main__":
    unittest.main()
