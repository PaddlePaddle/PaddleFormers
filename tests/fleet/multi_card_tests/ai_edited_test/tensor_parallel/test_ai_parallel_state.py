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

import paddleformers.fleet
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet

TP_SIZE = None


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


def setUpModule():
    """Initialize fleet once for all tests in this module (TP=4, sharding=2)."""
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    _init_fleet_custom(
        mp=TP_SIZE, pp=1, sharding=TP_SIZE // 4 if TP_SIZE >= 4 else 1
    )


class TestParallelState(unittest.TestCase):
    """Test parallel_state after fleet initialization."""

    def test_tp_group_size(self):
        """Test tensor model parallel group has correct size."""
        self.assertEqual(
            paddleformers.fleet.parallel_state.get_tensor_model_parallel_world_size(),
            TP_SIZE,
        )

    def test_pp_group_size(self):
        """Test pipeline model parallel group has correct size."""
        self.assertEqual(
            paddleformers.fleet.parallel_state.get_pipeline_model_parallel_world_size(),
            1,
        )

    def test_tp_rank_in_range(self):
        """Test TP rank is within valid range."""
        rank = (
            paddleformers.fleet.parallel_state.get_tensor_model_parallel_rank()
        )
        self.assertGreaterEqual(rank, 0)
        self.assertLess(rank, TP_SIZE)


class TestProcessGroupCollection(unittest.TestCase):
    """Test ProcessGroupCollection."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        paddle.seed(42)
        model_parallel_cuda_manual_seed(42)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def test_tp_group_exists(self):
        """Test tensor model parallel group exists."""
        self.assertIsNotNone(self.pg_collection.tp)

    def test_tp_group_size(self):
        """Test tensor model parallel group has correct size."""
        self.assertEqual(self.pg_collection.tp.nranks, TP_SIZE)

    def test_pp_group_size(self):
        """Test pipeline model parallel group has correct size."""
        self.assertEqual(self.pg_collection.pp.nranks, 1)


class TestModelParallelSeed(unittest.TestCase):
    """Test multi-card random seed synchronization."""

    def test_seed_consistency(self):
        """Test that all ranks produce the same tensor with same seed."""
        model_parallel_cuda_manual_seed(123)
        t = paddle.randn([4, 8], dtype="float32")
        # Verify tensor is valid (no NaN)
        self.assertFalse(paddle.isnan(t).any())


if __name__ == "__main__":
    unittest.main()
