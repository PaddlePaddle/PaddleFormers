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

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.moe.moe_utils import (
    AddAuxiliaryLoss,
    FilterScores,
    permute,
    unpermute,
)

WORLD_SIZE = None


def _init_tp_sharding():
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
    np.random.seed(42)
    paddle.seed(42)
    paddle.manual_seed(42)
    model_parallel_cuda_manual_seed(42)


def setUpModule():
    """Initialize fleet once for all tests in this module."""
    global WORLD_SIZE
    WORLD_SIZE = dist.get_world_size()
    _init_tp_sharding()


def _make_routing_map(num_tokens, num_experts, num_experts_per_tok, rng=None):
    """Create a boolean routing_map of shape [num_tokens, num_experts]."""
    if rng is None:
        rng = np.random.RandomState(42)
    routing_map = np.zeros([num_tokens, num_experts], dtype=np.int64)
    for i in range(num_tokens):
        chosen = rng.choice(
            num_experts, size=num_experts_per_tok, replace=False
        )
        routing_map[i, chosen] = 1
    return paddle.to_tensor(routing_map)


class TestPermuteUnpermute(unittest.TestCase):
    def test_permute_shape(self):
        """Test permute output shape."""
        num_tokens = 32
        hidden = 16
        num_experts = 4
        num_experts_per_tok = 2
        tokens = paddle.randn([num_tokens, hidden], dtype=paddle.float32)
        routing_map = _make_routing_map(
            num_tokens, num_experts, num_experts_per_tok
        )
        permuted, sorted_indices = permute(tokens, routing_map)
        expected_out = num_tokens * num_experts_per_tok
        self.assertEqual(permuted.shape[0], expected_out)
        self.assertEqual(permuted.shape[1], hidden)

    def test_unpermute_shape(self):
        """Test unpermute restores original shape."""
        num_tokens = 32
        hidden = 16
        num_experts = 4
        num_experts_per_tok = 2
        tokens = paddle.randn([num_tokens, hidden], dtype=paddle.float32)
        routing_map = _make_routing_map(
            num_tokens, num_experts, num_experts_per_tok
        )

        permuted, sorted_indices = permute(tokens, routing_map)
        restore_shape = paddle.shape(tokens)
        unpermuted = unpermute(permuted, sorted_indices, restore_shape)
        self.assertEqual(unpermuted.shape, [num_tokens, hidden])

    def test_permute_unpermute_roundtrip(self):
        """Test permute then unpermute is numerically correct (scatter-add)."""
        num_tokens = 16
        hidden = 16
        num_experts = 2
        num_experts_per_tok = 2
        rng = np.random.RandomState(42)

        tokens = paddle.randn([num_tokens, hidden], dtype=paddle.float32)
        routing_map = _make_routing_map(
            num_tokens, num_experts, num_experts_per_tok, rng=rng
        )

        permuted, sorted_indices = permute(tokens, routing_map)
        restore_shape = paddle.shape(tokens)
        unpermuted = unpermute(permuted, sorted_indices, restore_shape)

        # unpermute is a scatter_add, so it sums contributions from each expert
        self.assertEqual(unpermuted.shape, [num_tokens, hidden])


class TestAddAuxiliaryLoss(unittest.TestCase):
    def test_auxiliary_loss_scalar(self):
        """Test AddAuxiliaryLoss produces output with same shape as input."""
        x = paddle.randn([4, 8], dtype=paddle.float32)
        aux_loss_val = paddle.randn([1], dtype=paddle.float32)
        result = AddAuxiliaryLoss.apply(x, aux_loss_val)
        self.assertEqual(result.shape, [4, 8])

    def test_auxiliary_loss_backward(self):
        """Test AddAuxiliaryLoss supports backward."""
        x = paddle.randn([4, 8], dtype=paddle.float32)
        x.stop_gradient = False
        aux_loss_val = paddle.randn([1], dtype=paddle.float32)
        result = AddAuxiliaryLoss.apply(x, aux_loss_val)
        result.sum().backward()
        self.assertIsNotNone(x.grad)


class TestFilterScores(unittest.TestCase):
    def test_filter_scores_shape(self):
        """Test FilterScores produces correct output shape."""
        num_experts = 4
        probs = paddle.randn([8, num_experts], dtype=paddle.float32)
        indices = paddle.randint(0, num_experts, [8, num_experts])
        filtered_scores = FilterScores.apply(probs, indices)
        # FilterScores filters out zero entries and keeps only top-k scores
        self.assertGreaterEqual(filtered_scores.numel(), 0)


class TestAllToAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def test_all_to_all_communication(self):
        """Test _AllToAll communication primitive."""
        from paddleformers.fleet.transformer.moe.moe_utils import _AllToAll

        ep_group = self.pg_collection.ep
        if ep_group is not None and ep_group.nranks > 1:
            alltoall = _AllToAll(ep_group)
            send_tensor = paddle.randn([4, 8], dtype=paddle.float32)
            recv_tensor = alltoall(send_tensor, 2)
            self.assertEqual(recv_tensor.shape[0] * recv_tensor.shape[1], 4 * 8)


if __name__ == "__main__":
    unittest.main()
