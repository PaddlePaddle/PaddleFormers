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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import (
    AllGatherGroupOp,
    _AllToAll,
    barrier_ep,
    reduce_scatter_group,
)


class TestBarrierEP(unittest.TestCase):
    """Tests for barrier_ep function."""

    def test_barrier_ep_with_mock_group(self):
        """Test barrier_ep calls paddle.distributed.barrier."""
        mock_group = MagicMock()
        with patch("paddle.distributed.barrier") as mock_barrier:
            barrier_ep(mock_group)
            mock_barrier.assert_called_once_with(mock_group)


class TestAllToAllForward(unittest.TestCase):
    """Tests for _AllToAll forward."""

    def test_forward_single_rank(self):
        """Test _AllToAll forward with world_size=1 returns input."""
        mock_group = MagicMock()
        with patch("paddle.distributed.get_world_size", return_value=1):
            x = paddle.randn([4, 8])
            out = _AllToAll.apply([4, 8], x, group=mock_group)
            self.assertTrue(paddle.allclose(x, out).item())


class TestReduceScatterGroup(unittest.TestCase):
    """Tests for reduce_scatter_group."""

    def test_single_rank(self):
        """Test reduce_scatter_group with parallelism=1 returns clone."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8])
        out = reduce_scatter_group(x, group=mock_group)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(paddle.allclose(x, out).item())


class TestAllGatherGroupOp(unittest.TestCase):
    """Tests for AllGatherGroupOp."""

    def test_forward_single_rank(self):
        """Test AllGatherGroupOp forward with parallelism=1."""
        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8])
        out = AllGatherGroupOp.apply(x, group=mock_group)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(paddle.allclose(x, out).item())


if __name__ == "__main__":
    unittest.main()
