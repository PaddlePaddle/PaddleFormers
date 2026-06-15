# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.tensor_parallel.mappings import (
    _CopyToModelParallelRegion,
    _GatherFromModelParallelRegion,
    _GatherFromSequenceParallelRegion,
    _ReduceFromModelParallelRegion,
    _ReduceScatterToSequenceParallelRegion,
    _ScatterToModelParallelRegion,
    _split_along_first_dim,
    _split_along_last_dim,
)


def _tensors_equal(a, b):
    """Helper to compare paddle tensors for equality."""
    return bool(paddle.equal(a, b).numpy().all())


class TestSplitAlongFirstDimSingleGPU(unittest.TestCase):
    """Tests for _split_along_first_dim with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        x = paddle.randn([4, 8])
        result = _split_along_first_dim(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))


class TestSplitAlongLastDimSingleGPU(unittest.TestCase):
    """Tests for _split_along_last_dim with single GPU."""

    def test_returns_input_when_world_size_1(self):
        """Should return input when world_size is 1."""
        mock_group = MagicMock()
        mock_group.world_size = 1
        mock_group.ranks = [0]
        x = paddle.randn([4, 8])
        result = _split_along_last_dim(x, mock_group)
        self.assertTrue(_tensors_equal(result, x))


class TestCopyToModelParallelRegion(unittest.TestCase):
    """Tests for _CopyToModelParallelRegion."""

    def test_forward_returns_input(self):
        """Forward should return input unchanged."""
        mock_group = MagicMock()
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _CopyToModelParallelRegion.forward(ctx, x, mock_group)
        self.assertTrue(_tensors_equal(result, x))
        self.assertEqual(ctx.group, mock_group)

    def test_backward_with_none_group(self):
        """Backward with None group should return grad_output."""
        ctx = MagicMock()
        ctx.group = None
        grad = paddle.randn([4, 8])
        result = _CopyToModelParallelRegion.backward(ctx, grad)
        self.assertTrue(_tensors_equal(result, grad))


class TestReduceFromModelParallelRegion(unittest.TestCase):
    """Tests for _ReduceFromModelParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _ReduceFromModelParallelRegion.forward(ctx, x, None)
        self.assertTrue(_tensors_equal(result, x))

    def test_forward_with_single_rank_group(self):
        """Forward with single-rank group should return input."""
        ctx = MagicMock()
        mock_group = MagicMock()
        mock_group.nranks = 1
        x = paddle.randn([4, 8])
        result = _ReduceFromModelParallelRegion.forward(ctx, x, mock_group)
        self.assertTrue(_tensors_equal(result, x))

    def test_backward_returns_grad_output(self):
        """Backward should return grad_output unchanged."""
        ctx = MagicMock()
        grad = paddle.randn([4, 8])
        result = _ReduceFromModelParallelRegion.backward(ctx, grad)
        self.assertTrue(_tensors_equal(result, grad))


class TestScatterToModelParallelRegion(unittest.TestCase):
    """Tests for _ScatterToModelParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _ScatterToModelParallelRegion.forward(ctx, x, None)
        self.assertTrue(_tensors_equal(result, x))


class TestGatherFromModelParallelRegion(unittest.TestCase):
    """Tests for _GatherFromModelParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _GatherFromModelParallelRegion.forward(ctx, x, None)
        self.assertTrue(_tensors_equal(result, x))


class TestGatherFromSequenceParallelRegion(unittest.TestCase):
    """Tests for _GatherFromSequenceParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _GatherFromSequenceParallelRegion.forward(ctx, x, None, True, None, False)
        self.assertTrue(_tensors_equal(result, x))


class TestReduceScatterToSequenceParallelRegion(unittest.TestCase):
    """Tests for _ReduceScatterToSequenceParallelRegion."""

    def test_forward_with_none_group(self):
        """Forward with None group should return input."""
        ctx = MagicMock()
        x = paddle.randn([4, 8])
        result = _ReduceScatterToSequenceParallelRegion.forward(ctx, x, None, None, False)
        self.assertTrue(_tensors_equal(result, x))


if __name__ == "__main__":
    unittest.main()
