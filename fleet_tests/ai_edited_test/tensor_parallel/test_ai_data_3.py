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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.tensor_parallel.data import (
    _MAX_DATA_DIM,
    _build_key_size_numel_dictionaries,
    _check_data_types,
    broadcast_data,
)


def _make_group(world_size=2, rank=0):
    """Create a mock process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank = rank
    group.nranks = world_size
    group.ranks = list(range(world_size))
    return group


class TestMaxDataDim(unittest.TestCase):
    """Tests for _MAX_DATA_DIM constant."""

    def test_max_data_dim_value(self):
        """Test _MAX_DATA_DIM is 5."""
        self.assertEqual(_MAX_DATA_DIM, 5)


class TestCheckDataTypes(unittest.TestCase):
    """Tests for _check_data_types function."""

    def test_matching_types_pass(self):
        """Test no assertion when all data types match."""
        data = {
            "a": paddle.randn([2, 3]),
            "b": paddle.randn([2, 3]),
        }
        _check_data_types(["a", "b"], data, paddle.float32)

    def test_mismatched_types_raise(self):
        """Test assertion when data types do not match."""
        data = {
            "a": paddle.randn([2, 3]).cast("float32"),
            "b": paddle.randn([2, 3]).cast("float64"),
        }
        with self.assertRaises(AssertionError):
            _check_data_types(["a", "b"], data, paddle.float32)


class TestBuildKeySizeNumelDictionaries(unittest.TestCase):
    """Tests for _build_key_size_numel_dictionaries function."""

    @patch("paddleformers.fleet.tensor_parallel.data.paddle.distributed.broadcast")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.tensor")
    @patch(
        "paddleformers.fleet.tensor_parallel.data.get_tensor_model_parallel_group_if_none"
    )
    def test_build_on_rank_zero(
        self, mock_get_group, mock_tensor, mock_broadcast
    ):
        """Test building dictionaries on rank 0."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group

        # Make paddle.tensor return something that supports .cpu()
        mock_sizes = MagicMock()
        mock_sizes.cpu.return_value = paddle.to_tensor(
            [4, 2, 0, 0, 0, 4, 2, 0, 0, 0], dtype=paddle.int32
        )
        mock_tensor.return_value = mock_sizes

        data = {
            "w": paddle.randn([4, 2]),
        }
        key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
            ["w"], data
        )
        self.assertIn("w", key_size)
        self.assertEqual(key_numel["w"], 8)
        self.assertEqual(total_numel, 8)

    @patch("paddleformers.fleet.tensor_parallel.data.paddle.distributed.broadcast")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.tensor")
    @patch(
        "paddleformers.fleet.tensor_parallel.data.get_tensor_model_parallel_group_if_none"
    )
    def test_dim_assertion_raises(
        self, mock_get_group, mock_tensor, mock_broadcast
    ):
        """Test assertion when tensor dim >= MAX_DATA_DIM."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group

        mock_sizes = MagicMock()
        mock_sizes.cpu.return_value = paddle.zeros([10], dtype=paddle.int32)
        mock_tensor.return_value = mock_sizes

        # Create a 6-D tensor (>= _MAX_DATA_DIM=5)
        data = {"w": paddle.randn([2, 2, 2, 2, 2, 2])}
        with self.assertRaises(AssertionError):
            _build_key_size_numel_dictionaries(["w"], data)


class TestBroadcastData(unittest.TestCase):
    """Tests for broadcast_data function."""

    @patch("paddleformers.fleet.tensor_parallel.data.paddle.distributed.broadcast")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.narrow")
    @patch(
        "paddleformers.fleet.tensor_parallel.data.get_tensor_model_parallel_group_if_none"
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries"
    )
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.concat")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.empty")
    def test_broadcast_data_rank_zero(
        self,
        mock_empty,
        mock_concat,
        mock_build,
        mock_get_group,
        mock_narrow,
        mock_broadcast,
    ):
        """Test broadcast_data on rank zero packs data."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        mock_build.return_value = (
            {"w": [4, 2]},
            {"w": 8},
            8,
        )
        mock_concat.return_value = paddle.randn([8])
        mock_narrow.return_value = paddle.randn([4, 2])

        data = {"w": paddle.randn([4, 2])}
        result = broadcast_data(["w"], data, paddle.float32)
        self.assertIn("w", result)

    @patch("paddleformers.fleet.tensor_parallel.data.paddle.distributed.broadcast")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.narrow")
    @patch(
        "paddleformers.fleet.tensor_parallel.data.get_tensor_model_parallel_group_if_none"
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries"
    )
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.empty")
    def test_broadcast_data_non_zero_rank(
        self,
        mock_empty,
        mock_build,
        mock_get_group,
        mock_narrow,
        mock_broadcast,
    ):
        """Test broadcast_data on non-zero rank creates empty tensor."""
        group = _make_group(world_size=2, rank=1)
        mock_get_group.return_value = group
        mock_build.return_value = (
            {"w": [4, 2]},
            {"w": 8},
            8,
        )
        mock_empty.return_value = paddle.randn([8])
        mock_narrow.return_value = paddle.randn([4, 2])

        data = {}
        result = broadcast_data(["w"], data, paddle.float32)
        self.assertIn("w", result)


class TestBroadcastDataMultipleKeys(unittest.TestCase):
    """Tests for broadcast_data with multiple keys."""

    @patch("paddleformers.fleet.tensor_parallel.data.paddle.distributed.broadcast")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.narrow")
    @patch(
        "paddleformers.fleet.tensor_parallel.data.get_tensor_model_parallel_group_if_none"
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries"
    )
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.concat")
    @patch("paddleformers.fleet.tensor_parallel.data.paddle.empty")
    def test_multiple_keys(
        self,
        mock_empty,
        mock_concat,
        mock_build,
        mock_get_group,
        mock_narrow,
        mock_broadcast,
    ):
        """Test broadcast_data with multiple keys."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        mock_build.return_value = (
            {"w1": [4, 2], "w2": [4, 2]},
            {"w1": 8, "w2": 8},
            16,
        )
        mock_concat.return_value = paddle.randn([16])
        mock_narrow.side_effect = [
            paddle.randn([4, 2]),
            paddle.randn([4, 2]),
        ]

        data = {
            "w1": paddle.randn([4, 2]),
            "w2": paddle.randn([4, 2]),
        }
        result = broadcast_data(["w1", "w2"], data, paddle.float32)
        self.assertIn("w1", result)
        self.assertIn("w2", result)


if __name__ == "__main__":
    unittest.main()
