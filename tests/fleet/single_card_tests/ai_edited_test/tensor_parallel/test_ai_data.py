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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestCheckDataTypes(unittest.TestCase):
    """Tests for _check_data_types in data.py."""

    def test_check_data_types_pass(self):
        """All keys have matching dtype should pass without assertion."""
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {
            "a": paddle.zeros([2, 3], dtype=paddle.float32),
            "b": paddle.ones([4, 5], dtype=paddle.float32),
        }
        # Should not raise
        _check_data_types(["a", "b"], data, paddle.float32)

    def test_check_data_types_fail(self):
        """Mismatched dtype should raise AssertionError."""
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {
            "a": paddle.zeros([2, 3], dtype=paddle.float32),
            "b": paddle.ones([4, 5], dtype=paddle.float16),
        }
        with self.assertRaises(AssertionError):
            _check_data_types(["a", "b"], data, paddle.float32)


class TestBuildKeySizeNumelDictionaries(unittest.TestCase):
    """Tests for _build_key_size_numel_dictionaries in data.py."""

    def test_build_key_size_numel_on_rank0(self):
        """Test building key size/numel dicts on rank 0."""
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0, 1]

        data = {
            "w1": paddle.zeros([4, 8], dtype=paddle.float32),
            "w2": paddle.ones([16], dtype=paddle.float32),
        }

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast"),
        ):
            key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
                ["w1", "w2"], data, tp_group=mock_group
            )

        self.assertEqual(key_size["w1"], [4, 8])
        self.assertEqual(key_numel["w1"], 32)
        self.assertEqual(key_size["w2"], [16])
        self.assertEqual(key_numel["w2"], 16)
        self.assertEqual(total_numel, 48)

    def test_build_key_size_numel_on_non_rank0(self):
        """Test building key size/numel dicts on non-rank 0 (receives broadcast)."""
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 1
        mock_group.ranks = [0, 1]

        # On non-rank 0, data values are not used for sizes
        data = {
            "w1": paddle.zeros([4, 8], dtype=paddle.float32),
            "w2": paddle.ones([16], dtype=paddle.float32),
        }

        def mock_broadcast(tensor, src, group=None):
            # Simulate broadcast by filling with expected sizes
            # w1: [4, 8, 0, 0, 0], w2: [16, 0, 0, 0, 0]
            expected = paddle.to_tensor([4, 8, 0, 0, 0, 16, 0, 0, 0, 0], dtype=paddle.int32)
            tensor.copy_(expected, False)

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
        ):
            key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(
                ["w1", "w2"], data, tp_group=mock_group
            )

        self.assertEqual(key_size["w1"], [4, 8])
        self.assertEqual(key_numel["w1"], 32)
        self.assertEqual(key_size["w2"], [16])
        self.assertEqual(key_numel["w2"], 16)
        self.assertEqual(total_numel, 48)


class TestBroadcastData(unittest.TestCase):
    """Tests for broadcast_data in data.py."""

    def test_broadcast_data_rank0(self):
        """Test broadcast_data on rank 0."""
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import broadcast_data

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0, 1]

        data = {
            "w1": paddle.randn([2, 4], dtype=paddle.float32),
            "w2": paddle.randn([8], dtype=paddle.float32),
        }

        def mock_broadcast(tensor, src, group=None):
            # No-op for rank 0 in single-process test
            pass

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
            patch(
                "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries",
                return_value=(
                    {"w1": [2, 4], "w2": [8]},
                    {"w1": 8, "w2": 8},
                    16,
                ),
            ),
        ):
            output = broadcast_data(["w1", "w2"], data, paddle.float32, tp_group=mock_group)

        self.assertIn("w1", output)
        self.assertIn("w2", output)
        self.assertEqual(output["w1"].shape, [2, 4])
        self.assertEqual(output["w2"].shape, [8])

    def test_broadcast_data_non_rank0(self):
        """Test broadcast_data on non-rank 0."""
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import broadcast_data

        mock_group = MagicMock()
        mock_group.rank = 1
        mock_group.ranks = [0, 1]

        data = {
            "w1": paddle.zeros([2, 4], dtype=paddle.float32),
            "w2": paddle.zeros([8], dtype=paddle.float32),
        }

        def mock_broadcast(tensor, src, group=None):
            pass

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
            patch(
                "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries",
                return_value=(
                    {"w1": [2, 4], "w2": [8]},
                    {"w1": 8, "w2": 8},
                    16,
                ),
            ),
        ):
            output = broadcast_data(["w1", "w2"], data, paddle.float32, tp_group=mock_group)

        self.assertIn("w1", output)
        self.assertIn("w2", output)


class TestMaxDataDim(unittest.TestCase):
    """Tests for _MAX_DATA_DIM constant."""

    def test_max_data_dim_value(self):
        from paddleformers.fleet.tensor_parallel.data import _MAX_DATA_DIM

        self.assertEqual(_MAX_DATA_DIM, 5)
