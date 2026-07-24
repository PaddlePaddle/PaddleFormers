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


class TestCheckDataTypesExtra(unittest.TestCase):
    """Additional tests for _check_data_types."""

    def test_single_key_matching_dtype(self):
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {"a": paddle.zeros([2, 3], dtype=paddle.float32)}
        _check_data_types(["a"], data, paddle.float32)

    def test_empty_keys(self):
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {}
        _check_data_types([], data, paddle.float32)

    def test_float16_dtype(self):
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {"a": paddle.zeros([2, 3], dtype=paddle.float16)}
        _check_data_types(["a"], data, paddle.float16)

    def test_int32_dtype(self):
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {"a": paddle.zeros([2, 3], dtype=paddle.int32)}
        _check_data_types(["a"], data, paddle.int32)

    def test_multiple_keys_mixed_fails(self):
        from paddleformers.fleet.tensor_parallel.data import _check_data_types

        data = {
            "a": paddle.zeros([2, 3], dtype=paddle.float32),
            "b": paddle.zeros([2, 3], dtype=paddle.float16),
        }
        with self.assertRaises(AssertionError):
            _check_data_types(["a", "b"], data, paddle.float32)


class TestBuildKeySizeNumelExtra(unittest.TestCase):
    """Additional tests for _build_key_size_numel_dictionaries."""

    def test_single_key_on_rank0(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0]

        data = {"w": paddle.zeros([4, 8, 16], dtype=paddle.float32)}

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast"),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries(
                    ["w"], data, tp_group=mock_group
                )
            )

        self.assertEqual(key_size["w"], [4, 8, 16])
        self.assertEqual(key_numel["w"], 512)
        self.assertEqual(total_numel, 512)

    def test_empty_keys(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0]

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast"),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries([], {}, tp_group=mock_group)
            )

        self.assertEqual(len(key_size), 0)
        self.assertEqual(total_numel, 0)

    def test_large_tensor_dim(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0]

        # Test with a 4D tensor
        data = {"t": paddle.zeros([2, 3, 4, 5], dtype=paddle.float32)}

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast"),
        ):
            key_size, key_numel, total_numel = (
                _build_key_size_numel_dictionaries(
                    ["t"], data, tp_group=mock_group
                )
            )

        self.assertEqual(key_size["t"], [2, 3, 4, 5])
        self.assertEqual(key_numel["t"], 120)

    def test_dim_too_large_raises(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import (
            _build_key_size_numel_dictionaries,
        )

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0]

        # 6D tensor should exceed _MAX_DATA_DIM=5
        data = {"t": paddle.zeros([1, 2, 3, 4, 5, 6], dtype=paddle.float32)}

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast"),
            self.assertRaises(AssertionError),
        ):
            _build_key_size_numel_dictionaries(["t"], data, tp_group=mock_group)


class TestBroadcastDataExtra(unittest.TestCase):
    """Additional tests for broadcast_data."""

    def test_broadcast_single_key(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import broadcast_data

        mock_group = MagicMock()
        mock_group.rank = 0
        mock_group.ranks = [0]

        data = {"w": paddle.randn([3, 4], dtype=paddle.float32)}

        def mock_broadcast(tensor, src, group=None):
            pass

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
            patch(
                "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries",
                return_value=({"w": [3, 4]}, {"w": 12}, 12),
            ),
        ):
            output = broadcast_data(
                ["w"], data, paddle.float32, tp_group=mock_group
            )

        self.assertEqual(output["w"].shape, [3, 4])

    def test_broadcast_non_rank0_receives_data(self):
        import paddle.distributed as dist

        from paddleformers.fleet.tensor_parallel.data import broadcast_data

        mock_group = MagicMock()
        mock_group.rank = 1
        mock_group.ranks = [0, 1]

        data = {"w": paddle.zeros([2, 3], dtype=paddle.float32)}

        def mock_broadcast(tensor, src, group=None):
            pass

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch.object(dist, "broadcast", side_effect=mock_broadcast),
            patch(
                "paddleformers.fleet.tensor_parallel.data._build_key_size_numel_dictionaries",
                return_value=({"w": [2, 3]}, {"w": 6}, 6),
            ),
        ):
            output = broadcast_data(
                ["w"], data, paddle.float32, tp_group=mock_group
            )

        self.assertIn("w", output)
        self.assertEqual(output["w"].shape, [2, 3])


class TestGetTensorModelParallelGroupIfNoneInData(unittest.TestCase):
    """Tests for the tp_group default behavior used in data.py functions."""

    def test_get_tensor_model_parallel_group_none_no_dist(self):
        from paddleformers.fleet.utils import (
            get_tensor_model_parallel_group_if_none,
        )

        with patch("paddle.distributed.is_initialized", return_value=False):
            result = get_tensor_model_parallel_group_if_none(None)
            self.assertIsNone(result)

    def test_get_tensor_model_parallel_group_single_rank(self):
        from paddleformers.fleet.utils import (
            get_tensor_model_parallel_group_if_none,
        )

        mock_group = MagicMock()
        mock_group.ranks = [0]

        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_tensor_model_parallel_group_if_none(mock_group)
            self.assertEqual(result, mock_group)


class TestMaxDataDimConstants(unittest.TestCase):
    """Tests for constants used in data module."""

    def test_max_data_dim(self):
        from paddleformers.fleet.tensor_parallel.data import _MAX_DATA_DIM

        self.assertIsInstance(_MAX_DATA_DIM, int)
        self.assertGreater(_MAX_DATA_DIM, 0)

    def test_max_data_dim_value(self):
        from paddleformers.fleet.tensor_parallel.data import _MAX_DATA_DIM

        self.assertEqual(_MAX_DATA_DIM, 5)


if __name__ == "__main__":
    unittest.main()
