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


class TestGetBatchOnThisCpRank(unittest.TestCase):
    """Tests for get_batch_on_this_cp_rank in utils.py."""

    def test_with_tensor_input(self):
        """Test with tensor input returns result."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        # Mock ContextParallelScatterOp
        with patch(
            "paddleformers.fleet.utils.ContextParallelScatterOp"
        ) as mock_cp:
            mock_instance = MagicMock()
            mock_cp.apply = MagicMock(return_value=paddle.randn([2, 4]))
            inp = paddle.randn([2, 8])
            result = get_batch_on_this_cp_rank(inp)

    def test_with_dict_input(self):
        """Test with dict input processes specified keys."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with patch(
            "paddleformers.fleet.utils.ContextParallelScatterOp"
        ) as mock_cp:
            mock_cp.apply = MagicMock(return_value=paddle.randn([2, 4]))
            inputs = {
                "input_ids": paddle.randn([2, 8]),
                "position_ids": paddle.randn([2, 8]),
                "labels": paddle.randn([2, 8]),
                "other": paddle.randn([2, 8]),
            }
            result = get_batch_on_this_cp_rank(inputs)
            self.assertIsInstance(result, dict)

    def test_with_list_input_raises(self):
        """Test with list input raises AssertionError."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(AssertionError):
            get_batch_on_this_cp_rank([paddle.randn([2, 8])])

    def test_with_invalid_type_raises(self):
        """Test with invalid input type raises ValueError."""
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(ValueError):
            get_batch_on_this_cp_rank("not_valid")


class TestNvtxDecorator(unittest.TestCase):
    """Tests for nvtx_decorator function."""

    def test_nvtx_decorator_returns_callable(self):
        """Test nvtx_decorator returns a callable."""
        from paddleformers.fleet.utils import nvtx_decorator

        result = nvtx_decorator()
        self.assertTrue(callable(result))

    def test_nvtx_decorator_with_message(self):
        """Test nvtx_decorator with custom message."""
        from paddleformers.fleet.utils import nvtx_decorator

        result = nvtx_decorator(message="Custom Range")
        self.assertTrue(callable(result))

    def test_nvtx_decorator_applied_to_function(self):
        """Test nvtx_decorator applied to a function."""
        from paddleformers.fleet.utils import nvtx_decorator

        @nvtx_decorator()
        def my_function():
            return 42

        # Function should still work (NVTX is disabled by default)
        result = my_function()
        self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
