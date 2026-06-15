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


class TestPipelineUtilsProfilePipelineDetails(unittest.TestCase):
    """Tests for profile_pipeline_details in pp_utils/utils.py."""

    @patch("paddle.base.core.is_compiled_with_cuda", return_value=True)
    @patch("paddle.device.cuda.memory_allocated", return_value=2e9)
    @patch("paddle.device.cuda.memory_reserved", return_value=4e9)
    def test_profile_with_cuda(self, mock_reserved, mock_allocated, mock_cuda):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            profile_pipeline_details,
        )

        mock_logger = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.pp_utils.utils.get_sync_logger",
            return_value=mock_logger,
        ):
            profile_pipeline_details("gpu_test")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("gpu_test", call_args)
            self.assertIn("memory_allocated_size", call_args)

    @patch("paddle.base.core.is_compiled_with_cuda", return_value=False)
    def test_profile_without_cuda(self, mock_cuda):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            profile_pipeline_details,
        )

        mock_logger = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.pp_utils.utils.get_sync_logger",
            return_value=mock_logger,
        ):
            profile_pipeline_details("cpu_test")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("cpu_test", call_args)
            self.assertIn("memory_allocated_size=0.00", call_args)


class TestPipelineUtilsP2pNumberDtype(unittest.TestCase):
    """Tests for paddle_2_number and number_2_dtype in pipeline_parallel/utils.py via pp_utils."""

    def test_paddle_2_number_all_types(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import paddle_2_number

        self.assertEqual(paddle_2_number(paddle.float16), 0)
        self.assertEqual(paddle_2_number(paddle.float32), 1)
        self.assertEqual(paddle_2_number(paddle.float64), 2)
        self.assertEqual(paddle_2_number(paddle.int32), 3)
        self.assertEqual(paddle_2_number(paddle.int64), 4)
        self.assertEqual(paddle_2_number(paddle.bfloat16), 5)
        self.assertEqual(paddle_2_number(paddle.bool), 6)

    def test_number_2_dtype_all_types(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(0), "float16")
        self.assertEqual(number_2_dtype(1), "float32")
        self.assertEqual(number_2_dtype(2), "float64")
        self.assertEqual(number_2_dtype(3), "int32")
        self.assertEqual(number_2_dtype(4), "int64")
        self.assertEqual(number_2_dtype(5), "bfloat16")
        self.assertEqual(number_2_dtype(6), "bool")


class TestPipelineUtilsDictTupleConversion(unittest.TestCase):
    """Tests for dict/tuple conversion in pipeline_parallel/pp_utils/utils.py."""

    def test_dict_to_tuple_helper_non_dict(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        tensor = paddle.randn([2, 3])
        result = dict_to_tuple_helper(tensor)
        self.assertIs(result, tensor)

    def test_dict_to_tuple_helper_dict(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        t = paddle.randn([2, 3])
        result = dict_to_tuple_helper({"key": t})
        self.assertIsInstance(result, tuple)

    def test_tuple_to_dict_helper_single_with_key(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        t = paddle.randn([2, 3])
        t.key = "my_key"
        # tuple_to_dict_helper expects a tuple when using key
        result, use_dict = tuple_to_dict_helper((t,))
        self.assertTrue(use_dict)

    def test_tuple_to_dict_helper_single_no_key(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        t = paddle.randn([2, 3])
        result, use_dict = tuple_to_dict_helper(t)
        self.assertFalse(use_dict)

    def test_convert_tensor_dict_to_tuple_with_list(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = convert_tensor_dict_to_tuple({"a": [t1, t2]})
        self.assertEqual(len(result), 2)
        self.assertTrue(hasattr(result[0], "key"))
        self.assertIn("a", result[0].key)

    def test_convert_tensor_tuple_to_dict_with_space_key(self):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        t1 = paddle.randn([2, 3])
        t1.key = "layer 0"
        t2 = paddle.randn([2, 3])
        t2.key = "layer 1"
        result = convert_tensor_tuple_to_dict((t1, t2))
        self.assertIn("layer", result)
        self.assertEqual(len(result["layer"]), 2)
        # key attribute should be deleted
        self.assertFalse(hasattr(t1, "key"))


if __name__ == "__main__":
    unittest.main()
