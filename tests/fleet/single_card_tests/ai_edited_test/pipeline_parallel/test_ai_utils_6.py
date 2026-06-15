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


class TestPaddle2Number(unittest.TestCase):
    """Tests for paddle_2_number conversion."""

    def test_float16(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float16), 0)

    def test_float32(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float32), 1)

    def test_float64(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.float64), 2)

    def test_int32(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.int32), 3)

    def test_int64(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.int64), 4)

    def test_bfloat16(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.bfloat16), 5)

    def test_bool(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        self.assertEqual(paddle_2_number(paddle.bool), 6)

    def test_invalid_dtype(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            paddle_2_number,
        )

        with self.assertRaises(AssertionError):
            paddle_2_number("not_a_dtype")


class TestNumber2Dtype(unittest.TestCase):
    """Tests for number_2_dtype conversion."""

    def test_float16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(0), "float16")

    def test_float32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(1), "float32")

    def test_float64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(2), "float64")

    def test_int32(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(3), "int32")

    def test_int64(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(4), "int64")

    def test_bfloat16(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(5), "bfloat16")

    def test_bool(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        self.assertEqual(number_2_dtype(6), "bool")

    def test_invalid_number(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import number_2_dtype

        with self.assertRaises(AssertionError):
            number_2_dtype(99)


class TestPaddle2NumberRoundTrip(unittest.TestCase):
    """Round trip tests for dtype conversion."""

    def test_round_trip_all_types(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            number_2_dtype,
            paddle_2_number,
        )

        dtypes = [
            paddle.float16,
            paddle.float32,
            paddle.float64,
            paddle.int32,
            paddle.int64,
            paddle.bfloat16,
            paddle.bool,
        ]
        for dtype in dtypes:
            num = paddle_2_number(dtype)
            dtype_str = number_2_dtype(num)
            self.assertIsInstance(dtype_str, str)
            self.assertTrue(len(dtype_str) > 0)


class TestProfilePipelineDetails(unittest.TestCase):
    """Tests for profile_pipeline_details function."""

    # TODO(hushenwei2000): enable this test after migrate to paddle pp
    # Importing paddleformers.fleet triggers paddleformers.fleet/context_parallel_utils.py at module
    # level which calls paddle.cuda.get_device_capability(), and that fails when
    # is_compiled_with_cuda is patched to False.
    # @patch("paddle.base.core.is_compiled_with_cuda", return_value=False)
    # def test_profile_no_cuda(self, mock_cuda):
    #     from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
    #         profile_pipeline_details,
    #     )
    #
    #     mock_logger = MagicMock()
    #     with patch(
    #         "paddleformers.fleet.pipeline_parallel.pp_utils.utils.get_sync_logger",
    #         return_value=mock_logger,
    #     ):
    #         profile_pipeline_details("test message")
    #         mock_logger.info.assert_called_once()
    #         call_args = mock_logger.info.call_args[0][0]
    #         self.assertIn("test message", call_args)
    #         self.assertIn("memory_allocated_size=0.00", call_args)

    @patch("paddle.base.core.is_compiled_with_cuda", return_value=True)
    @patch("paddle.device.cuda.memory_allocated", return_value=1e9)
    @patch("paddle.device.cuda.memory_reserved", return_value=2e9)
    def test_profile_with_cuda(self, mock_reserved, mock_allocated, mock_cuda):
        from paddleformers.fleet.pipeline_parallel.pp_utils.utils import (
            profile_pipeline_details,
        )

        mock_logger = MagicMock()
        with patch(
            "paddleformers.fleet.pipeline_parallel.pp_utils.utils.get_sync_logger",
            return_value=mock_logger,
        ):
            profile_pipeline_details("cuda_test")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("cuda_test", call_args)


class TestDictToTupleHelper(unittest.TestCase):
    """Tests for dict_to_tuple_helper - converts dict to tuple."""

    def test_dict_input_converted(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        tensor = paddle.randn([2, 3])
        tensor.key = "test_key"
        data = {"test_key": tensor}
        result = dict_to_tuple_helper(data)
        # Returns a tuple of tensors
        self.assertIsInstance(result, tuple)

    def test_non_dict_passthrough(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            dict_to_tuple_helper,
        )

        # Non-dict input passes through directly
        tensor = paddle.randn([2, 3])
        result = dict_to_tuple_helper(tensor)
        self.assertIs(result, tensor)


class TestTupleToDictHelper(unittest.TestCase):
    """Tests for tuple_to_dict_helper - converts tuple to dict if tensors have .key."""

    def test_tuple_without_key_passthrough(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = tuple_to_dict_helper((t1, t2))
        # Real paddle.Tensor doesn't have .key, so use_dict=False
        self.assertFalse(result[1])

    def test_none_single_input(self):
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        # None doesn't have .key, so use_dict=False
        result = tuple_to_dict_helper(None)
        self.assertFalse(result[1])


class TestConvertTensorDictToTuple(unittest.TestCase):
    """Tests for convert_tensor_dict_to_tuple."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        tensor = paddle.randn([2, 3])
        tensor.key = "test_key"
        result = convert_tensor_dict_to_tuple({"test_key": tensor})
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_list_value(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        result = convert_tensor_dict_to_tuple({"key": [t1, t2]})
        self.assertEqual(len(result), 2)


class TestConvertTensorTupleToDict(unittest.TestCase):
    """Tests for convert_tensor_tuple_to_dict."""

    def test_single_tensor(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        tensor = paddle.randn([2, 3])
        tensor.key = "single_key"
        result = convert_tensor_tuple_to_dict((tensor,))
        self.assertIn("single_key", result)

    def test_multi_tensor_with_spaces(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        t1 = paddle.randn([2, 3])
        t1.key = "key 0"
        t2 = paddle.randn([2, 3])
        t2.key = "key 1"
        result = convert_tensor_tuple_to_dict((t1, t2))
        self.assertIn("key", result)
        self.assertIsInstance(result["key"], list)
        self.assertEqual(len(result["key"]), 2)


class TestTupleToDictHelperWithKey(unittest.TestCase):
    """Tests for tuple_to_dict_helper with keyed tensors."""

    def test_tuple_with_key(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        t1 = paddle.randn([2, 3])
        t1.key = "key1"
        t2 = paddle.randn([2, 3])
        t2.key = "key2"
        result = tuple_to_dict_helper((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertTrue(result[1])  # use_dict=True


if __name__ == "__main__":
    unittest.main()
