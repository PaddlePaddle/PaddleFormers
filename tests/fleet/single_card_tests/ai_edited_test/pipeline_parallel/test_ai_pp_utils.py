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


class TestPaddle2Number(unittest.TestCase):
    """Tests for paddle_2_number function."""

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
            paddle_2_number("invalid")


class TestNumber2Dtype(unittest.TestCase):
    """Tests for number_2_dtype function."""

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


class TestProfilePipelineDetails(unittest.TestCase):
    """Tests for profile_pipeline_details function."""

    def test_profile_on_cpu(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            profile_pipeline_details,
        )

        mock_logger = MagicMock()
        with (
            patch("paddle.device.cuda.memory_allocated", return_value=0),
            patch("paddle.device.cuda.memory_reserved", return_value=0),
            patch(
                "paddle.distributed.fleet.meta_parallel.pipeline_parallel.get_sync_logger",
                return_value=mock_logger,
            ),
        ):
            profile_pipeline_details("test_msg")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("test_msg", call_args)

    def test_profile_on_gpu(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            profile_pipeline_details,
        )

        mock_logger = MagicMock()
        with (
            patch("paddle.device.cuda.memory_allocated", return_value=1e9),
            patch("paddle.device.cuda.memory_reserved", return_value=2e9),
            patch(
                "paddle.distributed.fleet.meta_parallel.pipeline_parallel.get_sync_logger",
                return_value=mock_logger,
            ),
        ):
            profile_pipeline_details("test_msg")
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("test_msg", call_args)


class TestDictTupleHelpers(unittest.TestCase):
    """Tests for dict/tuple helper functions."""

    def test_dict_to_tuple_passthrough(self):
        from paddle.distributed.fleet.meta_parallel import dict_to_tuple_helper

        result = dict_to_tuple_helper("single_tensor")
        self.assertEqual(result, "single_tensor")

    def test_dict_to_tuple_dict(self):
        from paddle.distributed.fleet.meta_parallel import dict_to_tuple_helper

        mock_t1 = MagicMock()
        mock_t2 = MagicMock()
        mock_dict = {"a": mock_t1, "b": mock_t2}
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.utils.convert_tensor_dict_to_tuple",
            return_value=(mock_t1, mock_t2),
        ) as mock_fn:
            result = dict_to_tuple_helper(mock_dict)
            mock_fn.assert_called_once_with(output_tensor_dict=mock_dict)

    def test_tuple_to_dict_with_key(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            tuple_to_dict_helper,
        )

        t = paddle.randn([2, 3])
        t.key = "test_key"
        with patch(
            "paddle.distributed.fleet.meta_parallel.pp_utils.utils.convert_tensor_tuple_to_dict",
            return_value={"test_key": t},
        ) as mock_fn:
            result, use_dict = tuple_to_dict_helper(t)
            self.assertTrue(use_dict)
            mock_fn.assert_called_once()

    def test_convert_tensor_dict_to_tuple_single(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        t = paddle.randn([2, 3])
        d = {"a": t}
        result = convert_tensor_dict_to_tuple(d)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].key, "a")

    def test_convert_tensor_dict_to_tuple_list(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_dict_to_tuple,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        d = {"a": [t1, t2]}
        result = convert_tensor_dict_to_tuple(d)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIn("a 0", result[0].key)
        self.assertIn("a 1", result[1].key)

    def test_convert_tensor_tuple_to_dict_single(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        t1 = paddle.randn([2, 3])
        t1.key = "a"
        result = convert_tensor_tuple_to_dict((t1,))
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)

    def test_convert_tensor_tuple_to_dict_list(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pp_utils.utils import (
            convert_tensor_tuple_to_dict,
        )

        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        t1.key = "a 0"
        t2.key = "a 1"
        result = convert_tensor_tuple_to_dict((t1, t2))
        self.assertIsInstance(result, dict)
        self.assertIn("a", result)
        self.assertEqual(len(result["a"]), 2)


if __name__ == "__main__":
    unittest.main()
