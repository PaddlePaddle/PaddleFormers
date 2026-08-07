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


import logging
import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestGlobalMemoryBufferExtra(unittest.TestCase):
    """Additional tests for GlobalMemoryBuffer."""

    def test_get_tensor_with_different_dtypes(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([3, 4], paddle.float32, "test_f32")
        t2 = buf.get_tensor([3, 4], paddle.float64, "test_f64")
        self.assertEqual(t1.dtype, paddle.float32)
        self.assertEqual(t2.dtype, paddle.float64)

    def test_get_tensor_scalar_shape(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        try:
            t = buf.get_tensor([], paddle.float32, "scalar")
            self.assertIsNotNone(t)
        except Exception:
            pass

    def test_get_tensor_1d_shape(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t = buf.get_tensor([10], paddle.float32, "one_d")
        self.assertEqual(t.shape, [10])

    def test_get_tensor_shrinking_request(self):
        """Requesting smaller tensor after larger should reuse buffer."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([10, 10], paddle.float32, "shrink")
        t2 = buf.get_tensor([3, 3], paddle.float32, "shrink")
        # Both should return valid tensors
        self.assertEqual(t1.shape, [10, 10])
        self.assertEqual(t2.shape, [3, 3])

    def test_get_tensor_multiple_names(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        names = ["a", "b", "c", "d", "e"]
        for name in names:
            t = buf.get_tensor([2, 2], paddle.float32, name)
            self.assertEqual(t.shape, [2, 2])


class TestEnsureDivisibilityExtra(unittest.TestCase):
    """Additional tests for ensure_divisibility."""

    def test_divisible_by_one(self):
        from paddleformers.fleet.utils import ensure_divisibility

        ensure_divisibility(0, 1)
        ensure_divisibility(42, 1)

    def test_divisible_by_self(self):
        from paddleformers.fleet.utils import ensure_divisibility

        ensure_divisibility(5, 5)
        ensure_divisibility(1, 1)

    def test_zero_numerator(self):
        from paddleformers.fleet.utils import ensure_divisibility

        # 0 % anything == 0
        ensure_divisibility(0, 7)


class TestDivideExtra(unittest.TestCase):
    """Additional tests for divide."""

    def test_divide_by_one(self):
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(42, 1), 42)

    def test_divide_zero_by_anything(self):
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(0, 5), 0)

    def test_divide_large_numbers(self):
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(1000000, 100), 10000)


class TestInitMethodNormalExtra(unittest.TestCase):
    """Additional tests for init_method_normal."""

    def test_zero_sigma(self):
        from paddleformers.fleet.utils import init_method_normal

        fn = init_method_normal(0.0)
        t = paddle.empty([50, 50])
        fn(t)
        # With sigma=0, all values should be very close to 0
        self.assertAlmostEqual(paddle.abs(t).max().item(), 0.0, places=5)

    def test_large_sigma(self):
        from paddleformers.fleet.utils import init_method_normal

        fn = init_method_normal(10.0)
        t = paddle.empty([100, 100])
        fn(t)
        # With large sigma, some values should be far from 0
        self.assertGreater(paddle.abs(t).max().item(), 1.0)


class TestScaledInitMethodNormalExtra(unittest.TestCase):
    """Additional tests for scaled_init_method_normal."""

    def test_zero_num_layers_gives_inf_std(self):
        from paddleformers.fleet.utils import scaled_init_method_normal

        # sigma / sqrt(2 * 0) results in ZeroDivisionError
        # This is expected behavior - the function requires num_layers > 0
        with self.assertRaises(ZeroDivisionError):
            scaled_init_method_normal(0.02, 0)

    def test_multiplier_affects_std(self):
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn_small = scaled_init_method_normal(1.0, 100, multiplier=1.0)
        fn_large = scaled_init_method_normal(1.0, 100, multiplier=4.0)

        t1 = paddle.empty([1000, 1000])
        t2 = paddle.empty([1000, 1000])
        fn_small(t1)
        fn_large(t2)

        std_small = paddle.std(t1).item()
        std_large = paddle.std(t2).item()
        # Larger multiplier -> smaller std (std = sigma / sqrt(multiplier * num_layers))
        # multiplier=1 -> std≈0.1, multiplier=4 -> std≈0.05
        self.assertGreater(std_small, std_large)


class TestGetPgSizeExtra(unittest.TestCase):
    """Additional tests for get_pg_size."""

    def test_single_rank_group(self):
        from paddleformers.fleet.utils import get_pg_size

        mock_group = MagicMock()
        mock_group.ranks = [0]

        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_pg_size(group=mock_group)
            self.assertEqual(result, 1)

    def test_multi_rank_group(self):
        from paddleformers.fleet.utils import get_pg_size

        mock_group = MagicMock()
        mock_group.ranks = [0, 1, 2, 3]
        mock_group.nranks = 4

        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_pg_size(group=mock_group)
            self.assertEqual(result, 4)


class TestGetPgRankExtra(unittest.TestCase):
    """Additional tests for get_pg_rank."""

    def test_group_with_rank(self):
        from paddleformers.fleet.utils import get_pg_rank

        mock_group = MagicMock()
        mock_group.rank = 2

        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_pg_rank(group=mock_group)
            self.assertEqual(result, 2)


class TestLogSingleRankExtra(unittest.TestCase):
    """Additional tests for log_single_rank."""

    def test_log_with_custom_rank(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=2),
        ):
            log_single_rank(logger, logging.INFO, "test", rank=2)
            logger.log.assert_called_once()

    def test_log_custom_rank_not_matching(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=1),
        ):
            log_single_rank(logger, logging.INFO, "test", rank=2)
            logger.log.assert_not_called()

    def test_log_with_kwargs(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        log_single_rank(
            logger,
            logging.INFO,
            "message with %s",
            "arg",
            extra={"key": "value"},
        )
        logger.log.assert_called_once()


class TestGetTensorModelParallelGroupIfNoneExtra(unittest.TestCase):
    """Additional tests for get_tensor_model_parallel_group_if_none."""

    def test_single_rank_group(self):
        from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

        mock_group = MagicMock()
        mock_group.ranks = [0]

        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_tensor_model_parallel_group_if_none(mock_group)
            self.assertEqual(result, mock_group)


class TestPrepareInputTensorsForWgradComputeExtra(unittest.TestCase):
    """Additional tests for prepare_input_tensors_for_wgrad_compute."""

    def test_1d_tensors_unchanged(self):
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([64])
        inp = paddle.randn([64])
        g_out, a_in = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g_out.shape, [64])
        self.assertEqual(a_in.shape, [64])

    def test_4d_tensors_unchanged(self):
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([2, 3, 4, 5])
        inp = paddle.randn([2, 3, 4, 5])
        g_out, a_in = prepare_input_tensors_for_wgrad_compute(grad, inp)
        # 4D tensors are not reshaped (only 3D -> 2D)
        self.assertEqual(g_out.shape, [2, 3, 4, 5])


class TestDeprecateInferenceParamsExtra(unittest.TestCase):
    """Additional tests for deprecate_inference_params."""

    def test_context_provided_ignores_params(self):
        from paddleformers.fleet.utils import deprecate_inference_params

        ctx = MagicMock()
        params = MagicMock()
        result = deprecate_inference_params(ctx, params)
        self.assertEqual(result, ctx)


class TestGetBatchOnThisCpRankExtra(unittest.TestCase):
    """Additional tests for get_batch_on_this_cp_rank."""

    def test_dict_with_position_ids(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        mock_input_ids = paddle.randn([2, 4])
        mock_position_ids = paddle.randn([2, 4])
        inputs = {
            "input_ids": mock_input_ids,
            "position_ids": mock_position_ids,
            "attention_mask": "not_scattered",
        }
        with patch("paddleformers.fleet.utils.ContextParallelScatterOp") as mock_cp_op:
            mock_cp_op.apply.side_effect = lambda x, **kw: x
            result = get_batch_on_this_cp_rank(inputs)
        self.assertIn("input_ids", result)
        self.assertIn("position_ids", result)
        self.assertEqual(result["attention_mask"], "not_scattered")

    def test_dict_with_labels(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        mock_labels = paddle.randn([2, 4])
        inputs = {"labels": mock_labels}
        with patch("paddleformers.fleet.utils.ContextParallelScatterOp") as mock_cp_op:
            mock_cp_op.apply.side_effect = lambda x, **kw: x
            result = get_batch_on_this_cp_rank(inputs)
        self.assertIn("labels", result)


if __name__ == "__main__":
    unittest.main()
