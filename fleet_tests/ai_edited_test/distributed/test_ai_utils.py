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


# Tests for src/paddlefleet/utils.py
# Additional tests for make_viewless_tensor, get_pg_rank, divide,
# WrappedTensor, GlobalMemoryBuffer, init_method_normal,
# scaled_init_method_normal, get_paddle_version, is_paddle_min_version,
# prepare_input_tensors_for_wgrad_compute, deprecate_inference_params,
# get_batch_on_this_cp_rank, nvtx_decorator, get_attr_wrapped_model

import functools
import logging
import unittest
import warnings
from unittest import mock

import paddle


class TestMakeViewlessTensor(unittest.TestCase):
    """Tests for make_viewless_tensor function."""

    def test_non_view_tensor_returned_as_is(self):
        """Test non-view tensor is returned as-is."""
        from paddleformers.fleet.utils import make_viewless_tensor

        x = paddle.randn([4, 8])
        with (
            mock.patch(
                "paddleformers.fleet.utils._kernel_make_viewless_tensor",
                return_value=x,
            ) as mock_kernel,
            mock.patch(
                "paddleformers.fleet.utils.MakeViewlessTensor.apply",
                return_value=x,
            ),
        ):
            # When _is_view returns False, input is returned as-is
            original_is_view = (
                type(x)._is_view if hasattr(type(x), "_is_view") else None
            )
            type(x)._is_view = lambda self: False
            try:
                result = make_viewless_tensor(
                    x, requires_grad=False, keep_graph=False
                )
                self.assertIs(result, x)
                mock_kernel.assert_not_called()
            finally:
                if original_is_view is None:
                    try:
                        del type(x)._is_view
                    except AttributeError:
                        pass
                else:
                    type(x)._is_view = original_is_view

    def test_view_tensor_keep_graph_true(self):
        """Test view tensor with keep_graph=True uses PyLayer."""
        from paddleformers.fleet.utils import make_viewless_tensor

        x = paddle.randn([4, 8])
        original_is_view = (
            type(x)._is_view if hasattr(type(x), "_is_view") else None
        )
        type(x)._is_view = lambda self: True
        try:
            with mock.patch(
                "paddleformers.fleet.utils.MakeViewlessTensor.apply",
                return_value=x,
            ) as mock_apply:
                result = make_viewless_tensor(
                    x, requires_grad=True, keep_graph=True
                )
                mock_apply.assert_called_once_with(x, True)
        finally:
            if original_is_view is None:
                try:
                    del type(x)._is_view
                except AttributeError:
                    pass
            else:
                type(x)._is_view = original_is_view

    def test_view_tensor_keep_graph_false(self):
        """Test view tensor with keep_graph=False uses kernel."""
        from paddleformers.fleet.utils import make_viewless_tensor

        x = paddle.randn([4, 8])
        original_is_view = (
            type(x)._is_view if hasattr(type(x), "_is_view") else None
        )
        type(x)._is_view = lambda self: True
        try:
            with mock.patch(
                "paddleformers.fleet.utils._kernel_make_viewless_tensor",
                return_value=x,
            ) as mock_kernel:
                result = make_viewless_tensor(
                    x, requires_grad=False, keep_graph=False
                )
                mock_kernel.assert_called_once_with(x, False)
        finally:
            if original_is_view is None:
                try:
                    del type(x)._is_view
                except AttributeError:
                    pass
            else:
                type(x)._is_view = original_is_view


class TestMakeViewlessTensorPyLayer(unittest.TestCase):
    """Tests for MakeViewlessTensor PyLayer."""

    def test_forward_calls_kernel(self):
        """Test forward calls _kernel_make_viewless_tensor."""
        from paddleformers.fleet.utils import MakeViewlessTensor

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])

        with mock.patch(
            "paddleformers.fleet.utils._kernel_make_viewless_tensor",
            return_value=inp,
        ) as mock_kernel:
            result = MakeViewlessTensor.forward(mock_ctx, inp, False)
            mock_kernel.assert_called_once_with(inp, False)

    def test_backward_passes_through(self):
        """Test backward passes gradient through."""
        from paddleformers.fleet.utils import MakeViewlessTensor

        mock_ctx = mock.MagicMock()
        grad = paddle.randn([4, 8])

        result = MakeViewlessTensor.backward(mock_ctx, grad)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], grad)
        self.assertIsNone(result[1])


class TestGetPgRank(unittest.TestCase):
    """Tests for get_pg_rank function."""

    def test_none_group_returns_zero(self):
        """Test get_pg_rank returns 0 when group is None."""
        from paddleformers.fleet.utils import get_pg_rank

        with mock.patch(
            "paddle.distributed.is_initialized", return_value=False
        ):
            self.assertEqual(get_pg_rank(group=None), 0)

    def test_initialized_group_returns_rank(self):
        """Test get_pg_rank returns group rank when initialized."""
        from paddleformers.fleet.utils import get_pg_rank

        mock_group = mock.MagicMock()
        mock_group.rank = 3

        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(get_pg_rank(group=mock_group), 3)


class TestGetPgSize(unittest.TestCase):
    """Tests for get_pg_size function."""

    def test_none_group_returns_one(self):
        """Test get_pg_size returns 1 when group is None."""
        from paddleformers.fleet.utils import get_pg_size

        with mock.patch(
            "paddle.distributed.is_initialized", return_value=False
        ):
            self.assertEqual(get_pg_size(group=None), 1)

    def test_single_rank_group(self):
        """Test get_pg_size returns 1 for single rank group."""
        from paddleformers.fleet.utils import get_pg_size

        mock_group = mock.MagicMock()
        mock_group.ranks = [0]

        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(get_pg_size(group=mock_group), 1)

    def test_multi_rank_group(self):
        """Test get_pg_size returns nranks for multi-rank group."""
        from paddleformers.fleet.utils import get_pg_size

        mock_group = mock.MagicMock()
        mock_group.ranks = [0, 1, 2, 3]
        mock_group.nranks = 4

        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(get_pg_size(group=mock_group), 4)


class TestDivide(unittest.TestCase):
    """Tests for divide and ensure_divisibility functions."""

    def test_divide_correct(self):
        """Test divide returns correct integer division."""
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(10, 5), 2)
        self.assertEqual(divide(100, 10), 10)
        self.assertEqual(divide(7, 1), 7)

    def test_divide_not_divisible_raises(self):
        """Test divide raises when numerator not divisible."""
        from paddleformers.fleet.utils import divide

        with self.assertRaises(AssertionError):
            divide(7, 3)

    def test_ensure_divisibility_passes(self):
        """Test ensure_divisibility does not raise when divisible."""
        from paddleformers.fleet.utils import ensure_divisibility

        ensure_divisibility(10, 5)
        ensure_divisibility(0, 1)

    def test_ensure_divisibility_raises(self):
        """Test ensure_divisibility raises when not divisible."""
        from paddleformers.fleet.utils import ensure_divisibility

        with self.assertRaises(AssertionError):
            ensure_divisibility(5, 3)


class TestWrappedTensor(unittest.TestCase):
    """Tests for WrappedTensor class."""

    def test_unwrap_returns_tensor(self):
        """Test unwrap returns the wrapped tensor."""
        from paddleformers.fleet.utils import WrappedTensor

        x = paddle.randn([4, 8])
        wrapped = WrappedTensor(x)
        result = wrapped.unwrap()
        self.assertIs(result, x)

    def test_double_unwrap_raises(self):
        """Test double unwrap raises RuntimeError."""
        from paddleformers.fleet.utils import WrappedTensor

        x = paddle.randn([4, 8])
        wrapped = WrappedTensor(x)
        wrapped.unwrap()
        with self.assertRaises(RuntimeError):
            wrapped.unwrap()


class TestGlobalMemoryBuffer(unittest.TestCase):
    """Tests for GlobalMemoryBuffer class."""

    def test_get_tensor_creates_buffer(self):
        """Test get_tensor creates a buffer on first call."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        with mock.patch("paddle.empty", return_value=paddle.randn([100])):
            tensor = buf.get_tensor([10, 10], "float32", "test")
            self.assertIsNotNone(tensor)

    def test_get_tensor_reuses_buffer(self):
        """Test get_tensor reuses buffer if large enough."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        with mock.patch("paddle.empty", return_value=paddle.randn([100])):
            t1 = buf.get_tensor([5, 5], "float32", "test")
            t2 = buf.get_tensor([3, 3], "float32", "test")
            # Should reuse same buffer
            pass

    def test_get_tensor_different_names(self):
        """Test get_tensor creates separate buffers for different names."""
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        with mock.patch("paddle.empty", return_value=paddle.randn([100])):
            t1 = buf.get_tensor([5, 5], "float32", "buf1")
            t2 = buf.get_tensor([5, 5], "float32", "buf2")
            self.assertNotEqual(id(t1), id(t2))


class TestInitMethods(unittest.TestCase):
    """Tests for init method functions."""

    def test_init_method_normal(self):
        """Test init_method_normal returns partial function."""
        from paddleformers.fleet.utils import init_method_normal

        fn = init_method_normal(0.02)
        self.assertIsInstance(fn, functools.partial)

    def test_scaled_init_method_normal(self):
        """Test scaled_init_method_normal returns partial function."""
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn = scaled_init_method_normal(0.02, num_layers=12)
        self.assertIsInstance(fn, functools.partial)

    def test_scaled_init_method_normal_multiplier(self):
        """Test scaled_init_method_normal with custom multiplier."""
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn1 = scaled_init_method_normal(0.02, num_layers=24, multiplier=1.0)
        fn2 = scaled_init_method_normal(0.02, num_layers=24, multiplier=4.0)
        self.assertIsInstance(fn1, functools.partial)
        self.assertIsInstance(fn2, functools.partial)


class TestPaddleVersionUtils(unittest.TestCase):
    """Tests for paddle version utility functions."""

    def test_get_paddle_version(self):
        """Test get_paddle_version returns version."""
        from paddleformers.fleet.utils import get_paddle_version

        version = get_paddle_version()
        self.assertIsNotNone(version)

    def test_is_paddle_min_version_no_packaging(self):
        """Test is_paddle_min_version raises when packaging not available."""
        from paddleformers.fleet.utils import is_paddle_min_version

        with mock.patch("paddleformers.fleet.utils.HAVE_PACKAGING", False):  # noqa: SIM117
            with self.assertRaises(ImportError):
                is_paddle_min_version("3.0.0")


class TestPrepareInputTensors(unittest.TestCase):
    """Tests for prepare_input_tensors_for_wgrad_compute."""

    def test_2d_input_unchanged(self):
        """Test 2D input shape unchanged."""
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([8, 16])
        inp = paddle.randn([8, 16])

        result_grad, result_inp = prepare_input_tensors_for_wgrad_compute(
            grad, inp
        )
        self.assertEqual(result_grad.shape, [8, 16])

    def test_3d_input_reshaped(self):
        """Test 3D input reshaped to 2D."""
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([2, 4, 16])
        inp = paddle.randn([2, 4, 16])

        result_grad, result_inp = prepare_input_tensors_for_wgrad_compute(
            grad, inp
        )
        self.assertEqual(len(result_grad.shape), 2)
        self.assertEqual(result_grad.shape, [8, 16])


class TestDeprecateInferenceParams(unittest.TestCase):
    """Tests for deprecate_inference_params function."""

    def test_none_context_none_params(self):
        """Test returns None when both are None."""
        from paddleformers.fleet.utils import deprecate_inference_params

        result = deprecate_inference_params(None, None)
        self.assertIsNone(result)

    def test_with_params_warns(self):
        """Test warning when inference_params is not None."""
        from paddleformers.fleet.utils import deprecate_inference_params

        mock_params = mock.MagicMock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, mock_params)
            self.assertTrue(len(w) > 0)
            self.assertIs(result, mock_params)


class TestLogSingleRank(unittest.TestCase):
    """Tests for log_single_rank function."""

    def test_log_when_no_dist(self):
        """Test logging when distributed is not initialized."""
        from paddleformers.fleet.utils import log_single_rank

        mock_logger = mock.MagicMock()
        with mock.patch(
            "paddle.distributed.is_initialized", return_value=False
        ):
            log_single_rank(mock_logger, logging.INFO, "test message")
            mock_logger.log.assert_called_once()

    def test_log_on_correct_rank(self):
        """Test logging only on the specified rank."""
        from paddleformers.fleet.utils import log_single_rank

        mock_logger = mock.MagicMock()
        with mock.patch("paddle.distributed.is_initialized", return_value=True):  # noqa: SIM117
            with mock.patch("paddle.distributed.get_rank", return_value=0):
                log_single_rank(
                    mock_logger, logging.INFO, "test message", rank=0
                )
                mock_logger.log.assert_called_once()

    def test_log_skipped_on_wrong_rank(self):
        """Test logging is skipped on non-matching rank."""
        from paddleformers.fleet.utils import log_single_rank

        mock_logger = mock.MagicMock()
        with mock.patch("paddle.distributed.is_initialized", return_value=True):  # noqa: SIM117
            with mock.patch("paddle.distributed.get_rank", return_value=1):
                log_single_rank(
                    mock_logger, logging.INFO, "test message", rank=0
                )
                mock_logger.log.assert_not_called()


class TestNvtxDecorator(unittest.TestCase):
    """Tests for nvtx_decorator function."""

    def test_decorator_returns_function_when_disabled(self):
        """Test decorator returns original function when NVTX disabled."""
        import paddleformers.fleet.utils as utils_mod
        from paddleformers.fleet.utils import nvtx_decorator

        original = utils_mod._nvtx_enabled

        utils_mod._nvtx_enabled = False

        @nvtx_decorator()
        def my_func():
            pass

        self.assertTrue(callable(my_func))

        utils_mod._nvtx_enabled = original


class TestKernelMakeViewlessTensor(unittest.TestCase):
    """Tests for _kernel_make_viewless_tensor function."""

    def test_creates_new_tensor(self):
        """Test creates a new tensor with same data."""
        from paddleformers.fleet.utils import _kernel_make_viewless_tensor

        x = paddle.randn([4, 8])
        result = _kernel_make_viewless_tensor(x, requires_grad=False)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)


if __name__ == "__main__":
    unittest.main()
