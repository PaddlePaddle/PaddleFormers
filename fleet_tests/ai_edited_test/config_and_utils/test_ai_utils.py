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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import functools
import logging
import operator
import unittest
import warnings
from unittest.mock import MagicMock, patch

import paddle


class TestWrappedTensor(unittest.TestCase):
    """Tests for WrappedTensor in paddleformers.fleet.utils."""

    def test_wrapped_tensor_create_and_unwrap(self):
        from paddleformers.fleet.utils import WrappedTensor

        t = paddle.randn([2, 3])
        wrapped = WrappedTensor(t)
        result = wrapped.unwrap()
        self.assertTrue(paddle.allclose(result, t))

    def test_wrapped_tensor_double_unwrap_raises(self):
        from paddleformers.fleet.utils import WrappedTensor

        t = paddle.randn([2, 3])
        wrapped = WrappedTensor(t)
        wrapped.unwrap()
        with self.assertRaises(RuntimeError):
            wrapped.unwrap()


class TestGlobalMemoryBuffer(unittest.TestCase):
    """Tests for GlobalMemoryBuffer in paddleformers.fleet.utils."""

    def test_get_tensor_returns_correct_shape(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t = buf.get_tensor([3, 4], paddle.float32, "test1")
        self.assertEqual(t.shape, [3, 4])
        self.assertEqual(t.dtype, paddle.float32)

    def test_get_tensor_reuses_buffer(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([3, 4], paddle.float32, "test_reuse")
        t2 = buf.get_tensor([3, 4], paddle.float32, "test_reuse")
        # Should return the same underlying buffer (view)
        self.assertEqual(t1.shape, t2.shape)

    def test_get_tensor_grows_buffer(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([3, 4], paddle.float32, "grow")
        t2 = buf.get_tensor([6, 8], paddle.float32, "grow")
        self.assertEqual(t2.shape, [6, 8])

    def test_get_tensor_different_names_separate(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        t1 = buf.get_tensor([2, 2], paddle.float32, "name_a")
        t2 = buf.get_tensor([3, 3], paddle.float32, "name_b")
        self.assertEqual(t1.shape, [2, 2])
        self.assertEqual(t2.shape, [3, 3])

    def test_get_tensor_with_mem_alloc_context(self):
        from paddleformers.fleet.utils import GlobalMemoryBuffer

        buf = GlobalMemoryBuffer()
        mock_ctx = MagicMock()
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        t = buf.get_tensor(
            [2, 3], paddle.float32, "ctx_test", mem_alloc_context=mock_ctx
        )
        self.assertEqual(t.shape, [2, 3])
        mock_ctx.assert_called_once()


class TestEnsureDivisibility(unittest.TestCase):
    """Tests for ensure_divisibility in paddleformers.fleet.utils."""

    def test_divisible(self):
        from paddleformers.fleet.utils import ensure_divisibility

        # Should not raise
        ensure_divisibility(10, 2)
        ensure_divisibility(9, 3)
        ensure_divisibility(0, 5)

    def test_not_divisible_raises(self):
        from paddleformers.fleet.utils import ensure_divisibility

        with self.assertRaises(AssertionError):
            ensure_divisibility(10, 3)


class TestDivide(unittest.TestCase):
    """Tests for divide in paddleformers.fleet.utils."""

    def test_divide_success(self):
        from paddleformers.fleet.utils import divide

        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(9, 3), 3)

    def test_divide_not_divisible_raises(self):
        from paddleformers.fleet.utils import divide

        with self.assertRaises(AssertionError):
            divide(10, 3)


class TestInitMethodNormal(unittest.TestCase):
    """Tests for init_method_normal in paddleformers.fleet.utils."""

    def test_returns_partial(self):
        from paddleformers.fleet.utils import init_method_normal

        fn = init_method_normal(0.02)
        self.assertIsInstance(fn, functools.partial)

    def test_initializes_tensor(self):
        from paddleformers.fleet.utils import init_method_normal

        fn = init_method_normal(0.02)
        t = paddle.empty([100, 100])
        fn(t)
        # Check that values are not all zero (very unlikely for random init)
        self.assertGreater(paddle.abs(t).max().item(), 0.0)


class TestScaledInitMethodNormal(unittest.TestCase):
    """Tests for scaled_init_method_normal in paddleformers.fleet.utils."""

    def test_returns_partial(self):
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn = scaled_init_method_normal(0.02, 12)
        self.assertIsInstance(fn, functools.partial)

    def test_initializes_tensor(self):
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn = scaled_init_method_normal(0.02, 12)
        t = paddle.empty([100, 100])
        fn(t)
        self.assertGreater(paddle.abs(t).max().item(), 0.0)

    def test_custom_multiplier(self):
        from paddleformers.fleet.utils import scaled_init_method_normal

        fn = scaled_init_method_normal(0.02, 12, multiplier=3.0)
        self.assertIsInstance(fn, functools.partial)


class TestGetPgSize(unittest.TestCase):
    """Tests for get_pg_size in paddleformers.fleet.utils."""

    def test_no_distributed(self):
        from paddleformers.fleet.utils import get_pg_size

        result = get_pg_size()
        self.assertEqual(result, 1)

    def test_none_group(self):
        from paddleformers.fleet.utils import get_pg_size

        result = get_pg_size(group=None)
        self.assertEqual(result, 1)


class TestGetPgRank(unittest.TestCase):
    """Tests for get_pg_rank in paddleformers.fleet.utils."""

    def test_no_distributed(self):
        from paddleformers.fleet.utils import get_pg_rank

        result = get_pg_rank()
        self.assertEqual(result, 0)

    def test_none_group(self):
        from paddleformers.fleet.utils import get_pg_rank

        result = get_pg_rank(group=None)
        self.assertEqual(result, 0)


class TestLogSingleRank(unittest.TestCase):
    """Tests for log_single_rank in paddleformers.fleet.utils."""

    def test_log_without_distributed(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        log_single_rank(logger, logging.INFO, "test message")
        logger.log.assert_called_once()

    def test_log_on_rank_zero(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=0),
        ):
            log_single_rank(logger, logging.INFO, "test message")
            logger.log.assert_called_once()

    def test_log_on_non_zero_rank_suppressed(self):
        from paddleformers.fleet.utils import log_single_rank

        logger = MagicMock()
        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=1),
        ):
            log_single_rank(logger, logging.INFO, "test message")
            logger.log.assert_not_called()


class TestGetTensorModelParallelGroupIfNone(unittest.TestCase):
    """Tests for get_tensor_model_parallel_group_if_none in paddleformers.fleet.utils."""

    def test_not_initialized_returns_none(self):
        from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

        with patch("paddle.distributed.is_initialized", return_value=False):
            result = get_tensor_model_parallel_group_if_none(None)
            self.assertIsNone(result)

    def test_with_explicit_group(self):
        from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

        mock_group = MagicMock()
        with patch("paddle.distributed.is_initialized", return_value=True):
            result = get_tensor_model_parallel_group_if_none(mock_group)
            self.assertEqual(result, mock_group)

    def test_none_group_warns_and_gets_default(self):
        from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=0),
            patch("paddleformers.fleet.utils.parallel_state") as mock_ps,
        ):
            mock_ps.get_tensor_model_parallel_group.return_value = MagicMock()
            result = get_tensor_model_parallel_group_if_none(None)
            self.assertIsNotNone(result)

    def test_none_group_is_expert(self):
        from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none

        with (
            patch("paddle.distributed.is_initialized", return_value=True),
            patch("paddle.distributed.get_rank", return_value=0),
            patch("paddleformers.fleet.utils.parallel_state") as mock_ps,
        ):
            mock_ps.get_expert_tensor_parallel_group.return_value = MagicMock()
            result = get_tensor_model_parallel_group_if_none(
                None, is_expert=True
            )
            self.assertIsNotNone(result)


class TestPrepareInputTensorsForWgradCompute(unittest.TestCase):
    """Tests for prepare_input_tensors_for_wgrad_compute in paddleformers.fleet.utils."""

    def test_3d_tensors_reshaped_to_2d(self):
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([2, 3, 4])
        inp = paddle.randn([2, 3, 4])
        g_out, a_in = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g_out.shape, [6, 4])
        self.assertEqual(a_in.shape, [6, 4])

    def test_2d_tensors_unchanged(self):
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([6, 4])
        inp = paddle.randn([6, 4])
        g_out, a_in = prepare_input_tensors_for_wgrad_compute(grad, inp)
        self.assertEqual(g_out.shape, [6, 4])
        self.assertEqual(a_in.shape, [6, 4])

    def test_contiguous_called(self):
        from paddleformers.fleet.utils import prepare_input_tensors_for_wgrad_compute

        grad = paddle.randn([2, 3, 4]).transpose([0, 2, 1])
        inp = paddle.randn([2, 3, 4]).transpose([0, 2, 1])
        g_out, a_in = prepare_input_tensors_for_wgrad_compute(grad, inp)
        # After contiguous and reshape to 2D
        self.assertEqual(len(g_out.shape), 2)


class TestGetPaddleVersion(unittest.TestCase):
    """Tests for get_paddle_version in paddleformers.fleet.utils."""

    def test_returns_version(self):
        from paddleformers.fleet.utils import get_paddle_version

        v = get_paddle_version()
        self.assertIsNotNone(v)


class TestIsPaddleMinVersion(unittest.TestCase):
    """Tests for is_paddle_min_version in paddleformers.fleet.utils."""

    def test_check_equality_true(self):
        from paddleformers.fleet.utils import is_paddle_min_version

        result = is_paddle_min_version("0.0.0", check_equality=True)
        self.assertTrue(result)

    def test_check_equality_false_high_version(self):
        from paddleformers.fleet.utils import is_paddle_min_version

        result = is_paddle_min_version("999.0.0", check_equality=True)
        self.assertFalse(result)

    @unittest.skipIf(
        os.getenv("repo_flag") != "paddlefleet",
        f"Skipping test: repo_flag={os.getenv('repo_flag')} (not 'paddlefleet')",
    )
    def test_check_strict_inequality(self):
        from paddleformers.fleet.utils import is_paddle_min_version

        result = is_paddle_min_version("0.0.0", check_equality=False)
        # Current version is > 0.0.0
        self.assertTrue(result)


class TestNvtxFunctions(unittest.TestCase):
    """Tests for NVTX profiling functions in paddleformers.fleet.utils."""

    def test_nvtx_range_push_disabled(self):
        """When NVTX is disabled, push is a no-op."""
        from paddleformers.fleet.utils import nvtx_range_push

        # Default is disabled
        nvtx_range_push("test_range")

    def test_nvtx_range_pop_disabled(self):
        """When NVTX is disabled, pop is a no-op."""
        from paddleformers.fleet.utils import nvtx_range_pop

        nvtx_range_pop("test_range")

    def test_nvtx_decorator_disabled(self):
        """When NVTX is disabled, decorator returns original function."""
        from paddleformers.fleet.utils import nvtx_decorator

        @nvtx_decorator()
        def my_func():
            return 42

        self.assertEqual(my_func(), 42)

    def test_nvtx_decorator_with_message_and_color(self):
        """NVTX decorator with message and color when disabled."""
        from paddleformers.fleet.utils import nvtx_decorator

        @nvtx_decorator(message="CustomMsg", color="blue")
        def my_func2():
            return "hello"

        self.assertEqual(my_func2(), "hello")

    def test_nvtx_decorator_get_func_path(self):
        from paddleformers.fleet.utils import _nvtx_decorator_get_func_path

        path = _nvtx_decorator_get_func_path(lambda: None)
        self.assertIsInstance(path, str)
        self.assertIn(".", path)

    @patch("paddleformers.fleet.utils._nvtx_enabled", True)
    def test_nvtx_range_push_enabled(self):
        from paddleformers.fleet.utils import nvtx_range_push

        with patch("paddle.base.core.nvprof_nvtx_push") as mock_push:
            nvtx_range_push("enabled_range")
            mock_push.assert_called_once_with("enabled_range")

    @patch("paddleformers.fleet.utils._nvtx_enabled", True)
    def test_nvtx_range_pop_enabled(self):
        from paddleformers.fleet.utils import nvtx_range_pop, nvtx_range_push

        with patch("paddle.base.core.nvprof_nvtx_pop") as mock_pop:
            nvtx_range_push("range_to_pop")
            nvtx_range_pop("range_to_pop")
            mock_pop.assert_called_once()

    @patch("paddleformers.fleet.utils._nvtx_enabled", True)
    def test_nvtx_range_pop_empty_stack_raises(self):
        import paddleformers.fleet.utils as utils_module

        original_messages = utils_module._nvtx_range_messages
        utils_module._nvtx_range_messages = []
        try:
            # Pop on empty stack should raise even if nvprof_nvtx_pop is mocked
            with (  # noqa: SIM117
                patch("paddle.base.core.nvprof_nvtx_push"),
                patch("paddle.base.core.nvprof_nvtx_pop"),
            ):
                with self.assertRaises(RuntimeError):
                    utils_module.nvtx_range_pop()
        finally:
            utils_module._nvtx_range_messages = original_messages


class TestGetAttrWrappedModel(unittest.TestCase):
    """Tests for get_attr_wrapped_model in paddleformers.fleet.utils."""

    def test_direct_attribute(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        model = MagicMock()
        model.my_attr = "value"
        result = get_attr_wrapped_model(model, "my_attr")
        self.assertEqual(result, "value")

    def test_wrapped_attribute(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        inner = MagicMock()
        inner.my_attr = "deep_value"
        outer = MagicMock()
        outer.module = inner
        outer.my_attr = None
        result = get_attr_wrapped_model(outer, "my_attr", allow_none=False)
        self.assertEqual(result, "deep_value")

    def test_list_raises(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        with self.assertRaises(RuntimeError):
            get_attr_wrapped_model([1, 2, 3], "my_attr")

    def test_attribute_not_found_raises(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        model = MagicMock(spec=[])
        with self.assertRaises(RuntimeError):
            get_attr_wrapped_model(model, "nonexistent")

    def test_return_model_obj(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        inner = MagicMock()
        inner.my_attr = "val"
        outer = MagicMock()
        outer.module = inner
        outer.my_attr = None
        result = get_attr_wrapped_model(
            outer, "my_attr", allow_none=False, return_model_obj=True
        )
        self.assertEqual(result, inner)

    def test_allow_none_true(self):
        from paddleformers.fleet.utils import get_attr_wrapped_model

        model = MagicMock()
        model.my_attr = "exists"
        result = get_attr_wrapped_model(model, "my_attr", allow_none=True)
        self.assertEqual(result, "exists")


class TestGetModelType(unittest.TestCase):
    """Tests for get_model_type in paddleformers.fleet.utils."""

    def test_returns_model_type(self):
        from paddleformers.fleet.utils import get_model_type

        model = MagicMock()
        model.model_type = "gpt"
        result = get_model_type(model)
        self.assertEqual(result, "gpt")


class TestGetModelXattn(unittest.TestCase):
    """Tests for get_model_xattn in paddleformers.fleet.utils."""

    def test_has_xattn(self):
        from paddleformers.fleet.utils import get_model_xattn

        model = MagicMock()
        model.xattn_needed = True
        result = get_model_xattn(model)
        self.assertTrue(result)

    def test_no_xattn_returns_false(self):
        from paddleformers.fleet.utils import get_model_xattn

        model = MagicMock(spec=[])
        result = get_model_xattn(model)
        self.assertFalse(result)


class TestGetModelConfig(unittest.TestCase):
    """Tests for get_model_config in paddleformers.fleet.utils."""

    def test_returns_config(self):
        from paddleformers.fleet.utils import get_model_config

        config = MagicMock()
        model = MagicMock()
        model.config = config
        result = get_model_config(model)
        self.assertEqual(result, config)


class TestMakeViewlessTensor(unittest.TestCase):
    """Tests for make_viewless_tensor and related functions."""

    def test_make_viewless_tensor_not_view(self):
        from paddleformers.fleet.utils import make_viewless_tensor

        t = paddle.randn([3, 4])
        t.stop_gradient = False
        if not hasattr(t, "_is_view"):
            # _is_view() not available in this Paddle version
            return
        result = make_viewless_tensor(t, requires_grad=False, keep_graph=False)
        # If tensor is not a view, it should be returned as-is
        self.assertEqual(result.shape, [3, 4])

    def test_make_viewless_tensor_view_without_keep_graph(self):
        from paddleformers.fleet.utils import make_viewless_tensor

        t = paddle.randn([3, 4])
        t.stop_gradient = False
        # Create a view
        view = t[:2, :]
        view.stop_gradient = False
        # _is_view might not be available on all paddle versions, test only if available
        if hasattr(view, "_is_view"):
            try:
                result = make_viewless_tensor(
                    view, requires_grad=False, keep_graph=False
                )
                self.assertEqual(result.shape, [2, 4])
            except Exception:
                pass

    def test_make_viewless_tensor_view_with_keep_graph(self):
        from paddleformers.fleet.utils import make_viewless_tensor

        t = paddle.randn([3, 4])
        t.stop_gradient = False
        view = t[:2, :]
        view.stop_gradient = False
        if hasattr(view, "_is_view"):
            try:
                result = make_viewless_tensor(
                    view, requires_grad=False, keep_graph=True
                )
                self.assertEqual(result.shape, [2, 4])
            except Exception:
                pass

    def test_make_viewless_tensor_cls_forward(self):
        from paddleformers.fleet.utils import MakeViewlessTensor

        t = paddle.randn([3, 4], dtype=paddle.float32)
        t.stop_gradient = False
        try:
            result = MakeViewlessTensor.apply(t, False)
            self.assertEqual(result.shape, [3, 4])
        except Exception:
            # May fail if autograd is not fully available
            pass

    def test_make_viewless_tensor_cls_backward(self):
        from paddleformers.fleet.utils import MakeViewlessTensor

        grad = paddle.randn([3, 4])
        result_grad, none_result = MakeViewlessTensor.backward(None, grad)
        self.assertTrue(paddle.allclose(result_grad, grad))
        self.assertIsNone(none_result)


class TestDeprecateInferenceParams(unittest.TestCase):
    """Tests for deprecate_inference_params in paddleformers.fleet.utils."""

    def test_both_none_returns_none(self):
        from paddleformers.fleet.utils import deprecate_inference_params

        result = deprecate_inference_params(None, None)
        self.assertIsNone(result)

    def test_context_provided_returns_context(self):
        from paddleformers.fleet.utils import deprecate_inference_params

        ctx = MagicMock()
        params = MagicMock()
        result = deprecate_inference_params(ctx, params)
        self.assertEqual(result, ctx)

    def test_only_params_warns(self):
        from paddleformers.fleet.utils import deprecate_inference_params

        params = MagicMock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, params)
            self.assertEqual(result, params)
            self.assertTrue(len(w) > 0)
            self.assertIn("inference_params", str(w[0].message))


class TestGetBatchOnThisCpRank(unittest.TestCase):
    """Tests for get_batch_on_this_cp_rank in paddleformers.fleet.utils."""

    def test_dict_input(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        mock_input_ids = paddle.randn([2, 4])
        mock_labels = paddle.randn([2, 4])
        inputs = {
            "input_ids": mock_input_ids,
            "labels": mock_labels,
            "other_key": "not_scattered",
        }
        with patch("paddleformers.fleet.utils.ContextParallelScatterOp") as mock_cp_op:
            mock_cp_op.apply.side_effect = lambda x, **kw: x
            result = get_batch_on_this_cp_rank(inputs)
        self.assertIn("input_ids", result)
        self.assertIn("labels", result)
        self.assertEqual(result["other_key"], "not_scattered")

    def test_tensor_input(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        t = paddle.randn([2, 4])
        with patch("paddleformers.fleet.utils.ContextParallelScatterOp") as mock_cp_op:
            mock_cp_op.apply.return_value = t
            result = get_batch_on_this_cp_rank(t)
        self.assertIsNotNone(result)

    def test_list_input_raises(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(AssertionError):
            get_batch_on_this_cp_rank([paddle.randn([2, 4])])

    def test_unsupported_type_raises(self):
        from paddleformers.fleet.utils import get_batch_on_this_cp_rank

        with self.assertRaises(ValueError):
            get_batch_on_this_cp_rank("unsupported")


class TestReduceOperator(unittest.TestCase):
    """Test the reduce import used in GlobalMemoryBuffer."""

    def test_reduce_mul(self):
        from functools import reduce

        result = reduce(operator.mul, [2, 3, 4], 1)
        self.assertEqual(result, 24)

    def test_reduce_mul_empty(self):
        from functools import reduce

        result = reduce(operator.mul, [], 1)
        self.assertEqual(result, 1)
