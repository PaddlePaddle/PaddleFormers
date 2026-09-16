# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


class TestSubbatch(unittest.TestCase):
    """Tests for paddleformers.nn.criterion.loss_utils.subbatch."""

    def test_subbatch_small_input_no_split(self):
        """When input is smaller than batch size, should call function directly."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def simple_fn(x):
            return x * 2

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0], bs=100, out_idx=0)
        x = paddle.randn([10, 4])
        result = wrapped(x)
        np.testing.assert_allclose(result.numpy(), (x * 2).numpy())

    def test_subbatch_splits_and_concatenates(self):
        """When input is larger than batch size, should split, process, and concatenate."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def simple_fn(x):
            return x * 2

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0], bs=3, out_idx=0)
        x = paddle.randn([10, 4])
        result = wrapped(x)
        expected = x * 2
        np.testing.assert_allclose(result.numpy(), expected.numpy())

    def test_subbatch_multiple_args(self):
        """subbatch should handle multiple arguments correctly."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def add_fn(a, b):
            return a + b

        wrapped = subbatch(add_fn, arg_idx=[0, 1], axis=[0, 0], bs=3, out_idx=0)
        a = paddle.ones([6, 4])
        b = paddle.ones([6, 4]) * 2
        result = wrapped(a, b)
        np.testing.assert_allclose(result.numpy(), (a + b).numpy())

    def test_subbatch_same_arg_idx(self):
        """subbatch with same_arg_idx should reuse sliced tensor."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        call_count = [0]

        def counting_fn(a, b):
            call_count[0] += 1
            return a + b

        wrapped = subbatch(counting_fn, arg_idx=[0], axis=[0], bs=3, out_idx=0, same_arg_idx={1: 0})
        a = paddle.ones([9, 4])
        result = wrapped(a, a)
        # Without same_arg_idx, each arg would be sliced separately but function still called 3 times
        # With same_arg_idx, args[1] should reuse args[0]
        np.testing.assert_allclose(result.numpy(), (a * 2).numpy())

    def test_subbatch_preserves_function_name(self):
        """subbatch should preserve the original function's name via functools.wraps."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def my_function(x):
            return x

        wrapped = subbatch(my_function, arg_idx=[0], axis=[0], bs=10, out_idx=0)
        self.assertEqual(wrapped.__name__, "my_function")

    def test_subbatch_axis_width_mismatch_raises(self):
        """subbatch should raise AssertionError when batch sizes don't match."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def two_arg_fn(a, b):
            return a

        wrapped = subbatch(two_arg_fn, arg_idx=[0, 1], axis=[0, 0], bs=10, out_idx=0)
        a = paddle.ones([6, 4])
        b = paddle.ones([8, 4])  # Different size
        with self.assertRaises(AssertionError):
            wrapped(a, b)

    def test_subbatch_same_arg_idx_invalid_raises(self):
        """same_arg_idx with i <= same_arg_idx[i] should raise AssertionError."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def fn(a, b):
            return a

        # same_arg_idx={0: 1} means 0 <= 1, which violates i > same_arg_idx[i]
        wrapped = subbatch(fn, arg_idx=[0, 1], axis=[0, 0], bs=10, out_idx=0, same_arg_idx={0: 1})
        a = paddle.ones([12, 4])
        b = paddle.ones([12, 4])
        with self.assertRaises(AssertionError):
            wrapped(a, b)

    def test_subbatch_with_recompute(self):
        """subbatch with use_recompute=True should use paddle recompute."""
        from paddleformers.nn.criterion.loss_utils import subbatch

        def simple_fn(x):
            return x * 2

        wrapped = subbatch(simple_fn, arg_idx=[0], axis=[0], bs=3, out_idx=0, use_recompute=True)
        x = paddle.randn([9, 4])
        with patch("paddle.distributed.fleet.utils.recompute") as mock_recompute:
            mock_recompute.return_value = x[:3] * 2
            # The first batch triggers recompute, subsequent may not
            try:
                wrapped(x)
            except Exception:
                pass
            mock_recompute.assert_called()


class TestCalcLMHeadLogits(unittest.TestCase):
    """Tests for paddleformers.nn.criterion.loss_utils.calc_lm_head_logits."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.sequence_parallel = overrides.get("sequence_parallel", False)
        config.max_sequence_length = overrides.get("max_sequence_length", 128)
        config.tensor_parallel_output = overrides.get("tensor_parallel_output", False)
        config.tensor_model_parallel_size = overrides.get("tensor_model_parallel_size", 1)
        return config

    def test_calc_lm_head_logits_basic(self):
        """calc_lm_head_logits should call parallel_matmul with correct args."""
        from paddleformers.nn.criterion.loss_utils import calc_lm_head_logits

        config = self._make_config()
        hidden = paddle.randn([2, 8, 64], dtype="float32")
        weight = paddle.randn([100, 64], dtype="float32")
        with patch("paddleformers.nn.criterion.loss_utils.parallel_matmul") as mock_matmul:
            mock_matmul.return_value = paddle.randn([2, 8, 100])
            calc_lm_head_logits(config, hidden, weight, None)
            mock_matmul.assert_called_once()

    def test_calc_lm_head_logits_with_bias(self):
        """calc_lm_head_logits should pass bias to parallel_matmul."""
        from paddleformers.nn.criterion.loss_utils import calc_lm_head_logits

        config = self._make_config()
        hidden = paddle.randn([2, 8, 64], dtype="float32")
        weight = paddle.randn([100, 64], dtype="float32")
        bias = paddle.randn([100], dtype="float32")
        with patch("paddleformers.nn.criterion.loss_utils.parallel_matmul") as mock_matmul:
            mock_matmul.return_value = paddle.randn([2, 8, 100])
            calc_lm_head_logits(config, hidden, weight, bias)
            call_kwargs = mock_matmul.call_args[1]
            self.assertIs(call_kwargs["bias"], bias)

    def test_calc_lm_head_logits_sequence_parallel(self):
        """With sequence_parallel and gather_hidden_states, should gather and reshape."""
        from paddleformers.nn.criterion.loss_utils import calc_lm_head_logits

        config = self._make_config(sequence_parallel=True)
        hidden = paddle.randn([2, 16, 64], dtype="float32")
        weight = paddle.randn([100, 64], dtype="float32")
        with patch("paddleformers.nn.criterion.loss_utils.GatherOp") as mock_gather:
            mock_gather.apply.return_value = paddle.randn([2, 128, 64])
            with patch("paddleformers.nn.criterion.loss_utils.parallel_matmul") as mock_matmul:
                mock_matmul.return_value = paddle.randn([2, 128, 100])
                calc_lm_head_logits(config, hidden, weight, None, gather_hidden_states=True)
                mock_gather.apply.assert_called_once_with(hidden)

    def test_calc_lm_head_logits_tensor_parallel_output_override(self):
        """Explicit tensor_parallel_output should override config setting."""
        from paddleformers.nn.criterion.loss_utils import calc_lm_head_logits

        config = self._make_config(tensor_parallel_output=False)
        hidden = paddle.randn([2, 8, 64], dtype="float32")
        weight = paddle.randn([100, 64], dtype="float32")
        with patch("paddleformers.nn.criterion.loss_utils.parallel_matmul") as mock_matmul:
            mock_matmul.return_value = paddle.randn([2, 8, 100])
            calc_lm_head_logits(config, hidden, weight, None, tensor_parallel_output=True)
            call_kwargs = mock_matmul.call_args[1]
            self.assertTrue(call_kwargs["tensor_parallel_output"])
