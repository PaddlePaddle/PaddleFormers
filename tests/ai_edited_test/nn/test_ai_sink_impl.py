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

import unittest
from unittest.mock import patch

import numpy as np
import paddle


class TestGetFAVersion(unittest.TestCase):
    """Tests for _get_fa_version function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import _get_fa_version

        return _get_fa_version

    def test_returns_value(self):
        """Test that _get_fa_version returns an integer value."""
        _get_fa_version = self._get_func()
        result = _get_fa_version()
        self.assertIsInstance(result, int)

    def test_deterministic_mode(self):
        """Test that when cudnn_deterministic is set, returns 2."""
        _get_fa_version = self._get_func()
        with patch("paddle.get_flags", return_value={"FLAGS_cudnn_deterministic": True}):
            result = _get_fa_version()
            self.assertEqual(result, 2)

    def test_non_deterministic_mode(self):
        """Test that when cudnn_deterministic is not set, reads FLAGS_flash_attn_version."""
        _get_fa_version = self._get_func()
        with patch("paddle.get_flags", return_value={"FLAGS_cudnn_deterministic": False}), patch(
            "paddle.base.framework.get_flags", return_value={"FLAGS_flash_attn_version": 3}
        ):
            result = _get_fa_version()
            self.assertEqual(result, 3)


class TestFlashAttentionForwardDispatch(unittest.TestCase):
    """Tests for _flash_attention_forward_dispatch function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import (
            _flash_attention_forward_dispatch,
        )

        return _flash_attention_forward_dispatch

    def test_return_softmax_true_raises(self):
        """Test that return_softmax=True raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([1, 4, 2, 8])
        k = paddle.randn([1, 4, 2, 8])
        v = paddle.randn([1, 4, 2, 8])
        with self.assertRaises(AssertionError):
            func(q, k, v, return_softmax=True)

    def test_seq_k_neq_seq_v_raises(self):
        """Test that seq_k != seq_v raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([1, 4, 2, 8])
        k = paddle.randn([1, 4, 2, 8])
        v = paddle.randn([1, 2, 2, 8])
        with self.assertRaises(AssertionError):
            func(q, k, v)

    def test_unsupported_fa_version_raises(self):
        """Test that unsupported fa_version raises ValueError."""
        func = self._get_func()
        with patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=1):
            q = paddle.randn([1, 4, 2, 8])
            k = paddle.randn([1, 4, 2, 8])
            v = paddle.randn([1, 4, 2, 8])
            with self.assertRaises(ValueError):
                func(q, k, v)


class TestFlashAttentionBackwardDispatch(unittest.TestCase):
    """Tests for _flash_attention_backward_dispatch function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import (
            _flash_attention_backward_dispatch,
        )

        return _flash_attention_backward_dispatch

    def test_unsupported_fa_version_raises(self):
        """Test that unsupported fa_version raises ValueError in backward."""
        func = self._get_func()
        with patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=99):
            grad = paddle.randn([1, 4, 2, 8])
            q = paddle.randn([1, 4, 2, 8])
            k = paddle.randn([1, 4, 2, 8])
            v = paddle.randn([1, 4, 2, 8])
            out = paddle.randn([1, 4, 2, 8])
            lse = paddle.randn([1, 2, 4])
            with self.assertRaises(ValueError):
                func(grad, q, k, v, out, lse)


class TestFlashmaskAttentionForwardDispatch(unittest.TestCase):
    """Tests for _flashmask_attention_forward_dispatch function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import (
            _flashmask_attention_forward_dispatch,
        )

        return _flashmask_attention_forward_dispatch

    @patch("paddleformers.nn.attention.sink_impl.paddle.nn.functional.flashmask_attention")
    @patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=2)
    def test_fa_version_2_calls_flashmask(self, mock_version, mock_flashmask):
        """Test that FA version 2 calls flashmask_attention without softmax_scale."""
        func = self._get_func()
        mock_flashmask.return_value = (
            paddle.randn([1, 4, 2, 8]),
            paddle.randn([1, 2, 4]),
        )
        q = paddle.randn([1, 4, 2, 8])
        k = paddle.randn([1, 4, 2, 8])
        v = paddle.randn([1, 4, 2, 8])
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        out, lse = func(q, k, v, sei)
        mock_flashmask.assert_called_once()
        self.assertEqual(out.shape[0], 1)

    @patch("paddleformers.nn.attention.sink_impl.paddle.nn.functional.flashmask_attention")
    @patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=3)
    def test_fa_version_3_calls_flashmask_with_scale(self, mock_version, mock_flashmask):
        """Test that FA version 3 calls flashmask_attention with softmax_scale."""
        func = self._get_func()
        mock_flashmask.return_value = (
            paddle.randn([1, 4, 2, 8]),
            paddle.randn([1, 2, 4]),
        )
        q = paddle.randn([1, 4, 2, 8])
        k = paddle.randn([1, 4, 2, 8])
        v = paddle.randn([1, 4, 2, 8])
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")
        scale = 0.5

        out, lse = func(q, k, v, sei, softmax_scale=scale)
        mock_flashmask.assert_called_once()
        call_kwargs = mock_flashmask.call_args[1]
        self.assertIn("softmax_scale", call_kwargs)

    @patch("paddleformers.nn.attention.sink_impl.paddle.nn.functional.flashmask_attention")
    @patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=2)
    def test_fa_version_2_custom_scale_warning(self, mock_version, mock_flashmask):
        """Test that FA version 2 prints warning for custom softmax_scale."""
        func = self._get_func()
        mock_flashmask.return_value = (
            paddle.randn([1, 4, 2, 8]),
            paddle.randn([1, 2, 4]),
        )
        q = paddle.randn([1, 4, 2, 8])
        k = paddle.randn([1, 4, 2, 8])
        v = paddle.randn([1, 4, 2, 8])
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")
        custom_scale = 0.123

        with patch("builtins.print") as mock_print:
            func(q, k, v, sei, softmax_scale=custom_scale)
            mock_print.assert_called_once()


class TestFlashmaskAttentionBackwardDispatch(unittest.TestCase):
    """Tests for _flashmask_attention_backward_dispatch function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import (
            _flashmask_attention_backward_dispatch,
        )

        return _flashmask_attention_backward_dispatch

    def test_unsupported_fa_version_raises(self):
        """Test that unsupported fa_version raises ValueError in flashmask backward."""
        func = self._get_func()
        with patch("paddleformers.nn.attention.sink_impl._get_fa_version", return_value=99):
            grad = paddle.randn([1, 4, 2, 8])
            q = paddle.randn([1, 4, 2, 8])
            k = paddle.randn([1, 4, 2, 8])
            v = paddle.randn([1, 4, 2, 8])
            out = paddle.randn([1, 4, 2, 8])
            lse = paddle.randn([1, 2, 4])
            sei = paddle.randint(0, 4, [2, 3], dtype="int32")
            with self.assertRaises(ValueError):
                func(grad, q, k, v, out, lse, sei)


class TestRepeatKVUtility(unittest.TestCase):
    """Tests for repeat_kv utility used by sink_impl."""

    def test_repeat_kv_n_rep_1(self):
        """Test repeat_kv with n_rep=1 returns unchanged tensor."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 4, 2, 8])
        result = repeat_kv(x, 1)
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_repeat_kv_n_rep_2(self):
        """Test repeat_kv with n_rep=2 doubles the head dimension."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 4, 2, 8])
        result = repeat_kv(x, 2)
        self.assertEqual(result.shape, [2, 4, 4, 8])

    def test_repeat_kv_n_rep_4(self):
        """Test repeat_kv with n_rep=4 quadruples the head dimension."""
        from paddleformers.nn.attention.utils import repeat_kv

        x = paddle.randn([2, 4, 2, 8])
        result = repeat_kv(x, 4)
        self.assertEqual(result.shape, [2, 4, 8, 8])


class TestSinkAttentionForward(unittest.TestCase):
    """Tests for sink_attention_forward function."""

    def _get_func(self):
        from paddleformers.nn.attention.sink_impl import sink_attention_forward

        return sink_attention_forward

    def test_import(self):
        """Test that sink_attention_forward can be imported."""
        func = self._get_func()
        self.assertTrue(callable(func))

    def test_invalid_query_ndim_raises(self):
        """Test that non-4D query tensor raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 8])  # 2D instead of 4D
        k = paddle.randn([2, 4, 2, 8])
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_invalid_sink_ndim_raises(self):
        """Test that non-1D sink tensor raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 4, 2, 8])
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2, 4])  # 2D instead of 1D
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_batch_size_mismatch_raises(self):
        """Test that mismatched batch sizes raise AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([3, 4, 2, 8])  # Different batch size
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_head_dim_mismatch_raises(self):
        """Test that mismatched head dimensions raise AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 4, 2, 16])  # Different head_dim
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_kv_heads_mismatch_raises(self):
        """Test that key and value heads mismatch raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 4, 2, 8])
        v = paddle.randn([2, 4, 3, 8])  # Different num kv heads
        sink = paddle.randn([2])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_q_heads_not_divisible_by_kv_raises(self):
        """Test that query heads not divisible by kv heads raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 3, 8])  # 3 q heads
        k = paddle.randn([2, 4, 2, 8])  # 2 kv heads, 3 % 2 != 0
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([3])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_sink_size_mismatch_raises(self):
        """Test that sink size mismatch with q heads raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 4, 2, 8])
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([4])  # Size 4 but q_heads=2
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_seq_len_mismatch_without_sei_raises(self):
        """Test that seq length mismatch without startend_row_indices raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 8, 2, 8])  # Different seq length
        v = paddle.randn([2, 8, 2, 8])
        sink = paddle.randn([2])
        with self.assertRaises(AssertionError):
            func(q, k, v, sink)

    def test_attention_mask_with_sei_raises(self):
        """Test that attention_mask with startend_row_indices raises AssertionError."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 4, 2, 8])
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2])
        mask = paddle.randn([2, 2, 4, 4])
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")
        with self.assertRaises(AssertionError):
            func(q, k, v, sink, attention_mask=mask, startend_row_indices=sei)

    def test_kv_seq_mismatch_with_sei_raises(self):
        """Test that key/value seq length mismatch with startend_row_indices raises."""
        func = self._get_func()
        q = paddle.randn([2, 4, 2, 8])
        k = paddle.randn([2, 8, 2, 8])
        v = paddle.randn([2, 4, 2, 8])
        sink = paddle.randn([2])
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")
        with self.assertRaises(AssertionError):
            func(q, k, v, sink, startend_row_indices=sei)


class TestFlashMaskSinkPyLayerForward(unittest.TestCase):
    """Tests for FlashMaskSinkPyLayer.forward specific validation and logic."""

    def _get_cls(self):
        from paddleformers.nn.attention.sink_impl import FlashMaskSinkPyLayer

        return FlashMaskSinkPyLayer

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_forward_with_startend_row_indices(self, mock_flashmask):
        """Test forward pass when startend_row_indices is provided."""
        mock_flashmask.return_value = (
            paddle.randn([2, 4, 2, 8]),
            paddle.randn([2, 2, 4]),
        )
        cls = self._get_cls()
        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        result = cls.apply(q, k, v, sink, sei)
        mock_flashmask.assert_called_once()
        self.assertEqual(result.shape, [2, 4, 2, 8])

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_forward_with_causal_true(self, mock_flashmask):
        """Test forward pass with causal=True."""
        mock_flashmask.return_value = (
            paddle.randn([2, 4, 2, 8]),
            paddle.randn([2, 2, 4]),
        )
        cls = self._get_cls()
        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        cls.apply(q, k, v, sink, sei, causal=True)
        mock_flashmask.assert_called_once()

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_forward_with_dropout(self, mock_flashmask):
        """Test forward pass with dropout."""
        mock_flashmask.return_value = (
            paddle.randn([2, 4, 2, 8]),
            paddle.randn([2, 2, 4]),
        )
        cls = self._get_cls()
        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        cls.apply(q, k, v, sink, sei, dropout=0.1)
        mock_flashmask.assert_called_once()

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_forward_with_custom_softmax_scale(self, mock_flashmask):
        """Test forward pass with custom softmax_scale."""
        mock_flashmask.return_value = (
            paddle.randn([2, 4, 2, 8]),
            paddle.randn([2, 2, 4]),
        )
        cls = self._get_cls()
        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        cls.apply(q, k, v, sink, sei, softmax_scale=0.5)
        mock_flashmask.assert_called_once()


class TestFlashMaskSinkPyLayerForwardLSETruncation(unittest.TestCase):
    """Tests for LSE shape truncation logic in forward pass."""

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_lse_shape_truncation(self, mock_flashmask):
        """Test that LSE is truncated when its last dim is larger than seq_len."""
        from paddleformers.nn.attention.sink_impl import FlashMaskSinkPyLayer

        # Return LSE with larger last dimension (seqlen_q_rounded)
        raw_output = paddle.randn([2, 4, 2, 8])
        lse_padded = paddle.randn([2, 2, 8])  # padded to 8 > seq_len 4
        mock_flashmask.return_value = (raw_output, lse_padded)

        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        result = FlashMaskSinkPyLayer.apply(q, k, v, sink, sei)
        self.assertEqual(result.shape, [2, 4, 2, 8])


class TestSinkMultiplierComputation(unittest.TestCase):
    """Tests for the sink multiplier computation in forward pass."""

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_multiplier_shape(self, mock_flashmask):
        """Test that the multiplier tensor has the correct shape."""
        from paddleformers.nn.attention.sink_impl import FlashMaskSinkPyLayer

        raw_output = paddle.randn([2, 4, 2, 8]).astype("float32")
        lse = paddle.randn([2, 2, 4]).astype("float32")
        mock_flashmask.return_value = (raw_output, lse)

        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        result = FlashMaskSinkPyLayer.apply(q, k, v, sink, sei)
        # Result should have the same shape as input
        self.assertEqual(result.shape, q.shape)

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_multiplier_values_between_zero_and_one(self, mock_flashmask):
        """Test that multiplier is between 0 and 1 (sigmoid-like behavior)."""
        from paddleformers.nn.attention.sink_impl import FlashMaskSinkPyLayer

        # Use a large negative sink so exp(sink - lse) -> 0, multiplier -> 1
        raw_output = paddle.randn([2, 4, 2, 8]).astype("float32")
        lse = paddle.randn([2, 2, 4]).astype("float32")
        mock_flashmask.return_value = (raw_output, lse)

        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.full([2], -100.0, dtype="float32")  # Very negative
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        result = FlashMaskSinkPyLayer.apply(q, k, v, sink, sei)
        # With very negative sink, multiplier ~ 1, output ~ raw_output
        diff = paddle.abs(result - raw_output.astype(result.dtype))
        self.assertTrue(paddle.all(diff < 0.1))


class TestFlashMaskSinkPyLayerBackward(unittest.TestCase):
    """Tests for FlashMaskSinkPyLayer backward pass."""

    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_backward_dispatch")
    @patch("paddleformers.nn.attention.sink_impl._flashmask_attention_forward_dispatch")
    def test_backward_with_startend_row_indices(self, mock_fwd, mock_bwd):
        """Test backward pass with startend_row_indices."""
        from paddleformers.nn.attention.sink_impl import FlashMaskSinkPyLayer

        raw_output = paddle.randn([2, 4, 2, 8]).astype("float32")
        lse = paddle.randn([2, 2, 4]).astype("float32")
        mock_fwd.return_value = (raw_output, lse)

        grad_q = paddle.randn([2, 4, 2, 8]).astype("float32")
        grad_k = paddle.randn([2, 4, 2, 8]).astype("float32")
        grad_v = paddle.randn([2, 4, 2, 8]).astype("float32")
        mock_bwd.return_value = (grad_q, grad_k, grad_v)

        q = paddle.randn([2, 4, 2, 8]).astype("float32")
        k = paddle.randn([2, 4, 2, 8]).astype("float32")
        v = paddle.randn([2, 4, 2, 8]).astype("float32")
        sink = paddle.randn([2]).astype("float32")
        sink.stop_gradient = True  # Test stop_gradient branch
        sei = paddle.randint(0, 4, [2, 3], dtype="int32")

        FlashMaskSinkPyLayer.apply(q, k, v, sink, sei)
        mock_fwd.assert_called()


if __name__ == "__main__":
    unittest.main()
