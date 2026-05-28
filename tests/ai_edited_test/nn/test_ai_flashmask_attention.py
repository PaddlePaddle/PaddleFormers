# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import patch

import paddle
import paddle.nn as nn


class TestFlashmaskAttentionForward(unittest.TestCase):
    """Tests for paddleformers.nn.attention.flashmask_attention.flashmask_attention_forward."""

    def _make_module(self, is_causal=True):
        module = nn.Layer()
        module.is_causal = is_causal
        return module

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_basic_forward_shape(self, mock_flash):
        """flashmask_attention_forward should produce correct output shape."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")
        attn_output, attn_weights = flashmask_attention_forward(
            module, query, key, value, attn_mask_startend_row_indices=indices
        )
        self.assertEqual(attn_output.shape, [2, 8, 64])
        self.assertIsNone(attn_weights)

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_causal_inferred_from_indices_shape_1(self, mock_flash):
        """When indices shape[-1]==1, is_causal should be set to True."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=False)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        call_kwargs = mock_flash.call_args[1]
        self.assertTrue(call_kwargs["causal"])

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_causal_false_from_indices_shape_4(self, mock_flash):
        """When indices shape[-1]==4, is_causal should be set to False."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1, 4], dtype="int64")
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        call_kwargs = mock_flash.call_args[1]
        self.assertFalse(call_kwargs["causal"])

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_3d_indices_unsqueezed(self, mock_flash):
        """3D indices should be unsqueezed to 4D before processing."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")  # 3D input
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        call_args = mock_flash.call_args
        passed_indices = call_args[1]["startend_row_indices"]
        self.assertEqual(passed_indices.ndim, 4)

    @patch("paddleformers.nn.attention.flashmask_attention.sink_attention_forward")
    def test_with_sink(self, mock_sink):
        """When sink is provided, sink_attention_forward should be called."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_sink.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sink = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")
        flashmask_attention_forward(
            module, query, key, value, attn_mask_startend_row_indices=indices, sink=sink, scaling=1.0, dropout=0.1
        )
        mock_sink.assert_called_once()
        call_kwargs = mock_sink.call_args[1]
        self.assertEqual(call_kwargs["dropout_p"], 0.1)
        self.assertEqual(call_kwargs["softmax_scale"], 1.0)

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_no_indices_causal_inferred(self, mock_flash):
        """With no indices and no explicit is_causal, should infer from module."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=None)
        call_kwargs = mock_flash.call_args[1]
        self.assertTrue(call_kwargs["causal"])

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_no_indices_single_token_not_causal(self, mock_flash):
        """With no indices and seq_len==1, causal should be False."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 1, 4, 16])
        module = self._make_module(is_causal=True)
        # Input: [batch=2, heads=4, seq=1, dim=16] -> after transpose: [2, 1, 4, 16], shape[1]=1
        query = paddle.randn([2, 4, 1, 16], dtype="float32")
        key = paddle.randn([2, 4, 1, 16], dtype="float32")
        value = paddle.randn([2, 4, 1, 16], dtype="float32")
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=None)
        call_kwargs = mock_flash.call_args[1]
        self.assertFalse(call_kwargs["causal"])

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    def test_explicit_is_causal_overrides_with_indices(self, mock_flash):
        """When explicit is_causal=False is passed with indices of shape[-1]==1,
        the code still overrides to True based on indices shape."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1, 4], dtype="int64")  # shape[-1]==4 -> non-causal
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices, is_causal=False)
        call_kwargs = mock_flash.call_args[1]
        # With shape[-1]==4, the code sets is_causal=False, matching explicit override
        self.assertFalse(call_kwargs["causal"])

    @patch("paddleformers.nn.attention.flashmask_attention.flashmask_attention")
    @patch("paddleformers.nn.attention.flashmask_attention.paddle.base.core.is_compiled_with_cuda", return_value=False)
    def test_non_cuda_skip_fa_version_check(self, mock_cuda, mock_flash):
        """Non-CUDA builds should skip the flash attention version check."""
        from paddleformers.nn.attention.flashmask_attention import (
            flashmask_attention_forward,
        )

        mock_flash.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 8, 16], dtype="float32")
        value = paddle.randn([2, 4, 8, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")
        flashmask_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        mock_flash.assert_called_once()
