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


class TestSDPAAttentionForward(unittest.TestCase):
    """Tests for paddleformers.nn.attention.sdpa_attention.sdpa_attention_forward."""

    def _make_module(self, is_causal=True):
        module = nn.Layer()
        module.is_causal = is_causal
        return module

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_basic_forward_shape(self, mock_sdpa):
        """sdpa_attention_forward should produce correct output shape."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        # Input shape: [batch, heads, seq, dim]
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        attn_output, attn_weights = sdpa_attention_forward(module, query, key, value)
        # Output shape: [batch, seq, heads * dim]
        self.assertEqual(attn_output.shape, [2, 8, 64])
        self.assertIsNone(attn_weights)

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_with_attention_mask(self, mock_sdpa):
        """sdpa_attention_forward with attention_mask should pass it through."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        mask = paddle.zeros([2, 1, 8, 4], dtype="float32")
        sdpa_attention_forward(module, query, key, value, attention_mask=mask)
        mock_sdpa.assert_called_once()
        call_kwargs = mock_sdpa.call_args
        # The 4th positional arg is attention_mask
        self.assertIs(call_kwargs[0][3], mask)

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_is_causal_inferred_multi_token(self, mock_sdpa):
        """When seq_len > 1 and no mask, is_causal should be inferred from module."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertTrue(call_kwargs["is_causal"])

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_is_causal_single_token(self, mock_sdpa):
        """When seq_len == 1, is_causal should be False regardless of module setting."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 1, 4, 16])
        module = self._make_module(is_causal=True)
        # Input: [batch=2, heads=4, seq=1, dim=16] -> after transpose: [2, 1, 4, 16], shape[1]=1
        query = paddle.randn([2, 4, 1, 16], dtype="float32")
        key = paddle.randn([2, 4, 1, 16], dtype="float32")
        value = paddle.randn([2, 4, 1, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertFalse(call_kwargs["is_causal"])

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_explicit_is_causal_false(self, mock_sdpa):
        """Explicitly passing is_causal=False should override inference."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value, is_causal=False)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertFalse(call_kwargs["is_causal"])

    @patch("paddleformers.nn.attention.sdpa_attention.sink_attention_forward")
    def test_with_sink(self, mock_sink):
        """When sink is provided, sink_attention_forward should be called."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sink.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sink = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value, sink=sink, scaling=1.0)
        mock_sink.assert_called_once()

    @patch("paddleformers.nn.attention.sdpa_attention._gen_from_sparse_attn_mask_indices")
    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_with_startend_row_indices_3d(self, mock_sdpa, mock_gen):
        """3D attn_mask_startend_row_indices should be unsqueezed to 4D."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        mock_gen.return_value = paddle.zeros([2, 1, 8, 4], dtype="bool")
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1], dtype="int64")
        sdpa_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        mock_gen.assert_called_once()
        # is_causal should be True because shape[-1] == 1
        call_kwargs = mock_sdpa.call_args[1]
        self.assertTrue(call_kwargs["is_causal"])

    @patch("paddleformers.nn.attention.sdpa_attention._gen_from_sparse_attn_mask_indices")
    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_with_startend_row_indices_4d_non_causal(self, mock_sdpa, mock_gen):
        """4D attn_mask_startend_row_indices with shape[-1]==4 should set is_causal=False."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        mock_gen.return_value = paddle.zeros([2, 1, 8, 4], dtype="bool")
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        indices = paddle.zeros([2, 8, 1, 4], dtype="int64")
        sdpa_attention_forward(module, query, key, value, attn_mask_startend_row_indices=indices)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertFalse(call_kwargs["is_causal"])

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_dropout_passed_through(self, mock_sdpa):
        """dropout parameter should be passed to scaled_dot_product_attention."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value, dropout=0.1)
        # dropout is passed as the 4th positional arg (after query, key, value, mask)
        call_args = mock_sdpa.call_args[0]
        self.assertAlmostEqual(call_args[4], 0.1)

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_training_passed_through(self, mock_sdpa):
        """module.training should be passed to scaled_dot_product_attention."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        module.training = True
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertTrue(call_kwargs["training"])

    @patch("paddleformers.nn.attention.sdpa_attention.nn.functional.scaled_dot_product_attention")
    def test_enable_gqa(self, mock_sdpa):
        """enable_gqa should always be True."""
        from paddleformers.nn.attention.sdpa_attention import sdpa_attention_forward

        mock_sdpa.return_value = paddle.randn([2, 8, 4, 16])
        module = self._make_module(is_causal=True)
        query = paddle.randn([2, 4, 8, 16], dtype="float32")
        key = paddle.randn([2, 4, 4, 16], dtype="float32")
        value = paddle.randn([2, 4, 4, 16], dtype="float32")
        sdpa_attention_forward(module, query, key, value)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertTrue(call_kwargs["enable_gqa"])
