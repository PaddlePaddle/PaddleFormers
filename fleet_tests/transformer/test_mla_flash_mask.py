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

"""
Test MLA (Multi-Latent Attention) with FlashMask.

This test file covers the MLA + flashmask code path in dot_product_attention.py,
specifically the handling of different query/key head_dim vs value head_dim cases.
"""

import unittest

import numpy as np
import paddle

from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestMLAFlashMaskWithBackward(unittest.TestCase):
    """Test backward pass for MLA with FlashMask."""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def _create_config(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        bf16: bool = True,
    ):
        """Create a TransformerConfig for MLA testing."""
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

        config.qk_nope_head_dim = qk_nope_head_dim
        config.qk_rope_head_dim = qk_rope_head_dim
        config.v_head_dim = v_head_dim
        config.head_dim = qk_nope_head_dim + qk_rope_head_dim

        config.num_key_value_heads = num_attention_heads
        config.softmax_scale = None
        config.use_bias = True
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.fp16 = False
        config.bf16 = bf16
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"

        return config

    def _create_attn_mask_startend_row_indices(
        self, batch_size: int, num_heads: int, seq_len: int, causal: bool = True
    ):
        """Create attention mask startend row indices for flashmask."""
        if causal:
            start_indices = np.zeros(
                (batch_size, 1, seq_len, 1), dtype=np.int32
            )
            end_indices = np.arange(1, seq_len + 1, dtype=np.int32).reshape(
                1, 1, seq_len, 1
            )
            end_indices = np.broadcast_to(
                end_indices, (batch_size, 1, seq_len, 1)
            )
            indices = np.concatenate([start_indices, end_indices], axis=-1)
        else:
            start_indices = np.zeros(
                (batch_size, 1, seq_len, 1), dtype=np.int32
            )
            end_indices = np.full(
                (batch_size, 1, seq_len, 1), seq_len, dtype=np.int32
            )
            indices = np.concatenate([start_indices, end_indices], axis=-1)

        return paddle.to_tensor(indices)

    def test_backward_with_padding(self):
        """Test backward pass when q_head_dim != v_head_dim (requires padding)."""
        config = self._create_config(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,  # v_head_dim < q_head_dim
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        batch_size = 2
        seq_len = 16
        num_heads = 4
        q_head_dim = 192
        v_head_dim = 128

        # Create tensors with requires_grad=True
        query = paddle.randn(
            (batch_size, seq_len, num_heads, q_head_dim), dtype=paddle.bfloat16
        )
        query.stop_gradient = False
        key = paddle.randn(
            (batch_size, seq_len, num_heads, q_head_dim), dtype=paddle.bfloat16
        )
        key.stop_gradient = False
        value = paddle.randn(
            (batch_size, seq_len, num_heads, v_head_dim), dtype=paddle.bfloat16
        )
        value.stop_gradient = False

        attn_mask_startend_row_indices = (
            self._create_attn_mask_startend_row_indices(
                batch_size, num_heads, seq_len, causal=True
            )
        )

        # Forward pass
        output = attention(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        grad_output = paddle.randn_like(output)
        output.backward(grad_output)

        # Check gradients exist and have correct shapes
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

    def test_backward_without_padding(self):
        """Test backward pass when q_head_dim == v_head_dim (no padding)."""
        config = self._create_config(
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=128,  # v_head_dim == q_head_dim
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 128

        query = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        query.stop_gradient = False
        key = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        key.stop_gradient = False
        value = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        value.stop_gradient = False

        attn_mask_startend_row_indices = (
            self._create_attn_mask_startend_row_indices(
                batch_size, num_heads, seq_len, causal=True
            )
        )

        output = attention(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attn_mask_type=AttnMaskType.causal,
        )

        grad_output = paddle.randn_like(output)
        output.backward(grad_output)

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)


if __name__ == "__main__":
    unittest.main()
