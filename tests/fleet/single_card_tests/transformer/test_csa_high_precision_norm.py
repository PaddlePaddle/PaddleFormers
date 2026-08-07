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

"""Tests for CSA Compressor + CSAIndexer with swa_high_precision_norm=True."""

import types
import unittest

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.transformer.csa_attention import (
    Compressor,
    CompressorSublayersSpec,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)

# =========================================================================
# Helpers
# =========================================================================


class _Linear(nn.Layer):
    """Linear layer with weight shape [in_size, out_size] to match
    linear_bf16_fp32 which does x @ weight directly (no transpose)."""

    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[in_size, out_size],
            dtype="float32",
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return (
            paddle.matmul(x.cast("float32"), self.weight).cast(x.dtype),
            None,
        )


class _Norm(nn.Layer):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__()
        size = hidden_size or 1
        self.weight = self.create_parameter(
            shape=[size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = 1e-5

    def forward(self, x, **kwargs):
        x_f32 = x.cast("float32")
        normed = (
            x_f32
            * paddle.rsqrt(x_f32.square().mean(-1, keepdim=True) + self.eps)
            * self.weight
        )
        return normed.cast(x.dtype)


def _make_config(
    hidden_size=256,
    head_dim=128,
    pos_dim=64,
    compress_ratio=2,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=4,
    q_lora_rank=128,
    swa_high_precision_norm=True,
    use_fp8_qat=False,
    use_fast_hadamard=False,
    high_precision_rope=False,
):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        qk_pos_emb_head_dim=pos_dim,
        init_method=None,
        init_method_std=0.02,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        use_fp8_qat=use_fp8_qat,
        swa_high_precision_norm=swa_high_precision_norm,
        use_fast_hadamard=use_fast_hadamard,
        high_precision_rope=high_precision_rope,
        q_lora_rank=q_lora_rank,
        dsa_index_n_heads=index_n_heads,
        dsa_index_head_dim=index_head_dim,
        dsa_index_topk=index_topk,
    )


# =========================================================================
# Tests
# =========================================================================


class TestCSAHighPrecisionNorm(unittest.TestCase):
    """Test Compressor and CSAIndexer with swa_high_precision_norm=True."""

    def setUp(self):
        paddle.seed(42)
        self.config = _make_config()
        self.hidden_size = 256
        self.head_dim = 128
        self.compress_ratio = 2
        self.b = 2
        self.sq = 64

    def test_compressor_forward_no_rotate(self):
        """Compressor with swa_high_precision_norm=True, rotate=False produces correct output."""
        comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        compressor = Compressor(
            config=self.config,
            sublayers_spec=comp_spec,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, self.sq, self.hidden_size]).astype("bfloat16")
        kv = compressor(x)

        n_compressed = self.sq // self.compress_ratio
        self.assertEqual(kv.shape, [self.b, n_compressed, self.head_dim])
        self.assertEqual(kv.dtype, paddle.bfloat16)
        self.assertFalse(paddle.isnan(kv).any().item())

    def test_compressor_forward_with_rotate(self):
        """Compressor with swa_high_precision_norm=True, rotate=True runs correctly."""
        comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        compressor = Compressor(
            config=self.config,
            sublayers_spec=comp_spec,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rotate=True,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, self.sq, self.hidden_size]).astype("bfloat16")
        kv = compressor(x)

        n_compressed = self.sq // self.compress_ratio
        self.assertEqual(kv.shape, [self.b, n_compressed, self.head_dim])
        self.assertEqual(kv.dtype, paddle.bfloat16)
        self.assertFalse(paddle.isnan(kv).any().item())

    def test_compressor_short_sequence_returns_none(self):
        """Compressor returns None when seq_len < compress_ratio."""
        comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        compressor = Compressor(
            config=self.config,
            sublayers_spec=comp_spec,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, 1, self.hidden_size]).astype("bfloat16")
        kv = compressor(x)
        self.assertIsNone(kv)

    def test_indexer_forward(self):
        """CSAIndexer with swa_high_precision_norm=True runs forward correctly."""
        indexer_comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        indexer_spec = CSAIndexerSublayersSpec(
            linear_wq_b=_Linear,
            linear_weights_proj=_Linear,
            compressor=LayerSpec(Compressor, sublayers_spec=indexer_comp_spec),
        )
        indexer = CSAIndexer(
            config=self.config,
            sublayers_spec=indexer_spec,
            compress_ratio=self.compress_ratio,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, self.sq, self.hidden_size]).astype("bfloat16")
        qr = paddle.randn([self.b, self.sq, self.config.q_lora_rank]).astype(
            "bfloat16"
        )
        index_scores, topk_indices = indexer(x, qr, mask=None)

        n_compressed = self.sq // self.compress_ratio
        self.assertEqual(index_scores.shape, [self.b, self.sq, n_compressed])
        self.assertEqual(
            topk_indices.shape, [self.b, self.sq, self.config.dsa_index_topk]
        )
        self.assertFalse(paddle.isnan(index_scores).any().item())

    def test_compressor_backward(self):
        """Compressor with swa_high_precision_norm=True supports backward pass."""
        comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        compressor = Compressor(
            config=self.config,
            sublayers_spec=comp_spec,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, self.sq, self.hidden_size]).astype("bfloat16")
        x.stop_gradient = False
        kv = compressor(x)
        loss = kv.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.isnan(x.grad).any().item())

    def test_indexer_backward(self):
        """CSAIndexer with swa_high_precision_norm=True supports backward pass."""
        indexer_comp_spec = CompressorSublayersSpec(
            linear_wkv=_Linear,
            linear_wgate=_Linear,
            norm=_Norm,
        )
        indexer_spec = CSAIndexerSublayersSpec(
            linear_wq_b=_Linear,
            linear_weights_proj=_Linear,
            compressor=LayerSpec(Compressor, sublayers_spec=indexer_comp_spec),
        )
        indexer = CSAIndexer(
            config=self.config,
            sublayers_spec=indexer_spec,
            compress_ratio=self.compress_ratio,
            rotary_pos_emb=None,
        )

        x = paddle.randn([self.b, self.sq, self.hidden_size]).astype("bfloat16")
        x.stop_gradient = False
        qr = paddle.randn([self.b, self.sq, self.config.q_lora_rank]).astype(
            "bfloat16"
        )
        qr.stop_gradient = False

        index_scores, topk_indices = indexer(x, qr, mask=None)
        loss = index_scores.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(paddle.isnan(x.grad).any().item())


if __name__ == "__main__":
    unittest.main()
