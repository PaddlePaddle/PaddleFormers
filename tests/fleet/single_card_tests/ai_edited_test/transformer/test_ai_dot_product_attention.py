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


import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_modules": None,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestDotProductAttentionConstructor(unittest.TestCase):
    """Tests for DotProductAttention constructor."""

    def test_basic_construction(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsInstance(attn, DotProductAttention)

    def test_context_parallel_raises(self):
        config = _make_config(context_parallel_size=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsInstance(attn, DotProductAttention)

    def test_layer_number_clamped_to_one(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.layer_number, 0)

    def test_softmax_scale_default(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        import math

        expected = 1.0 / math.sqrt(32)
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)

    def test_softmax_scale_custom(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.5,
        )
        self.assertEqual(attn.softmax_scale, 0.5)

    def test_query_key_layer_scaling(self):
        config = _make_config(apply_query_key_layer_scaling=True)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        import math

        expected = 1.0 / math.sqrt(32) / 1
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)

    def test_softmax_type_vanilla(self):
        config = _make_config(softmax_type="vanilla")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNone(attn.softmax_offset)

    def test_softmax_type_off_by_one(self):
        config = _make_config(softmax_type="off-by-one")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(attn.softmax_offset.shape, [4])

    def test_softmax_type_invalid(self):
        config = _make_config(softmax_type="invalid")
        with self.assertRaises(ValueError):
            DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

    def test_sliding_window(self):
        config = _make_config(sliding_window=(3, 3), window_attn_skip_freq=None)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.scale_mask_softmax)

    def test_custom_attention_dropout(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            attention_dropout=0.5,
        )
        self.assertIsNotNone(attn.attention_dropout)


class TestDotProductAttentionForward(unittest.TestCase):
    """Tests for DotProductAttention forward pass."""

    def setUp(self):
        self.config = _make_config()
        self.attn = DotProductAttention(
            config=self.config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.attn.eval()

    def test_forward_fp32(self):
        # fp32 path - uses matmul-based attention
        q = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        k = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        v = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        out = self.attn(q, k, v, None)
        self.assertEqual(out.shape, [2, 4, 128])

    def test_forward_attention_bias_raises(self):
        q = paddle.randn([2, 4, 4, 32])
        k = paddle.randn([2, 4, 4, 32])
        v = paddle.randn([2, 4, 4, 32])
        bias = paddle.randn([2, 4, 4, 32])
        with self.assertRaises(AssertionError):
            self.attn(q, k, v, None, attention_bias=bias)


class TestDotProductAttentionGQA(unittest.TestCase):
    """Tests for DotProductAttention with Group Query Attention."""

    def test_gqa_forward(self):
        config = _make_config(num_key_value_heads=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attn.eval()
        # num_attention_heads=4, num_key_value_heads=2
        q = paddle.randn([2, 4, 4, 32])
        k = paddle.randn([2, 4, 2, 32])
        v = paddle.randn([2, 4, 2, 32])
        out = attn(q, k, v, None)
        self.assertEqual(out.shape, [2, 4, 128])


class TestDotProductAttentionFP16(unittest.TestCase):
    """Tests for DotProductAttention forward with fp16 input."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_fp16(self):
        self.addCleanup(paddle.device.set_device, "gpu:0")
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attn.eval()
        paddle.device.set_device("gpu:0")
        q = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        k = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        v = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        out = attn(q, k, v, None)
        self.assertEqual(out.shape, [2, 4, 128])
        paddle.device.set_device("cpu")


class TestDotProductAttentionBF16(unittest.TestCase):
    """Tests for DotProductAttention forward with bf16 input."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_forward_bf16(self):
        self.addCleanup(paddle.device.set_device, "gpu:0")
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attn.eval()
        paddle.device.set_device("gpu:0")
        q = paddle.randn([2, 4, 4, 32], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 4, 32], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 4, 32], dtype=paddle.bfloat16)
        out = attn(q, k, v, None)
        self.assertEqual(out.shape, [2, 4, 128])
        paddle.device.set_device("cpu")


class TestDotProductAttentionEager(unittest.TestCase):
    """Tests for DotProductAttention with _attn_implementation='eager'."""

    def _make_eager_config(self, **overrides):
        config = _make_config(**overrides)
        config._attn_implementation = "eager"
        return config

    def test_eager_forward_fp32(self):
        config = self._make_eager_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attn.eval()
        q = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        k = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        v = paddle.randn([2, 4, 4, 32], dtype=paddle.float32)
        out = attn(q, k, v, None)
        self.assertEqual(out.shape, [2, 4, 128])

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_eager_fp16_skips_sdpa(self):
        """fp16 + eager should bypass scaled_dot_product_attention."""
        config = self._make_eager_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attn.eval()
        paddle.device.set_device("gpu:0")
        q = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        k = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        v = paddle.randn([2, 4, 4, 32], dtype=paddle.float16)
        with patch(
            "paddle.nn.functional.scaled_dot_product_attention",
            side_effect=AssertionError(
                "SDPA should not be called in eager mode"
            ),
        ) as mock_sdpa:
            out = attn(q, k, v, None)
            mock_sdpa.assert_not_called()
        self.assertEqual(out.shape, [2, 4, 128])
        paddle.device.set_device("cpu")

    def test_eager_packed_seq_raises(self):
        """eager + packed_seq_params should raise ValueError."""
        config = self._make_eager_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        q = paddle.randn([2, 4, 4, 32])
        k = paddle.randn([2, 4, 4, 32])
        v = paddle.randn([2, 4, 4, 32])
        with self.assertRaises(ValueError) as ctx:
            attn(q, k, v, None, packed_seq_params=MagicMock())
        self.assertIn("packed_seq_params", str(ctx.exception))


class TestDotProductAttentionContextParallel(unittest.TestCase):
    """Tests for DotProductAttention with context parallelism (formerly CPDotProductAttention)."""

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size"
    )
    def test_cp_packed_seq_raises(self, mock_get_cp_world_size):
        mock_get_cp_world_size.return_value = 2
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        q = paddle.randn([2, 4, 4, 32])
        k = paddle.randn([2, 4, 4, 32])
        v = paddle.randn([2, 4, 4, 32])
        with self.assertRaises(AssertionError):
            attn(q, k, v, None, packed_seq_params=MagicMock())


if __name__ == "__main__":
    unittest.main()
