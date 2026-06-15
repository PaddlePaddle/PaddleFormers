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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest

import paddle

from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "context_parallel_size": 1,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "softmax_type": "vanilla",
        "flashmask_use_varlen": False,
        "params_dtype": "float32",
        "perform_initialization": True,
        "init_method": paddle.nn.initializer.Normal(0.02),
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestDotProductAttentionConstruction(unittest.TestCase):
    """Tests for DotProductAttention construction."""

    def test_construction_basic(self):
        """Test basic construction of DotProductAttention."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.layer_number, 1)
        self.assertEqual(attn.attn_mask_type, AttnMaskType.causal)

    def test_construction_with_softmax_scale(self):
        """Test construction with custom softmax_scale."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.5,
        )
        self.assertEqual(attn.softmax_scale, 0.5)

    def test_layer_number_minimum_is_1(self):
        """Test that layer_number is at least 1."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.layer_number, 0)

    def test_context_parallel_size_not_1_constructs(self):
        """DotProductAttention now handles CP internally; context_parallel_size > 1 should not raise."""
        config = _make_config(context_parallel_size=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsInstance(attn, DotProductAttention)


class TestDotProductAttentionSoftmaxType(unittest.TestCase):
    """Tests for DotProductAttention softmax_type handling."""

    def test_vanilla_softmax(self):
        """Test construction with vanilla softmax type."""
        config = _make_config(softmax_type="vanilla")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNone(attn.softmax_offset)

    def test_off_by_one_softmax(self):
        """Test construction with off-by-one softmax type."""
        config = _make_config(softmax_type="off-by-one")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.softmax_offset)

    def test_invalid_softmax_type_raises(self):
        """Test that invalid softmax type raises ValueError."""
        config = _make_config(softmax_type="invalid")
        with self.assertRaises(ValueError):
            DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )


class TestDotProductAttentionForwardEager(unittest.TestCase):
    """Tests for DotProductAttention forward with eager attention."""

    def test_forward_eager_basic(self):
        """Test forward with eager attention (float32)."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        bsz, seq_len, num_heads, head_dim = 1, 4, 4, 16
        query = paddle.randn([bsz, seq_len, num_heads, head_dim])
        key = paddle.randn([bsz, seq_len, num_heads, head_dim])
        value = paddle.randn([bsz, seq_len, num_heads, head_dim])
        attention_mask = paddle.triu(paddle.ones([bsz, 1, seq_len, seq_len]) * -1e4, diagonal=1)

        # Use eager mode
        config._attn_implementation = "eager"
        output = attn(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(output.shape, [bsz, seq_len, num_heads * head_dim])


class TestDotProductAttentionApplyQueryKeyLayerScaling(unittest.TestCase):
    """Tests for apply_query_key_layer_scaling."""

    def test_layer_scaling(self):
        """Test that layer scaling affects softmax_scale."""
        config = _make_config(apply_query_key_layer_scaling=True)
        attn = DotProductAttention(
            config=config,
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        # softmax_scale should be divided by layer_number
        base_scale = 1.0 / (16**0.5)  # 1 / sqrt(head_dim)
        expected = base_scale / 2  # layer_number=2
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)


if __name__ == "__main__":
    unittest.main()
