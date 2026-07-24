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

import unittest

import paddle

from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
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


class TestDotProductAttentionGroupQueryAttention(unittest.TestCase):
    """Tests for DotProductAttention with group query attention."""

    def test_gqa_construction(self):
        """Test construction with GQA (fewer kv heads than query heads)."""
        config = _make_config(num_key_value_heads=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.num_query_groups_per_partition, 2)
        self.assertEqual(attn.num_attention_heads_per_partition, 4)

    def test_gqa_forward_eager(self):
        """Test forward with GQA in eager mode."""
        config = _make_config(num_key_value_heads=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        bsz, seq_len, num_heads, head_dim = 1, 4, 4, 16
        num_kv_heads = 2
        query = paddle.randn([bsz, seq_len, num_heads, head_dim])
        key = paddle.randn([bsz, seq_len, num_kv_heads, head_dim])
        value = paddle.randn([bsz, seq_len, num_kv_heads, head_dim])
        attention_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len]) * -1e4, diagonal=1
        )

        config._attn_implementation = "eager"
        output = attn(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(output.shape, [bsz, seq_len, num_heads * head_dim])


class TestDotProductAttentionLearnableSoftmax(unittest.TestCase):
    """Tests for DotProductAttention with learnable softmax offset."""

    def test_learnable_softmax_construction(self):
        """Test construction with learnable softmax type (perform_initialization=False)."""
        config = _make_config(
            softmax_type="learnable",
            perform_initialization=False,
        )
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        # With perform_initialization=False, init_method is not called,
        # so softmax_offset remains as a registered Parameter
        self.assertIsNotNone(attn.softmax_offset)
        self.assertIsInstance(attn.softmax_offset, paddle.nn.Parameter)

    def test_vanilla_softmax_offset_is_none(self):
        """Test that vanilla softmax type has None softmax_offset."""
        config = _make_config(softmax_type="vanilla")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNone(attn.softmax_offset)

    def test_off_by_one_softmax_offset(self):
        """Test that off-by-one softmax type has zero offset."""
        config = _make_config(softmax_type="off-by-one")
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.softmax_offset)
        self.assertEqual(
            attn.softmax_offset.shape, [4]
        )  # num_attention_heads_per_partition


class TestDotProductAttentionSlidingWindow(unittest.TestCase):
    """Tests for DotProductAttention with sliding window."""

    def test_sliding_window_construction(self):
        """Test construction with sliding window."""
        config = _make_config(
            sliding_window=2,
            window_attn_skip_freq=None,
        )
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        # Should have sliding_window set in scale_mask_softmax
        self.assertIsNotNone(attn.scale_mask_softmax)

    def test_no_sliding_window(self):
        """Test construction without sliding window."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.scale_mask_softmax)


class TestDotProductAttentionCustomDropout(unittest.TestCase):
    """Tests for DotProductAttention with custom attention dropout."""

    def test_custom_attention_dropout(self):
        """Test construction with custom attention_dropout."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            attention_dropout=0.1,
        )
        self.assertIsNotNone(attn.attention_dropout)


if __name__ == "__main__":
    unittest.main()
