# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType


class TestDotProductAttentionInit(unittest.TestCase):
    """Tests for DotProductAttention initialization."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_context_parallel_size_must_be_1(self, mock_pg, mock_softmax):
        """DotProductAttention should assert context_parallel_size == 1."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.context_parallel_size = 2  # Not 1

        with self.assertRaises(AssertionError):
            DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_softmax_scale_default(self, mock_pg, mock_softmax):
        """When softmax_scale is None, it should compute from hidden_size."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "float32"
        config.init_method = MagicMock()

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=None,
        )
        import math

        expected = 1.0 / math.sqrt(64)
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_layer_number_clamped_to_1(self, mock_pg, mock_softmax):
        """layer_number should be clamped to at least 1."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "float32"
        config.init_method = MagicMock()

        attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.layer_number, 0)


class TestDotProductAttentionSoftmaxOffset(unittest.TestCase):
    """Tests for DotProductAttention softmax_offset initialization."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_vanilla_softmax_offset_is_none(self, mock_pg, mock_softmax):
        """When softmax_type is vanilla, softmax_offset should be None."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "float32"
        config.init_method = MagicMock()
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNone(attn.softmax_offset)


class TestDotProductAttentionForwardAssertions(unittest.TestCase):
    """Tests for DotProductAttention.forward assertion checks."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_forward_rejects_attention_bias(self, mock_pg, mock_softmax):
        """forward should reject attention_bias."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "float32"
        config.init_method = MagicMock()
        config._attn_implementation = "flash"
        config.gpt_model_use_experimental_version = False

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        with self.assertRaises(AssertionError):
            attn.forward(
                query=paddle.randn([1, 4, 8, 64]),
                key=paddle.randn([1, 4, 8, 64]),
                value=paddle.randn([1, 4, 8, 64]),
                attention_mask=None,
                attention_bias=paddle.randn([1, 1, 4, 4]),
            )


if __name__ == "__main__":
    unittest.main()
