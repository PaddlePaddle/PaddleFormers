# Copyright (c) 2026 PaddleFaddle Authors. All Rights Reserved.
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


class TestDotProductAttentionEagerMode(unittest.TestCase):
    """Tests for DotProductAttention with eager attention implementation."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_eager_mode_rejects_packed_seq(self, mock_pg, mock_softmax):
        """Eager mode should reject packed_seq_params."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.gpt_model_use_experimental_version = False
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
        config.params_dtype = "bfloat16"
        config.init_method = MagicMock()
        config._attn_implementation = "eager"

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_packed = MagicMock()
        with self.assertRaises(ValueError):
            attn.forward(
                query=paddle.randn([1, 4, 8, 64]),
                key=paddle.randn([1, 4, 8, 64]),
                value=paddle.randn([1, 4, 8, 64]),
                attention_mask=None,
                packed_seq_params=mock_packed,
            )


class TestDotProductAttentionQueryKeyLayerScaling(unittest.TestCase):
    """Tests for DotProductAttention with apply_query_key_layer_scaling."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_layer_scaling_divides_softmax_scale(self, mock_pg, mock_softmax):
        """apply_query_key_layer_scaling should divide softmax_scale by layer_number."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.gpt_model_use_experimental_version = False
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.apply_query_key_layer_scaling = True
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "bfloat16"
        config.init_method = MagicMock()

        attn = DotProductAttention(
            config=config,
            layer_number=3,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=None,
        )
        import math

        base_scale = 1.0 / math.sqrt(64)
        expected = base_scale / 3  # Divided by layer_number
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)


class TestDotProductAttentionSoftmaxTypes(unittest.TestCase):
    """Tests for DotProductAttention with different softmax types."""

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_invalid_softmax_type_raises(self, mock_pg, mock_softmax):
        """Invalid softmax_type should raise ValueError."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.gpt_model_use_experimental_version = False
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
        config.softmax_type = "invalid_type"
        config.perform_initialization = True
        config.params_dtype = "bfloat16"
        config.init_method = MagicMock()
        config.add_full_attention_sink_bias = False
        config.add_swa_attention_sink_bias = False

        with self.assertRaises(ValueError):
            DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )

    @patch("paddleformers.fleet.transformer.dot_product_attention.FusedScaleMaskSoftmax")
    @patch("paddleformers.fleet.transformer.dot_product_attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_off_by_one_softmax_type(self, mock_pg, mock_softmax):
        """off-by-one softmax_type should create softmax_offset tensor."""
        mock_pg.return_value = MagicMock(tp=MagicMock(world_size=1, rank=0))
        config = MagicMock()
        config.gpt_model_use_experimental_version = False
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
        config.softmax_type = "off-by-one"
        config.perform_initialization = True
        config.params_dtype = "bfloat16"
        config.init_method = MagicMock()

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertIsNotNone(attn.softmax_offset)
        self.assertTrue(paddle.is_tensor(attn.softmax_offset))


if __name__ == "__main__":
    unittest.main()
