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
from unittest.mock import MagicMock, patch

from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)


class TestSelfAttentionGatedAttentionAttribute(unittest.TestCase):
    """Tests for SelfAttention gated_attention attribute."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch(
        "paddleformers.fleet.transformer.attention.get_pg_size", return_value=1
    )
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_gated_attention_attribute(self, mock_pg, mock_size, mock_build):
        """Test that gated_attention attribute is set from config."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.gated_attention = True
        config.recompute_granularity = None
        config.recompute_modules = None
        config.use_bias = False
        config.attention_bias = False
        config.softmax_scale = None
        config.init_method = MagicMock()
        config.output_layer_init_method = MagicMock()
        config.tensor_model_parallel_size = 1
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec(
            qkv_proj=MagicMock(),
            core_attention=MagicMock(),
            o_proj=MagicMock(),
        )

        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
        )
        self.assertTrue(attn.gated_attention)


class TestSelfAttentionRRFlashAttention(unittest.TestCase):
    """Tests for SelfAttention rr_flash_attention attribute."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch(
        "paddleformers.fleet.transformer.attention.get_pg_size", return_value=1
    )
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_rr_flash_attention_with_list(self, mock_pg, mock_size, mock_build):
        """Test rr_flash_attention is set when config provides list."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.gated_attention = False
        config.recompute_granularity = None
        config.recompute_modules = None
        config.use_bias = False
        config.attention_bias = False
        config.softmax_scale = None
        config.init_method = MagicMock()
        config.output_layer_init_method = MagicMock()
        config.tensor_model_parallel_size = 1
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec(
            qkv_proj=MagicMock(),
            core_attention=MagicMock(),
            o_proj=MagicMock(),
        )

        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
        )
        # Default should be False
        self.assertFalse(attn.use_rr_flash_attention)


if __name__ == "__main__":
    unittest.main()
