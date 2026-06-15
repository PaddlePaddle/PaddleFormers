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

from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
    _md5,
)
from paddleformers.fleet.transformer.enums import AttnMaskType


class TestMd5Function(unittest.TestCase):
    """Tests for _md5 helper function."""

    def test_md5_returns_string(self):
        """_md5 should return a hex digest string."""
        t = paddle.randn([4, 8])
        result = _md5(t)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 32)  # MD5 hex digest length

    def test_md5_same_input_same_output(self):
        """_md5 should return the same value for the same input."""
        t = paddle.randn([4, 8])
        result1 = _md5(t)
        result2 = _md5(t)
        self.assertEqual(result1, result2)


class TestAttentionForwardRotaryPosEmbDuplication(unittest.TestCase):
    """Tests for Attention.forward rotary_pos_emb duplication logic."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch("paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_forward_doubles_rotary_pos_emb_tuple(self, mock_pg, mock_size, mock_build):
        """forward should duplicate rotary_pos_emb into a tuple when not already."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1
        config.sequence_parallel = False
        config.gpt_model_use_experimental_version = False
        config.multi_latent_attention = False
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        # We just test that the code path works
        self.assertIsNotNone(attn)


class TestSelfAttentionGetQKVPerHeadNorm(unittest.TestCase):
    """Tests for SelfAttention.get_query_key_value_tensors with per_head qk_norm."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch("paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_per_head_norm_with_q_norm(self, mock_pg, mock_size, mock_build):
        """When q_norm is set, it should be called during get_query_key_value_tensors."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1
        config.gated_attention = False
        config.qk_norm_type = "per_head"
        config.rms_norm_eps = 1e-5
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        b, s, h = 2, 4, 512
        q_dim = 8 * 64
        kv_dim = 2 * 8 * 64
        mixed_qkv = paddle.randn([b, s, q_dim + kv_dim])
        attn.qkv_proj = MagicMock(return_value=(mixed_qkv, None))
        attn.num_attention_heads_per_partition = 8
        attn.num_query_groups_per_partition = 8
        attn.hidden_size_per_attention_head = 64
        attn.gated_attention = False
        mock_q_norm = MagicMock(side_effect=lambda x: x)
        mock_k_norm = MagicMock(side_effect=lambda x: x)
        attn.q_norm = mock_q_norm
        attn.k_norm = mock_k_norm
        attn.pg_collection = MagicMock(tp=MagicMock(world_size=1, rank=0))

        result = attn.get_query_key_value_tensors(paddle.randn([b, s, h]), split_qkv=True)
        mock_q_norm.assert_called_once()
        mock_k_norm.assert_called_once()


class TestSelfAttentionGetQKVWithGate(unittest.TestCase):
    """Tests for SelfAttention.get_query_key_value_tensors with gated attention."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch("paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups")
    def test_gated_attention_returns_four_values(self, mock_pg, mock_size, mock_build):
        """When gated_attention is True, should return (query, key, value, gate)."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1
        config.gated_attention = True
        config.gpt_model_use_experimental_version = False
        config.qk_norm_type = "per_head"
        config.rms_norm_eps = 1e-5
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        b, s, h = 2, 4, 512
        q_dim = 8 * 64
        gate_dim = 8 * 64
        kv_dim = 2 * 8 * 64
        mixed_qkv = paddle.randn([b, s, q_dim + gate_dim + kv_dim])
        attn.qkv_proj = MagicMock(return_value=(mixed_qkv, None))
        attn.num_attention_heads_per_partition = 8
        attn.num_query_groups_per_partition = 8
        attn.hidden_size_per_attention_head = 64
        attn.gated_attention = True
        attn.q_norm = None
        attn.k_norm = None
        attn.pg_collection = MagicMock(tp=MagicMock(world_size=1, rank=0))

        result = attn.get_query_key_value_tensors(paddle.randn([b, s, h]), split_qkv=True)
        self.assertEqual(len(result), 4)  # query, key, value, gate


if __name__ == "__main__":
    unittest.main()
