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

import paddle

from paddleformers.fleet.transformer.attention import (
    Attention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
    SelfAttentionVHA,
    SelfAttentionVHASublayersSpec,
    _apply_ec_complex_3d_mrope,
)


class TestSelfAttentionSublayersSpec(unittest.TestCase):
    """Tests for SelfAttentionSublayersSpec dataclass."""

    def test_default_values(self):
        spec = SelfAttentionSublayersSpec()
        self.assertIsNone(spec.qkv_proj)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)
        self.assertIsNone(spec.q_norm)
        self.assertIsNone(spec.k_norm)


class TestCrossAttentionSublayersSpec(unittest.TestCase):
    """Tests for CrossAttentionSublayersSpec dataclass."""

    def test_default_values(self):
        spec = CrossAttentionSublayersSpec()
        self.assertIsNone(spec.linear_q)
        self.assertIsNone(spec.linear_kv)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)


class TestApplyEcComplex3dMrope(unittest.TestCase):
    """Tests for _apply_ec_complex_3d_mrope function."""

    def test_basic_rope_apply(self):
        """Test basic EC complex 3D MRoPE application."""
        batch, seq_len, num_heads, head_dim = 2, 4, 2, 8
        query = paddle.randn([batch, seq_len, num_heads, head_dim])
        key = paddle.randn([batch, seq_len, num_heads, head_dim])
        position_ids = paddle.zeros([batch, seq_len, 3], dtype="int64")
        for i in range(batch):
            for j in range(seq_len):
                position_ids[i, j, :] = j

        # mrope_section must satisfy: point_num >= mrope_section[0]
        # point_num = head_dim // 2 = 4, so mrope_section[0] <= 4
        q_out, k_out = _apply_ec_complex_3d_mrope(
            query, key, position_ids, head_dim=head_dim, mrope_section=[2, 1, 1]
        )
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)

    def test_rope_with_mrope_section(self):
        """Test with custom mrope_section."""
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        query = paddle.randn([batch, seq_len, num_heads, head_dim])
        key = paddle.randn([batch, seq_len, num_heads, head_dim])
        position_ids = paddle.zeros([batch, seq_len, 3], dtype="int64")
        for j in range(seq_len):
            position_ids[0, j, :] = j

        q_out, k_out = _apply_ec_complex_3d_mrope(
            query,
            key,
            position_ids,
            head_dim=head_dim,
            mrope_section=[4, 2, 2],
        )
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)

    def test_rope_with_position_ids_padding(self):
        """Test MRoPE when position_ids is shorter than query."""
        batch, seq_len, num_heads, head_dim = 1, 8, 2, 8
        query = paddle.randn([batch, seq_len, num_heads, head_dim])
        key = paddle.randn([batch, seq_len, num_heads, head_dim])
        # Position ids shorter than query seq_len
        position_ids = paddle.zeros([batch, seq_len - 2, 3], dtype="int64")
        for j in range(seq_len - 2):
            position_ids[0, j, :] = j

        q_out, k_out = _apply_ec_complex_3d_mrope(
            query, key, position_ids, head_dim=head_dim, mrope_section=[2, 1, 1]
        )
        self.assertEqual(q_out.shape, query.shape)
        self.assertEqual(k_out.shape, key.shape)

    def test_rope_with_custom_rope_theta(self):
        """Test with custom rope_theta."""
        batch, seq_len, num_heads, head_dim = 1, 4, 2, 8
        query = paddle.randn([batch, seq_len, num_heads, head_dim])
        key = paddle.randn([batch, seq_len, num_heads, head_dim])
        position_ids = paddle.zeros([batch, seq_len, 3], dtype="int64")
        for j in range(seq_len):
            position_ids[0, j, :] = j

        q_out, k_out = _apply_ec_complex_3d_mrope(
            query,
            key,
            position_ids,
            head_dim=head_dim,
            rope_theta=500000.0,
            mrope_section=[2, 1, 1],
        )
        self.assertEqual(q_out.shape, query.shape)


class TestAttentionAbstract(unittest.TestCase):
    """Tests for Attention abstract class."""

    def test_set_for_recompute_input_layernorm_raises(self):
        """Test that set_for_recompute_input_layernorm raises NotImplementedError."""

        # Create a concrete subclass to test the method
        class ConcreteAttention(Attention):
            def get_query_key_value_tensors(self, *args, **kwargs):
                return None

        with patch.object(Attention, "__init__", lambda self, *a, **kw: None):
            attn = ConcreteAttention.__new__(ConcreteAttention)
            with self.assertRaises(NotImplementedError):
                attn.set_for_recompute_input_layernorm()


class TestSelfAttentionRecompute(unittest.TestCase):
    """Tests for SelfAttention recompute configuration."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_selective_recompute_with_list(
        self, mock_pg, mock_size, mock_build
    ):
        """Test SelfAttention with selective recompute (list)."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.recompute_granularity = "selective"
        config.recompute_modules = ["core_attn"]
        config.recompute_num_layers = None
        config.recompute_method = None
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
        self.assertTrue(attn.recompute_core_attention)

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_selective_recompute_with_dict(
        self, mock_pg, mock_size, mock_build
    ):
        """Test SelfAttention with selective recompute (dict)."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.recompute_granularity = "selective"
        config.recompute_modules = {"core_attn": 2}
        config.recompute_num_layers = None
        config.recompute_method = "block"
        config.use_bias = False
        config.attention_bias = False
        config.softmax_scale = None
        config.init_method = MagicMock()
        config.output_layer_init_method = MagicMock()
        config.tensor_model_parallel_size = 1
        config.sliding_window = None
        config.use_vha_attention = False
        # Required for need_recompute_in_block
        config.num_empty_layers_add_in_head = 0
        config.num_hidden_layers = 2
        config.num_empty_layers_add_in_tail = 0
        config.pipeline_model_parallel_size = 1
        config.virtual_pipeline_model_parallel_size = 1

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
        self.assertTrue(attn.recompute_core_attention)


class TestSelfAttentionGatedAttention(unittest.TestCase):
    """Tests for SelfAttention gated attention configuration."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_gated_attention_enabled(self, mock_pg, mock_size, mock_build):
        """Test SelfAttention with gated attention enabled."""
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
        config.recompute_num_layers = None
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


class TestVHAAttentionInit(unittest.TestCase):
    """Tests for VHA (Virtual Head Attention) initialization and methods."""

    def _make_vha_attn(self, mock_pg, mock_build):
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        config = MagicMock()
        config.head_dim = 16
        config.num_attention_heads = 8
        config.num_key_value_heads = 4
        config.hidden_size = 64
        config.v_head_dim = 16
        config.vha_postmix_rank = 2
        config.vha_q_lora_rank = 16
        config.swa_vha_q_lora_rank = 16
        config.swa_vha_postmix_rank = None
        config.recompute_granularity = None
        config.recompute_modules = None
        config.use_bias = False
        config.attention_bias = False
        config.softmax_scale = None
        config.init_method = MagicMock()
        config.output_layer_init_method = MagicMock()
        config.tensor_model_parallel_size = 1
        config.sliding_window = None
        config.attention_value_scale = None
        config.gated_attention = False

        spec = SelfAttentionVHASublayersSpec(
            q_proj=MagicMock(),
            k_proj=MagicMock(),
            v_proj=MagicMock(),
            core_attention=MagicMock(),
            o_proj=MagicMock(),
        )

        return SelfAttentionVHA(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
        )

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_vha_creates_parameters(self, mock_pg, mock_size, mock_build):
        """VHA init should create vha_premix_weight, vha_postmix_U, vha_postmix_V."""
        attn = self._make_vha_attn(mock_pg, mock_build)
        self.assertEqual(list(attn.vha_premix_weight.shape), [4, 16, 16])
        self.assertEqual(list(attn.vha_postmix_U.shape), [8, 2])
        self.assertEqual(list(attn.vha_postmix_V.shape), [8, 2])

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_vha_premix_output_shape(self, mock_pg, mock_size, mock_build):
        """_apply_vha_premix should expand query from kv_heads to num_heads."""
        attn = self._make_vha_attn(mock_pg, mock_build)
        # Input: [batch, seq, num_attention_heads // num_key_value_heads, q_head_dim] = [2,4,2,16]
        query = paddle.randn([2, 4, 2, 16])
        out = attn._apply_vha_premix(query)
        # Output: [batch, seq, num_attention_heads, head_dim]
        self.assertEqual(list(out.shape), [2, 4, 8, 16])

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch("paddleformers.fleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_vha_postmix_identity_at_init(self, mock_pg, mock_size, mock_build):
        """_apply_vha_postmix should be near-identity at init (V=0)."""
        attn = self._make_vha_attn(mock_pg, mock_build)
        attn_out = paddle.randn([2, 4, 8 * 16])
        result = attn._apply_vha_postmix(attn_out)
        self.assertEqual(list(result.shape), list(attn_out.shape))
        diff = (result - attn_out).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
