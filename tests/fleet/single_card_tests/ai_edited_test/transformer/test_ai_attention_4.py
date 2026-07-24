# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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
    CrossAttention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.enums import AttnMaskType


def _make_self_attn(**attrs):
    """Create SelfAttention with mocked __init__."""
    with patch.object(SelfAttention, "__init__", lambda self, *a, **kw: None):
        attn = SelfAttention.__new__(SelfAttention)
        object.__setattr__(attn, "_sub_layers", {})
        object.__setattr__(attn, "_parameters", {})
        object.__setattr__(attn, "_buffers", {})
        object.__setattr__(attn, "_non_persistable_buffers", set())
        for k, v in attrs.items():
            object.__setattr__(attn, k, v)
        return attn


class TestAttentionSetForRecomputeInputLayernorm(unittest.TestCase):
    """Tests for Attention.set_for_recompute_input_layernorm."""

    def test_raises_not_implemented(self):
        """set_for_recompute_input_layernorm should raise NotImplementedError on base class."""
        with patch.object(
            SelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = SelfAttention.__new__(SelfAttention)
            with self.assertRaises(NotImplementedError):
                attn.set_for_recompute_input_layernorm()


class TestCrossAttentionSublayersSpecDefaults(unittest.TestCase):
    """Tests for CrossAttentionSublayersSpec default values."""

    def test_default_values_are_none(self):
        """All fields of CrossAttentionSublayersSpec should default to None."""
        spec = CrossAttentionSublayersSpec()
        self.assertIsNone(spec.linear_q)
        self.assertIsNone(spec.linear_kv)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)


class TestCrossAttentionInitValidation(unittest.TestCase):
    """Tests for CrossAttention initialization validation."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch(
        "paddleformers.fleet.transformer.attention.get_pg_size", return_value=1
    )
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_cross_attention_rejects_gqa(self, mock_pg, mock_size, mock_build):
        """CrossAttention should reject group query attention."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 4  # GQA
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1
        config.sliding_window = None

        spec = CrossAttentionSublayersSpec()
        with self.assertRaises(ValueError):
            CrossAttention(
                config=config,
                sublayers_spec=spec,
                layer_number=1,
                attn_mask_type=AttnMaskType.padding,
            )


class TestSelfAttentionBackwardDW(unittest.TestCase):
    """Tests for SelfAttention.backward_dw."""

    def test_backward_dw_calls_qkv_and_output(self):
        """backward_dw should call _backward_qkv_proj and _backward_output_proj."""
        attn = _make_self_attn()
        object.__setattr__(attn, "qkv_proj", MagicMock())
        object.__setattr__(attn, "o_proj", MagicMock())
        attn.backward_dw()
        attn.qkv_proj.backward_dw.assert_called_once()
        attn.o_proj.backward_dw.assert_called_once()


class TestAttentionInitAttributes(unittest.TestCase):
    """Tests for Attention attribute setup via concrete subclasses."""

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch(
        "paddleformers.fleet.transformer.attention.get_pg_size", return_value=1
    )
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_attention_sets_layer_number(self, mock_pg, mock_size, mock_build):
        """SelfAttention should set layer_number from constructor."""
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
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=3,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(attn.layer_number, 3)

    @patch("paddleformers.fleet.transformer.attention.build_spec_layer")
    @patch(
        "paddleformers.fleet.transformer.attention.get_pg_size", return_value=1
    )
    @patch(
        "paddleformers.fleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_attention_sets_attention_type(
        self, mock_pg, mock_size, mock_build
    ):
        """SelfAttention should have attention_type='self'."""
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
        config.sliding_window = None
        config.use_vha_attention = False

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        self.assertEqual(attn.attention_type, "self")


if __name__ == "__main__":
    unittest.main()
