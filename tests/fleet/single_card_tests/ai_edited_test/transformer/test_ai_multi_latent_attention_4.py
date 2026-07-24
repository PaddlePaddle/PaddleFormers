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

from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)


def _make_mla_self_attn(**attrs):
    """Create MLASelfAttention with mocked __init__."""
    with patch.object(
        MLASelfAttention, "__init__", lambda self, *a, **kw: None
    ):
        attn = MLASelfAttention.__new__(MLASelfAttention)
        object.__setattr__(attn, "_sub_layers", {})
        object.__setattr__(attn, "_parameters", {})
        object.__setattr__(attn, "_buffers", {})
        object.__setattr__(attn, "_non_persistable_buffers", set())
        for k, v in attrs.items():
            object.__setattr__(attn, k, v)
        return attn


class TestMLASelfAttentionBackwardDW(unittest.TestCase):
    """Tests for MLASelfAttention.backward_dw."""

    def test_backward_dw_calls_all_proj(self):
        """backward_dw should call _backward_kv_proj, _backward_q_proj, _backward_output_proj."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "_backward_kv_proj", MagicMock())
        object.__setattr__(attn, "_backward_q_proj", MagicMock())
        object.__setattr__(attn, "_backward_output_proj", MagicMock())
        attn.backward_dw()
        attn._backward_kv_proj.assert_called_once()
        attn._backward_q_proj.assert_called_once()
        attn._backward_output_proj.assert_called_once()


class TestMLASelfAttentionBackwardKVProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_kv_proj."""

    def test_backward_kv_proj_calls_both_layers(self):
        """_backward_kv_proj should call backward_dw on kv_b_proj and kv_a_proj_with_mqa."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "kv_b_proj", MagicMock())
        object.__setattr__(attn, "kv_a_proj_with_mqa", MagicMock())
        attn._backward_kv_proj()
        attn.kv_b_proj.backward_dw.assert_called_once()
        attn.kv_a_proj_with_mqa.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardQProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_q_proj."""

    def test_backward_q_proj_with_lora_rank(self):
        """_backward_q_proj should call q_a_proj and q_b_proj when q_lora_rank is not None."""
        config = MagicMock()
        config.q_lora_rank = 768
        attn = _make_mla_self_attn(config=config)
        object.__setattr__(attn, "q_a_proj", MagicMock())
        object.__setattr__(attn, "q_b_proj", MagicMock())
        attn._backward_q_proj()
        attn.q_a_proj.backward_dw.assert_called_once()
        attn.q_b_proj.backward_dw.assert_called_once()

    def test_backward_q_proj_without_lora_rank(self):
        """_backward_q_proj should call q_proj when q_lora_rank is None."""
        config = MagicMock()
        config.q_lora_rank = None
        attn = _make_mla_self_attn(config=config)
        object.__setattr__(attn, "q_proj", MagicMock())
        attn._backward_q_proj()
        attn.q_proj.backward_dw.assert_called_once()


class TestMLASelfAttentionBackwardOutputProj(unittest.TestCase):
    """Tests for MLASelfAttention._backward_output_proj."""

    def test_backward_output_proj_calls_o_proj(self):
        """_backward_output_proj should call backward_dw on o_proj."""
        attn = _make_mla_self_attn()
        object.__setattr__(attn, "o_proj", MagicMock())
        attn._backward_output_proj()
        attn.o_proj.backward_dw.assert_called_once()


class TestMLASelfAttentionSublayersSpecDefaults(unittest.TestCase):
    """Tests for MLASelfAttentionSublayersSpec defaults."""

    def test_default_q_a_layernorm_is_none(self):
        """q_a_layernorm should default to None."""
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.q_a_layernorm)

    def test_default_kv_a_layernorm_is_none(self):
        """kv_a_layernorm should default to None."""
        spec = MLASelfAttentionSublayersSpec()
        self.assertIsNone(spec.kv_a_layernorm)


if __name__ == "__main__":
    unittest.main()
