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

from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class TestTransformerConfigDefaults(unittest.TestCase):
    """Tests for TransformerConfig default values."""

    def test_default_num_hidden_layers(self):
        """Test default num_hidden_layers."""
        config = TransformerConfig()
        self.assertEqual(config.num_hidden_layers, 1)

    def test_default_num_nextn_predict_layers(self):
        """Test default num_nextn_predict_layers."""
        config = TransformerConfig()
        self.assertEqual(config.num_nextn_predict_layers, 0)

    def test_default_hidden_size(self):
        """Test default hidden_size."""
        config = TransformerConfig()
        self.assertEqual(config.hidden_size, 0)

    def test_custom_values(self):
        """Test TransformerConfig with custom values."""
        config = TransformerConfig(
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 8)


class TestTransformerConfigRecompute(unittest.TestCase):
    """Tests for TransformerConfig recompute settings."""

    def test_recompute_granularity_default(self):
        """Test default recompute_granularity."""
        config = TransformerConfig()
        self.assertIsNone(config.recompute_granularity)

    def test_recompute_granularity_selective(self):
        """Test recompute_granularity='selective'."""
        config = TransformerConfig(
            recompute_granularity="selective",
            recompute_method="block",
            recompute_num_layers=2,
            recompute_modules=["core_attn"],
        )
        self.assertEqual(config.recompute_granularity, "selective")

    def test_recompute_granularity_full(self):
        """Test recompute_granularity='full'."""
        config = TransformerConfig(
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=2,
        )
        self.assertEqual(config.recompute_granularity, "full")


class TestTransformerConfigAttentionSettings(unittest.TestCase):
    """Tests for TransformerConfig attention-related settings."""

    def test_num_attention_heads(self):
        """Test num_attention_heads setting."""
        config = TransformerConfig(num_attention_heads=8)
        self.assertEqual(config.num_attention_heads, 8)

    def test_num_key_value_heads(self):
        """Test num_key_value_heads setting."""
        config = TransformerConfig(num_attention_heads=8, num_key_value_heads=2)
        self.assertEqual(config.num_key_value_heads, 2)

    def test_head_dim(self):
        """Test head_dim setting."""
        config = TransformerConfig(head_dim=32)
        self.assertEqual(config.head_dim, 32)


class TestTransformerConfigFP8Settings(unittest.TestCase):
    """Tests for TransformerConfig fp8 settings."""

    def test_fp8_default(self):
        """Test default fp8 setting."""
        config = TransformerConfig()
        self.assertFalse(config.fp8)


class TestTransformerConfigNormalization(unittest.TestCase):
    """Tests for TransformerConfig normalization setting."""

    def test_normalization_rmsnorm(self):
        """Test normalization='RMSNorm'."""
        config = TransformerConfig(normalization="RMSNorm")
        self.assertEqual(config.normalization, "RMSNorm")

    def test_normalization_layernorm(self):
        """Test normalization='LayerNorm'."""
        config = TransformerConfig(normalization="LayerNorm")
        self.assertEqual(config.normalization, "LayerNorm")


class TestTransformerConfigMLASettings(unittest.TestCase):
    """Tests for TransformerConfig MLA settings."""

    def test_multi_latent_attention_default(self):
        """Test default multi_latent_attention setting."""
        config = TransformerConfig()
        self.assertFalse(config.multi_latent_attention)

    def test_q_lora_rank(self):
        """Test q_lora_rank setting."""
        config = TransformerConfig(q_lora_rank=32)
        self.assertEqual(config.q_lora_rank, 32)

    def test_kv_lora_rank(self):
        """Test kv_lora_rank setting."""
        config = TransformerConfig(kv_lora_rank=16)
        self.assertEqual(config.kv_lora_rank, 16)


class TestTransformerConfigMoESettings(unittest.TestCase):
    """Tests for TransformerConfig MoE settings."""

    def test_n_routed_experts_default(self):
        """Test default n_routed_experts."""
        config = TransformerConfig()
        self.assertIsNone(config.n_routed_experts)

    def test_moe_intermediate_size(self):
        """Test moe_intermediate_size setting."""
        config = TransformerConfig(moe_intermediate_size=256)
        self.assertEqual(config.moe_intermediate_size, 256)


if __name__ == "__main__":
    unittest.main()
