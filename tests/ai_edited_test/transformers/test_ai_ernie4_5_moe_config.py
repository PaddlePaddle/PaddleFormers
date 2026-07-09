# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import unittest

from paddleformers.transformers.ernie4_5_moe.configuration import Ernie4_5_MoeConfig


class TestErnie4_5_MoeConfig(unittest.TestCase):
    """Tests for Ernie4_5_MoeConfig."""

    def test_default_config(self):
        config = Ernie4_5_MoeConfig()
        self.assertEqual(config.model_type, "ernie4_5_moe")
        self.assertEqual(config.vocab_size, 103424)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.intermediate_size, 12288)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertFalse(config.use_cache)
        self.assertTrue(config.use_rmsnorm)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)
        self.assertFalse(config.use_bias)
        self.assertEqual(config.rope_theta, 10000)
        self.assertEqual(config.moe_num_experts, 16)
        self.assertEqual(config.moe_k, 2)
        self.assertTrue(config.moe_use_aux_free)

    def test_custom_config(self):
        config = Ernie4_5_MoeConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=6,
            num_attention_heads=8,
        )
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.intermediate_size, 4096)
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.num_attention_heads, 8)

    def test_head_dim_computed(self):
        config = Ernie4_5_MoeConfig(hidden_size=2560, num_attention_heads=2)
        self.assertEqual(config.head_dim, 1280)

    def test_head_dim_explicit(self):
        config = Ernie4_5_MoeConfig(hidden_size=2560, num_attention_heads=2, head_dim=64)
        self.assertEqual(config.head_dim, 64)

    def test_moe_layer_end_index_default(self):
        """moe_layer_end_index should default to num_hidden_layers - 1 when -1 is passed."""
        config = Ernie4_5_MoeConfig(num_hidden_layers=10, moe_layer_end_index=-1)
        self.assertEqual(config.moe_layer_end_index, 9)

    def test_moe_layer_end_index_explicit(self):
        config = Ernie4_5_MoeConfig(num_hidden_layers=10, moe_layer_end_index=5)
        self.assertEqual(config.moe_layer_end_index, 5)

    def test_tie_word_embeddings_default(self):
        """tie_word_embeddings should default to True."""
        config = Ernie4_5_MoeConfig()
        self.assertTrue(config.tie_word_embeddings)

    def test_to_json_string(self):
        config = Ernie4_5_MoeConfig()
        json_str = config.to_json_string()
        parsed = json.loads(json_str)
        self.assertIn("model_type", parsed)
        self.assertEqual(parsed["model_type"], "ernie4_5_moe")

    def test_to_json_string_full(self):
        config = Ernie4_5_MoeConfig()
        json_str = config.to_json_string(use_diff=False)
        parsed = json.loads(json_str)
        self.assertIn("vocab_size", parsed)
        self.assertIn("hidden_size", parsed)


if __name__ == "__main__":
    unittest.main()
