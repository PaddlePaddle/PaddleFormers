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

import unittest

from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config


class TestMiniMaxM2Config(unittest.TestCase):
    """Tests for MiniMaxM2Config."""

    def test_default_config(self):
        config = MiniMaxM2Config()
        self.assertEqual(config.model_type, "minimax_m2")
        self.assertEqual(config.vocab_size, 200064)
        self.assertEqual(config.hidden_size, 3072)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.moe_intermediate_size, 1536)
        self.assertEqual(config.num_hidden_layers, 62)
        self.assertEqual(config.num_attention_heads, 48)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.max_position_embeddings, 196608)
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertTrue(config.use_cache)
        self.assertEqual(config.rope_theta, 5000000)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_group, 1)
        self.assertEqual(config.topk_group, 1)
        self.assertEqual(config.first_k_dense_replace, 0)
        self.assertTrue(config.use_qk_norm)
        self.assertEqual(config.qk_norm_type, "per_layer")
        self.assertTrue(config.use_mtp)
        self.assertEqual(config.num_mtp_modules, 3)
        self.assertEqual(config.mtp_transformer_layers, 1)
        self.assertTrue(config.use_routing_bias)
        self.assertEqual(config.moe_layer_freq, 1)
        self.assertEqual(config.scoring_func, "sigmoid")

    def test_custom_config(self):
        config = MiniMaxM2Config(
            vocab_size=100000,
            hidden_size=2048,
            num_hidden_layers=32,
        )
        self.assertEqual(config.vocab_size, 100000)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 32)

    def test_rope_scaling_type_migration(self):
        """Test that 'type' field in rope_scaling is moved to 'rope_type'."""
        config = MiniMaxM2Config(rope_scaling={"type": "linear", "factor": 2.0})
        self.assertEqual(config.rope_scaling["rope_type"], "linear")

    def test_keys_to_ignore_at_inference(self):
        self.assertEqual(MiniMaxM2Config.keys_to_ignore_at_inference, ["past_key_values"])


if __name__ == "__main__":
    unittest.main()
