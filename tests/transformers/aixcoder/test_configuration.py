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

"""Tests for AiXcoder configuration."""

import unittest

from paddleformers.transformers.aixcoder.configuration import AixcoderConfig


class AixcoderConfigTest(unittest.TestCase):
    """Test AiXcoder configuration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_dict = {
            "hidden_size": 4096,
            "intermediate_size": 14464,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-5,
            "use_cache": True,
            "tie_word_embeddings": False,
            "rope_theta": 500000.0,
            "rope_scaling": None,
            "vocab_size": 49152,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 2,
        }

    def test_config_initialization(self):
        """Test configuration initialization."""
        config = AixcoderConfig(**self.config_dict)

        # Check basic attributes
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 14464)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.vocab_size, 49152)
        self.assertEqual(config.model_type, "aixcoder")

    def test_config_defaults(self):
        """Test default configuration values."""
        config = AixcoderConfig()

        # Check default values
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 11008)  # Default is 11008
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 32)  # Default is num_attention_heads
        self.assertEqual(config.vocab_size, 49152)  # Default is 49152
        self.assertEqual(config.rope_theta, 10000.0)  # Default is 10000.0
        self.assertIsNone(config.rope_scaling)
        self.assertEqual(config.use_cache, True)
        self.assertEqual(config.tie_word_embeddings, False)

    def test_config_rope_scaling(self):
        """Test RoPE scaling configuration."""
        # Test with valid RoPE scaling
        rope_scaling = {"type": "linear", "factor": 2.0}
        config = AixcoderConfig(rope_scaling=rope_scaling)
        self.assertEqual(config.rope_scaling, rope_scaling)

        # Test None rope_scaling (default)
        config = AixcoderConfig(rope_scaling=None)
        self.assertIsNone(config.rope_scaling)

        # Test rope_scaling_type and rope_scaling_factor
        config = AixcoderConfig(rope_scaling_type="linear", rope_scaling_factor=2.0)
        self.assertEqual(config.rope_scaling_type, "linear")
        self.assertEqual(config.rope_scaling_factor, 2.0)

    def test_config_special_tokens(self):
        """Test special token configuration."""
        # Test with custom token ids
        config = AixcoderConfig(bos_token_id=10, eos_token_id=20, pad_token_id=30)
        self.assertEqual(config.bos_token_id, 10)
        self.assertEqual(config.eos_token_id, 20)
        self.assertEqual(config.pad_token_id, 30)

        # Test default token ids
        config = AixcoderConfig()
        self.assertIsNotNone(config.bos_token_id)
        self.assertIsNotNone(config.eos_token_id)

    def test_config_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = AixcoderConfig(**self.config_dict)
        config_dict = config.to_dict()

        # Check key attributes are preserved
        self.assertEqual(config_dict["hidden_size"], 4096)
        self.assertEqual(config_dict["num_hidden_layers"], 32)
        self.assertEqual(config_dict["model_type"], "aixcoder")
        self.assertEqual(config_dict["vocab_size"], 49152)

    def test_config_from_pretrained(self):
        """Test loading configuration from pretrained."""
        # Save config to temp file and reload
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config = AixcoderConfig(**self.config_dict)
            config.save_pretrained(temp_dir)

            # Check config.json exists
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))

            # Load config
            loaded_config = AixcoderConfig.from_pretrained(temp_dir)

            # Check attributes match
            self.assertEqual(loaded_config.hidden_size, config.hidden_size)
            self.assertEqual(loaded_config.num_hidden_layers, config.num_hidden_layers)
            self.assertEqual(loaded_config.vocab_size, config.vocab_size)

    def test_config_update(self):
        """Test configuration update."""
        config = AixcoderConfig()

        # Update some attributes
        config.hidden_size = 2048
        config.num_hidden_layers = 16

        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 16)


if __name__ == "__main__":
    unittest.main()
