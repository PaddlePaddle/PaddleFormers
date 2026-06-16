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
from dataclasses import fields


class TestGPTConfigDefaults(unittest.TestCase):
    """Test GPTConfig default field values."""

    def test_default_vocab_size(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.vocab_size, 1024)

    def test_default_position_embedding_type(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.position_embedding_type, "rope")

    def test_default_rotary_percent(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.rotary_percent, 1.0)

    def test_default_rotary_base(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.rotary_base, 10000)

    def test_default_rope_scaling(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.rope_scaling, 1.0)

    def test_default_max_sequence_length(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertEqual(config.max_sequence_length, 64)

    def test_default_tie_word_embeddings(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertFalse(config.tie_word_embeddings)

    def test_default_moe_expert_fusion(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertFalse(config.moe_expert_fusion)

    def test_default_parallel_output(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertTrue(config.parallel_output)

    def test_default_layer_types(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        self.assertIsNone(config.layer_types)


class TestGPTConfigCustomValues(unittest.TestCase):
    """Test GPTConfig with custom field values."""

    def test_custom_vocab_size(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(vocab_size=32000)
        self.assertEqual(config.vocab_size, 32000)

    def test_custom_position_embedding_type(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(position_embedding_type="learned_absolute")
        self.assertEqual(config.position_embedding_type, "learned_absolute")

    def test_custom_rope_scaling(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(rope_scaling=2.0)
        self.assertEqual(config.rope_scaling, 2.0)

    def test_custom_max_sequence_length(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(max_sequence_length=2048)
        self.assertEqual(config.max_sequence_length, 2048)

    def test_custom_tie_word_embeddings_true(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(tie_word_embeddings=True)
        self.assertTrue(config.tie_word_embeddings)

    def test_custom_parallel_output_false(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(parallel_output=False)
        self.assertFalse(config.parallel_output)

    def test_custom_layer_types(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(layer_types=["attention", "attention"])
        self.assertEqual(config.layer_types, ["attention", "attention"])


class TestGPTConfigInheritance(unittest.TestCase):
    """Test that GPTConfig properly inherits from TransformerConfig."""

    def test_is_dataclass(self):
        import dataclasses

        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        self.assertTrue(dataclasses.is_dataclass(GPTConfig))

    def test_inherits_from_transformer_config(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        self.assertTrue(issubclass(GPTConfig, TransformerConfig))

    def test_has_transformer_config_fields(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig()
        # These fields come from TransformerConfig
        self.assertTrue(hasattr(config, "hidden_size"))
        self.assertTrue(hasattr(config, "num_hidden_layers"))
        self.assertTrue(hasattr(config, "num_attention_heads"))

    def test_gpt_config_has_all_gpt_fields(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        gpt_field_names = {
            "vocab_size",
            "position_embedding_type",
            "rotary_percent",
            "rotary_base",
            "rope_scaling",
            "max_sequence_length",
            "tie_word_embeddings",
            "moe_expert_fusion",
            "parallel_output",
            "layer_types",
        }
        config_fields = {f.name for f in fields(GPTConfig)}
        self.assertTrue(gpt_field_names.issubset(config_fields))


class TestGPTConfigEdgeCases(unittest.TestCase):
    """Test GPTConfig edge cases."""

    def test_zero_vocab_size(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(vocab_size=0)
        self.assertEqual(config.vocab_size, 0)

    def test_large_vocab_size(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(vocab_size=256000)
        self.assertEqual(config.vocab_size, 256000)

    def test_negative_rotary_percent(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(rotary_percent=-0.5)
        self.assertEqual(config.rotary_percent, -0.5)

    def test_zero_rotary_base(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(rotary_base=0)
        self.assertEqual(config.rotary_base, 0)

    def test_empty_layer_types(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(layer_types=[])
        self.assertEqual(config.layer_types, [])

    def test_multiple_kwargs_override(self):
        from paddleformers.fleet.models.gpt.gpt_config import GPTConfig

        config = GPTConfig(
            vocab_size=50000,
            position_embedding_type="none",
            rotary_percent=0.5,
            rotary_base=500000,
            rope_scaling=4.0,
            max_sequence_length=4096,
            tie_word_embeddings=True,
            moe_expert_fusion=True,
            parallel_output=False,
        )
        self.assertEqual(config.vocab_size, 50000)
        self.assertEqual(config.position_embedding_type, "none")
        self.assertEqual(config.rotary_percent, 0.5)
        self.assertEqual(config.rotary_base, 500000)
        self.assertEqual(config.rope_scaling, 4.0)
        self.assertEqual(config.max_sequence_length, 4096)
        self.assertTrue(config.tie_word_embeddings)
        self.assertTrue(config.moe_expert_fusion)
        self.assertFalse(config.parallel_output)


if __name__ == "__main__":
    unittest.main()
