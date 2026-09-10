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

from paddleformers.cli.train.ernie_pretrain.models.ernie.configuration import (
    ERNIE_PRETRAINED_INIT_CONFIGURATION,
    ERNIE_PRETRAINED_RESOURCE_FILES_MAP,
    ErnieMoEConfig,
)


class TestErnieMoEConfig(unittest.TestCase):
    """Tests for ErnieMoEConfig class."""

    def test_default_construction(self):
        """Test creating ErnieMoEConfig with default values."""
        config = ErnieMoEConfig()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.model_type, "ernie")
        self.assertFalse(config.use_cache)
        self.assertFalse(config.use_recompute)
        self.assertTrue(config.use_flash_attn)
        self.assertFalse(config.use_mem_eff_attn)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)
        self.assertTrue(config.use_rmsnorm)
        self.assertTrue(config.fuse_rms_norm)
        self.assertFalse(config.fuse_ln)
        self.assertFalse(config.use_bias)

    def test_custom_construction(self):
        """Test creating ErnieMoEConfig with custom values."""
        config = ErnieMoEConfig(
            vocab_size=65536,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
        )
        self.assertEqual(config.vocab_size, 65536)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)

    def test_use_moe_property(self):
        """Test the use_moe property."""
        config_no_moe = ErnieMoEConfig(moe_num_experts=0)
        self.assertFalse(config_no_moe.use_moe)

        config_moe = ErnieMoEConfig(moe_num_experts=8)
        self.assertTrue(config_moe.use_moe)

    def test_fp8_configs_default(self):
        """Test default fp8_configs structure."""
        config = ErnieMoEConfig()
        self.assertIn("quant_scheme", config.fp8_configs)
        self.assertEqual(config.fp8_configs["quant_scheme"], "DelayedScaling")
        self.assertIn("recipe", config.fp8_configs)
        self.assertIn("layers", config.fp8_configs)

    def test_fp8_configs_custom_override(self):
        """Test fp8_configs with custom overrides."""
        custom_recipe = {"amax_history_len": 512}
        config = ErnieMoEConfig(fp8_configs={"recipe": custom_recipe})
        # Should merge with defaults
        self.assertEqual(config.fp8_configs["recipe"]["amax_history_len"], 512)
        self.assertIn("format", config.fp8_configs["recipe"])

    def test_fp8_mem_configs_default(self):
        """Test default fp8_mem_configs."""
        config = ErnieMoEConfig()
        self.assertIn("shared_expert", config.fp8_mem_configs)
        self.assertFalse(config.fp8_mem_configs["shared_expert"])
        self.assertIn("dequant_input", config.fp8_mem_configs)

    def test_fp8_fused_ops_configs_default(self):
        """Test default fp8_fused_ops_configs."""
        config = ErnieMoEConfig()
        self.assertIn("stack_quant", config.fp8_fused_ops_configs)
        self.assertFalse(config.fp8_fused_ops_configs["stack_quant"])
        self.assertTrue(config.fp8_fused_ops_configs["split_group_gemm"])

    def test_moe_params(self):
        """Test MoE-related parameters."""
        config = ErnieMoEConfig(
            moe_num_experts=8,
            moe_layer_interval=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.01,
            moe_k=2,
        )
        self.assertEqual(config.moe_num_experts, 8)
        self.assertEqual(config.moe_layer_interval, 4)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertAlmostEqual(config.router_aux_loss_coef, 0.01)
        self.assertEqual(config.moe_k, 2)

    def test_moe_layer_end_index_default(self):
        """Test moe_layer_end_index default behavior (should be num_hidden_layers-1 when -1)."""
        config = ErnieMoEConfig(num_hidden_layers=10, moe_layer_end_index=-1)
        self.assertEqual(config.moe_layer_end_index, 9)

    def test_moe_layer_end_index_custom(self):
        """Test moe_layer_end_index with custom value."""
        config = ErnieMoEConfig(num_hidden_layers=10, moe_layer_end_index=5)
        self.assertEqual(config.moe_layer_end_index, 5)

    def test_scoring_func_softmax(self):
        """Test scoring_func with softmax."""
        config = ErnieMoEConfig(scoring_func="softmax")
        self.assertEqual(config.scoring_func, "softmax")

    def test_scoring_func_sigmoid(self):
        """Test scoring_func with sigmoid."""
        config = ErnieMoEConfig(scoring_func="sigmoid")
        self.assertEqual(config.scoring_func, "sigmoid")

    def test_use_recompute_attn_disables_recompute(self):
        """Test that use_recompute_attn=True disables use_recompute."""
        config = ErnieMoEConfig(use_recompute_attn=True, use_recompute=True)
        self.assertFalse(config.use_recompute)
        self.assertTrue(config.use_recompute_attn)

    def test_async_a2a_requires_quant_before_a2a(self):
        """Test that use_async_a2a=True requires use_quant_before_a2a=True."""
        with self.assertRaises(AssertionError):
            ErnieMoEConfig(use_async_a2a=True, use_quant_before_a2a=False)

    def test_async_a2a_with_quant(self):
        """Test use_async_a2a works when use_quant_before_a2a is True."""
        config = ErnieMoEConfig(use_async_a2a=True, use_quant_before_a2a=True)
        self.assertTrue(config.use_async_a2a)
        self.assertTrue(config.use_quant_before_a2a)

    def test_to_json_string(self):
        """Test to_json_string produces valid JSON."""
        config = ErnieMoEConfig()
        json_str = config.to_json_string(use_diff=False)
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, dict)
        self.assertIn("vocab_size", parsed)
        self.assertIn("hidden_size", parsed)

    def test_tie_word_embeddings_default(self):
        """Test that tie_word_embeddings defaults to False."""
        config = ErnieMoEConfig()
        self.assertFalse(config.tie_word_embeddings)

    def test_model_type(self):
        """Test model_type is 'ernie'."""
        config = ErnieMoEConfig()
        self.assertEqual(config.model_type, "ernie")

    def test_attribute_map(self):
        """Test the attribute_map mappings."""
        self.assertEqual(ErnieMoEConfig.attribute_map["n_positions"], "max_position_embeddings")
        self.assertEqual(ErnieMoEConfig.attribute_map["n_embd"], "hidden_size")
        self.assertEqual(ErnieMoEConfig.attribute_map["n_layer"], "num_hidden_layers")
        self.assertEqual(ErnieMoEConfig.attribute_map["n_head"], "num_attention_heads")
        self.assertEqual(ErnieMoEConfig.attribute_map["n_inner"], "intermediate_size")

    def test_insert_empty_layer_none_becomes_empty_list(self):
        """Test insert_empty_layer None becomes empty list."""
        config = ErnieMoEConfig(insert_empty_layer=None)
        self.assertEqual(config.insert_empty_layer, [])

    def test_insert_empty_layer_list(self):
        """Test insert_empty_layer with a list."""
        config = ErnieMoEConfig(insert_empty_layer=[1, 2])
        self.assertEqual(config.insert_empty_layer, [1, 2])

    def test_insert_empty_layer_invalid(self):
        """Test insert_empty_layer with non-list raises assertion."""
        with self.assertRaises(AssertionError):
            ErnieMoEConfig(insert_empty_layer="invalid")


class TestPretrainedInitConfiguration(unittest.TestCase):
    """Tests for ERNIE_PRETRAINED_INIT_CONFIGURATION."""

    def test_contains_tiny_random_ernie(self):
        """Test that tiny-random-ernie config exists."""
        self.assertIn("ernie/tiny-random-ernie", ERNIE_PRETRAINED_INIT_CONFIGURATION)

    def test_tiny_random_ernie_values(self):
        """Test tiny-random-ernie config values."""
        cfg = ERNIE_PRETRAINED_INIT_CONFIGURATION["ernie/tiny-random-ernie"]
        self.assertEqual(cfg["hidden_size"], 768)
        self.assertEqual(cfg["num_attention_heads"], 2)
        self.assertEqual(cfg["num_hidden_layers"], 2)
        self.assertEqual(cfg["vocab_size"], 32000)


class TestPretrainedResourceFilesMap(unittest.TestCase):
    """Tests for ERNIE_PRETRAINED_RESOURCE_FILES_MAP."""

    def test_has_model_state(self):
        """Test model_state key exists."""
        self.assertIn("model_state", ERNIE_PRETRAINED_RESOURCE_FILES_MAP)


if __name__ == "__main__":
    unittest.main()
