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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock

from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestTransformerConfigDefaults(unittest.TestCase):
    """Test default values of TransformerConfig."""

    def test_default_num_hidden_layers(self):
        cfg = _make_config()
        self.assertEqual(cfg.num_hidden_layers, 2)

    def test_default_hidden_size(self):
        cfg = _make_config()
        self.assertEqual(cfg.hidden_size, 128)

    def test_default_num_attention_heads(self):
        cfg = _make_config()
        self.assertEqual(cfg.num_attention_heads, 4)

    def test_default_intermediate_size_is_four_hidden(self):
        cfg = _make_config(hidden_size=64)
        self.assertEqual(cfg.intermediate_size, 4 * 64)

    def test_default_head_dim_computed(self):
        cfg = _make_config(hidden_size=128, num_attention_heads=4)
        self.assertEqual(cfg.head_dim, 32)

    def test_default_num_key_value_heads(self):
        cfg = _make_config()
        self.assertEqual(cfg.num_key_value_heads, 4)

    def test_default_normalization(self):
        cfg = _make_config()
        self.assertEqual(cfg.normalization, "RMSNorm")

    def test_default_rms_norm_eps(self):
        cfg = _make_config()
        self.assertEqual(cfg.rms_norm_eps, 1e-5)

    def test_default_hidden_dropout(self):
        cfg = _make_config()
        self.assertEqual(cfg.hidden_dropout_prob, 0.0)

    def test_default_attention_dropout(self):
        cfg = _make_config()
        self.assertEqual(cfg.attention_dropout, 0.0)

    def test_default_gated_linear_unit(self):
        cfg = _make_config()
        self.assertFalse(cfg.gated_linear_unit)

    def test_default_use_bias(self):
        cfg = _make_config()
        self.assertFalse(cfg.use_bias)

    def test_default_init_method_is_set(self):
        cfg = _make_config()
        self.assertIsNotNone(cfg.init_method)
        self.assertIsNotNone(cfg.output_layer_init_method)
        self.assertIsNotNone(cfg.embedding_init_method)


class TestTransformerConfigPostInitValidation(unittest.TestCase):
    """Test validation logic in __post_init__."""

    def test_intermediate_size_none_becomes_four_hidden(self):
        cfg = _make_config(hidden_size=256, intermediate_size=None)
        self.assertEqual(cfg.intermediate_size, 1024)

    def test_intermediate_size_explicit(self):
        cfg = _make_config(intermediate_size=512)
        self.assertEqual(cfg.intermediate_size, 512)

    def test_head_dim_none_computed_from_hidden_and_heads(self):
        cfg = _make_config(hidden_size=256, num_attention_heads=8, head_dim=None)
        self.assertEqual(cfg.head_dim, 32)

    def test_head_dim_explicit(self):
        cfg = _make_config(head_dim=64)
        self.assertEqual(cfg.head_dim, 64)

    def test_num_key_value_heads_none_equals_num_attention_heads(self):
        cfg = _make_config(num_attention_heads=4, num_key_value_heads=None)
        self.assertEqual(cfg.num_key_value_heads, 4)

    def test_num_key_value_heads_must_be_divisible_by_tp_size(self):
        with self.assertRaises(ValueError):
            _make_config(num_key_value_heads=3, tensor_model_parallel_size=2)

    def test_num_key_value_heads_valid_with_tp_size(self):
        cfg = _make_config(num_key_value_heads=4, tensor_model_parallel_size=2)
        self.assertEqual(cfg.num_key_value_heads, 4)

    def test_apply_query_key_layer_scaling_sets_fp32_softmax(self):
        cfg = _make_config(
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
        )
        self.assertTrue(cfg.attention_softmax_in_fp32)

    def test_recompute_granularity_empty_string_becomes_none(self):
        cfg = _make_config(recompute_granularity="")
        self.assertIsNone(cfg.recompute_granularity)

    def test_recompute_granularity_full_requires_method(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="full")

    def test_recompute_granularity_full_valid_block(self):
        cfg = _make_config(
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.assertEqual(cfg.recompute_granularity, "full")

    def test_recompute_granularity_full_valid_uniform(self):
        cfg = _make_config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        self.assertEqual(cfg.recompute_granularity, "full")

    def test_recompute_granularity_selective_requires_modules(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="selective")

    def test_recompute_granularity_invalid(self):
        with self.assertRaises(AssertionError):
            _make_config(recompute_granularity="invalid")

    def test_recompute_selective_valid(self):
        cfg = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        self.assertEqual(cfg.recompute_granularity, "selective")

    def test_embedding_init_method_std_defaults_to_init_std(self):
        cfg = _make_config(init_method_std=0.03, embedding_init_method_std=None)
        self.assertEqual(cfg.embedding_init_method_std, 0.03)


class TestTransformerConfigMoEValidation(unittest.TestCase):
    """Test MoE related validation."""

    def test_moe_layer_freq_default_when_none(self):
        cfg = _make_config(
            n_routed_experts=8,
            first_k_dense_replace=None,
            moe_layer_freq=None,
        )
        self.assertEqual(cfg.moe_layer_freq, 1)

    def test_first_k_dense_replace_and_moe_layer_freq_both_set_raises(self):
        with self.assertRaises(ValueError):
            _make_config(
                first_k_dense_replace=2,
                moe_layer_freq=[1, 0],
            )

    def test_first_k_dense_replace_creates_pattern(self):
        cfg = _make_config(
            num_hidden_layers=6,
            first_k_dense_replace=2,
            moe_layer_freq=2,
        )
        self.assertEqual(len(cfg.moe_layer_freq), 6)
        self.assertEqual(cfg.moe_layer_freq[0], 0)
        self.assertEqual(cfg.moe_layer_freq[1], 0)

    def test_first_k_dense_replace_all_moe(self):
        cfg = _make_config(
            num_hidden_layers=5,
            first_k_dense_replace=1,
        )
        self.assertEqual(len(cfg.moe_layer_freq), 5)
        self.assertEqual(cfg.moe_layer_freq[0], 0)
        for i in range(1, 5):
            self.assertEqual(cfg.moe_layer_freq[i], 1)


class TestTransformerConfigMLARoPEFusion(unittest.TestCase):
    """Test MLA + RoPE fusion validation."""

    def test_mla_rope_fusion_valid_with_yarn(self):
        cfg = _make_config(
            multi_latent_attention=True,
            apply_rope_fusion=True,
            rope_type="yarn",
        )
        self.assertTrue(cfg.apply_rope_fusion)


class TestTransformerConfigFromConfig(unittest.TestCase):
    """Test from_config class method."""

    def test_from_config_creates_instance(self):
        mock_cfg = MagicMock()
        mock_cfg.__dict__ = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "hidden_act": "gelu",
        }
        cfg = TransformerConfig.from_config(mock_cfg)
        self.assertEqual(cfg.hidden_size, 64)
        self.assertEqual(cfg.num_attention_heads, 2)

    def test_from_config_hidden_act_string(self):
        mock_cfg = MagicMock()
        mock_cfg.__dict__ = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "hidden_act": "silu",
        }
        cfg = TransformerConfig.from_config(mock_cfg)
        self.assertTrue(callable(cfg.hidden_act))

    def test_from_config_hidden_act_gelu_pytorch_tanh(self):
        mock_cfg = MagicMock()
        mock_cfg.__dict__ = {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "hidden_act": "gelu_pytorch_tanh",
        }
        cfg = TransformerConfig.from_config(mock_cfg)
        self.assertTrue(callable(cfg.hidden_act))


class TestTransformerConfigGet(unittest.TestCase):
    """Test the get method."""

    def test_get_existing_key(self):
        cfg = _make_config(hidden_size=64)
        self.assertEqual(cfg.get("hidden_size"), 64)

    def test_get_missing_key_returns_default(self):
        cfg = _make_config()
        self.assertIsNone(cfg.get("nonexistent"))
        self.assertEqual(cfg.get("nonexistent", 42), 42)


class TestTransformerConfigHiddenAct(unittest.TestCase):
    """Test hidden_act processing in _process_attribute."""

    def test_hidden_act_callable(self):
        cfg = _make_config()
        func = lambda x: x
        cfg._process_attribute("hidden_act", func)
        self.assertEqual(cfg.hidden_act, func)

    def test_hidden_act_invalid_type_raises(self):
        cfg = _make_config()
        with self.assertRaises(TypeError):
            cfg._process_attribute("hidden_act", 123)


class TestTransformerConfigTransformRules(unittest.TestCase):
    """Test transform_rules field mapping."""

    def test_transform_rules_exist(self):
        self.assertIn("index_n_heads", TransformerConfig.transform_rules)
        self.assertEqual(
            TransformerConfig.transform_rules["index_n_heads"],
            "dsa_index_n_heads",
        )

    def test_all_dsa_fields_in_transform_rules(self):
        expected_keys = [
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "indexer_loss_coeff",
            "indexer_use_sparse_loss",
            "indexer_rotary_interleaved",
        ]
        for key in expected_keys:
            self.assertIn(key, TransformerConfig.transform_rules)


class TestTransformerConfigEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_hidden_layers(self):
        # num_hidden_layers=0 causes ZeroDivisionError in output_layer_init_method
        with self.assertRaises(ZeroDivisionError):
            _make_config(num_hidden_layers=0)

    def test_large_hidden_size(self):
        cfg = _make_config(hidden_size=4096, num_attention_heads=32)
        self.assertEqual(cfg.head_dim, 128)

    def test_gated_linear_unit_doubles_intermediate(self):
        cfg = _make_config(
            hidden_size=128,
            gated_linear_unit=True,
            intermediate_size=256,
        )
        # intermediate_size is not doubled in __post_init__, only in MLP
        self.assertEqual(cfg.intermediate_size, 256)

    def test_mtp_config_defaults(self):
        cfg = _make_config()
        self.assertEqual(cfg.num_nextn_predict_layers, 0)
        self.assertFalse(cfg.train_mtp_only)
        self.assertEqual(cfg.mtp_loss_scaling_factor, 0.3)

    def test_dsa_config_defaults(self):
        cfg = _make_config()
        self.assertIsNone(cfg.dsa_index_n_heads)
        self.assertEqual(cfg.dsa_index_head_dim, 128)
        self.assertEqual(cfg.dsa_index_topk, 2048)

    def test_mla_config_defaults(self):
        cfg = _make_config()
        self.assertEqual(cfg.q_lora_rank, 512)
        self.assertEqual(cfg.kv_lora_rank, 512)
        self.assertEqual(cfg.rope_type, "yarn")

    def test_softmax_type_default(self):
        cfg = _make_config()
        self.assertEqual(cfg.softmax_type, "vanilla")

    def test_fp8_default_none(self):
        cfg = _make_config()
        self.assertIsNone(cfg.fp8)

    def test_window_attn_defaults(self):
        cfg = _make_config()
        self.assertIsNone(cfg.sliding_window)
        self.assertIsNone(cfg.window_attn_skip_freq)


if __name__ == "__main__":
    unittest.main()
