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

import importlib.util
import os
import sys
import types
import unittest

# Direct import to avoid __init__.py triggering workflow.py which requires AutoTokenizer
_MODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "paddleformers", "cli", "train", "deepseek_v3_pretrain"
)
_MODULE_DIR = os.path.abspath(_MODULE_DIR)


def _load_module(name, path):
    """Load a module directly from file path without going through __init__.py."""
    full_name = f"paddleformers.cli.train.deepseek_v3_pretrain.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-populate the package in sys.modules to prevent __init__.py from loading
_pkg_name = "paddleformers.cli.train.deepseek_v3_pretrain"
if _pkg_name not in sys.modules:
    _pkg_mod = types.ModuleType(_pkg_name)
    _pkg_mod.__path__ = [_MODULE_DIR]
    _pkg_mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _pkg_mod

# Load the configuration module directly
_config_mod = _load_module("configuration", os.path.join(_MODULE_DIR, "configuration.py"))
DeepseekV2FastConfig = _config_mod.DeepseekV2FastConfig


class TestDeepseekV2FastConfig(unittest.TestCase):
    """Test DeepseekV2FastConfig class."""

    def setUp(self):
        self.config = DeepseekV2FastConfig()

    def test_default_values(self):
        """Test that default values are correctly set."""
        self.assertEqual(self.config.vocab_size, 102400)
        self.assertEqual(self.config.hidden_size, 4096)
        self.assertEqual(self.config.intermediate_size, 11008)
        self.assertEqual(self.config.moe_intermediate_size, 1407)
        self.assertEqual(self.config.num_hidden_layers, 30)
        self.assertEqual(self.config.num_nextn_predict_layers, 0)
        self.assertEqual(self.config.num_nextn_predict_lambda, 1.0)
        self.assertEqual(self.config.num_attention_heads, 32)
        self.assertEqual(self.config.num_key_value_heads, 32)
        self.assertIsNone(self.config.n_shared_experts)
        self.assertIsNone(self.config.n_routed_experts)
        self.assertEqual(self.config.ep_size, 1)
        self.assertEqual(self.config.routed_scaling_factor, 1.0)
        self.assertEqual(self.config.kv_lora_rank, 512)
        self.assertEqual(self.config.q_lora_rank, 1536)
        self.assertEqual(self.config.qk_rope_head_dim, 64)
        self.assertEqual(self.config.v_head_dim, 128)
        self.assertEqual(self.config.qk_nope_head_dim, 128)
        self.assertEqual(self.config.topk_method, "gready")
        self.assertIsNone(self.config.n_group)
        self.assertIsNone(self.config.topk_group)
        self.assertIsNone(self.config.num_experts_per_tok)
        self.assertEqual(self.config.moe_layer_freq, 1)
        self.assertEqual(self.config.first_k_dense_replace, 0)
        self.assertFalse(self.config.norm_topk_prob)
        self.assertEqual(self.config.scoring_func, "softmax")
        self.assertTrue(self.config.seq_aux)
        self.assertEqual(self.config.hidden_act, "silu")
        self.assertEqual(self.config.max_position_embeddings, 2048)
        self.assertEqual(self.config.seq_length, 32768)
        self.assertEqual(self.config.initializer_range, 0.02)
        self.assertEqual(self.config.rms_norm_eps, 1e-6)
        self.assertTrue(self.config.use_cache)
        self.assertIsNone(self.config.pad_token_id)
        self.assertEqual(self.config.bos_token_id, 100000)
        self.assertEqual(self.config.eos_token_id, 100001)
        self.assertEqual(self.config.pretraining_tp, 1)
        self.assertFalse(self.config.tie_word_embeddings)
        self.assertEqual(self.config.rope_theta, 10000.0)
        self.assertIsNone(self.config.rope_scaling)
        self.assertFalse(self.config.attention_bias)
        self.assertEqual(self.config.attention_dropout, 0.0)
        self.assertFalse(self.config.speculate_model_type)
        self.assertFalse(self.config.using_flex_token)
        self.assertFalse(self.config.use_dualpipev)
        self.assertFalse(self.config.send_mtp_embed)
        self.assertFalse(self.config.using_post_norm_recompute)
        self.assertFalse(self.config.stepped_recompute_fwd_gate_up)
        self.assertEqual(self.config.recompute_fwd_gate_up, 0)
        self.assertEqual(self.config.recompute_fa3, 0)
        self.assertFalse(self.config.is_split_group_gemm)
        self.assertFalse(self.config.fakse_gate_restrict_balance)
        self.assertEqual(self.config.adaptive_remained_O1_recompute_ratio, 0)
        self.assertTrue(self.config.offline_quant_expert_weight)
        self.assertEqual(self.config.mlp_bwd_subbatch_rows, 0)
        self.assertEqual(self.config.mlp_fwd_subbatch_rows, 0)
        self.assertEqual(self.config.output_subbatch_rows, 0)
        self.assertTrue(self.config.dsv3_use_fp8_gemm)
        self.assertTrue(self.config.dsv3_use_atten_recompute)
        self.assertFalse(self.config.use_ds_gemm)
        self.assertTrue(self.config.dsv3_use_fp8_dispatch)
        self.assertEqual(self.config.fa_version, 3)
        self.assertIsNone(self.config.max_sequence_length)

    def test_model_type(self):
        """Test that model_type is set correctly."""
        self.assertEqual(self.config.model_type, "deepseek_v3")

    def test_keys_to_ignore_at_inference(self):
        """Test keys_to_ignore_at_inference attribute."""
        self.assertEqual(self.config.keys_to_ignore_at_inference, ["past_key_values"])

    def test_custom_initialization(self):
        """Test config with custom parameter values."""
        config = DeepseekV2FastConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5504,
            moe_intermediate_size=1407,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=4,
            n_shared_experts=2,
            n_routed_experts=64,
            ep_size=4,
            routed_scaling_factor=2.5,
            topk_method="group_limited_greedy",
            n_group=8,
            topk_group=4,
            num_experts_per_tok=6,
            norm_topk_prob=True,
            scoring_func="sigmoid",
            seq_aux=False,
        )
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.intermediate_size, 5504)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(config.n_shared_experts, 2)
        self.assertEqual(config.n_routed_experts, 64)
        self.assertEqual(config.ep_size, 4)
        self.assertEqual(config.routed_scaling_factor, 2.5)
        self.assertEqual(config.topk_method, "group_limited_greedy")
        self.assertEqual(config.n_group, 8)
        self.assertEqual(config.topk_group, 4)
        self.assertEqual(config.num_experts_per_tok, 6)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.scoring_func, "sigmoid")
        self.assertFalse(config.seq_aux)

    def test_num_key_value_heads_backward_compatibility(self):
        """Test that num_key_value_heads defaults to num_attention_heads when None."""
        config = DeepseekV2FastConfig(num_attention_heads=16, num_key_value_heads=None)
        self.assertEqual(config.num_key_value_heads, 16)

    def test_use_fp8_default(self):
        """Test that use_fp8 is False by default."""
        self.assertFalse(self.config.use_fp8)

    def test_flex_token_config(self):
        """Test flex token related configuration."""
        config = DeepseekV2FastConfig(using_flex_token=True)
        self.assertTrue(config.using_flex_token)

    def test_dualpipev_config(self):
        """Test dualpipev configuration."""
        config = DeepseekV2FastConfig(use_dualpipev=True)
        self.assertTrue(config.use_dualpipev)

    def test_moe_layer_freq_config(self):
        """Test moe_layer_freq with different values."""
        config = DeepseekV2FastConfig(moe_layer_freq=3)
        self.assertEqual(config.moe_layer_freq, 3)

    def test_first_k_dense_replace(self):
        """Test first_k_dense_replace configuration."""
        config = DeepseekV2FastConfig(first_k_dense_replace=3)
        self.assertEqual(config.first_k_dense_replace, 3)

    def test_max_sequence_length(self):
        """Test max_sequence_length configuration."""
        config = DeepseekV2FastConfig(max_sequence_length=8192)
        self.assertEqual(config.max_sequence_length, 8192)

    @unittest.skip("DeepseekV2FastConfig loaded via importlib may have different PretrainedConfig identity")
    def test_inheritance(self):
        """Test that DeepseekV2FastConfig inherits from PretrainedConfig."""
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        self.assertIsInstance(self.config, PretrainedConfig)

    def test_nextn_predict_config(self):
        """Test nextn predict layers configuration."""
        config = DeepseekV2FastConfig(
            num_nextn_predict_layers=3,
            num_nextn_predict_lambda=0.5,
        )
        self.assertEqual(config.num_nextn_predict_layers, 3)
        self.assertEqual(config.num_nextn_predict_lambda, 0.5)

    def test_subbatch_rows_config(self):
        """Test subbatch rows configuration."""
        config = DeepseekV2FastConfig(
            mlp_bwd_subbatch_rows=128,
            mlp_fwd_subbatch_rows=256,
            output_subbatch_rows=64,
        )
        self.assertEqual(config.mlp_bwd_subbatch_rows, 128)
        self.assertEqual(config.mlp_fwd_subbatch_rows, 256)
        self.assertEqual(config.output_subbatch_rows, 64)

    def test_recompute_config(self):
        """Test recompute-related configuration."""
        config = DeepseekV2FastConfig(
            recompute_fwd_gate_up=1,
            recompute_fa3=1,
            using_post_norm_recompute=True,
            stepped_recompute_fwd_gate_up=True,
        )
        self.assertEqual(config.recompute_fwd_gate_up, 1)
        self.assertEqual(config.recompute_fa3, 1)
        self.assertTrue(config.using_post_norm_recompute)
        self.assertTrue(config.stepped_recompute_fwd_gate_up)


if __name__ == "__main__":
    unittest.main()
