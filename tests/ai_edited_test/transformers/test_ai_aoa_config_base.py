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

import unittest


class _MockConfig:
    """Simple mock config object for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestMoEAOAConfigParams(unittest.TestCase):
    """Tests for MoEAOAConfigParams dataclass."""

    def test_default_values(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigParams

        params = MoEAOAConfigParams()
        self.assertEqual(params.num_hidden_layers, 0)
        self.assertEqual(params.num_attention_heads, 0)
        self.assertEqual(params.num_key_value_heads, 0)
        self.assertEqual(params.num_experts, 0)
        self.assertFalse(params.using_sonic_moe)
        self.assertFalse(params.moe_expert_fusion)
        self.assertFalse(params.fp8)
        self.assertFalse(params.fd_fallback)
        self.assertFalse(params.tie_word_embeddings)
        self.assertEqual(params.num_head_empty_layers, 0)
        self.assertEqual(params.first_k_dense_replace, 0)
        self.assertEqual(params.num_nextn_predict_layers, 0)
        self.assertFalse(params.attention_bias)
        self.assertFalse(params.multi_latent_attention)
        self.assertFalse(params.use_qk_norm)
        self.assertTrue(params.has_shared_experts)
        self.assertEqual(params.model_prefix, "model.")
        self.assertEqual(params.extra_statements, [])

    def test_custom_values(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigParams

        params = MoEAOAConfigParams(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=8,
            using_sonic_moe=True,
            moe_expert_fusion=True,
            fp8=True,
            tie_word_embeddings=True,
            num_head_empty_layers=2,
            first_k_dense_replace=4,
            attention_bias=True,
            multi_latent_attention=True,
            use_qk_norm=True,
            has_shared_experts=False,
        )
        self.assertEqual(params.num_hidden_layers, 32)
        self.assertEqual(params.num_attention_heads, 32)
        self.assertEqual(params.num_key_value_heads, 8)
        self.assertEqual(params.num_experts, 8)
        self.assertTrue(params.using_sonic_moe)
        self.assertTrue(params.moe_expert_fusion)
        self.assertTrue(params.fp8)
        self.assertTrue(params.tie_word_embeddings)
        self.assertEqual(params.num_head_empty_layers, 2)
        self.assertEqual(params.first_k_dense_replace, 4)
        self.assertTrue(params.attention_bias)
        self.assertTrue(params.multi_latent_attention)
        self.assertTrue(params.use_qk_norm)
        self.assertFalse(params.has_shared_experts)

    def test_extra_statements(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigParams

        params = MoEAOAConfigParams(extra_statements=["custom.stmt1", "custom.stmt2"])
        self.assertEqual(len(params.extra_statements), 2)


class TestMoEAOAConfigGeneratorBasicWeights(unittest.TestCase):
    """Tests for basic weight statement generation."""

    def test_get_basic_weight_statements_no_tie(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(tie_word_embeddings=False)
        stmts = MoEAOAConfigGenerator._get_basic_weight_statements(params)
        self.assertEqual(len(stmts), 3)
        self.assertIn("model.norm.weight -> model.norm.weight", stmts[0])
        self.assertIn("model.embed_tokens.weight -> model.embedding.embed_tokens.weight", stmts[1])
        self.assertIn("lm_head.weight -> model.lm_head.weight", stmts[2])

    def test_get_basic_weight_statements_tied(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(tie_word_embeddings=True)
        stmts = MoEAOAConfigGenerator._get_basic_weight_statements(params)
        self.assertEqual(len(stmts), 3)
        self.assertIn("model.embed_tokens.weight -> model.lm_head.weight", stmts[2])

    def test_get_basic_weight_statements_custom_prefix(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(model_prefix="")
        stmts = MoEAOAConfigGenerator._get_basic_weight_statements(params)
        self.assertIn("norm.weight -> norm.weight", stmts[0])


class TestMoEAOAConfigGeneratorDenseLayers(unittest.TestCase):
    """Tests for dense layer statement generation."""

    def test_no_dense_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(first_k_dense_replace=0)
        stmts = MoEAOAConfigGenerator._get_dense_layer_statements(params)
        self.assertEqual(stmts, [])

    def test_with_dense_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            first_k_dense_replace=2,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        stmts = MoEAOAConfigGenerator._get_dense_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        self.assertIn("layers.1.input_layernorm.weight", stmts[0])
        self.assertIn("fused_ffn", "".join(stmts))

    def test_dense_layer_with_offset(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            first_k_dense_replace=2,
            num_head_empty_layers=3,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        stmts = MoEAOAConfigGenerator._get_dense_layer_statements(params)
        self.assertIn("model.layers.0", "".join(stmts))
        self.assertIn("layers.3.", "".join(stmts))
        self.assertIn("layers.4.", "".join(stmts))


class TestMoEAOAConfigGeneratorAttention(unittest.TestCase):
    """Tests for attention statement generation."""

    def test_standard_attention(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_bias=False,
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(params, 0, "model.layers.0", "model.layers.0")
        self.assertEqual(len(stmts), 1)
        self.assertIn("fused_qkv", stmts[0])
        self.assertIn("num_heads=32", stmts[0])

    def test_attention_with_bias(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_attention_heads=32,
            num_key_value_heads=8,
            attention_bias=True,
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(params, 0, "model.layers.0", "model.layers.0")
        self.assertEqual(len(stmts), 2)
        self.assertIn("q_proj.bias", stmts[1])
        self.assertIn("axis=0", stmts[1])

    def test_mla_attention(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            multi_latent_attention=True,
            use_qk_norm=False,
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(params, 0, "model.layers.0", "model.layers.0")
        self.assertEqual(len(stmts), 4)
        self.assertIn("kv_a_proj_with_mqa", stmts[0])
        self.assertIn("q_b_proj", stmts[3])

    def test_mla_attention_with_qk_norm(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            multi_latent_attention=True,
            use_qk_norm=True,
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(params, 0, "model.layers.0", "model.layers.0")
        self.assertEqual(len(stmts), 6)
        self.assertIn("q_a_layernorm", "".join(stmts))


class TestMoEAOAConfigGeneratorMoELayers(unittest.TestCase):
    """Tests for MoE layer statement generation."""

    def test_moe_layer_basic(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=8,
            has_shared_experts=True,
        )
        stmts = MoEAOAConfigGenerator._get_moe_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("gate.weight", combined)
        self.assertIn("shared_experts", combined)

    def test_moe_layer_no_shared_experts(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=8,
            has_shared_experts=False,
        )
        stmts = MoEAOAConfigGenerator._get_moe_expert_statements(params, "model.layers.0", "model.layers.0")
        combined = "\n".join(stmts)
        self.assertNotIn("shared_experts", combined)

    def test_routed_expert_normal(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(using_sonic_moe=False)
        stmts = MoEAOAConfigGenerator._get_routed_expert_statements(params, "model.layers.0", "model.layers.0")
        combined = "\n".join(stmts)
        self.assertIn("$EXPERT_ID", combined)
        self.assertIn("^T", combined)

    def test_routed_expert_sonic_moe(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(using_sonic_moe=True)
        stmts = MoEAOAConfigGenerator._get_routed_expert_statements(params, "model.layers.0", "model.layers.0")
        combined = "\n".join(stmts)
        self.assertIn("axis=0", combined)
        self.assertNotIn("^T", combined)


class TestMoEAOAConfigGeneratorMTP(unittest.TestCase):
    """Tests for MTP layer statement generation."""

    def test_no_mtp_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(num_nextn_predict_layers=0)
        stmts = MoEAOAConfigGenerator._get_mtp_layer_statements(params)
        self.assertEqual(stmts, [])

    def test_with_mtp_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=4,
            num_nextn_predict_layers=2,
        )
        stmts = MoEAOAConfigGenerator._get_mtp_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("eh_proj", combined)
        self.assertIn("enorm", combined)
        self.assertIn("hnorm", combined)


class TestMoEAOAConfigGeneratorGroupedGEMM(unittest.TestCase):
    """Tests for grouped GEMM statement generation."""

    def test_no_grouped_gemm_no_fd(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_experts=4,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=False,
            fd_fallback=False,
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        self.assertEqual(stmts, [])

    def test_grouped_gemm_enabled(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_experts=4,
            moe_expert_fusion=True,
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("grouped_gemm_experts", combined)

    def test_fd_fallback(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_experts=4,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=False,
            fd_fallback=True,
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("axis=0", combined)


class TestMoEAOAConfigGeneratorExtractParams(unittest.TestCase):
    """Tests for _extract_params."""

    def test_extract_with_n_routed_experts(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
            n_routed_experts=8,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 8)

    def test_extract_with_num_experts(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=16,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 16)

    def test_extract_no_experts(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 0)

    def test_extract_with_fd_fallback(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
            fd_fallback=True,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertTrue(params.fd_fallback)


class TestMoEAOAConfigGeneratorBuildAoaConfig(unittest.TestCase):
    """Tests for _build_aoa_config."""

    def test_build_basic(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=4,
            has_shared_experts=True,
        )
        result = MoEAOAConfigGenerator._build_aoa_config(params)
        self.assertIn("aoa_statements", result)
        self.assertIsInstance(result["aoa_statements"], list)
        self.assertTrue(len(result["aoa_statements"]) > 0)

    def test_build_with_extra_statements(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            extra_statements=["stmt1", "stmt2"],
        )
        result = MoEAOAConfigGenerator._build_aoa_config(params)
        self.assertIn("stmt1", result["aoa_statements"])
        self.assertIn("stmt2", result["aoa_statements"])


class TestMoEAOAConfigGeneratorGenAoaConfig(unittest.TestCase):
    """Tests for gen_aoa_config main entry point."""

    def test_gen_aoa_config(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=4,
        )
        result = MoEAOAConfigGenerator.gen_aoa_config(config)
        self.assertIn("aoa_statements", result)
        self.assertTrue(len(result["aoa_statements"]) > 0)


class TestMoEAOAConfigGeneratorInverse(unittest.TestCase):
    """Tests for inverse AOA config generation."""

    def test_inv_basic_weights_no_tie(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(tie_word_embeddings=False)
        stmts = MoEAOAConfigGenerator._get_inv_basic_weight_statements(params)
        self.assertTrue(len(stmts) > 0)
        self.assertIn("lm_head.weight -> lm_head.weight", "".join(stmts))

    def test_inv_basic_weights_tied(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(tie_word_embeddings=True)
        stmts = MoEAOAConfigGenerator._get_inv_basic_weight_statements(params)
        self.assertIn("-> _", "".join(stmts))

    def test_inv_build_aoa_config(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=4,
        )
        result = MoEAOAConfigGenerator._build_inv_aoa_config(params)
        self.assertIn("aoa_statements", result)
        self.assertTrue(len(result["aoa_statements"]) > 0)

    def test_gen_inv_aoa_config(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        config = _MockConfig(
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=4,
        )
        result = MoEAOAConfigGenerator.gen_inv_aoa_config(config)
        self.assertIn("aoa_statements", result)
        self.assertTrue(len(result["aoa_statements"]) > 0)

    def test_inv_dense_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(first_k_dense_replace=2, num_head_empty_layers=1)
        stmts = MoEAOAConfigGenerator._get_inv_dense_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("fused_ffn", combined)

    def test_inv_mtp_layers(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(num_hidden_layers=4, num_nextn_predict_layers=2, num_head_empty_layers=0)
        stmts = MoEAOAConfigGenerator._get_inv_mtp_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("eh_proj", combined)

    def test_inv_moe_layer_statements(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_hidden_layers=4,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_experts=4,
            has_shared_experts=True,
        )
        stmts = MoEAOAConfigGenerator._get_inv_moe_layer_statements(params)
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("gate.weight", combined)

    def test_inv_grouped_gemm_layer(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_experts=4,
            moe_expert_fusion=True,
        )
        stmts = MoEAOAConfigGenerator._get_inv_grouped_gemm_layer_statements(params, "model.layers.0")
        self.assertTrue(len(stmts) > 0)
        combined = "\n".join(stmts)
        self.assertIn("weight1", combined)
        self.assertIn("weight2", combined)

    def test_inv_grouped_gemm_none(self):
        from paddleformers.transformers.aoa_config_base import (
            MoEAOAConfigGenerator,
            MoEAOAConfigParams,
        )

        params = MoEAOAConfigParams(
            num_experts=4,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=False,
            fd_fallback=False,
        )
        stmts = MoEAOAConfigGenerator._get_inv_grouped_gemm_layer_statements(params, "model.layers.0")
        self.assertEqual(stmts, [])


class TestMoEAOAConfigGeneratorModelPrefix(unittest.TestCase):
    """Tests for _get_model_prefix."""

    def test_default_prefix(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        prefix = MoEAOAConfigGenerator._get_model_prefix(_MockConfig())
        self.assertEqual(prefix, "model.")

    def test_base_model_class(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        class TestGen(MoEAOAConfigGenerator):
            pass

        TestGen.base_model_class = TestGen
        prefix = TestGen._get_model_prefix(_MockConfig())
        self.assertEqual(prefix, "")


class TestMoEAOAConfigGeneratorHasSharedExperts(unittest.TestCase):
    """Tests for _has_shared_experts."""

    def test_default_has_shared(self):
        from paddleformers.transformers.aoa_config_base import MoEAOAConfigGenerator

        self.assertTrue(MoEAOAConfigGenerator._has_shared_experts(_MockConfig()))


if __name__ == "__main__":
    unittest.main()
