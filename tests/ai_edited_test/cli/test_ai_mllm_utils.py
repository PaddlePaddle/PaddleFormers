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

import sys
import types
import unittest
from unittest.mock import MagicMock

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.utils.mllm_utils import (
    _ALL_MODULES,
    MLLMModelMapping,
    ModelKeys,
    MultiModelKeys,
    freeze_model_parameters,
    get_multimodel_lora_target_modules,
    get_multimodel_target_modules,
    register_multimodel_keys,
)


class TestMLLMModelMapping(unittest.TestCase):
    """Tests for MLLMModelMapping class"""

    def test_all_mappings_defined(self):
        self.assertEqual(MLLMModelMapping.qwen2_5_vl, "qwen2_5_vl")
        self.assertEqual(MLLMModelMapping.qwen3_vl, "qwen3_vl")
        self.assertEqual(MLLMModelMapping.qwen3_vl_moe, "qwen3_vl_moe")
        self.assertEqual(MLLMModelMapping.paddleocr_vl, "paddleocr_vl")
        self.assertEqual(MLLMModelMapping.ernie4_5_moe_vl, "ernie4_5_moe_vl")
        self.assertEqual(MLLMModelMapping.glm4v_moe, "glm4v_moe")


class TestModelKeys(unittest.TestCase):
    """Tests for ModelKeys dataclass"""

    def test_default_values(self):
        keys = ModelKeys(model_dtype="test_model")
        self.assertEqual(keys.model_dtype, "test_model")
        self.assertIsNone(keys.embedding)
        self.assertIsNone(keys.module_list)
        self.assertIsNone(keys.lm_head)
        self.assertIsNone(keys.q_proj)
        self.assertIsNone(keys.k_proj)
        self.assertIsNone(keys.v_proj)
        self.assertIsNone(keys.o_proj)
        self.assertIsNone(keys.mlp)

    def test_custom_values(self):
        keys = ModelKeys(
            model_dtype="test",
            embedding="model.embed_tokens",
            lm_head="lm_head",
        )
        self.assertEqual(keys.embedding, "model.embed_tokens")
        self.assertEqual(keys.lm_head, "lm_head")


class TestMultiModelKeys(unittest.TestCase):
    """Tests for MultiModelKeys dataclass"""

    def test_default_values(self):
        keys = MultiModelKeys(model_dtype="test")
        self.assertEqual(keys.llm, [])
        self.assertEqual(keys.aligner, [])
        self.assertEqual(keys.vision, [])

    def test_string_converted_to_list(self):
        keys = MultiModelKeys(model_dtype="test", llm="model.language_model")
        self.assertEqual(keys.llm, ["model.language_model"])

    def test_none_converted_to_empty_list(self):
        keys = MultiModelKeys(model_dtype="test", llm=None)
        self.assertEqual(keys.llm, [])

    def test_list_preserved(self):
        keys = MultiModelKeys(
            model_dtype="test",
            llm=["model.language_model", "lm_head"],
            vision=["model.visual"],
        )
        self.assertEqual(keys.llm, ["model.language_model", "lm_head"])
        self.assertEqual(keys.vision, ["model.visual"])

    def test_post_init_converts_strings(self):
        keys = MultiModelKeys(
            model_dtype="test",
            aligner="model.visual.merger",
        )
        self.assertEqual(keys.aligner, ["model.visual.merger"])


class TestRegisterAndGetMultimodelKeys(unittest.TestCase):
    """Tests for register_multimodel_keys and get_multimodel_target_modules"""

    def test_get_existing_model(self):
        # qwen2_5_vl should already be registered
        result = get_multimodel_target_modules("qwen2_5_vl")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, MultiModelKeys)

    def test_get_nonexistent_model(self):
        result = get_multimodel_target_modules("nonexistent_model_xyz")
        self.assertIsNone(result)

    def test_get_none_model_type(self):
        result = get_multimodel_target_modules(None)
        self.assertIsNone(result)

    def test_get_empty_model_type(self):
        result = get_multimodel_target_modules("")
        self.assertIsNone(result)

    def test_register_new_model(self):
        test_key = "test_unique_model_for_testing_123"
        keys = MultiModelKeys(
            model_dtype=test_key,
            llm=["model"],
            vision=["visual"],
        )
        register_multimodel_keys(keys, exist_ok=True)
        result = get_multimodel_target_modules(test_key)
        self.assertIsNotNone(result)
        self.assertEqual(result.llm, ["model"])

    def test_register_duplicate_raises_error(self):
        keys = MultiModelKeys(model_dtype="qwen2_5_vl", llm=["model"])
        with self.assertRaises(ValueError) as ctx:
            register_multimodel_keys(keys, exist_ok=False)
        self.assertIn("already been registered", str(ctx.exception))


class TestGetMultimodelLoraTargetModules(unittest.TestCase):
    """Tests for get_multimodel_lora_target_modules"""

    def test_unregistered_model_returns_original(self):
        model = MagicMock()
        model.config.model_type = "unknown_model_xyz"

        target_modules = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
        result = get_multimodel_lora_target_modules(model, target_modules, "freeze_vision")
        self.assertEqual(result, target_modules)

    def test_registered_model_filters_modules(self):
        model = MagicMock()
        model.config.model_type = "qwen2_5_vl"

        target_modules = [
            "model.visual.patch_embed.weight",
            "model.language_model.layers.0.self_attn.weight",
            "lm_head.weight",
        ]
        result = get_multimodel_lora_target_modules(model, target_modules, "freeze_vision")
        # "model.visual" matches vision, so "model.visual.patch_embed.weight" should be removed
        self.assertNotIn("model.visual.patch_embed.weight", result)
        self.assertIn("model.language_model.layers.0.self_attn.weight", result)
        self.assertIn("lm_head.weight", result)

    def test_freeze_aligner_filters_aligner_modules(self):
        model = MagicMock()
        model.config.model_type = "qwen2_5_vl"

        target_modules = [
            "model.visual.merger.weight",
            "model.language_model.layers.0.self_attn.weight",
        ]
        result = get_multimodel_lora_target_modules(model, target_modules, "freeze_aligner")
        # "model.visual.merger" matches aligner prefix
        self.assertNotIn("model.visual.merger.weight", result)
        self.assertIn("model.language_model.layers.0.self_attn.weight", result)

    def test_no_freeze_config_returns_all(self):
        model = MagicMock()
        model.config.model_type = "qwen2_5_vl"

        target_modules = [
            "model.visual.merger.weight",
            "model.language_model.layers.0.self_attn.weight",
        ]
        result = get_multimodel_lora_target_modules(model, target_modules, "")
        self.assertEqual(len(result), 2)

    def test_freeze_llm_filters_llm_modules(self):
        model = MagicMock()
        model.config.model_type = "qwen2_5_vl"

        target_modules = [
            "model.language_model.layers.0.self_attn.weight",
            "model.visual.merger.weight",
            "lm_head.weight",
        ]
        result = get_multimodel_lora_target_modules(model, target_modules, "freeze_llm")
        # Both "model.language_model" and "lm_head" are llm modules
        self.assertNotIn("model.language_model.layers.0.self_attn.weight", result)
        self.assertNotIn("lm_head.weight", result)
        self.assertIn("model.visual.merger.weight", result)


class TestFreezeModelParameters(unittest.TestCase):
    """Tests for freeze_model_parameters"""

    def test_model_without_config(self):
        model = MagicMock(spec=[])
        # Model without config attribute should return early
        freeze_model_parameters(model, "freeze_vision")
        # Should not raise any error

    def test_model_without_model_type(self):
        model = MagicMock()
        model.config = MagicMock(spec=["some_attr"])
        # Model config without model_type should return early
        freeze_model_parameters(model, "freeze_vision")

    def test_unregistered_model_type(self):
        model = MagicMock()
        model.config.model_type = "unknown_model_xyz"
        model.config.__class__ = type("Config", (), {"model_type": "unknown_model_xyz"})
        model.named_parameters = MagicMock(return_value=[])
        freeze_model_parameters(model, "freeze_vision")
        # Should not raise any error


class TestAllModules(unittest.TestCase):
    """Tests for _ALL_MODULES constant"""

    def test_contains_expected_modules(self):
        self.assertEqual(_ALL_MODULES, ["vision", "aligner", "llm"])


if __name__ == "__main__":
    unittest.main()
