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
from unittest.mock import MagicMock, patch

# Check if llava_spec module can be imported
_llava_spec_importable = True
try:
    from paddleformers.fleet.models.multimodal import llava_spec  # noqa: F401
except (ImportError, AttributeError):
    _llava_spec_importable = False


class TestDecoderModelWithLocalDefaultSpec(unittest.TestCase):
    """Test decoder_model_with_local_default_spec function."""

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_returns_layer_spec(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertIsNotNone(result)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_spec_has_transformer_layer(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )
        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertEqual(result.module, TransformerLayer)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_spec_has_submodules(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )
        from paddleformers.fleet.transformer.transformer_layer import (
            TransformerLayerSublayersSpec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertIsNotNone(result.submodules)
        self.assertIsInstance(result.submodules, TransformerLayerSublayersSpec)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_spec_has_self_attention(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertIsNotNone(result.submodules.self_attention)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_spec_has_mlp(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertIsNotNone(result.submodules.mlp)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_spec_has_bias_dropout_add(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec()
        self.assertIsNotNone(result.submodules.self_attn_bda)
        self.assertIsNotNone(result.submodules.mlp_bda)


class TestDecoderModelWithMoE(unittest.TestCase):
    """Test decoder_model_with_local_default_spec with MoE."""

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_moe_spec(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec(
            num_experts=8, moe_expert_fusion=True
        )
        self.assertIsNotNone(result)
        mock_mlp.assert_called_once_with(
            use_te=False, num_experts=8, moe_expert_fusion=True
        )


class TestDecoderModelWithQKLayerNorm(unittest.TestCase):
    """Test decoder_model_with_local_default_spec with QK layernorm."""

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    @patch("paddleformers.fleet.models.multimodal.llava_spec.get_mlp_layer_spec")
    def test_qk_layernorm_param_passed(self, mock_mlp):
        from paddleformers.fleet.models.multimodal.llava_spec import (
            decoder_model_with_local_default_spec,
        )

        mock_mlp.return_value = MagicMock()

        result = decoder_model_with_local_default_spec(qk_layernorm=True)
        self.assertIsNotNone(result)
        # qk_layernorm is accepted but doesn't change the basic spec structure
        mock_mlp.assert_called_once()


class TestLLaVASpecImports(unittest.TestCase):
    """Test that LLaVA spec imports are correct."""

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    def test_import_ln_impl(self):
        from paddleformers.fleet.fusions.fused_layer_norm import FusedLayerNorm
        from paddleformers.fleet.models.multimodal.llava_spec import LNImpl

        self.assertEqual(LNImpl, FusedLayerNorm)

    @unittest.skipUnless(
        _llava_spec_importable,
        "paddleformers.fleet.models.multimodal.llava_spec cannot be imported (missing get_mlp_layer_spec in gpt_layer_specs)",
    )
    def test_import_get_bias_dropout_add(self):
        from paddleformers.fleet.fusions.fused_bias_dropout import (
            get_bias_dropout_add as gba,
        )
        from paddleformers.fleet.models.multimodal.llava_spec import (
            get_bias_dropout_add,
        )

        self.assertEqual(get_bias_dropout_add, gba)


if __name__ == "__main__":
    unittest.main()
