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

import paddle

from paddleformers.fleet.models.vision.clip_vit_model import (
    CLIPViTModel,
    get_num_image_embeddings,
)


class TestCLIPViTModelInitVariants(unittest.TestCase):
    """Test CLIPViTModel initialization variants."""

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_siglip_model_init(self, mock_block, mock_log):
        """Test SigLIP model init with no class token."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="siglip",
            class_token_len=0,
            add_class_token=False,
            img_h=14,
            img_w=14,
            patch_dim=14,
        )
        # SigLIP has ln_post instead of ln_pre
        self.assertIsNotNone(model.ln_post)
        self.assertIsNone(model.ln_pre)
        self.assertFalse(model.add_class_token)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_internvit_model_init(self, mock_block, mock_log):
        """Test InternViT model init."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="internvit",
            img_h=14,
            img_w=14,
            patch_dim=14,
        )
        # InternViT has no ln_pre and no ln_post by default
        self.assertIsNone(model.ln_pre)
        self.assertIsNone(model.ln_post)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_internvit300m_model_init(self, mock_block, mock_log):
        """Test InternViT 300M model init."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="internvit300M",
            img_h=14,
            img_w=14,
            patch_dim=14,
        )
        self.assertIsNone(model.ln_pre)
        self.assertIsNone(model.ln_post)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_clip_model_init(self, mock_block, mock_log):
        """Test CLIP model init with ln_pre."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="clip",
            img_h=14,
            img_w=14,
            patch_dim=14,
        )
        # CLIP has ln_pre
        self.assertIsNotNone(model.ln_pre)
        self.assertIsNone(model.ln_post)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_model_type_attribute(self, mock_block, mock_log):
        """Test model_type is set correctly."""
        from paddleformers.fleet.transformer.enums import ModelType

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            img_h=14,
            img_w=14,
            patch_dim=14,
        )
        self.assertEqual(model.model_type, ModelType.encoder_or_decoder)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_seq_length_computation(self, mock_block, mock_log):
        """Test seq_length is computed correctly."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            img_h=28,
            img_w=28,
            patch_dim=14,
            add_class_token=True,
            class_token_len=1,
        )
        # (28/14)*(28/14) + 1 = 4 + 1 = 5
        self.assertEqual(model.seq_length, 5)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_num_patches_computation(self, mock_block, mock_log):
        """Test num_patches is computed correctly."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            img_h=336,
            img_w=336,
            patch_dim=14,
        )
        # (336/14) * (336/14) = 24 * 24 = 576
        self.assertEqual(model.num_patches, 576)

    @patch(
        "paddleformers.fleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddleformers.fleet.models.vision.clip_vit_model.TransformerBlock")
    def test_unsupported_model_subtype_value_error(self, mock_block, mock_log):
        """Test that unsupported model_subtype causes ValueError in init."""
        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        # The assertion at the beginning catches "unsupported" before ValueError
        with self.assertRaises(AssertionError):
            CLIPViTModel(
                transformer_config=mock_config,
                transformer_layer_spec=MagicMock(),
                model_subtype="unsupported_type",
                img_h=14,
                img_w=14,
                patch_dim=14,
            )


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
class TestCLIPViTModelForwardCUDA(unittest.TestCase):
    """Test CLIPViTModel forward on CUDA - model construction tests."""

    def test_forward_siglip_no_class_token(self):
        """Test SigLIP model construction without class token."""
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec(config=config)
        model = CLIPViTModel(
            transformer_config=config,
            transformer_layer_spec=spec,
            img_h=14,
            img_w=14,
            patch_dim=14,
            model_subtype="siglip",
            class_token_len=0,
            add_class_token=False,
        )
        # Verify model attributes
        self.assertIsNotNone(model.ln_post)
        self.assertFalse(model.add_class_token)
        self.assertEqual(model.seq_length, 1)

    def test_forward_with_ln_post(self):
        """Test SigLIP model construction with ln_post."""
        from paddleformers.fleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec(config=config)
        model = CLIPViTModel(
            transformer_config=config,
            transformer_layer_spec=spec,
            img_h=14,
            img_w=14,
            patch_dim=14,
            model_subtype="siglip",
            class_token_len=0,
            add_class_token=False,
        )
        # SigLIP has ln_post
        self.assertIsNotNone(model.ln_post)
        self.assertIsNone(model.ln_pre)


class TestGetNumImageEmbeddingsEdgeCases(unittest.TestCase):
    """Additional edge case tests for get_num_image_embeddings."""

    def test_radio_without_class_token(self):
        """Test RADIO model with class token disabled."""
        result = get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="radio",
            disable_vision_class_token=True,
            class_token_len=8,
            pixel_shuffle=False,
        )
        num_patches = (224 // 16) * (224 // 16)
        self.assertEqual(result, num_patches)

    def test_internvit300m_without_class_token(self):
        """Test InternViT 300M without class token."""
        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="internvit300M",
            disable_vision_class_token=True,
            class_token_len=1,
            pixel_shuffle=False,
        )
        num_patches = (336 // 14) * (336 // 14)
        self.assertEqual(result, num_patches)

    def test_tile_tags_non_qwen_medium_tiles(self):
        """Test tile tags with non-qwen tokenizer and medium tiles."""
        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
            use_tile_tags=True,
            max_num_tiles=12,
            tokenizer_type="llama3p1",
        )
        # 577 + 5 = 582 (no qwen padding for llama3p1)
        self.assertEqual(result, 582)

    def test_cradio_g_without_class_token(self):
        """Test cradio-g without class token."""
        result = get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="cradio-g",
            disable_vision_class_token=True,
            class_token_len=1,
            pixel_shuffle=False,
        )
        num_patches = (224 // 16) * (224 // 16)
        self.assertEqual(result, num_patches)


if __name__ == "__main__":
    unittest.main()
