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
from dataclasses import fields

from paddleformers.cli.train.ernie_pretrain.model_config import ModelConfig


class TestModelConfig(unittest.TestCase):
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test that ModelConfig has expected default values."""
        config = ModelConfig()
        self.assertIsNone(config.model_name_or_path)
        self.assertIsNone(config.tokenizer_name_or_path)
        self.assertFalse(config.use_fast_layer_norm)
        self.assertAlmostEqual(config.hidden_dropout_prob, 0.1)
        self.assertAlmostEqual(config.attention_probs_dropout_prob, 0.1)
        self.assertTrue(config.continue_training)
        self.assertEqual(config.stage, "SFT")
        self.assertEqual(config.fine_tuning, "LoRA")
        self.assertFalse(config.lora)
        self.assertIsNone(config.lora_path)
        self.assertEqual(config.lora_rank, 8)
        self.assertFalse(config.use_quick_lora)
        self.assertFalse(config.rslora)
        self.assertAlmostEqual(config.lora_plus_scale, 1.0)
        self.assertFalse(config.pissa)
        self.assertFalse(config.lora_use_mixer)
        self.assertFalse(config.use_mora)
        self.assertFalse(config.lorapro)
        self.assertEqual(config.lorapro_x_mode, "zero")
        self.assertAlmostEqual(config.lorapro_scaling_factor, 2.0)
        self.assertFalse(config.vera)
        self.assertEqual(config.vera_rank, 8)
        self.assertFalse(config.lokr)
        self.assertIsNone(config.lokr_path)
        self.assertEqual(config.lokr_dim, 8)
        self.assertFalse(config.prefix_tuning)
        self.assertIsNone(config.prefix_path)
        self.assertEqual(config.num_prefix_tokens, 128)
        self.assertFalse(config.reft)
        self.assertFalse(config.save_to_aistudio)
        self.assertIsNone(config.aistudio_repo_id)
        self.assertTrue(config.aistudio_repo_private)
        self.assertEqual(config.aistudio_repo_license, "Apache License 2.0")
        self.assertIsNone(config.aistudio_token)
        self.assertFalse(config.neftune)
        self.assertAlmostEqual(config.neftune_noise_alpha, 5.0)
        self.assertFalse(config.flash_mask)
        self.assertEqual(config._attn_implementation, "flashmask")
        self.assertFalse(config.use_long_sequence_strategies)
        self.assertAlmostEqual(config.rope_scaling_factor, 1.0)
        self.assertIsNone(config.strategy_type)
        self.assertIsNone(config.strategy_name)
        self.assertIsNone(config.weight_quantize_algo)
        self.assertEqual(config.qlora_weight_blocksize, 64)
        self.assertFalse(config.qlora_weight_double_quant)
        self.assertEqual(config.qlora_weight_double_quant_block_size, 256)
        self.assertFalse(config.apply_hadamard)
        self.assertEqual(config.hadamard_block_size, 32)
        self.assertFalse(config.quant_input_grad)
        self.assertFalse(config.quant_weight_grad)
        self.assertEqual(config.apply_online_actscale_step, 200)
        self.assertAlmostEqual(config.actscale_moving_rate, 0.01)
        self.assertEqual(config.fp8_format_type, "hybrid")
        self.assertTrue(config.use_attn_mask_startend_row_indices)
        self.assertEqual(config.pp_seg_method, "layer:DecoderLayer|EmptyLayer")

    def test_custom_values(self):
        """Test that ModelConfig accepts custom values."""
        config = ModelConfig(
            model_name_or_path="test-model",
            hidden_dropout_prob=0.2,
            stage="DPO",
            lora=True,
            lora_rank=16,
        )
        self.assertEqual(config.model_name_or_path, "test-model")
        self.assertAlmostEqual(config.hidden_dropout_prob, 0.2)
        self.assertEqual(config.stage, "DPO")
        self.assertTrue(config.lora)
        self.assertEqual(config.lora_rank, 16)

    def test_is_dataclass(self):
        """Test that ModelConfig is a dataclass."""
        from dataclasses import is_dataclass

        self.assertTrue(is_dataclass(ModelConfig))

    def test_all_fields_have_defaults(self):
        """Test that all fields have default values so constructor works without args."""
        ModelConfig()
        for f in fields(ModelConfig):
            self.assertTrue(
                f.default is not None or f.default_factory is not None or f.default is False or f.default == 0,
                f"Field {f.name} may not have a default",
            )

    def test_field_count(self):
        """Test the expected number of fields in ModelConfig."""
        all_fields = fields(ModelConfig)
        # ModelConfig has many fields, just check a reasonable minimum
        self.assertGreaterEqual(len(all_fields), 40)

    def test_lora_params(self):
        """Test LoRA-related parameters."""
        config = ModelConfig(lora=True, lora_rank=32, use_quick_lora=True, rslora=True)
        self.assertTrue(config.lora)
        self.assertEqual(config.lora_rank, 32)
        self.assertTrue(config.use_quick_lora)
        self.assertTrue(config.rslora)

    def test_quantization_params(self):
        """Test quantization-related parameters."""
        config = ModelConfig(
            weight_quantize_algo="nf4",
            qlora_weight_blocksize=128,
            qlora_weight_double_quant=True,
        )
        self.assertEqual(config.weight_quantize_algo, "nf4")
        self.assertEqual(config.qlora_weight_blocksize, 128)
        self.assertTrue(config.qlora_weight_double_quant)

    def test_fp8_params(self):
        """Test FP8-related parameters."""
        config = ModelConfig(fp8_format_type="e4m3", apply_hadamard=True)
        self.assertEqual(config.fp8_format_type, "e4m3")
        self.assertTrue(config.apply_hadamard)


if __name__ == "__main__":
    unittest.main()
