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
from dataclasses import fields

# Mock the missing transformers module to avoid import errors
if "transformers.tokenization_utils_tokenizers" not in sys.modules:
    _mock_mod = types.ModuleType("transformers.tokenization_utils_tokenizers")
    _mock_mod.TokenizersBackend = type("TokenizersBackend", (), {})
    sys.modules["transformers.tokenization_utils_tokenizers"] = _mock_mod

from paddleformers.cli.train.dpo.dpo_argument import (
    DPOConfig,
    DPODataArgument,
    DPOModelArgument,
    DPOTrainingArguments,
)


class TestDPOConfig(unittest.TestCase):
    """Tests for DPOConfig dataclass"""

    def test_default_values(self):
        config = DPOConfig()
        self.assertEqual(config.beta, 0.1)
        self.assertEqual(config.simpo_gamma, 0.5)
        self.assertEqual(config.label_smoothing, 0.0)
        self.assertEqual(config.loss_type, "sigmoid")
        self.assertEqual(config.pref_loss_ratio, 1.0)
        self.assertEqual(config.sft_loss_ratio, 0.0)
        self.assertEqual(config.dpop_lambda, 50)
        self.assertEqual(config.ref_model_update_steps, -1)
        self.assertFalse(config.reference_free)
        self.assertFalse(config.lora)
        self.assertEqual(config.offset_alpha, 0.0)
        self.assertFalse(config.normalize_logps)
        self.assertFalse(config.ignore_eos_token)

    def test_custom_values(self):
        config = DPOConfig(
            beta=0.5,
            simpo_gamma=1.0,
            label_smoothing=0.1,
            loss_type="hinge",
            pref_loss_ratio=0.8,
            sft_loss_ratio=0.2,
            dpop_lambda=100,
            ref_model_update_steps=10,
            reference_free=True,
            lora=True,
            offset_alpha=0.5,
            normalize_logps=True,
            ignore_eos_token=True,
        )
        self.assertEqual(config.beta, 0.5)
        self.assertEqual(config.simpo_gamma, 1.0)
        self.assertEqual(config.label_smoothing, 0.1)
        self.assertEqual(config.loss_type, "hinge")
        self.assertEqual(config.pref_loss_ratio, 0.8)
        self.assertEqual(config.sft_loss_ratio, 0.2)
        self.assertEqual(config.dpop_lambda, 100)
        self.assertEqual(config.ref_model_update_steps, 10)
        self.assertTrue(config.reference_free)
        self.assertTrue(config.lora)
        self.assertEqual(config.offset_alpha, 0.5)
        self.assertTrue(config.normalize_logps)
        self.assertTrue(config.ignore_eos_token)

    def test_fields_count(self):
        all_fields = [f.name for f in fields(DPOConfig)]
        self.assertEqual(len(all_fields), 13)

    def test_loss_types(self):
        for loss_type in ["sigmoid", "hinge", "ipo", "kto_pair", "sppo_hard", "nca_pair", "dpop", "or", "simpo"]:
            config = DPOConfig(loss_type=loss_type)
            self.assertEqual(config.loss_type, loss_type)


class TestDPODataArgument(unittest.TestCase):
    """Tests for DPODataArgument dataclass"""

    def test_default_values(self):
        arg = DPODataArgument()
        self.assertEqual(arg.max_seq_len, 4096)
        self.assertEqual(arg.max_prompt_len, 2048)
        self.assertEqual(arg.num_samples_each_epoch, 6000000)
        self.assertEqual(arg.buffer_size, 1000)

    def test_custom_values(self):
        arg = DPODataArgument(
            max_seq_len=8192,
            max_prompt_len=4096,
            num_samples_each_epoch=1000,
            buffer_size=500,
        )
        self.assertEqual(arg.max_seq_len, 8192)
        self.assertEqual(arg.max_prompt_len, 4096)
        self.assertEqual(arg.num_samples_each_epoch, 1000)
        self.assertEqual(arg.buffer_size, 500)

    def test_inherits_data_config(self):
        arg = DPODataArgument()
        # Inherited from DataConfig
        self.assertIsNone(arg.dataset_name_or_path)
        self.assertIsNone(arg.train_dataset_type)
        self.assertIsNone(arg.train_dataset_path)
        self.assertEqual(arg.eval_dataset_type, "erniekit")
        self.assertEqual(arg.dataset_type, "iterable")
        self.assertTrue(arg.use_template)
        self.assertTrue(arg.encode_one_turn)
        self.assertFalse(arg.packing)
        self.assertTrue(arg.greedy_intokens)
        self.assertTrue(arg.random_shuffle)


class TestDPOModelArgument(unittest.TestCase):
    """Tests for DPOModelArgument dataclass"""

    def test_default_values(self):
        arg = DPOModelArgument()
        self.assertIsNone(arg.model_name_or_path)
        self.assertIsNone(arg.tokenizer_name_or_path)
        self.assertEqual(arg.download_hub, "aistudio")
        self.assertFalse(arg.flash_mask)
        self.assertIsNone(arg.weight_quantize_algo)
        self.assertTrue(arg.use_attn_mask_startend_row_indices)
        self.assertEqual(arg.lora_rank, 8)
        self.assertIsNone(arg.lora_path)
        self.assertFalse(arg.rslora)
        self.assertEqual(arg.lora_plus_scale, 1.0)
        self.assertEqual(arg.lora_alpha, -1)
        self.assertFalse(arg.rslora_plus)
        self.assertEqual(arg._attn_implementation, "flashmask")

    def test_custom_model_path(self):
        arg = DPOModelArgument(model_name_or_path="/path/to/model")
        self.assertEqual(arg.model_name_or_path, "/path/to/model")

    def test_lora_config(self):
        arg = DPOModelArgument(
            lora_rank=16,
            rslora=True,
            lora_plus_scale=2.0,
            lora_alpha=4,
        )
        self.assertEqual(arg.lora_rank, 16)
        self.assertTrue(arg.rslora)
        self.assertEqual(arg.lora_plus_scale, 2.0)
        self.assertEqual(arg.lora_alpha, 4)

    def test_download_hub_options(self):
        for hub in ["huggingface", "aistudio", "modelscope"]:
            arg = DPOModelArgument(download_hub=hub)
            self.assertEqual(arg.download_hub, hub)


class TestDPOTrainingArguments(unittest.TestCase):
    """Tests for DPOTrainingArguments dataclass"""

    def test_default_values(self):
        args = DPOTrainingArguments(output_dir="/tmp/test_output", bf16=True)
        self.assertEqual(args.num_of_gpus, -1)
        self.assertTrue(args.unified_checkpoint)
        self.assertEqual(args.unified_checkpoint_config, "")
        self.assertFalse(args.autotuner_benchmark)
        self.assertFalse(args.use_intermediate_api)
        self.assertEqual(args.num_hidden_layers, 2)

    def test_autotuner_benchmark_mode(self):
        args = DPOTrainingArguments(
            output_dir="/tmp/test_output", autotuner_benchmark=True, disable_tqdm=False, bf16=True
        )
        self.assertEqual(args.num_train_epochs, 1)
        self.assertEqual(args.max_steps, 5)
        self.assertTrue(args.do_train)
        self.assertFalse(args.do_export)
        self.assertFalse(args.do_predict)
        self.assertFalse(args.do_eval)
        self.assertTrue(args.overwrite_output_dir)
        self.assertFalse(args.load_best_model_at_end)

    def test_max_steps_sets_num_train_epochs(self):
        args = DPOTrainingArguments(output_dir="/tmp/test_output", max_steps=100, bf16=True)
        self.assertEqual(args.num_train_epochs, 1)


if __name__ == "__main__":
    unittest.main()
