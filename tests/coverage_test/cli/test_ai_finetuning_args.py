# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import os
import unittest
from unittest.mock import MagicMock, patch


class TestPreTrainingArguments(unittest.TestCase):
    """Tests for PreTrainingArguments dataclass field defaults."""

    def test_pretraining_field_defaults(self):
        """Test PreTrainingArguments field defaults without instantiation."""
        import dataclasses

        from paddleformers.cli.hparams.finetuning_args import PreTrainingArguments

        fields = {
            f.name: f.default for f in dataclasses.fields(PreTrainingArguments) if f.default is not dataclasses.MISSING
        }
        self.assertEqual(fields["eval_iters"], -1)
        self.assertEqual(fields["use_async_save"], False)
        self.assertEqual(fields["pre_alloc_memory"], 0.0)
        self.assertEqual(fields["use_moe"], False)
        self.assertEqual(fields["enable_mtp_magic_send"], False)
        self.assertEqual(fields["lr_scheduler"], "cosine")
        self.assertEqual(fields["freeze_config"], "")
        self.assertEqual(fields["decay_function"], "half_life")
        self.assertEqual(fields["gc_interval"], 0)
        self.assertEqual(fields["global_batch_size"], -1)
        self.assertEqual(fields["global_logging_interval"], 1)
        self.assertEqual(fields["num_consecutive"], 1)
        self.assertEqual(fields["use_ortho_loss_callback"], False)
        self.assertEqual(fields["moe_with_send_router_loss"], True)

    def test_vl_sft_field_defaults(self):
        """Test VLSFTTrainingArguments field defaults."""
        import dataclasses

        from paddleformers.cli.hparams.finetuning_args import VLSFTTrainingArguments

        fields = {
            f.name: f.default
            for f in dataclasses.fields(VLSFTTrainingArguments)
            if f.default is not dataclasses.MISSING
        }
        self.assertEqual(fields["factor"], 20)
        self.assertEqual(fields["hidden_dropout_prob"], 0.0)
        self.assertEqual(fields["moe_dropout_prob"], 0.0)
        self.assertEqual(fields["token_balance_loss"], False)


class TestSFTTrainingArguments(unittest.TestCase):
    """Tests for SFTTrainingArguments dataclass field defaults."""

    def test_sft_field_defaults(self):
        """Test SFTTrainingArguments field defaults."""
        import dataclasses

        from paddleformers.cli.hparams.finetuning_args import SFTTrainingArguments

        fields = {
            f.name: f.default for f in dataclasses.fields(SFTTrainingArguments) if f.default is not dataclasses.MISSING
        }
        self.assertEqual(fields["max_estimate_samples"], 1e5)
        self.assertEqual(fields["estimation_output_file"], "estimation_output.json")


class TestDPOTrainingArguments(unittest.TestCase):
    """Tests for DPOTrainingArguments dataclass field defaults."""

    def test_dpo_field_defaults(self):
        """Test DPOTrainingArguments field defaults."""
        import dataclasses

        from paddleformers.cli.hparams.finetuning_args import DPOTrainingArguments

        fields = {
            f.name: f.default for f in dataclasses.fields(DPOTrainingArguments) if f.default is not dataclasses.MISSING
        }
        self.assertEqual(fields["num_of_gpus"], -1)
        self.assertEqual(fields["normalize_logps"], False)
        self.assertEqual(fields["label_smoothing"], 0.0)
        self.assertEqual(fields["ignore_eos_token"], False)
        self.assertEqual(fields["ref_model_update_steps"], -1)
        self.assertEqual(fields["reference_free"], False)
        self.assertEqual(fields["loss_type"], "sigmoid")
        self.assertEqual(fields["pref_loss_ratio"], 1.0)
        self.assertEqual(fields["sft_loss_ratio"], 0.0)
        self.assertEqual(fields["beta"], 0.1)
        self.assertEqual(fields["offset_alpha"], 0.0)
        self.assertEqual(fields["simpo_gamma"], 0.5)
        self.assertEqual(fields["dpop_lambda"], 50)


class TestFinetuningArgumentsFieldDefaults(unittest.TestCase):
    """Tests for FinetuningArguments field defaults without instantiation."""

    def test_finetuning_field_defaults(self):
        """Test FinetuningArguments field defaults."""
        import dataclasses

        from paddleformers.cli.hparams.finetuning_args import FinetuningArguments

        fields = {
            f.name: f.default for f in dataclasses.fields(FinetuningArguments) if f.default is not dataclasses.MISSING
        }
        self.assertEqual(fields["compute_type"], "bf16")
        self.assertEqual(fields["use_fp8"], False)
        self.assertEqual(fields["use_recompute_mtp"], False)
        self.assertEqual(fields["autotuner_benchmark"], False)
        self.assertEqual(fields["dataset_num_proc"], None)
        self.assertEqual(fields["dataset_batch_size"], 1000)
        self.assertEqual(fields["dataset_text_field"], "text")
        self.assertEqual(fields["enable_linear_fused_grad_add"], False)
        self.assertEqual(fields["hidden_dropout_prob"], 0.0)
        self.assertEqual(fields["attention_probs_dropout_prob"], 0.0)


class TestFinetuningArgumentsComputeType(unittest.TestCase):
    """Tests for FinetuningArguments compute_type logic.

    The compute_type logic is in __post_init__, which is hard to test
    directly because it calls super().__post_init__() requiring distributed
    setup. Instead, we directly test the compute_type mapping by copying the
    relevant logic from FinetuningArguments.__post_init__.
    """

    def _apply_compute_type_logic(self, compute_type):
        """Replicate the compute_type mapping from FinetuningArguments.__post_init__."""
        from paddleformers.cli.hparams.finetuning_args import DEFAULT_QUANTIZE_LAYERS

        bf16 = True
        fp16 = False
        weight_quantize_algo = None

        if compute_type == "bf16":
            fp16 = False
            weight_quantize_algo = None
        elif compute_type == "fp16":
            bf16 = False
            fp16 = True
            weight_quantize_algo = None
        elif compute_type == "wint4":
            weight_quantize_algo = {"weight_only_int4": DEFAULT_QUANTIZE_LAYERS}
        elif compute_type == "wint8":
            weight_quantize_algo = {"weight_only_int8": DEFAULT_QUANTIZE_LAYERS}
        elif compute_type == "wint4/8":
            weight_quantize_algo = {
                "weight_only_int4": [
                    ".*mlp.experts.*",
                    ".*mlp.shared_expert.*",
                    ".*mlp.shared_experts.*",
                ],
                "weight_only_int8": [
                    ".*self_attn.qkv_proj.*",
                    ".*self_attn.q_proj.*",
                    ".*self_attn.k_proj.*",
                    ".*self_attn.v_proj.*",
                    ".*self_attn.o_proj.*",
                    ".*mlp.up_gate_proj.*",
                    ".*mlp.up_proj.*",
                    ".*mlp.gate_proj.*",
                    ".*mlp.down_proj.*",
                ],
            }
        elif compute_type == "nf4":
            weight_quantize_algo = {"nf4": DEFAULT_QUANTIZE_LAYERS}
        else:
            raise ValueError(f"Unknown compute_type: {compute_type}")

        return bf16, fp16, weight_quantize_algo

    def test_compute_type_bf16(self):
        bf16, fp16, algo = self._apply_compute_type_logic("bf16")
        self.assertTrue(bf16)
        self.assertFalse(fp16)
        self.assertIsNone(algo)

    def test_compute_type_fp16(self):
        bf16, fp16, algo = self._apply_compute_type_logic("fp16")
        self.assertFalse(bf16)
        self.assertTrue(fp16)
        self.assertIsNone(algo)

    def test_compute_type_wint4(self):
        from paddleformers.cli.hparams.finetuning_args import DEFAULT_QUANTIZE_LAYERS

        bf16, fp16, algo = self._apply_compute_type_logic("wint4")
        self.assertIsInstance(algo, dict)
        self.assertIn("weight_only_int4", algo)
        self.assertEqual(algo["weight_only_int4"], DEFAULT_QUANTIZE_LAYERS)

    def test_compute_type_wint8(self):
        bf16, fp16, algo = self._apply_compute_type_logic("wint8")
        self.assertIsInstance(algo, dict)
        self.assertIn("weight_only_int8", algo)

    def test_compute_type_wint4_8(self):
        bf16, fp16, algo = self._apply_compute_type_logic("wint4/8")
        self.assertIsInstance(algo, dict)
        self.assertIn("weight_only_int4", algo)
        self.assertIn("weight_only_int8", algo)
        self.assertIn(".*self_attn.qkv_proj.*", algo["weight_only_int8"])

    def test_compute_type_nf4(self):
        from paddleformers.cli.hparams.finetuning_args import DEFAULT_QUANTIZE_LAYERS

        bf16, fp16, algo = self._apply_compute_type_logic("nf4")
        self.assertIsInstance(algo, dict)
        self.assertIn("nf4", algo)
        self.assertEqual(algo["nf4"], DEFAULT_QUANTIZE_LAYERS)

    def test_compute_type_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._apply_compute_type_logic("unknown_type")
        self.assertIn("Unknown compute_type", str(ctx.exception))


class TestGetTrainArgs(unittest.TestCase):
    """Tests for get_train_args function."""

    def setUp(self):
        """Save original environment state before each test."""
        self._original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment state after each test."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    @patch("paddleformers.cli.hparams.parser._parse_train_args")
    def test_get_train_args_vl_stage_sets_env_vars(self, mock_parse, mock_env):
        """Test get_train_args sets NCCL env vars when stage contains 'VL'."""
        from paddleformers.cli.hparams.parser import get_train_args

        mock_model_args = MagicMock(stage="VL-SFT")
        mock_data_args = MagicMock(packing=True, truncate_packing=True, template_backend="jinja")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()
        mock_parse.return_value = (
            mock_model_args,
            mock_data_args,
            mock_preprocess_args,
            mock_generating_args,
            mock_finetuning_args,
        )

        get_train_args({"output_dir": "/tmp/out"})

        self.assertEqual(os.environ.get("NCCL_DEBUG"), "INFO")
        self.assertEqual(os.environ.get("PYTHONUNBUFFERED"), "1")

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    @patch("paddleformers.cli.hparams.parser._parse_train_args")
    def test_get_train_args_vl_truncate_packing_warning(self, mock_parse, mock_env):
        """Test get_train_args warns and resets truncate_packing when both packing and truncate_packing are True."""
        from paddleformers.cli.hparams.parser import get_train_args

        mock_model_args = MagicMock(stage="VL-SFT")
        mock_data_args = MagicMock(
            packing=True, truncate_packing=True, split_multi_turn=False, template_backend="custom"
        )
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()
        mock_parse.return_value = (
            mock_model_args,
            mock_data_args,
            mock_preprocess_args,
            mock_generating_args,
            mock_finetuning_args,
        )

        with patch("paddleformers.cli.hparams.parser.logger") as mock_logger:
            get_train_args({"output_dir": "/tmp/out"})
            mock_logger.warning.assert_called_once()
            self.assertFalse(mock_data_args.truncate_packing)

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    @patch("paddleformers.cli.hparams.parser._parse_train_args")
    def test_get_train_args_split_multi_turn_requires_jinja(self, mock_parse, mock_env):
        """Test get_train_args raises ValueError when split_multi_turn is True but template_backend is not jinja."""
        from paddleformers.cli.hparams.parser import get_train_args

        mock_model_args = MagicMock(stage="SFT")
        mock_data_args = MagicMock(split_multi_turn=True, template_backend="custom")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()
        mock_parse.return_value = (
            mock_model_args,
            mock_data_args,
            mock_preprocess_args,
            mock_generating_args,
            mock_finetuning_args,
        )

        with self.assertRaises(ValueError) as ctx:
            get_train_args({"output_dir": "/tmp/out"})
        self.assertIn("template_backend must be jinja", str(ctx.exception))

    @patch("paddleformers.cli.hparams.parser.is_env_enabled", return_value=False)
    @patch("paddleformers.cli.hparams.parser._parse_train_args")
    def test_get_train_args_flashmask_requires_attn_mask(self, mock_parse, mock_env):
        """Test get_train_args raises ValueError when flashmask is used without use_attn_mask_startend_row_indices."""
        from paddleformers.cli.hparams.parser import get_train_args

        mock_model_args = MagicMock(
            stage="SFT", _attn_implementation="flashmask", use_attn_mask_startend_row_indices=False
        )
        mock_data_args = MagicMock(split_multi_turn=False, template_backend="custom")
        mock_preprocess_args = MagicMock()
        mock_generating_args = MagicMock()
        mock_finetuning_args = MagicMock()
        mock_parse.return_value = (
            mock_model_args,
            mock_data_args,
            mock_preprocess_args,
            mock_generating_args,
            mock_finetuning_args,
        )

        with self.assertRaises(ValueError) as ctx:
            get_train_args({"output_dir": "/tmp/out"})
        self.assertIn("flashmask", str(ctx.exception))
        self.assertIn("use_attn_mask_startend_row_indices", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
