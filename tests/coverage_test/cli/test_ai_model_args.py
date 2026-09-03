# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.model_args import (
    ErniePretrainArgument,
    FP8FusedOpsConfigs,
    FP8MemConfigs,
    ModelArguments,
    VisionArguments,
)


class TestVisionArguments(unittest.TestCase):
    """Tests for VisionArguments dataclass."""

    def test_defaults(self):
        args = VisionArguments()
        self.assertEqual(args.attn_implementation, "eager")
        self.assertTrue(args.attn_sep)
        self.assertEqual(args.depth, 32)
        self.assertEqual(args.embed_dim, 1280)
        self.assertEqual(args.hidden_act, "quick_gelu")
        self.assertEqual(args.hidden_size, 1280)
        self.assertEqual(args.in_channels, 3)
        self.assertEqual(args.mlp_ratio, 4)
        self.assertEqual(args.model_type, "DFNRope_vision_transformer")
        self.assertEqual(args.num_heads, 16)
        self.assertEqual(args.patch_size, 14)
        self.assertEqual(args.spatial_merge_size, 2)
        self.assertEqual(args.tensor_model_parallel_size, 4)
        self.assertEqual(args.vit_num_recompute_layers, 10000)

    def test_custom_values(self):
        args = VisionArguments(depth=64, embed_dim=2560, num_heads=32)
        self.assertEqual(args.depth, 64)
        self.assertEqual(args.embed_dim, 2560)
        self.assertEqual(args.num_heads, 32)


class TestFP8MemConfigs(unittest.TestCase):
    """Tests for FP8MemConfigs dataclass."""

    def test_defaults(self):
        args = FP8MemConfigs()
        self.assertFalse(args.shared_expert)
        self.assertFalse(args.recompute_fwd_gate_up)
        self.assertFalse(args.dequant_input)
        self.assertFalse(args.offline_quant_expert_weight)


class TestFP8FusedOpsConfigs(unittest.TestCase):
    """Tests for FP8FusedOpsConfigs dataclass."""

    def test_defaults(self):
        args = FP8FusedOpsConfigs()
        self.assertFalse(args.stack_quant)
        self.assertFalse(args.swiglu_probs_bwd)
        self.assertTrue(args.split_group_gemm)
        self.assertTrue(args.spaq)
        self.assertTrue(args.transpose_split_quant)


class TestErniePretrainArgument(unittest.TestCase):
    """Tests for ErniePretrainArgument dataclass."""

    def test_defaults(self):
        args = ErniePretrainArgument()
        self.assertFalse(args.use_quant_before_a2a)
        self.assertFalse(args.use_async_a2a)
        self.assertFalse(args.use_rms_qkv_recompute)
        self.assertFalse(args.use_recompute)
        self.assertEqual(args.num_nextn_predict_layers, 0)
        self.assertFalse(args.use_fp8_mlp)
        self.assertEqual(args.num_hidden_layers, 2)
        self.assertEqual(args.moe_k, 2)
        self.assertEqual(args.moe_gate, "top2_fused")

    def test_custom_values(self):
        args = ErniePretrainArgument(
            use_recompute=True,
            num_hidden_layers=12,
            moe_num_experts=8,
        )
        self.assertTrue(args.use_recompute)
        self.assertEqual(args.num_hidden_layers, 12)
        self.assertEqual(args.moe_num_experts, 8)


class TestModelArguments(unittest.TestCase):
    """Tests for ModelArguments dataclass."""

    def test_defaults(self):
        args = ModelArguments()
        self.assertIsNone(args.model_name_or_path)
        self.assertIsNone(args.tokenizer_name_or_path)
        self.assertTrue(args.continue_training)
        self.assertEqual(args.stage, "SFT")
        self.assertTrue(args.use_mem_eff_attn)
        self.assertTrue(args.use_attn_mask_startend_row_indices)

    def test_lora_auto_set(self):
        """Test that fine_tuning='LoRA' auto-sets lora=True."""
        args = ModelArguments(fine_tuning="LoRA")
        self.assertTrue(args.lora)

    def test_lora_auto_unset(self):
        """Test that fine_tuning='Full' auto-sets lora=False."""
        args = ModelArguments(fine_tuning="Full")
        self.assertFalse(args.lora)

    def test_lora_auto_unset_case_insensitive(self):
        """Test that fine_tuning='full' (lowercase) auto-sets lora=False."""
        args = ModelArguments(fine_tuning="full")
        self.assertFalse(args.lora)

    def test_lora_explicit_values(self):
        """Test explicit LoRA-related parameters."""
        args = ModelArguments(
            fine_tuning="LoRA",
            lora_rank=16,
            lora_alpha=32,
        )
        self.assertTrue(args.lora)
        self.assertEqual(args.lora_rank, 16)
        self.assertEqual(args.lora_alpha, 32)

    def test_rslora_defaults(self):
        args = ModelArguments()
        self.assertFalse(args.rslora)
        self.assertFalse(args.rslora_plus)
        self.assertEqual(args.lora_plus_scale, 1.0)

    def test_neftune_defaults(self):
        args = ModelArguments()
        self.assertFalse(args.neftune)
        self.assertEqual(args.neftune_noise_alpha, 5.0)

    def test_moe_defaults(self):
        args = ModelArguments()
        self.assertEqual(args.moe_group, "dummy")
        self.assertFalse(args.moe_group_experts)
        self.assertEqual(args.moe_orthogonal_loss_lambda, 0.0)
        self.assertFalse(args.moe_use_hard_gate)
        self.assertIsNone(args.moe_use_aux_free)

    def test_pp_seg_method(self):
        args = ModelArguments()
        self.assertEqual(args.pp_seg_method, "layer:DecoderLayer|EmptyLayer")

    def test_vision_config_default(self):
        args = ModelArguments()
        self.assertIsInstance(args.vision_config, VisionArguments)

    def test_ernie_model_config_default(self):
        args = ModelArguments()
        self.assertIsInstance(args.ernie_model_config, ErniePretrainArgument)

    def test_token_ids(self):
        args = ModelArguments()
        self.assertEqual(args.bos_token_id, 0)
        self.assertEqual(args.eos_token_id, 1)
        self.assertEqual(args.max_position_embeddings, 4096)

    def test_moe_gate(self):
        args = ModelArguments()
        self.assertEqual(args.moe_gate, "top2_fused")

    def test_download_hub_default(self):
        args = ModelArguments()
        self.assertIsNone(args.download_hub)

    def test_copy_custom_file_list_default(self):
        args = ModelArguments()
        self.assertEqual(args.copy_custom_file_list, "")

    def test_attn_implementation(self):
        args = ModelArguments()
        self.assertEqual(args._attn_implementation, "flashmask")

    def test_fuse_softmax_mask(self):
        args = ModelArguments()
        self.assertFalse(args.fuse_softmax_mask)

    def test_fuse_gate_detach_matmul(self):
        args = ModelArguments()
        self.assertTrue(args.fuse_gate_detach_matmul)

    def test_use_sparse_flash_attn(self):
        args = ModelArguments()
        self.assertTrue(args.use_sparse_flash_attn)

    def test_use_global_causal_attn(self):
        args = ModelArguments()
        self.assertFalse(args.use_global_causal_attn)

    def test_rope_3d(self):
        args = ModelArguments()
        self.assertTrue(args.rope_3d)

    def test_model_with_dpo_criterion(self):
        args = ModelArguments()
        self.assertFalse(args.model_with_dpo_criterion)

    def test_loss_subbatch_seqlen(self):
        args = ModelArguments()
        self.assertEqual(args.loss_subbatch_seqlen, 32768)

    def test_moe_multimodal_dispatch(self):
        args = ModelArguments()
        self.assertEqual(args.moe_multimodal_dispatch_use_allgather, "v2-alltoall-unpad")


if __name__ == "__main__":
    unittest.main()
