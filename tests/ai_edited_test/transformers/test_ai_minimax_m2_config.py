# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest


class TestMiniMaxM2Config(unittest.TestCase):
    """Tests for transformers/minimax_m2/configuration.py"""

    def test_default_config(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config()
        self.assertEqual(config.model_type, "minimax_m2")
        self.assertEqual(config.vocab_size, 200064)
        self.assertEqual(config.hidden_size, 3072)
        self.assertEqual(config.num_hidden_layers, 62)
        self.assertEqual(config.num_attention_heads, 48)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.moe_intermediate_size, 1536)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.n_routed_experts, 256)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertTrue(config.use_cache)
        self.assertEqual(config.hidden_act, "silu")
        self.assertAlmostEqual(config.rms_norm_eps, 1e-6)
        self.assertAlmostEqual(config.rope_theta, 5000000)

    def test_custom_config(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(
            vocab_size=100,
            hidden_size=2048,
            num_hidden_layers=12,
            num_attention_heads=16,
            rope_theta=1000000,
        )
        self.assertEqual(config.vocab_size, 100)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertAlmostEqual(config.rope_theta, 1000000)

    def test_moe_config(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(
            num_experts_per_tok=4,
            n_routed_experts=64,
            n_group=2,
            topk_group=2,
            first_k_dense_replace=2,
        )
        self.assertEqual(config.num_experts_per_tok, 4)
        self.assertEqual(config.n_routed_experts, 64)
        self.assertEqual(config.n_group, 2)

    def test_rope_scaling_bc_type(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(rope_scaling={"type": "linear", "factor": 2.0})
        self.assertEqual(config.rope_scaling["rope_type"], "linear")

    def test_config_with_sliding_window(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(sliding_window=4096)
        self.assertEqual(config.sliding_window, 4096)

    def test_mtp_config(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(use_mtp=False)
        self.assertFalse(config.use_mtp)

    def test_scoring_func(self):
        from paddleformers.transformers.minimax_m2.configuration import MiniMaxM2Config

        config = MiniMaxM2Config(scoring_func="softmax")
        self.assertEqual(config.scoring_func, "softmax")
