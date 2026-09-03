# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from paddleformers.cli.hparams.server_args import ServerArguments


class TestServerArguments(unittest.TestCase):
    """Tests for ServerArguments dataclass."""

    def test_defaults(self):
        """Test ServerArguments default values."""
        args = ServerArguments()
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8188)
        self.assertEqual(args.metrics_port, 8001)
        self.assertEqual(args.engine_worker_queue_port, 8002)

    def test_model_defaults(self):
        """Test model-related defaults."""
        args = ServerArguments()
        self.assertEqual(args.max_model_len, 2048)
        self.assertEqual(args.max_num_seqs, 8)
        self.assertEqual(args.use_warmup, 0)
        self.assertAlmostEqual(args.gpu_memory_utilization, 0.9)
        self.assertIsNone(args.quantization)
        self.assertFalse(args.enable_mm)
        self.assertEqual(args.limit_mm_per_prompt, "{'image': 1, 'video': 1}")
        self.assertEqual(args.reasoning_parser, "ernie-45-vl")
        self.assertEqual(args.max_num_batched_tokens, 384)

    def test_cache_defaults(self):
        """Test cache-related defaults."""
        args = ServerArguments()
        self.assertEqual(args.block_size, 64)
        self.assertAlmostEqual(args.kv_cache_ratio, 0.75)

    def test_torch_defaults(self):
        """Test torch-related defaults."""
        args = ServerArguments()
        self.assertIsNone(args.load_choices)

    def test_tool_call_defaults(self):
        """Test tool call defaults."""
        args = ServerArguments()
        self.assertIsNone(args.tool_call_parser)

    def test_custom_host(self):
        """Test custom host."""
        args = ServerArguments(host="0.0.0.0")
        self.assertEqual(args.host, "0.0.0.0")

    def test_custom_ports(self):
        """Test custom port values."""
        args = ServerArguments(port=9090, metrics_port=9001, engine_worker_queue_port=9002)
        self.assertEqual(args.port, 9090)
        self.assertEqual(args.metrics_port, 9001)
        self.assertEqual(args.engine_worker_queue_port, 9002)

    def test_custom_model_params(self):
        """Test custom model parameters."""
        args = ServerArguments(
            max_model_len=4096,
            max_num_seqs=16,
            use_warmup=1,
            gpu_memory_utilization=0.95,
        )
        self.assertEqual(args.max_model_len, 4096)
        self.assertEqual(args.max_num_seqs, 16)
        self.assertEqual(args.use_warmup, 1)
        self.assertAlmostEqual(args.gpu_memory_utilization, 0.95)

    def test_custom_quantization(self):
        """Test custom quantization setting."""
        args = ServerArguments(quantization="wint4")
        self.assertEqual(args.quantization, "wint4")

    def test_enable_mm_true(self):
        """Test enabling multimodal support."""
        args = ServerArguments(enable_mm=True)
        self.assertTrue(args.enable_mm)

    def test_custom_limit_mm_per_prompt(self):
        """Test custom multimodal per-prompt limits."""
        args = ServerArguments(limit_mm_per_prompt="{'image': 10, 'video': 3}")
        self.assertEqual(args.limit_mm_per_prompt, "{'image': 10, 'video': 3}")

    def test_custom_reasoning_parser(self):
        """Test custom reasoning parser."""
        args = ServerArguments(reasoning_parser="custom_parser")
        self.assertEqual(args.reasoning_parser, "custom_parser")

    def test_custom_max_num_batched_tokens(self):
        """Test custom max_num_batched_tokens."""
        args = ServerArguments(max_num_batched_tokens=512)
        self.assertEqual(args.max_num_batched_tokens, 512)

    def test_custom_cache_params(self):
        """Test custom cache parameters."""
        args = ServerArguments(block_size=128, kv_cache_ratio=0.9)
        self.assertEqual(args.block_size, 128)
        self.assertAlmostEqual(args.kv_cache_ratio, 0.9)

    def test_custom_load_choices(self):
        """Test custom load_choices."""
        args = ServerArguments(load_choices="default_v1")
        self.assertEqual(args.load_choices, "default_v1")

    def test_custom_tool_call_parser(self):
        """Test custom tool_call_parser."""
        args = ServerArguments(tool_call_parser="custom_tool_parser")
        self.assertEqual(args.tool_call_parser, "custom_tool_parser")


if __name__ == "__main__":
    unittest.main()
