# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle


class TestLMHead(unittest.TestCase):
    """Tests for paddleformers.nn.lm_head.LMHead."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.vocab_size = overrides.get("vocab_size", 100)
        config.hidden_size = overrides.get("hidden_size", 64)
        config.tensor_model_parallel_size = overrides.get("tensor_model_parallel_size", 1)
        config.lm_head_bias = overrides.get("lm_head_bias", False)
        config.use_fused_head_and_loss_fn = overrides.get("use_fused_head_and_loss_fn", False)
        config.sequence_parallel = False
        config.max_sequence_length = 128
        config.tensor_parallel_output = False
        config.get = lambda key, default=None: overrides.get(key, default)
        return config

    def test_lm_head_init_no_bias(self):
        """LMHead without bias should have bias=None."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(lm_head_bias=False)
        head = LMHead(config)
        self.assertIsNone(head.bias)
        self.assertFalse(head.vocab_parallel)
        self.assertEqual(head.weight.shape, [100, 64])

    def test_lm_head_init_with_bias(self):
        """LMHead with bias should create a bias parameter."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(lm_head_bias=True)
        head = LMHead(config)
        self.assertIsNotNone(head.bias)
        self.assertEqual(head.bias.shape, [100])

    def test_lm_head_init_vocab_parallel(self):
        """LMHead with tp_size>1 should enable vocab_parallel and reduce vocab_size."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(vocab_size=100, tensor_model_parallel_size=2)
        head = LMHead(config)
        self.assertTrue(head.vocab_parallel)
        self.assertEqual(head.weight.shape, [50, 64])

    def test_lm_head_init_vocab_not_divisible(self):
        """LMHead should raise ValueError when vocab_size is not divisible by tp_size."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(vocab_size=101, tensor_model_parallel_size=2)
        with self.assertRaises(ValueError):
            LMHead(config)

    def test_lm_head_forward_normal(self):
        """Normal forward pass should call calc_lm_head_logits."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config()
        head = LMHead(config)
        x = paddle.randn([2, 8, 64], dtype="float32")
        with patch("paddleformers.nn.lm_head.calc_lm_head_logits", return_value=paddle.randn([2, 8, 100])) as mock_fn:
            head(x)
            mock_fn.assert_called_once()
            # Verify key arguments
            call_kwargs = mock_fn.call_args
            self.assertEqual(call_kwargs[0][0], config)
            self.assertTrue(call_kwargs[1]["gather_hidden_states"])

    def test_lm_head_forward_fused(self):
        """With use_fused_head_and_loss_fn, should return (hidden_states, weight, bias, True)."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(use_fused_head_and_loss_fn=True)
        head = LMHead(config)
        x = paddle.randn([2, 8, 64], dtype="float32")
        result = head(x)
        self.assertEqual(len(result), 4)
        hidden_states, weight, bias, flag = result
        np.testing.assert_allclose(hidden_states.numpy(), x.numpy())
        self.assertTrue(flag)

    def test_lm_head_extra_repr(self):
        """extra_repr should contain key attributes."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config()
        head = LMHead(config)
        repr_str = head.extra_repr()
        self.assertIn("hidden_size=64", repr_str)
        self.assertIn("vocab_size=100", repr_str)
        self.assertIn("vocab_parallel=False", repr_str)

    def test_lm_head_extra_repr_parallel(self):
        """extra_repr with vocab_parallel should show True."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(tensor_model_parallel_size=2)
        head = LMHead(config)
        repr_str = head.extra_repr()
        self.assertIn("vocab_parallel=True", repr_str)

    def test_lm_head_sharded_state_dict_single_gpu(self):
        """sharded_state_dict with tp_size=1 should call parent's method."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(tensor_model_parallel_size=1)
        head = LMHead(config)
        # The parent's sharded_state_dict should work fine
        result = head.sharded_state_dict(structured_name_prefix="test.")
        self.assertIsInstance(result, dict)

    def test_lm_head_sharded_state_dict_multi_gpu(self):
        """sharded_state_dict with tp_size>1 should use build_sharded_state_dict."""
        from paddleformers.nn.lm_head import LMHead

        config = self._make_config(tensor_model_parallel_size=2)
        head = LMHead(config)
        with patch("paddleformers.nn.lm_head.build_sharded_state_dict") as mock_build:
            mock_build.return_value = {"test": "value"}
            head.sharded_state_dict(structured_name_prefix="test.")
            mock_build.assert_called_once()
