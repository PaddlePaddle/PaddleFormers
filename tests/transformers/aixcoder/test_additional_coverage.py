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

"""Additional tests for improving AiXcoder coverage."""

import unittest

import paddle

from paddleformers.transformers.aixcoder.configuration import AixcoderConfig
from paddleformers.transformers.aixcoder.modeling import (
    AixcoderForCausalLM,
    AixcoderModel,
)


class AixcoderAdditionalCoverageTest(unittest.TestCase):
    """Additional tests to improve code coverage."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AixcoderConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
        )
        paddle.seed(42)

    def test_forward_simple(self):
        """Simple forward pass test."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        # Single token input
        input_ids = paddle.ones([1, 1], dtype="int64")

        with paddle.no_grad():
            # Test __call__ method
            outputs = model(input_ids)

        self.assertIsNotNone(outputs)

    def test_forward_with_kwargs(self):
        """Test forward with various kwargs."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 5])

        # Test with use_cache=True
        with paddle.no_grad():
            outputs = model(input_ids, use_cache=True)
        self.assertIsNotNone(outputs)

        # Test with use_cache=False
        with paddle.no_grad():
            outputs = model(input_ids, use_cache=False)
        self.assertIsNotNone(outputs)

        # Test with output_hidden_states=True
        with paddle.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        self.assertIsNotNone(outputs)

    def test_model_string_representation(self):
        """Test model __str__ and __repr__ methods."""
        model = AixcoderForCausalLM(self.config)

        # Test string representation
        str_repr = str(model)
        self.assertIsInstance(str_repr, str)
        self.assertIn("AixcoderForCausalLM", str_repr)

    def test_model_parameter_count(self):
        """Test model parameter counting."""
        model = AixcoderForCausalLM(self.config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)

    def test_model_state_dict(self):
        """Test model state_dict operations."""
        model = AixcoderForCausalLM(self.config)

        # Get state dict
        state_dict = model.state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)

        # Check key components are present
        has_embed = any("embed" in key for key in state_dict.keys())
        has_layers = any("layers" in key for key in state_dict.keys())
        has_lm_head = any("lm_head" in key or "criterion" in key for key in state_dict.keys())

        self.assertTrue(has_embed)
        self.assertTrue(has_layers)
        self.assertTrue(has_lm_head)

    def test_model_gradient_accumulation(self):
        """Test model with gradient accumulation."""
        model = AixcoderForCausalLM(self.config)
        model.train()

        input_ids = paddle.randint(0, 1000, [2, 5])
        labels = paddle.randint(0, 1000, [2, 5])

        # Forward pass 1
        outputs1 = model(input_ids, labels=labels)
        if hasattr(outputs1, "loss"):
            loss1 = outputs1.loss
        else:
            loss1 = outputs1[0] if isinstance(outputs1, tuple) else None

        # Forward pass 2 (simulating gradient accumulation)
        outputs2 = model(input_ids, labels=labels)
        if hasattr(outputs2, "loss"):
            loss2 = outputs2.loss
        else:
            loss2 = outputs2[0] if isinstance(outputs2, tuple) else None

        # Both losses should be computed
        if loss1 is not None:
            self.assertIsNotNone(loss2)

    def test_model_eval_mode_consistency(self):
        """Test model behavior consistency in eval mode."""
        model = AixcoderForCausalLM(self.config)

        # Set to eval mode
        model.eval()
        self.assertFalse(model.training)

        # Set to train mode
        model.train()
        self.assertTrue(model.training)

        # Back to eval
        model.eval()
        self.assertFalse(model.training)

    def test_model_with_position_embeddings(self):
        """Test model with custom position embeddings."""
        model = AixcoderModel(self.config)
        model.eval()

        batch_size = 1
        seq_len = 10
        input_ids = paddle.randint(0, 1000, [batch_size, seq_len])

        # Create custom position ids
        position_ids = paddle.arange(seq_len).unsqueeze(0)

        with paddle.no_grad():
            outputs = model(input_ids, position_ids=position_ids)

        self.assertIsNotNone(outputs)
        self.assertEqual(outputs[0].shape[1], seq_len)

    def test_model_batch_processing(self):
        """Test model with different batch sizes."""
        model = AixcoderModel(self.config)
        model.eval()

        seq_len = 5

        # Test batch size 1
        input_ids = paddle.randint(0, 1000, [1, seq_len])
        with paddle.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs[0].shape[0], 1)

        # Test batch size 4
        input_ids = paddle.randint(0, 1000, [4, seq_len])
        with paddle.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs[0].shape[0], 4)

        # Test batch size 8
        input_ids = paddle.randint(0, 1000, [8, seq_len])
        with paddle.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs[0].shape[0], 8)

    def test_causal_lm_logits_shape(self):
        """Test CausalLM logits shape across different inputs."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        test_cases = [
            (1, 1),  # Single token
            (2, 5),  # Small batch
            (1, 50),  # Long sequence
            (4, 10),  # Medium batch and sequence
        ]

        for batch_size, seq_len in test_cases:
            input_ids = paddle.randint(0, 1000, [batch_size, seq_len])

            with paddle.no_grad():
                outputs = model(input_ids)

            # Get logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # Check shape
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], seq_len)
            self.assertEqual(logits.shape[2], self.config.vocab_size)


if __name__ == "__main__":
    unittest.main()
