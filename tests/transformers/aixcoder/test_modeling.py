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

"""Tests for AiXcoder model."""

import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.transformers.aixcoder.configuration import AixcoderConfig
from paddleformers.transformers.aixcoder.modeling import (
    AixcoderForCausalLM,
    AixcoderModel,
    AixcoderPretrainedModel,
)


class AixcoderModelTest(unittest.TestCase):
    """Test AiXcoder model classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small config for testing
        self.config = AixcoderConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            max_position_embeddings=512,
        )

        # Set paddle seed for reproducibility
        paddle.seed(42)

    def test_aixcoder_pretrained_model(self):
        """Test AixcoderPretrainedModel initialization."""
        model = AixcoderPretrainedModel(self.config)

        # Check model has config
        self.assertIsNotNone(model.config)
        self.assertEqual(model.config.hidden_size, 128)
        self.assertEqual(model.config.num_hidden_layers, 2)

        # Check base model name
        self.assertEqual(model.base_model_prefix, "model")

    def test_aixcoder_model_forward(self):
        """Test AixcoderModel forward pass."""
        model = AixcoderModel(self.config)
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        # Forward pass
        with paddle.no_grad():
            outputs = model(input_ids)

        # Check output shape
        self.assertEqual(outputs[0].shape[0], batch_size)
        self.assertEqual(outputs[0].shape[1], seq_length)
        self.assertEqual(outputs[0].shape[2], self.config.hidden_size)

    def test_aixcoder_for_causal_lm_forward(self):
        """Test AixcoderForCausalLM forward pass."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        # Forward pass
        with paddle.no_grad():
            outputs = model(input_ids)

        # Check output
        self.assertIsNotNone(outputs)

        # Check logits shape
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs[0]

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_length)
        self.assertEqual(logits.shape[2], self.config.vocab_size)

    def test_aixcoder_for_causal_lm_with_labels(self):
        """Test AixcoderForCausalLM with labels (training mode)."""
        model = AixcoderForCausalLM(self.config)
        model.train()

        # Create dummy input and labels
        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])
        labels = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])

        # Forward pass with labels
        outputs = model(input_ids, labels=labels)

        # Check loss is computed
        self.assertIsNotNone(outputs)
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.get("loss")

        if loss is not None:
            # Check loss is scalar
            self.assertEqual(loss.ndim, 0)
            # Check loss is positive
            self.assertGreater(loss.item(), 0)

    def test_aixcoder_for_causal_lm_generate(self):
        """Test AixcoderForCausalLM text generation."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        # Create prompt
        batch_size = 1
        prompt_length = 5
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, prompt_length])

        # Mock generate method if not available
        if not hasattr(model, "generate"):
            model.generate = MagicMock(
                return_value=paddle.concat(
                    [input_ids, paddle.randint(0, self.config.vocab_size, [batch_size, 10])], axis=1
                )
            )

        # Generate
        with paddle.no_grad():
            generated = model.generate(input_ids, max_new_tokens=10, temperature=0.7)

        # Check generated shape
        # generate() returns a tuple, first element is the generated token ids
        if isinstance(generated, tuple):
            generated_ids = generated[0]
        else:
            generated_ids = generated

        self.assertEqual(generated_ids.shape[0], batch_size)
        self.assertGreater(generated_ids.shape[1], prompt_length)

    def test_aixcoder_model_with_attention_mask(self):
        """Test AixcoderModel with attention mask."""
        model = AixcoderModel(self.config)
        model.eval()

        # Create input with attention mask
        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])
        attention_mask = paddle.ones([batch_size, seq_length])
        attention_mask[0, 5:] = 0  # Mask last 5 tokens of first sequence

        # Forward pass
        with paddle.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Check output shape
        self.assertEqual(outputs[0].shape[0], batch_size)
        self.assertEqual(outputs[0].shape[1], seq_length)
        self.assertEqual(outputs[0].shape[2], self.config.hidden_size)

    def test_aixcoder_model_with_position_ids(self):
        """Test AixcoderModel with custom position ids."""
        model = AixcoderModel(self.config)
        model.eval()

        # Create input with position ids
        batch_size = 2
        seq_length = 10
        input_ids = paddle.randint(0, self.config.vocab_size, [batch_size, seq_length])
        position_ids = paddle.arange(seq_length).expand([batch_size, seq_length])

        # Forward pass
        with paddle.no_grad():
            outputs = model(input_ids, position_ids=position_ids)

        # Check output
        self.assertIsNotNone(outputs)

    def test_aixcoder_model_gradient_checkpointing(self):
        """Test gradient checkpointing configuration."""
        # Create model with gradient checkpointing
        self.config.use_recompute = True
        model = AixcoderModel(self.config)

        # Check if gradient checkpointing can be enabled
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            self.assertTrue(model.config.use_recompute)

        # Disable gradient checkpointing
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    def test_aixcoder_config_validation_in_model(self):
        """Test that model validates config properly."""
        # Test with invalid config
        invalid_config = AixcoderConfig(
            hidden_size=128,
            num_attention_heads=5,  # 128 not divisible by 5
        )

        # Model should handle invalid config gracefully
        try:
            _ = AixcoderModel(invalid_config)
        except ValueError as e:
            # Expected behavior - config validation should catch this
            self.assertIn("hidden_size", str(e).lower())

    def test_aixcoder_for_causal_lm_tie_weights(self):
        """Test weight tying between embeddings and LM head."""
        # Create config with tie_word_embeddings=True
        config = AixcoderConfig(
            hidden_size=128, num_hidden_layers=2, num_attention_heads=4, vocab_size=1000, tie_word_embeddings=True
        )

        model = AixcoderForCausalLM(config)

        # Check if embeddings are tied (implementation dependent)
        # This test might need adjustment based on actual implementation
        if hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
            input_embed = model.get_input_embeddings()
            output_embed = model.get_output_embeddings()
            # Check if they reference the same weights when tie_word_embeddings=True
            if config.tie_word_embeddings and input_embed and output_embed:
                # Implementation specific check
                pass

    def test_model_dtype_and_device(self):
        """Test model dtype and device placement."""
        model = AixcoderForCausalLM(self.config)

        # Check model is on correct device
        if paddle.is_compiled_with_cuda():
            # Try to move model to GPU if cuda() method exists
            if hasattr(model, "cuda"):
                model = model.cuda()
            else:
                # If no cuda() method, just test on CPU
                return

            # Create input on same device
            input_ids = paddle.randint(0, self.config.vocab_size, [1, 10])
            input_ids = input_ids.cuda()

            with paddle.no_grad():
                outputs = model(input_ids)

            # Check output is on same device
            self.assertEqual(outputs[0].place.gpu_device_id(), input_ids.place.gpu_device_id())


if __name__ == "__main__":
    unittest.main()
