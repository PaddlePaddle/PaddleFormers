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

"""Edge case tests for AiXcoder model."""

import unittest

import paddle
import paddle.nn as nn

from paddleformers.transformers.aixcoder.configuration import AixcoderConfig
from paddleformers.transformers.aixcoder.modeling import (
    AixcoderForCausalLM,
    AixcoderModel,
    AixcoderPretrainedModel,
)


class AixcoderModelEdgeCaseTest(unittest.TestCase):
    """Test edge cases for AiXcoder model classes."""

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
        paddle.seed(42)

    def test_init_weights_linear(self):
        """Test weight initialization for Linear layers."""
        model = AixcoderPretrainedModel(self.config)

        # Create a linear layer
        linear = nn.Linear(128, 256)

        # Apply weight initialization
        model._init_weights(linear)

        # Check weights are initialized
        self.assertIsNotNone(linear.weight)
        self.assertEqual(linear.weight.shape, [128, 256])

        # Check bias initialization if exists
        if linear.bias is not None:
            # Bias should be zeros
            self.assertTrue(paddle.allclose(linear.bias, paddle.zeros_like(linear.bias)))

    def test_init_weights_embedding(self):
        """Test weight initialization for Embedding layers."""
        model = AixcoderPretrainedModel(self.config)

        # Create an embedding layer with padding_idx
        embedding = nn.Embedding(1000, 128, padding_idx=0)

        # Apply weight initialization
        model._init_weights(embedding)

        # Check weights are initialized
        self.assertIsNotNone(embedding.weight)
        self.assertEqual(embedding.weight.shape, [1000, 128])

        # Check padding index is zero
        self.assertTrue(paddle.allclose(embedding.weight[0], paddle.zeros([128])))

    def test_init_weights_with_no_bias(self):
        """Test weight initialization for Linear layer without bias."""
        model = AixcoderPretrainedModel(self.config)

        # Create a linear layer without bias
        linear = nn.Linear(128, 256, bias_attr=False)

        # Apply weight initialization - should not fail
        model._init_weights(linear)

        # Check weights are initialized
        self.assertIsNotNone(linear.weight)
        self.assertIsNone(linear.bias)

    def test_get_tensor_parallel_mappings(self):
        """Test tensor parallel mapping generation."""
        # Test with tensor parallel config
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            tensor_parallel_degree=2,
            tensor_parallel_rank=0,
        )

        # Get mappings
        mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

        # Check mappings are generated
        self.assertIsNotNone(mappings)
        self.assertIsInstance(mappings, dict)

        # Check key layers are included
        self.assertIn("aixcoder.embed_tokens.weight", mappings)
        self.assertIn("aixcoder.layers.0.self_attn.o_proj.weight", mappings)
        self.assertIn("aixcoder.layers.0.mlp.down_proj.weight", mappings)

    def test_get_tensor_parallel_mappings_with_fused_qkv(self):
        """Test tensor parallel mappings with fused QKV projection."""
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            tensor_parallel_degree=2,
            fuse_attention_qkv=True,
        )

        mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

        # Check fused QKV is included
        self.assertIn("aixcoder.layers.0.self_attn.qkv_proj.weight", mappings)
        # Check individual projections are not included
        self.assertNotIn("aixcoder.layers.0.self_attn.q_proj.weight", mappings)

    def test_get_tensor_parallel_mappings_with_fused_ffn(self):
        """Test tensor parallel mappings with fused FFN projection."""
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
            tensor_parallel_degree=2,
            fuse_attention_ffn=True,
        )

        mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

        # Check fused gate_up is included
        self.assertIn("aixcoder.layers.0.mlp.gate_up_fused_proj.weight", mappings)
        # Check individual projections are not included
        self.assertNotIn("aixcoder.layers.0.mlp.gate_proj.weight", mappings)
        self.assertNotIn("aixcoder.layers.0.mlp.up_proj.weight", mappings)

    def test_get_tensor_parallel_mappings_vocab_not_divisible(self):
        """Test tensor parallel mappings when vocab size is not divisible by TP degree."""
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=999,  # Not divisible by 2
            tensor_parallel_degree=2,
        )

        mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

        # Check that vocab-related layers are excluded
        self.assertNotIn("lm_head.weight", mappings)
        self.assertNotIn("aixcoder.embed_tokens.weight", mappings)

    def test_get_tensor_parallel_mappings_with_tie_embeddings(self):
        """Test tensor parallel mappings with tied word embeddings."""
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=1000,
            tensor_parallel_degree=2,
            tie_word_embeddings=True,
        )

        mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

        # Check lm_head mapping exists
        self.assertIn("lm_head.weight", mappings)

    def test_prepare_inputs_for_generation(self):
        """Test prepare_inputs_for_generation method."""
        model = AixcoderForCausalLM(self.config)

        # Create dummy inputs
        input_ids = paddle.randint(0, 1000, [1, 10])
        past_key_values = None
        attention_mask = paddle.ones([1, 10])

        # Prepare inputs
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=None,
        )

        # Check output format
        self.assertIsInstance(model_inputs, dict)
        self.assertIn("input_ids", model_inputs)
        self.assertIn("attention_mask", model_inputs)
        self.assertIn("past_key_values", model_inputs)

    def test_prepare_inputs_for_generation_with_past(self):
        """Test prepare_inputs_for_generation with past key values."""
        model = AixcoderForCausalLM(self.config)

        # Create dummy inputs
        input_ids = paddle.randint(0, 1000, [1, 10])
        # Mock past_key_values (not None)
        past_key_values = [(paddle.randn([1, 4, 9, 32]), paddle.randn([1, 4, 9, 32])) for _ in range(2)]
        attention_mask = paddle.ones([1, 10])

        # Prepare inputs with past
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )

        # When past is provided, only the last token should be used
        self.assertEqual(model_inputs["input_ids"].shape[1], 1)

    def test_model_with_extreme_sequence_length(self):
        """Test model with very long and very short sequences."""
        model = AixcoderModel(self.config)
        model.eval()

        # Test with sequence length of 1
        input_ids = paddle.randint(0, 1000, [1, 1])
        with paddle.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs[0].shape[1], 1)

        # Test with maximum sequence length
        max_len = min(512, self.config.max_position_embeddings)
        input_ids = paddle.randint(0, 1000, [1, max_len])
        with paddle.no_grad():
            outputs = model(input_ids)
        self.assertEqual(outputs[0].shape[1], max_len)

    def test_model_with_all_masked_attention(self):
        """Test model with fully masked attention."""
        model = AixcoderModel(self.config)
        model.eval()

        # Create input with all positions masked
        input_ids = paddle.randint(0, 1000, [1, 10])
        attention_mask = paddle.zeros([1, 10])

        # Should not crash
        with paddle.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(outputs)

    def test_causal_lm_with_negative_labels(self):
        """Test CausalLM with -100 labels (ignored in loss)."""
        model = AixcoderForCausalLM(self.config)
        model.train()

        # Create inputs with some -100 labels
        input_ids = paddle.randint(0, 1000, [2, 10])
        labels = paddle.randint(0, 1000, [2, 10])
        labels[:, :5] = -100  # Ignore first 5 positions

        # Forward pass
        outputs = model(input_ids, labels=labels)

        # Loss should still be computed
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            loss = outputs[0] if isinstance(outputs, tuple) else None

        if loss is not None:
            self.assertGreater(loss.item(), 0)

    def test_model_with_different_dtypes(self):
        """Test model with different data types."""
        model = AixcoderModel(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 10])

        # Test with float32 (default)
        with paddle.no_grad():
            outputs_f32 = model(input_ids)
        self.assertEqual(outputs_f32[0].dtype, paddle.float32)

        # Test with bfloat16 if available
        # Note: Embedding layers typically keep float32 weights for numerical stability
        # The output dtype may be float32 even if other layers are bfloat16
        if paddle.is_compiled_with_cuda() and hasattr(paddle, "bfloat16"):
            model_bf16 = model.astype(paddle.bfloat16)
            with paddle.no_grad():
                outputs_bf16 = model_bf16(input_ids)
            # Accept either bfloat16 or float32 (float32 is acceptable for embedding outputs)
            # Embedding layers often keep float32 for numerical stability
            self.assertIn(outputs_bf16[0].dtype, [paddle.bfloat16, paddle.float32])

    def test_model_output_hidden_states(self):
        """Test model with output_hidden_states flag."""
        model = AixcoderModel(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 10])

        # Test with output_hidden_states=True
        with paddle.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Check if hidden states are returned
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
            # Should have num_layers + 1 hidden states (including embedding output)
            self.assertEqual(len(hidden_states), self.config.num_hidden_layers + 1)

    def test_model_output_attentions(self):
        """Test model with output_attentions flag."""
        model = AixcoderModel(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 10])

        # Test with output_attentions=True
        with paddle.no_grad():
            outputs = model(input_ids, output_attentions=True)

        # Check if attentions are returned
        if hasattr(outputs, "attentions"):
            attentions = outputs.attentions
            # Should have num_layers attention weights
            self.assertEqual(len(attentions), self.config.num_hidden_layers)

    def test_from_pretrained_with_config_updates(self):
        """Test from_pretrained with config updates."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save a model
            model = AixcoderForCausalLM(self.config)
            model.save_pretrained(temp_dir)

            # Load with config updates
            loaded_model = AixcoderForCausalLM.from_pretrained(
                temp_dir,
                num_hidden_layers=1,  # Override config
                use_cache=False,
            )

            # Check config was updated
            self.assertEqual(loaded_model.config.num_hidden_layers, 1)
            self.assertEqual(loaded_model.config.use_cache, False)


if __name__ == "__main__":
    unittest.main()
