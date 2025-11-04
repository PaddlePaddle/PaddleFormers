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

"""Final tests to reach 80% code coverage for AiXcoder."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.transformers.aixcoder.configuration import AixcoderConfig
from paddleformers.transformers.aixcoder.modeling import (
    AixcoderForCausalLM,
    AixcoderModel,
    AixcoderPretrainedModel,
    CausalLMOutputWithCrossAttentions,
)


class AixcoderFinalCoverageTest(unittest.TestCase):
    """Final tests to reach 80% coverage."""

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

    def test_embedding_getters_setters(self):
        """Test input/output embedding getters and setters."""
        model = AixcoderForCausalLM(self.config)

        # Test get_input_embeddings
        input_embeddings = model.get_input_embeddings()
        self.assertIsNotNone(input_embeddings)

        # Test set_input_embeddings
        new_embeddings = paddle.nn.Embedding(1000, 128)
        model.set_input_embeddings(new_embeddings)
        self.assertEqual(model.get_input_embeddings(), new_embeddings)

        # Test get_output_embeddings
        output_embeddings = model.get_output_embeddings()
        self.assertIsNotNone(output_embeddings)

        # Test set_output_embeddings
        new_output = paddle.nn.Linear(128, 1000)
        model.set_output_embeddings(new_output)
        self.assertEqual(model.get_output_embeddings(), new_output)

    def test_decoder_getters_setters(self):
        """Test decoder getter and setter."""
        model = AixcoderForCausalLM(self.config)

        # Test get_decoder
        decoder = model.get_decoder()
        self.assertIsNotNone(decoder)
        self.assertIsInstance(decoder, AixcoderModel)

        # Test set_decoder
        new_decoder = AixcoderModel(self.config)
        model.set_decoder(new_decoder)
        self.assertEqual(model.get_decoder(), new_decoder)

    def test_update_model_kwargs_for_generation_with_position_ids(self):
        """Test update_model_kwargs_for_generation with position_ids."""
        model = AixcoderForCausalLM(self.config)

        # Create model_kwargs with position_ids
        model_kwargs = {
            "position_ids": paddle.to_tensor([[0, 1, 2, 3, 4]]),
            "attention_mask": paddle.ones([1, 5]),
        }

        # Create mock outputs
        outputs = (paddle.randn([1, 5, 1000]),)

        # Update model_kwargs
        updated = model.update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

        # Check position_ids was updated
        self.assertIn("position_ids", updated)
        # Should have added one more position
        self.assertEqual(updated["position_ids"].shape[1], 6)

        # Check attention_mask was updated
        self.assertEqual(updated["attention_mask"].shape[1], 6)

    def test_update_model_kwargs_with_causal_lm_output(self):
        """Test update_model_kwargs with CausalLMOutputWithCrossAttentions."""
        model = AixcoderForCausalLM(self.config)

        # Create mock outputs with past_key_values (as tuple of tuples)
        mock_past = tuple((paddle.randn([1, 4, 5, 32]), paddle.randn([1, 4, 5, 32])) for _ in range(2))
        outputs = CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=paddle.randn([1, 5, 1000]),
            past_key_values=mock_past,
        )

        model_kwargs = {"attention_mask": paddle.ones([1, 5])}

        # Update model_kwargs
        updated = model.update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

        # Check past_key_values was updated
        self.assertIn("past_key_values", updated)
        self.assertEqual(updated["past_key_values"], mock_past)

    def test_prepare_inputs_for_generation_with_inputs_embeds(self):
        """Test prepare_inputs_for_generation with inputs_embeds."""
        model = AixcoderForCausalLM(self.config)

        # Test with inputs_embeds instead of input_ids
        inputs_embeds = paddle.randn([1, 5, 128])

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
        )

        # Should use inputs_embeds
        self.assertIn("inputs_embeds", model_inputs)
        self.assertEqual(model_inputs["inputs_embeds"].shape, inputs_embeds.shape)

    def test_get_tensor_parallel_split_mappings(self):
        """Test the static method _get_tensor_parallel_split_mappings."""
        # Test with fuse_qkv configuration
        config = AixcoderConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            fuse_attention_qkv=True,
            fuse_attention_ffn=True,
            tensor_parallel_degree=2,
        )

        # Mock the split function - it's imported from conversion_utils inside the method
        with patch("paddleformers.transformers.conversion_utils.split_or_merge_func") as mock_split:
            mock_fn = MagicMock()
            mock_split.return_value = mock_fn

            # Call the method
            mappings = AixcoderPretrainedModel._get_tensor_parallel_mappings(config, is_split=True)

            # Check mappings were created
            self.assertIsInstance(mappings, dict)

            # Should have mappings for fused layers
            # Check that mappings were created (actual content depends on implementation)
            # We're mainly testing that the method runs without errors
            self.assertGreater(len(mappings), 0)

    def test_from_pretrained_with_fused_layers(self):
        """Test from_pretrained with fused layer configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a model with fused configurations
            config = AixcoderConfig(
                hidden_size=128,
                num_hidden_layers=1,
                num_attention_heads=4,
                fuse_attention_qkv=True,
                fuse_attention_ffn=True,
            )
            model = AixcoderForCausalLM(config)
            model.save_pretrained(temp_dir)

            # Load with merged weights
            # merge_tensor_parallel is a classmethod in AixcoderPretrainedModel (inherited from PretrainedModel)
            with patch.object(AixcoderPretrainedModel, "merge_tensor_parallel"):
                loaded = AixcoderForCausalLM.from_pretrained(
                    temp_dir,
                    tensor_parallel_degree=1,
                )

                # Model should load successfully
                self.assertIsNotNone(loaded)

    def test_from_pretrained_merge_tensor_parallel(self):
        """Test from_pretrained with tensor parallel merging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config for testing merge path
            config = AixcoderConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                fuse_attention_qkv=True,
                fuse_attention_ffn=True,
            )
            model = AixcoderForCausalLM(config)
            model.save_pretrained(temp_dir)

            # Mock the merge function to trigger the merge path
            # merge_tensor_parallel is a classmethod in AixcoderPretrainedModel (inherited from PretrainedModel)
            with patch.object(AixcoderPretrainedModel, "merge_tensor_parallel") as mock_merge:
                # Make merge_tensor_parallel return some mock weights
                mock_state_dict = model.state_dict()

                # Add some fused keys to trigger the conversion
                mock_state_dict["aixcoder.layers.0.self_attn.qkv_proj.weight"] = paddle.randn([384, 128])
                mock_state_dict["aixcoder.layers.0.mlp.gate_up_fused_proj.weight"] = paddle.randn([512, 128])

                mock_merge.return_value = mock_state_dict

                # Load with specific config to trigger merge
                loaded = AixcoderForCausalLM.from_pretrained(
                    temp_dir,
                    tensor_parallel_degree=-1,  # This might trigger merge
                )

                self.assertIsNotNone(loaded)

    def test_forward_with_all_none_inputs(self):
        """Test forward pass with minimal inputs."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        # Just input_ids, everything else None/default
        input_ids = paddle.ones([1, 1], dtype="int64")

        with paddle.no_grad():
            outputs = model(
                input_ids,
                position_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                past_key_values=None,
                output_hidden_states=None,
                output_attentions=None,
                return_dict=None,
            )

        self.assertIsNotNone(outputs)

    def test_forward_with_return_dict_false(self):
        """Test forward with return_dict=False."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 5])

        with paddle.no_grad():
            outputs = model(input_ids, return_dict=False)

        # Should return tuple
        self.assertIsInstance(outputs, tuple)
        # First element should be logits
        self.assertEqual(outputs[0].shape[2], self.config.vocab_size)

    def test_model_with_inputs_embeds(self):
        """Test model forward with inputs_embeds instead of input_ids."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        # Use inputs_embeds directly
        inputs_embeds = paddle.randn([1, 5, 128])

        with paddle.no_grad():
            outputs = model(inputs_embeds=inputs_embeds)

        self.assertIsNotNone(outputs)
        if hasattr(outputs, "logits"):
            self.assertEqual(outputs.logits.shape[1], 5)
        else:
            self.assertEqual(outputs[0].shape[1], 5)

    def test_prepare_inputs_with_cache_length(self):
        """Test prepare_inputs_for_generation with cache."""
        model = AixcoderForCausalLM(self.config)

        # Create input_ids
        input_ids = paddle.randint(0, 1000, [1, 10])

        # Create past_key_values with cache_length
        cache_length = 5
        past_key_values = [
            (paddle.randn([1, 4, cache_length, 32]), paddle.randn([1, 4, cache_length, 32])) for _ in range(2)
        ]

        # Prepare inputs with cache
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # With past_key_values, should only use last token
        self.assertEqual(model_inputs["input_ids"].shape[1], 1)

        # Check past_key_values is passed through
        self.assertEqual(model_inputs["past_key_values"], past_key_values)
        self.assertTrue(model_inputs["use_cache"])

    def test_prepare_inputs_with_inputs_embeds_and_no_past(self):
        """Test prepare_inputs_for_generation with inputs_embeds and no past_key_values."""
        model = AixcoderForCausalLM(self.config)

        # Test with inputs_embeds and no past_key_values (should use inputs_embeds path)
        inputs_embeds = paddle.randn([1, 5, 128])
        input_ids = paddle.randint(0, 1000, [1, 5])

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
        )

        # Should use inputs_embeds when past_key_values is None
        self.assertIn("inputs_embeds", model_inputs)
        self.assertEqual(model_inputs["inputs_embeds"].shape, inputs_embeds.shape)

    def test_get_model_inputs_spec(self):
        """Test _get_model_inputs_spec method."""
        model = AixcoderForCausalLM(self.config)

        # Call the method
        spec = model._get_model_inputs_spec("float32")

        # Check spec is a dictionary with expected keys
        self.assertIsInstance(spec, dict)
        self.assertIn("input_ids", spec)
        self.assertIn("attention_mask", spec)
        self.assertIn("position_ids", spec)

    def test_forward_with_return_dict_true(self):
        """Test forward with return_dict=True."""
        model = AixcoderForCausalLM(self.config)
        model.eval()

        input_ids = paddle.randint(0, 1000, [1, 5])

        with paddle.no_grad():
            outputs = model(input_ids, return_dict=True)

        # Should return CausalLMOutputWithCrossAttentions object
        self.assertIsInstance(outputs, CausalLMOutputWithCrossAttentions)
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[2], self.config.vocab_size)


if __name__ == "__main__":
    unittest.main()
