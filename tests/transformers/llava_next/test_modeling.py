# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    floats_tensor,
    ids_tensor,
)


class LlavaNextModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=40,
        vocab_size=101,
        image_token_index=99,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_hidden_size=16,
        vision_intermediate_size=32,
        vision_num_hidden_layers=1,
        vision_num_attention_heads=4,
        image_size=16,
        patch_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_num_attention_heads = vision_num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = 3
        patch_grid = self.image_size // self.patch_size
        self.num_image_tokens = patch_grid * patch_grid + patch_grid * (patch_grid + 1)

    def get_config(self):
        text_config = {
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": "silu",
            "max_position_embeddings": 128,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "attention_bias": False,
        }
        vision_config = {
            "model_type": "siglip_vision_model",
            "hidden_size": self.vision_hidden_size,
            "intermediate_size": self.vision_intermediate_size,
            "num_hidden_layers": self.vision_num_hidden_layers,
            "num_attention_heads": self.vision_num_attention_heads,
            "num_channels": self.num_channels,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "hidden_act": "gelu_pytorch_tanh",
            "vision_use_head": False,
        }
        return LlavaNextConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_grid_pinpoints=[[self.image_size, self.image_size]],
            vision_feature_select_strategy="full",
            vision_feature_layer=-1,
            image_token_index=self.image_token_index,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [self.batch_size, 2, self.num_channels, self.image_size, self.image_size],
        )
        image_sizes = paddle.to_tensor([[self.image_size, self.image_size]] * self.batch_size, dtype=paddle.int64)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones(input_ids.shape, dtype=paddle.int64)
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, 0] = self.bos_token_id
        input_ids[:, -1] = self.eos_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_index
        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        labels[:, : 1 + self.num_image_tokens] = -100
        return config, input_ids, attention_mask, pixel_values, image_sizes, labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, pixel_values, image_sizes, labels = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "labels": labels,
        }
        return config, inputs_dict


class LlavaNextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = LlavaNextModel
    all_model_classes = (LlavaNextModel, LlavaNextForConditionalGeneration)
    all_generative_model_classes = {LlavaNextForConditionalGeneration: {LlavaNextModel, "llava_next"}}
    max_new_tokens = 3
    test_resize_embeddings = False

    @gpu_device_initializer(log_prefix="LlavaNextModelTest")
    def setUp(self):
        self.model_tester = LlavaNextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaNextConfig, has_text_modality=False)

    def _prepare_for_class(self, inputs_dict, model_class):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class)
        if model_class is LlavaNextModel:
            inputs_dict.pop("labels", None)
        return inputs_dict

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[self.model_tester.image_token_index]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        return logits_processor_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        return model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
            **inputs_dict,
        )

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        paddle.seed(0)
        return model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **self._get_logits_processor_kwargs(do_sample=True, config=model.config),
            **inputs_dict,
        )

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        return model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            trunc_input=False,
            **beam_kwargs,
            **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
            **inputs_dict,
        )

    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value[:batch_size, ...] if isinstance(value, paddle.Tensor) else value
            for key, value in inputs_dict.items()
            if key != "labels"
        }
        inputs_dict["logits_to_keep"] = 1
        config.text_config.eos_token_id = None
        config.text_config.forced_eos_token_id = None
        return config, inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

        config = LlavaNextConfig()
        self.assertEqual(config.vision_config.model_type, "siglip_vision_model")
        self.assertFalse(config.vision_config.vision_use_head)

        config = LlavaNextConfig(vision_config={"hidden_size": 16, "vision_use_head": False})
        self.assertEqual(config.vision_config.model_type, "siglip_vision_model")
        self.assertEqual(config.vision_config.hidden_size, 16)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            loaded_config = LlavaNextConfig.from_pretrained(tmpdirname)
        self.assertEqual(loaded_config.to_dict(), config.to_dict())

    def test_granite_text_tower_padding_mask_is_disabled_locally(self):
        config = self.model_tester.get_config()
        config.text_config.model_type = "granite"
        config.text_config.vocab_size = self.model_tester.vocab_size
        config.text_config.hidden_size = self.model_tester.hidden_size
        config.text_config.intermediate_size = self.model_tester.intermediate_size
        config.text_config.num_hidden_layers = self.model_tester.num_hidden_layers
        config.text_config.num_attention_heads = self.model_tester.num_attention_heads
        config.text_config.num_key_value_heads = self.model_tester.num_key_value_heads

        model = LlavaNextModel(config)
        self.assertIsNone(model.language_model.embed_tokens._padding_idx)

        from paddleformers.transformers import GraniteConfig, GraniteModel

        granite = GraniteModel(
            GraniteConfig(
                vocab_size=self.model_tester.vocab_size,
                hidden_size=self.model_tester.hidden_size,
                intermediate_size=self.model_tester.intermediate_size,
                num_hidden_layers=self.model_tester.num_hidden_layers,
                num_attention_heads=self.model_tester.num_attention_heads,
                num_key_value_heads=self.model_tester.num_key_value_heads,
                pad_token_id=0,
            )
        )
        self.assertEqual(granite.embed_tokens._padding_idx, 0)

    def test_flex_checkpoint_conversion_rules(self):
        config = self.model_tester.get_config()
        aoa = LlavaNextForConditionalGeneration._gen_aoa_config(config)["aoa_statements"]
        inv_aoa = LlavaNextForConditionalGeneration._gen_inv_aoa_config(config)["aoa_statements"]

        self.assertIn("image_newline -> model.image_newline", aoa)
        self.assertIn(
            "multi_modal_projector.linear_1.weight^T -> model.multi_modal_projector.linear_1.weight",
            aoa,
        )
        self.assertIn(
            "vision_tower.vision_model.embeddings.patch_embedding.weight -> model.vision_tower.embeddings.patch_embedding.weight",
            aoa,
        )
        self.assertIn("model.image_newline -> image_newline", inv_aoa)
        self.assertIn(
            "model.multi_modal_projector.linear_1.weight^T -> multi_modal_projector.linear_1.weight",
            inv_aoa,
        )

    def test_model_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaNextModel(config).eval()
        outputs = model(
            **self._prepare_for_class(inputs_dict, LlavaNextModel),
            return_dict=True,
        )
        self.assertEqual(
            outputs.last_hidden_state.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size],
        )
        self.assertEqual(
            outputs.image_hidden_states.shape,
            [self.model_tester.batch_size * self.model_tester.num_image_tokens, self.model_tester.hidden_size],
        )

    def test_conditional_generation_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaNextForConditionalGeneration(config).eval()
        outputs = model(
            **inputs_dict,
            logits_to_keep=1,
            return_dict=True,
        )
        self.assertEqual(
            outputs.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size],
        )
        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.image_hidden_states)

    def test_mismatching_num_image_tokens(self):
        config, input_ids, attention_mask, pixel_values, image_sizes, _ = self.model_tester.prepare_config_and_inputs()
        model = LlavaNextForConditionalGeneration(config).eval()
        input_ids[:, 1] = self.model_tester.pad_token_id
        with self.assertRaises(ValueError):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

    def test_prepare_inputs_for_generation_keeps_vision_inputs_only_first_step(self):
        config, input_ids, attention_mask, pixel_values, image_sizes, _ = self.model_tester.prepare_config_and_inputs()
        model = LlavaNextForConditionalGeneration(config).eval()

        first_step = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            is_first_iteration=True,
        )
        self.assertIs(first_step["pixel_values"], pixel_values)
        self.assertIs(first_step["image_sizes"], image_sizes)

        next_step = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=(),
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
        self.assertIsNone(next_step["pixel_values"])
        self.assertIsNone(next_step["image_sizes"])

    def test_expand_inputs_for_generation_expands_5d_pixel_values_by_sample(self):
        config, input_ids, attention_mask, pixel_values, image_sizes, _ = self.model_tester.prepare_config_and_inputs()
        model = LlavaNextForConditionalGeneration(config).eval()

        expanded_input_ids, expanded_kwargs = model.expand_inputs_for_generation(
            input_ids,
            expand_size=2,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        self.assertEqual(expanded_input_ids.shape[0], input_ids.shape[0] * 2)
        self.assertEqual(expanded_kwargs["pixel_values"].shape[0], pixel_values.shape[0] * 2)
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][0], pixel_values[0]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][1], pixel_values[0]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][2], pixel_values[1]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][3], pixel_values[1]))

    def test_expand_inputs_for_generation_expands_4d_pixel_values_by_patches(self):
        config, input_ids, attention_mask, _, image_sizes, _ = self.model_tester.prepare_config_and_inputs()
        model = LlavaNextForConditionalGeneration(config).eval()
        pixel_values_shape = [
            self.model_tester.batch_size * 2,
            self.model_tester.num_channels,
            self.model_tester.image_size,
            self.model_tester.image_size,
        ]
        pixel_values = paddle.arange(
            int(np.prod(pixel_values_shape)),
            dtype="float32",
        ).reshape(pixel_values_shape)

        _, expanded_kwargs = model.expand_inputs_for_generation(
            input_ids,
            expand_size=2,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        self.assertEqual(expanded_kwargs["pixel_values"].shape[0], pixel_values.shape[0] * 2)
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][0:2], pixel_values[0:2]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][2:4], pixel_values[0:2]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][4:6], pixel_values[2:4]))
        self.assertTrue(paddle.equal_all(expanded_kwargs["pixel_values"][6:8], pixel_values[2:4]))

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()
            beam_kwargs, _ = self._get_beam_scorer_and_kwargs(1, 1)
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)
            self.assertEqual(output_generate[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    @unittest.skip("LlavaNext generation requires input_ids and pixel_values on the first generation step.")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip("Group beam search is not compatible with current VLM implementation.")
    def test_group_beam_search_generate(self):
        pass


@unittest.skip("LlavaNext tiny checkpoint is not available yet.")
class LlavaNextIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = LlavaNextForConditionalGeneration
    hf_remote_test_model_path = "PaddleFormers/tiny-random-llava-next"
    paddlehub_remote_test_model_path = "PaddleFormers/tiny-random-llava-next"


if __name__ == "__main__":
    unittest.main()
