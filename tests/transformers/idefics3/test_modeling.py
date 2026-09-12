# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import copy
import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
)


class Idefics3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=3,
        is_training=False,
        image_token_id=4,
        pad_token_id=0,
        hidden_size=16,
        num_hidden_layers=1,
        vocab_size=32,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size

    def get_config(self):
        return Idefics3Config(
            image_token_id=self.image_token_id,
            pad_token_id=self.pad_token_id,
            scale_factor=2,
            text_config={
                "model_type": "llama",
                "hidden_size": self.hidden_size,
                "intermediate_size": 32,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": self.pad_token_id,
                "fuse_rms_norm": False,
            },
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        image_hidden_states = paddle.zeros([self.batch_size, 1, config.text_config.hidden_size], dtype="float32")
        return config, image_hidden_states

    def prepare_config_and_inputs_for_common(self):
        config, image_hidden_states = self.prepare_config_and_inputs()
        input_ids = paddle.to_tensor([[1, self.image_token_id, 5]], dtype="int64").expand([self.batch_size, -1])
        labels = paddle.to_tensor([[-100, -100, 5]], dtype="int64").expand([self.batch_size, -1])
        attention_mask = paddle.ones(input_ids.shape, dtype="int64")

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_hidden_states": image_hidden_states,
            "labels": labels,
        }
        return config, inputs_dict


class Idefics3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Idefics3ForConditionalGeneration`.
    """

    base_model_class = Idefics3Model
    all_model_classes = (Idefics3Model, Idefics3ForConditionalGeneration)
    all_generative_model_classes = {Idefics3ForConditionalGeneration: {Idefics3Model, "idefics3"}}
    max_new_tokens = 3
    has_attentions = False

    @gpu_device_initializer(log_prefix="Idefics3ModelTest", gpu_id=0)
    def setUp(self):
        super().setUp()
        self.model_tester = Idefics3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Idefics3Config, has_text_modality=False)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 2]],
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
        if config is not None and config.image_token_id < config.text_config.vocab_size:
            logits_processor_kwargs["bad_words_ids"].append([config.image_token_id])
        return logits_processor_kwargs

    def prepare_config_and_inputs_for_generate(self, batch_size=1):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {k: v[:batch_size, ...] if isinstance(v, paddle.Tensor) else v for k, v in inputs_dict.items()}
        inputs_dict.pop("labels", None)
        config.text_config.eos_token_id = None
        config.text_config.forced_eos_token_id = None
        return config, inputs_dict

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.check_config_can_be_init_without_params()
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_config = Idefics3Config(**config.to_dict())
        self.assertEqual(base_config.vocab_size, base_config.text_config.vocab_size)
        self.assertEqual(base_config.vision_config.model_type, "idefics3_vision")
        self.assertEqual(base_config.text_config.model_type, "llama")

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class).eval()
            with paddle.no_grad():
                expected = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_safetensors=False, save_checkpoint_format="")
                reloaded = model_class.from_pretrained(
                    tmpdirname,
                    convert_from_hf=False,
                    load_checkpoint_format="",
                    key_mapping={r"^(.*)$": r"\1"},
                ).eval()

                with paddle.no_grad():
                    actual = reloaded(**self._prepare_for_class(inputs_dict, model_class))[0]

            expected = expected.numpy()
            actual = actual.numpy()
            expected[np.isnan(expected)] = 0
            actual[np.isnan(actual)] = 0
            self.assertLessEqual(np.max(np.abs(expected - actual)), 1e-5)

    def test_resize_tokens_embeddings(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        no_label_inputs = {k: v for k, v in inputs_dict.items() if k != "labels"}

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = self._make_model_instance(config, model_class).eval()
            old_vocab_size = config.text_config.vocab_size

            model_embed = model.resize_token_embeddings(old_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, old_vocab_size + 10)
            self.assertEqual(model_embed.weight.shape[0], old_vocab_size + 10)

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(no_label_inputs, model_class))
            self.assertIsInstance(outputs[0], paddle.Tensor)

            model_embed = model.resize_token_embeddings(old_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, old_vocab_size - 15)
            self.assertEqual(model_embed.weight.shape[0], old_vocab_size - 15)

            # Input ids should be clamped to the maximum size of the reduced vocabulary
            no_label_inputs["input_ids"] = paddle.clip(no_label_inputs["input_ids"], max=old_vocab_size - 15 - 1)

            with paddle.no_grad():
                outputs = model(**self._prepare_for_class(no_label_inputs, model_class))
            self.assertIsInstance(outputs[0], paddle.Tensor)

    def test_text_config(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        base_config_dict = config.to_dict()
        base_config = Idefics3Config(**base_config_dict)

        self.assertEqual(base_config.vocab_size, base_config.text_config.vocab_size)
        base_config.vocab_size = 55
        self.assertEqual(base_config.vocab_size, 55)
        self.assertEqual(base_config.text_config.vocab_size, 55)

    def test_model_forward_with_precomputed_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Idefics3Model(config)
        model.eval()

        with paddle.no_grad():
            outputs = model(
                input_ids=inputs_dict["input_ids"],
                attention_mask=inputs_dict["attention_mask"],
                image_hidden_states=inputs_dict["image_hidden_states"],
            )

        self.assertEqual(
            outputs.last_hidden_state.shape, [self.model_tester.batch_size, 3, config.text_config.hidden_size]
        )
        self.assertEqual(
            outputs.image_hidden_states.shape, [self.model_tester.batch_size, 1, config.text_config.hidden_size]
        )

    def test_conditional_generation_forward_with_labels(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Idefics3ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            outputs = model(**inputs_dict)

        self.assertEqual(outputs.logits.shape, [self.model_tester.batch_size, 3, config.text_config.vocab_size])
        self.assertIsNotNone(outputs.loss)

    def test_image_token_count_mismatch_raises(self):
        config = self.model_tester.get_config()
        model = Idefics3Model(config)

        input_ids = paddle.to_tensor(
            [[1, self.model_tester.image_token_id, self.model_tester.image_token_id]], dtype="int64"
        )
        image_hidden_states = paddle.zeros([1, 1, config.text_config.hidden_size], dtype="float32")

        with self.assertRaises(ValueError):
            model(input_ids=input_ids, image_hidden_states=image_hidden_states)

    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()

            outputs = model.generate(
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.max_new_tokens,
                return_dict_in_generate=False,
                use_cache=True,
                trunc_input=False,
                **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
                **inputs_dict,
            )

            self.assertEqual(outputs[0].shape[0], inputs_dict["input_ids"].shape[0])
            self.assertEqual(outputs[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()

            outputs = model.generate(
                do_sample=False,
                decode_strategy="beam_search",
                num_beams=2,
                num_return_sequences=2,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.max_new_tokens,
                return_dict_in_generate=False,
                use_cache=True,
                trunc_input=False,
                **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
                **inputs_dict,
            )

            self.assertEqual(outputs[0].shape[0], inputs_dict["input_ids"].shape[0] * 2)
            self.assertEqual(outputs[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_group_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()

            outputs = model.generate(
                do_sample=False,
                decode_strategy="beam_search",
                num_beams=2,
                num_beam_groups=2,
                num_return_sequences=2,
                diversity_rate=2.0,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.max_new_tokens,
                return_dict_in_generate=False,
                use_cache=True,
                trunc_input=False,
                **self._get_logits_processor_kwargs(do_sample=False, config=model.config),
                **inputs_dict,
            )

            self.assertEqual(outputs[0].shape[0], inputs_dict["input_ids"].shape[0] * 2)
            self.assertEqual(outputs[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)

    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).eval()

            paddle.seed(0)
            outputs = model.generate(
                do_sample=True,
                decode_strategy="sampling",
                num_beams=1,
                num_return_sequences=2,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.max_new_tokens,
                return_dict_in_generate=False,
                use_cache=True,
                trunc_input=False,
                **self._get_logits_processor_kwargs(do_sample=True, config=model.config),
                **inputs_dict,
            )

            self.assertEqual(outputs[0].shape[0], inputs_dict["input_ids"].shape[0] * 2)
            self.assertEqual(outputs[0].shape[1], inputs_dict["input_ids"].shape[1] + self.max_new_tokens)


@unittest.skip("Idefics3 tiny checkpoint is not available yet.")
class Idefics3IntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = Idefics3ForConditionalGeneration
    hf_remote_test_model_path = "PaddleFormers/tiny-random-idefics3"
    paddlehub_remote_test_model_path = "PaddleFormers/tiny-random-idefics3"

    def test_model_from_pretrained_hf_hub(self):
        super().test_model_from_pretrained_hf_hub()

    def test_model_from_pretrained_paddle_hub(self):
        super().test_model_from_pretrained_paddle_hub()

    def test_model_from_config_paddle_hub(self):
        super().test_model_from_config_paddle_hub()

    def test_model_from_pretrained_with_cache_dir(self):
        super().test_model_from_pretrained_with_cache_dir()

    def test_pretrained_save_and_load(self):
        super().test_pretrained_save_and_load()


if __name__ == "__main__":
    unittest.main()
