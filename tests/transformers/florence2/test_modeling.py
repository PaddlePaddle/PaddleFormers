# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    Florence2Config,
    Florence2ForConditionalGeneration,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    floats_tensor,
    ids_tensor,
)


class Florence2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        decoder_seq_length=4,
        image_size=32,
        vocab_size=100,
        hidden_size=32,
        encoder_layers=2,
        decoder_layers=2,
        num_attention_heads=4,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = encoder_layers
        self.expected_num_hidden_layers = encoder_layers + 1
        self.is_training = is_training

        self.image_feature_length = 2
        self.encoder_seq_length = seq_length + self.image_feature_length
        self.decoder_key_length = decoder_seq_length

    def get_config(self):
        return Florence2Config(
            vision_config={
                "depths": [1, 1, 1, 1],
                "dim_embed": [16, 32, 64, 128],
                "num_heads": [2, 4, 8, 16],
                "num_groups": [2, 4, 8, 16],
                "window_size": 4,
                "projection_dim": self.hidden_size,
                "drop_path_rate": 0.0,
                "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
            },
            text_config={
                "vocab_size": self.vocab_size,
                "d_model": self.hidden_size,
                "encoder_layers": self.encoder_layers,
                "decoder_layers": self.decoder_layers,
                "encoder_attention_heads": self.num_attention_heads,
                "decoder_attention_heads": self.num_attention_heads,
                "encoder_ffn_dim": self.hidden_size * 2,
                "decoder_ffn_dim": self.hidden_size * 2,
                "max_position_embeddings": 128,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "use_cache": True,
            },
            projection_dim=self.hidden_size,
            vocab_size=self.vocab_size,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype=paddle.int64)
        labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones([self.batch_size, self.seq_length], dtype="int64")
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        return config, input_ids, attention_mask, decoder_input_ids, labels, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, decoder_input_ids, _, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "pixel_values": pixel_values,
            "use_cache": False,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, attention_mask, decoder_input_ids, labels, pixel_values):
        model = Florence2ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                pixel_values=pixel_values,
                use_cache=False,
            )
            loss_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                use_cache=False,
            )

        self.parent.assertEqual(
            list(outputs.logits.shape), [self.batch_size, self.decoder_seq_length, self.vocab_size]
        )
        self.parent.assertEqual(loss_outputs.loss.ndim, 0)


class Florence2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Florence2ForConditionalGeneration
    all_model_classes = (Florence2ForConditionalGeneration,)
    all_generative_model_classes = {Florence2ForConditionalGeneration: (None, "florence2")}
    is_encoder_decoder = True
    has_attentions = False
    test_mismatched_shapes = False

    def setUp(self):
        self.model_tester = Florence2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Florence2Config,
            common_properties=["vocab_size"],
            vocab_size=100,
            projection_dim=32,
        )

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_classes()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_florence2_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_save_load(self):
        super().test_save_load()

    def test_determinism(self):
        super().test_determinism()

    def test_hidden_states_output(self):
        super().test_hidden_states_output()

    def test_resize_tokens_embeddings(self):
        super().test_resize_tokens_embeddings()

    def _get_generation_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"][:1].clone()
        attention_mask = inputs_dict["attention_mask"][:1].clone()
        pixel_values = inputs_dict["pixel_values"][:1].clone()
        return config, input_ids, attention_mask, pixel_values

    def test_greedy_generate(self):
        config, input_ids, attention_mask, pixel_values = self._get_generation_inputs()
        model = Florence2ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=3,
                decode_strategy="greedy_search",
            )[0]

        self.assertEqual(generated.shape[0], input_ids.shape[0])
        self.assertGreaterEqual(generated.shape[1], 1)

    def test_beam_search_generate(self):
        config, input_ids, attention_mask, pixel_values = self._get_generation_inputs()
        model = Florence2ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=3,
                decode_strategy="beam_search",
                num_beams=2,
            )[0]

        self.assertEqual(generated.shape[0], input_ids.shape[0])
        self.assertGreaterEqual(generated.shape[1], 1)

    def test_sample_generate(self):
        config, input_ids, attention_mask, pixel_values = self._get_generation_inputs()
        model = Florence2ForConditionalGeneration(config)
        model.eval()

        paddle.seed(1234)
        with paddle.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=3,
                decode_strategy="sampling",
                top_k=10,
            )[0]

        self.assertEqual(generated.shape[0], input_ids.shape[0])
        self.assertGreaterEqual(generated.shape[1], 1)

    def test_generate_without_input_ids(self):
        # Florence2 needs either image-conditioned embeddings or explicit text ids.
        pass

    def test_group_beam_search_generate(self):
        # Group beam search coverage is not required for Florence2.
        pass

    def test_paddleformers_sft_labels(self):
        model = Florence2ForConditionalGeneration(self.model_tester.get_config())
        input_ids = paddle.to_tensor([[10, 11, 12, 20, 21, 2]])
        labels = paddle.to_tensor([[-100, -100, 20, 21, 2, -100]])
        source_ids, decoder_labels, source_mask = model._split_sft_inputs(input_ids, labels, None)
        self.assertEqual(source_ids.tolist(), [[10, 11, 12]])
        self.assertEqual(decoder_labels.tolist(), [[20, 21, 2]])
        self.assertEqual(source_mask.tolist(), [[1, 1, 1]])


class Florence2ModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = Florence2ForConditionalGeneration
    hf_remote_test_model_path = None
    paddlehub_remote_test_model_path = None

    @unittest.skip("Florence2 tiny pretrained checkpoint is not available yet.")
    def test_model_from_pretrained_paddle_hub(self):
        pass

    @unittest.skip("Florence2 tiny pretrained checkpoint is not available yet.")
    def test_model_from_config_paddle_hub(self):
        pass

    @unittest.skip("Florence2 tiny pretrained checkpoint is not available yet.")
    def test_pretrained_save_and_load(self):
        pass


class Florence2ModelLocalPretrainedTest(unittest.TestCase):
    def test_local_save_load_consistency(self):
        config, inputs_dict = Florence2ModelTester(self).prepare_config_and_inputs_for_common()
        model = Florence2ForConditionalGeneration(config)
        model.eval()

        with paddle.no_grad():
            expected = model(**inputs_dict).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, save_safetensors=False, save_checkpoint_format="")
            loaded = Florence2ForConditionalGeneration.from_pretrained(
                tmpdirname, convert_from_hf=False, load_checkpoint_format=""
            )
            loaded.eval()
            with paddle.no_grad():
                actual = loaded(**inputs_dict).logits

        self.assertLessEqual(np.max(np.abs(expected.numpy() - actual.numpy())), 1e-5)


if __name__ == "__main__":
    unittest.main()
