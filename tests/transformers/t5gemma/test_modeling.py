# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import (
    T5GemmaConfig,
    T5GemmaEncoderModel,
    T5GemmaForConditionalGeneration,
    T5GemmaForSequenceClassification,
    T5GemmaForTokenClassification,
    T5GemmaModel,
    T5GemmaModuleConfig,
)
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class T5GemmaModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        encoder_seq_length=7,
        decoder_seq_length=5,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=64,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=16,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
    ):
        self.parent: T5GemmaModelTest = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        self.seq_length = encoder_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size, dtype=paddle.int64)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        decoder_input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.encoder_seq_length])
            decoder_input_mask = random_attention_mask([self.batch_size, self.decoder_seq_length])

        sequence_labels = None
        token_labels = None
        lm_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size, dtype=paddle.int64)
            token_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.num_labels, dtype=paddle.int64)
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size, dtype=paddle.int64)

        config = self.get_config()
        return (
            config,
            input_ids,
            input_mask,
            decoder_input_ids,
            decoder_input_mask,
            sequence_labels,
            token_labels,
            lm_labels,
        )

    def get_module_config(self) -> T5GemmaModuleConfig:
        return T5GemmaModuleConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_activation=self.hidden_activation,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            sliding_window=self.sliding_window,
            final_logit_softcapping=self.final_logit_softcapping,
            attn_logit_softcapping=self.attn_logit_softcapping,
            _attn_implementation="eager",
        )

    def get_config(self) -> T5GemmaConfig:
        module_config = self.get_module_config()
        return T5GemmaConfig(
            encoder=module_config,
            decoder=module_config,
            is_encoder_decoder=True,
            vocab_size=self.vocab_size,
            tie_word_embeddings=self.tie_word_embeddings,
            _attn_implementation="eager",
        )

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            input_mask,
            decoder_input_ids,
            decoder_input_mask,
            sequence_labels,
            token_labels,
            lm_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_input_mask,
        }
        return config, inputs_dict

    def create_and_check_model(
        self,
        config: T5GemmaConfig,
        input_ids,
        input_mask,
        decoder_input_ids,
        decoder_input_mask,
        sequence_labels,
        token_labels,
        lm_labels,
    ):
        model = T5GemmaModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_mask,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.decoder_seq_length, self.hidden_size])

    def create_and_check_encoder_model(
        self,
        config: T5GemmaConfig,
        input_ids,
        input_mask,
        decoder_input_ids,
        decoder_input_mask,
        sequence_labels,
        token_labels,
        lm_labels,
    ):
        encoder_config = T5GemmaConfig(
            encoder=config.encoder,
            decoder=config.decoder,
            is_encoder_decoder=False,
            vocab_size=self.vocab_size,
            _attn_implementation="eager",
        )
        model = T5GemmaEncoderModel(encoder_config)
        model.eval()
        result = model(input_ids=input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.encoder_seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(
        self,
        config: T5GemmaConfig,
        input_ids,
        input_mask,
        decoder_input_ids,
        decoder_input_mask,
        sequence_labels,
        token_labels,
        lm_labels,
    ):
        model = T5GemmaModel(config)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_mask,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.decoder_seq_length, self.hidden_size])

    def create_and_check_lm_head_model(
        self, config, input_ids, input_mask, decoder_input_ids, decoder_input_mask, *args
    ):
        model = T5GemmaForConditionalGeneration(config)
        model.eval()

        result = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_mask,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.decoder_seq_length, self.vocab_size])

    def create_and_check_conditional_generation_loss(
        self,
        config,
        input_ids,
        input_mask,
        decoder_input_ids,
        decoder_input_mask,
        sequence_labels,
        token_labels,
        lm_labels,
    ):
        model = T5GemmaForConditionalGeneration(config)
        model.eval()
        result = model(input_ids=input_ids, attention_mask=input_mask, labels=lm_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.decoder_seq_length, self.vocab_size])
        self.parent.assertIsNotNone(result.loss)

    def check_model_position_ids(self, config, input_ids, input_mask, decoder_input_ids, decoder_input_mask, *args):
        model = T5GemmaForConditionalGeneration(config)
        model.eval()

        result_no_position_id = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_mask,
            return_dict=self.parent.return_dict,
        )
        encoder_position_ids = paddle.arange(input_ids.shape[-1], dtype="int64").expand(input_ids.shape)
        decoder_position_ids = paddle.arange(decoder_input_ids.shape[-1], dtype="int64").expand(
            decoder_input_ids.shape
        )
        result_position_id = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            position_ids=encoder_position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_mask,
            decoder_position_ids=decoder_position_ids,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


class T5GemmaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = T5GemmaModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False
    is_encoder_decoder = True

    all_model_classes = (T5GemmaModel, T5GemmaForConditionalGeneration)
    all_generative_model_classes = {T5GemmaForConditionalGeneration: (T5GemmaModel, "t5gemma")}
    pipeline_model_mapping = {
        "feature-extraction": T5GemmaEncoderModel,
        "text2text-generation": T5GemmaForConditionalGeneration,
        "text-classification": T5GemmaForSequenceClassification,
        "token-classification": T5GemmaForTokenClassification,
    }

    @gpu_device_initializer(log_prefix="T5GemmaModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = T5GemmaModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=T5GemmaConfig,
            common_properties=["vocab_size"],
            vocab_size=256,
        )

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        max_batch_size = 2
        input_ids = input_ids[:max_batch_size, :]
        attention_mask = attention_mask[:max_batch_size, :]
        max_length = 3
        return config, input_ids, attention_mask, max_length

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_t5gemma_conditional_generation_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_t5gemma_conditional_generation_loss(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_conditional_generation_loss(*config_and_inputs)

    def test_t5gemma_sequence_classification_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = self.model_tester.num_labels
        labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = T5GemmaForSequenceClassification(config)
        model.eval()
        result = model(**inputs_dict, labels=labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_t5gemma_token_classification_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = self.model_tester.num_labels
        labels = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length],
            self.model_tester.num_labels,
            dtype=paddle.int64,
        )
        model = T5GemmaForTokenClassification(config)
        model.eval()
        result = model(**inputs_dict, labels=labels, return_dict=True)
        self.assertEqual(
            result.logits.shape,
            [self.model_tester.batch_size, self.model_tester.decoder_seq_length, self.model_tester.num_labels],
        )

    def test_greedy_generate(self):
        pass

    def test_sample_generate(self):
        pass

    def test_beam_search_generate(self):
        pass

    def test_group_beam_search_generate(self):
        pass

    def test_generate_without_input_ids(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass

    def test_past_key_values_format(self):
        pass


class T5GemmaIntegrationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="T5GemmaIntegrationTest")
    def setUp(self):
        self.test_dtype = "float32"

    @unittest.skip("No published PaddleFormers tiny-random-t5gemma checkpoint is available yet.")
    def test_model_tiny_logits(self):
        input_ids = paddle.to_tensor([[2, 306, 4658, 278, 6593, 1]], dtype="int64")
        decoder_input_ids = paddle.to_tensor([[2, 1024, 312, 1]], dtype="int64")
        model = T5GemmaForConditionalGeneration.from_pretrained(
            "PaddleFormers/tiny-random-t5gemma",
            dtype=self.test_dtype,
            load_checkpoint_format="flex_checkpoint",
        )
        model.eval()
        with paddle.no_grad():
            logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True).logits
        self.assertEqual(logits.shape, [1, 4, model.config.vocab_size])


@unittest.skip("No published internal micro-random T5Gemma checkpoint is available yet.")
class T5GemmaGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "PaddleFormers/tiny-random-t5gemma"


class T5GemmaCompatibilityTest(unittest.TestCase):
    test_model_id = "hf-internal-testing/tiny-random-T5GemmaModel"

    @gpu_device_initializer(log_prefix="T5GemmaCompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import T5GemmaConfig as HFT5GemmaConfig
        from transformers import (
            T5GemmaForConditionalGeneration as HFT5GemmaForConditionalGeneration,
        )
        from transformers import T5GemmaModuleConfig as HFT5GemmaModuleConfig

        cls.torch_model_path = tempfile.TemporaryDirectory().name
        module_config = HFT5GemmaModuleConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            sliding_window=16,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            tie_word_embeddings=True,
            _attn_implementation="eager",
        )
        config = HFT5GemmaConfig(
            encoder=module_config,
            decoder=module_config,
            vocab_size=128,
            tie_word_embeddings=True,
            _attn_implementation="eager",
        )
        model = HFT5GemmaForConditionalGeneration(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_t5gemma_converter(self):
        input_ids = np.random.randint(3, 100, [1, 8])
        decoder_input_ids = np.random.randint(3, 100, [1, 5])

        import torch
        from transformers import (
            T5GemmaForConditionalGeneration as HFT5GemmaForConditionalGeneration,
        )

        torch_model = HFT5GemmaForConditionalGeneration.from_pretrained(
            self.torch_model_path,
            torch_dtype=torch.float32,
        )
        torch_model.eval()
        torch_logit = torch_model(
            input_ids=torch.tensor(input_ids),
            decoder_input_ids=torch.tensor(decoder_input_ids),
            return_dict=False,
        )[0]

        paddle_model = T5GemmaForConditionalGeneration.from_pretrained(
            self.torch_model_path,
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )
        paddle_model.eval()
        paddle_logit = paddle_model(
            input_ids=paddle.to_tensor(input_ids),
            decoder_input_ids=paddle.to_tensor(decoder_input_ids),
        )[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_t5gemma_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_ids = np.random.randint(3, 100, [1, 8])
            decoder_input_ids = np.random.randint(3, 100, [1, 5])

            import torch
            from transformers import (
                T5GemmaForConditionalGeneration as HFT5GemmaForConditionalGeneration,
            )

            torch_model = HFT5GemmaForConditionalGeneration.from_pretrained(
                self.torch_model_path,
                torch_dtype=torch.float32,
            )
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(
                input_ids=torch.tensor(input_ids),
                decoder_input_ids=torch.tensor(decoder_input_ids),
                return_dict=False,
            )[0]

            paddle_model = T5GemmaForConditionalGeneration.from_pretrained(
                tempdir,
                dtype="float32",
                load_checkpoint_format="flex_checkpoint",
            )
            paddle_model.eval()
            paddle_logit = paddle_model(
                input_ids=paddle.to_tensor(input_ids),
                decoder_input_ids=paddle.to_tensor(decoder_input_ids),
            )[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )
