# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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

import gc
import unittest

import paddle

from paddleformers.transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
)
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class Qwen2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        max_window_layers=3,
        use_sliding_window=True,
        sliding_window=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
    ):
        self.parent: Qwen2ModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scope = scope

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> Qwen2Config:
        return Qwen2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            max_window_layers=self.max_window_layers,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model with Llama->Qwen2
    def create_and_check_model(
        self, config: Qwen2Config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Qwen2Model(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_model_as_decoder with Llama->Qwen2
    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = Qwen2Model(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_for_causal_lm with Llama->Qwen2
    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = Qwen2ForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.create_and_check_decoder_model_past_large_inputs with Llama->Qwen2
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = Qwen2ForCausalLM(config=config)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.concat([input_ids, next_tokens], dim=-1)
        next_attention_mask = paddle.concat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Qwen2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Qwen2Model
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Qwen2Model, Qwen2ForCausalLM)
    all_generative_model_classes = {Qwen2ForCausalLM: {Qwen2Model, "qwen2"}}
    pipeline_model_mapping = {
        "feature-extraction": Qwen2Model,
        "text-classification": Qwen2ForSequenceClassification,
        "token-classification": Qwen2ForTokenClassification,
        "text-generation": Qwen2ForCausalLM,
        "zero-shot": Qwen2ForSequenceClassification,
    }

    def setUp(self):
        super().setUp()
        self.model_tester = Qwen2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2Config, hidden_size=37)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_Qwen2_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = Qwen2ForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_Qwen2_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = Qwen2ForSequenceClassification(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    def test_Qwen2_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(paddle.float32)
        model = Qwen2ForSequenceClassification(config)

        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels, return_dict=True)
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, self.model_tester.num_labels])

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTest.test_llama_token_classification_model with Llama->Qwen2,llama->Qwen2
    def test_Qwen2_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = paddle.not_equal(input_ids, paddle.ones_like(input_ids))
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = Qwen2ForTokenClassification(config=config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels, return_dict=True)
        self.assertEqual(
            result.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels],
        )

    @unittest.skip("Qwen2 buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("Qwen2 uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass


class Qwen2IntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen2ForCausalLM.from_pretrained("__internal_testing__/tiny-random-qwen2", dtype="float32")
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model(input_ids, return_dict=True).logits
        # Expected mean on dim = -1

        EXPECTED_MEAN = paddle.to_tensor(
            [[0.00008947, -0.00001425, 0.00035553, -0.00003941, 0.00068506, 0.00005345, 0.00060015, 0.00081522]]
        )
        paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-6, rtol=1e-6)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([0.26874602, 0.51205510, -0.00591420, 0.05831886, 0.18694536,
                                           0.04331543, 0.09623559, -0.10191102, 0.07565773, 0.13765232,
                                           0.03041580, 0.42183253, 0.40434697, 0.06868516, 0.02637704,
                                           -0.13485563, -0.01698003, 0.21499887, -0.03826120, 0.16291623,
                                           -0.27641180, -0.36975217, 0.34660554, -0.52724630, -0.41814676,
                                           0.00843160, -0.29562786, -0.07467390, 0.40502766, 0.13571614])  # fmt: skip
        print(out[0, 0, :30])
        paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-6, rtol=1e-6)

        del model
        paddle.device.cuda.empty_cache()
        gc.collect()
