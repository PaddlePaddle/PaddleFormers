# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle

from paddleformers.transformers import (
    GraniteConfig,
    GraniteForCausalLM,
    GraniteModel,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class GraniteModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=128,
        shared_intermediate_size=128,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        embedding_multiplier=1.0,
        attention_multiplier=0.125,
        residual_multiplier=1.0,
        logits_scaling=1.0,
        rope_theta=10000.0,
        batch_size=2,
        seq_length=10,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        use_input_mask=False,
        use_labels=False,
        return_dict=False,
    ):
        self.parent = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.embedding_multiplier = embedding_multiplier
        self.attention_multiplier = attention_multiplier
        self.residual_multiplier = residual_multiplier
        self.logits_scaling = logits_scaling
        self.rope_theta = rope_theta
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return GraniteConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            shared_intermediate_size=self.shared_intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            hidden_act=self.hidden_act,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            tie_word_embeddings=self.tie_word_embeddings,
            embedding_multiplier=self.embedding_multiplier,
            attention_multiplier=self.attention_multiplier,
            residual_multiplier=self.residual_multiplier,
            logits_scaling=self.logits_scaling,
            rope_theta=self.rope_theta,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = GraniteModel(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = GraniteModel(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_attention_mask = model(input_ids, attention_mask=None)[0]
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def create_and_check_lm_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = GraniteForCausalLM(config)
        model.eval()
        result = model(input_ids, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_loss(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = GraniteForCausalLM(config)
        model.eval()
        labels = input_ids.clone()
        labels[:, :2] = -100
        result = model(input_ids, labels=labels, return_dict=True)
        shift_logits = result.logits[:, :-1].reshape([-1, config.vocab_size])
        shift_labels = labels[:, 1:].reshape([-1])
        valid = shift_labels != -100
        safe_labels = paddle.where(valid, shift_labels, paddle.zeros_like(shift_labels))
        selected_log_probs = paddle.take_along_axis(
            paddle.nn.functional.log_softmax(shift_logits, axis=-1), safe_labels.unsqueeze(-1), axis=-1
        ).squeeze(-1)
        expected_loss = -(selected_log_probs * paddle.cast(valid, selected_log_probs.dtype)).sum() / paddle.cast(
            valid, selected_log_probs.dtype
        ).sum()
        self.parent.assertTrue(paddle.allclose(result.loss, expected_loss, rtol=1e-5, atol=1e-5))

    def create_and_check_loss_mask(self, config, input_ids):
        model = GraniteForCausalLM(config)
        model.eval()
        labels = input_ids.clone()
        loss_mask = paddle.ones_like(input_ids)
        loss_mask[:, 2:4] = 0
        result = model(input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)

        shift_logits = result.logits[:, :-1].reshape([-1, config.vocab_size])
        shift_labels = labels[:, 1:].reshape([-1])
        valid = (shift_labels != -100) & paddle.cast(loss_mask[:, 1:].reshape([-1]), paddle.bool)
        safe_labels = paddle.where(valid, shift_labels, paddle.zeros_like(shift_labels))
        selected_log_probs = paddle.take_along_axis(
            paddle.nn.functional.log_softmax(shift_logits, axis=-1), safe_labels.unsqueeze(-1), axis=-1
        ).squeeze(-1)
        valid_float = paddle.cast(valid, selected_log_probs.dtype)
        expected_loss = -(selected_log_probs * valid_float).sum() / paddle.clip(valid_float.sum(), min=1.0)
        self.parent.assertTrue(paddle.allclose(result.loss, expected_loss, rtol=1e-5, atol=1e-5))

        empty_loss = model(
            input_ids,
            labels=labels,
            loss_mask=paddle.zeros_like(input_ids),
            return_dict=True,
        ).loss
        self.parent.assertTrue(paddle.isfinite(empty_loss).item())
        self.parent.assertEqual(empty_loss.item(), 0.0)

    def create_and_check_generate(self, config, input_ids, input_mask):
        model = GraniteForCausalLM(config)
        model.eval()
        generated = model.generate(input_ids, max_new_tokens=5, decode_strategy="greedy_search")
        if isinstance(generated, tuple):
            generated = generated[0]
        self.parent.assertEqual(generated.shape, [input_ids.shape[0], 5])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class GraniteModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = GraniteModel
    return_dict = False
    use_labels = False

    all_model_classes = (GraniteModel, GraniteForCausalLM)
    all_generative_model_classes = {GraniteForCausalLM: (GraniteModel, "granite")}
    test_missing_keys = False
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_model_parallel = False

    @gpu_device_initializer(log_prefix="GraniteModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = GraniteModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteConfig, hidden_size=37)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        max_batch_size = 2
        sequence_length = input_ids.shape[-1] // 2
        input_ids = input_ids[:max_batch_size, :sequence_length]
        attention_mask = attention_mask[:max_batch_size, :sequence_length]
        max_length = 3

        return config, input_ids, attention_mask, max_length

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)

    def test_lm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_model(*config_and_inputs)

    def test_loss(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_loss(*config_and_inputs)

    def test_loss_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_loss_mask(*config_and_inputs[:2])

    def test_generate(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate(*config_and_inputs[:3])


class GraniteGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = None
