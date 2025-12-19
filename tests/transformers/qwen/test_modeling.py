# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from parameterized import parameterized_class

from paddleformers.transformers.qwen.configuration import QWenConfig
from paddleformers.transformers.qwen.modeling import QWenForCausalLM, QWenModel
from tests.transformers.test_generation_utils import GenerationTesterMixin

from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


class QWenModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        type_sequence_label_size=2,
        use_relative_position=True,
        num_labels=3,
        num_choices=4,
        num_classes=3,
        scope=None,
        multi_query=True,
        bias=False,
        parallel_attn=True,
        output_attentions=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_sequence_label_size = type_sequence_label_size
        self.use_relative_position = use_relative_position
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.multi_query = multi_query
        self.bias = bias
        self.parallel_attn = parallel_attn
        self.output_attentions = output_attentions

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64")

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None

        if self.parent and self.parent.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

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

    def get_config(self):
        return QWenConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            use_relative_position=self.use_relative_position,
            num_class=self.num_classes,
            num_labels=self.num_labels,
            num_choices=self.num_choices,
            multi_query=self.multi_query,
            bias=self.bias,
            parallel_attn=self.parallel_attn,
            output_attentions=self.output_attentions,
            seq_length=self.seq_length,
            kv_channels=self.hidden_size // self.num_attention_heads,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = QWenModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            return_dict=self.parent.return_dict,
        )
        result = model(input_ids, use_cache=True, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        result = model(input_ids, use_cache=True, output_attentions=True, return_dict=self.parent.return_dict)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        result = model(
            input_ids,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = QWenForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids,
            return_dict=self.parent.return_dict,
        )
        self.parent.assertEqual(result[0].shape, [])
        self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])


@parameterized_class(
    ("return_dict", "use_labels"),
    [
        [False, False],
        [False, True],
        [True, False],
        [True, True],
    ],
)
class QWenModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = QWenModel
    return_dict: bool = False
    use_labels: bool = False
    use_test_inputs_embeds: bool = True
    use_test_model_name_list = False

    all_model_classes = (
        QWenModel,
        QWenForCausalLM,
    )
    all_generative_model_classes = {QWenForCausalLM: (QWenModel, "qwen")}

    def setUp(self):
        self.model_tester = QWenModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_model(*config_and_inputs)
