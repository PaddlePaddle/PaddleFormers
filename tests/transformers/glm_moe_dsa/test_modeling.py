# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import unittest

import paddle

from paddleformers.transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class GlmMoeDsaModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=1024,
        head_dim=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=False,
        pretraining_tp=1,  # TP rank used when training with megatron
        dtype="float32",
        slow_but_exact=False,
        batch_size: int = 2,
        seq_length: int = 10,
        type_sequence_label_size=2,
        activation_function="gelu",
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        use_input_mask: bool = False,
        use_labels: bool = False,
        return_dict=False,
    ):
        self.parent: GlmMoeDsaModelTest = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.pretraining_tp = pretraining_tp
        self.dtype = dtype
        self.slow_but_exact = slow_but_exact

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_sequence_label_size = type_sequence_label_size
        self.activation_function = activation_function
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout

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

    def get_config(self) -> GlmMoeDsaConfig:
        return GlmMoeDsaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            rms_norm_eps=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            dtype=self.dtype,
            hidden_act=self.activation_function,
            multi_latent_attention=True,
        )

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

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = GlmMoeDsaForCausalLM(config)
        model.eval()

        # PipelineLayer expects a dict with "input_ids" key
        input_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(input_dict)
        # The result may have different shapes depending on pipeline configuration
        # Just verify that the model can be called successfully
        self.parent.assertIsNotNone(result)


class GlmMoeDsaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    return_dict = False
    use_labels = False

    all_model_classes = (GlmMoeDsaForCausalLM,)
    all_generative_model_classes = {GlmMoeDsaForCausalLM: (None, "GlmMoeDsa")}

    @gpu_device_initializer(log_prefix="GlmMoeDsaModelTest")
    def setUp(self):
        super().setUp()

        self.model_tester = GlmMoeDsaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GlmMoeDsaConfig, vocab_size=256, hidden_size=24)

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

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_GlmMoeDsa_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    # Override test_forward_signature since PipelineLayer uses 'input' instead of 'input_ids'
    def test_forward_signature(self):
        pass

    # PipelineLayer does not support standard generation methods
    def test_generate_without_input_ids(self):
        pass

    def test_attention_outputs(self):
        pass

    def test_beam_search_generate(self):
        pass

    def test_greedy_generate(self):
        pass

    def test_group_beam_search_generate(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass

    def test_sample_generate(self):
        pass

    def test_determinism(self):
        pass

    def test_model_name_list(self):
        pass

    def test_save_load(self):
        pass

    def test_hidden_states_output(self):
        pass


class GlmMoeDsaModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    @gpu_device_initializer(log_prefix="GlmMoeDsaModelIntegrationTest")
    def setUp(self):
        pass

    def test_inference_no_attention(self):
        # Integration test will be added when pretrained model is available
        pass

    def test_inference_with_attention(self):
        # Integration test will be added when pretrained model is available
        pass


if __name__ == "__main__":
    unittest.main()
