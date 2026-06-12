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

from paddleformers.transformers import MiniMaxConfig, MiniMaxForCausalLM, MiniMaxModel
from paddleformers.transformers.auto.modeling import AutoModelForCausalLM
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MiniMaxModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        block_size=4,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent: MiniMaxModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.block_size = block_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self) -> MiniMaxConfig:
        return MiniMaxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            layer_types=["full_attention", "linear_attention"],
            block_size=self.block_size,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def create_and_check_model(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(result.loss)

    def create_and_check_training_step(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxForCausalLM(config)
        model.train()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=True)
        result.loss.backward()
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def create_and_check_auto_model(self, config: MiniMaxConfig):
        model = AutoModelForCausalLM.from_config(config)
        self.parent.assertIsInstance(model, MiniMaxForCausalLM)

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}


class MiniMaxModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = MiniMaxModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (MiniMaxModel, MiniMaxForCausalLM)
    all_generative_model_classes = {MiniMaxForCausalLM: (MiniMaxModel, "minimax")}

    @gpu_device_initializer(log_prefix="MiniMaxModelTest")
    def setUp(self):
        super().setUp()

        self.model_tester = MiniMaxModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniMaxConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_training_step(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_training_step(*config_and_inputs)

    def test_auto_model_for_causal_lm(self):
        config = self.model_tester.get_config()
        self.model_tester.create_and_check_auto_model(config)
