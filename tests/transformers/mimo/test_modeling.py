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
    MiMoConfig,
    MiMoForCausalLM,
    MiMoForCausalLMDeprecated,
    MiMoModel,
)
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MiMoModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_nextn_predict_layers=1,
        is_training=True,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.is_training = is_training
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def get_config(self):
        return MiMoConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            use_cache=self.use_cache,
            use_sliding_window=False,
            max_window_layers=self.num_hidden_layers,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return self.get_config(), input_ids, input_mask, token_labels

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = MiMoModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])
        self.parent.assertEqual(len(model.mtp_layers), self.num_nextn_predict_layers)

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, token_labels):
        model = MiMoForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}


class MiMoModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = MiMoModel
    return_dict = False
    use_test_inputs_embeds = True

    all_model_classes = (MiMoForCausalLM, MiMoForCausalLMDeprecated)
    all_generative_model_classes = {MiMoForCausalLM: (MiMoModel, "mimo")}
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = MiMoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiMoConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @gpu_device_initializer()
    def test_model(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, input_mask, token_labels)

    @gpu_device_initializer()
    def test_for_causal_lm(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(config, input_ids, input_mask, token_labels)


if __name__ == "__main__":
    unittest.main()
