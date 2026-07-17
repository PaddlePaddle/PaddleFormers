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

from paddleformers.transformers import HunyuanConfig, HunyuanForCausalLM, HunyuanModel
from paddleformers.transformers.hunyuan.modeling import HunyuanForSequenceClassification, HunyuanForTokenClassification
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import ModelTesterMixin, ids_tensor


class HunyuanModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ):
        self.parent: HunyuanModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_training = False

    def get_config(self) -> HunyuanConfig:
        return HunyuanConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=self.use_cache,
            attention_dropout=0.0,
            attention_bias=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            dtype="float32",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)
        attention_mask = paddle.ones([self.batch_size, self.seq_length], dtype=paddle.int64)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = HunyuanModel(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config, input_ids, _attention_mask):
        model = HunyuanModel(config)
        model.eval()

        # All masks are semantically causal and fully valid. This isolates the
        # accepted rank (2-D, 3-D, or 4-D) from padding semantics.
        mask_2d = paddle.ones([self.batch_size, self.seq_length], dtype=paddle.int64)
        causal_mask = paddle.tril(
            paddle.ones([self.batch_size, self.seq_length, self.seq_length], dtype=paddle.int64)
        )
        mask_4d = causal_mask.unsqueeze(1)

        result_2d = model(input_ids, attention_mask=mask_2d)[0]
        result_3d = model(input_ids, attention_mask=causal_mask)[0]
        result_4d = model(input_ids, attention_mask=mask_4d)[0]
        result_none = model(input_ids)[0]

        self.parent.assertTrue(paddle.allclose(result_2d, result_3d, atol=1e-6, rtol=1e-6))
        self.parent.assertTrue(paddle.allclose(result_2d, result_4d, atol=1e-6, rtol=1e-6))
        self.parent.assertTrue(paddle.allclose(result_2d, result_none, atol=1e-6, rtol=1e-6))

    def create_and_check_lm_head_model(self, config, input_ids, attention_mask):
        model = HunyuanForCausalLM(config)
        model.eval()

        logits_output = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(logits_output.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

        loss_output = model(input_ids, attention_mask=attention_mask, labels=input_ids, return_dict=True)
        self.parent.assertIsNotNone(loss_output.loss)
        self.parent.assertEqual(loss_output.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, attention_mask):
        model = HunyuanForCausalLM(config)
        model.eval()
        implicit_positions = model(input_ids, attention_mask=attention_mask, return_dict=True).logits
        position_ids = paddle.arange(self.seq_length, dtype="int64").expand([self.batch_size, self.seq_length])
        explicit_positions = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        ).logits
        self.parent.assertTrue(paddle.equal_all(implicit_positions, explicit_positions).item())

    def create_and_check_cache(self, config, input_ids, attention_mask):
        model = HunyuanForCausalLM(config)
        model.eval()
        prefix_length = self.seq_length - 1
        prefix_ids = input_ids[:, :prefix_length]
        prefix_mask = attention_mask[:, :prefix_length]

        cached = model(prefix_ids, attention_mask=prefix_mask, use_cache=True, return_dict=True)
        next_position_ids = paddle.full([self.batch_size, 1], prefix_length, dtype="int64")
        incremental = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            position_ids=next_position_ids,
            past_key_values=cached.past_key_values,
            use_cache=True,
            return_dict=True,
        ).logits
        full = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).logits[:, -1:]
        self.parent.assertTrue(paddle.allclose(incremental, full, atol=1e-5, rtol=1e-5))

    def create_and_check_gqa_model(self, config, input_ids, attention_mask):
        # Set KV heads before model construction; attention layers cache this
        # configuration during initialization.
        config.num_key_value_heads = max(1, config.num_attention_heads // 2)
        model = HunyuanForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertEqual(model.model.layers[0].self_attn.num_key_value_heads, config.num_key_value_heads)


class HunyuanModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = HunyuanModel
    all_model_classes = (HunyuanModel, HunyuanForCausalLM)
    all_generative_model_classes = {HunyuanForCausalLM: (HunyuanModel, "hunyuan")}
    use_test_inputs_embeds = False
    use_test_model_name_list = False
    has_attentions = False

    @gpu_device_initializer(log_prefix="HunyuanModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = HunyuanModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HunyuanConfig, vocab_size=99, hidden_size=32)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)
        return config, input_ids, attention_mask, 3

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_model_attention_mask(self):
        self.model_tester.create_and_check_model_attention_mask(*self.model_tester.prepare_config_and_inputs())

    def test_model_position_ids(self):
        self.model_tester.check_model_position_ids(*self.model_tester.prepare_config_and_inputs())

    def test_hunyuan_lm_head_model(self):
        self.model_tester.create_and_check_lm_head_model(*self.model_tester.prepare_config_and_inputs())

    def test_hunyuan_cache(self):
        self.model_tester.create_and_check_cache(*self.model_tester.prepare_config_and_inputs())

    def test_hunyuan_gqa_model(self):
        self.model_tester.create_and_check_gqa_model(*self.model_tester.prepare_config_and_inputs())

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict["attention_mask"]
            with paddle.no_grad():
                ids_output = model(input_ids, attention_mask=attention_mask, use_cache=False)[0]
                embeds_output = model(
                    inputs_embeds=model.get_input_embeddings()(input_ids),
                    attention_mask=attention_mask,
                    use_cache=False,
                )[0]
            self.assertTrue(paddle.allclose(ids_output, embeds_output, atol=1e-6, rtol=1e-6))

    def test_hunyuan_sequence_classification_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        model = HunyuanForSequenceClassification(config)
        model.eval()
        labels = ids_tensor([self.model_tester.batch_size], config.num_labels, dtype=paddle.int64)
        result = model(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            labels=labels,
            return_dict=True,
        )
        self.assertEqual(result.logits.shape, [self.model_tester.batch_size, config.num_labels])
        self.assertIsNotNone(result.loss)

    def test_hunyuan_token_classification_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        model = HunyuanForTokenClassification(config)
        model.eval()
        labels = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels, dtype=paddle.int64
        )
        result = model(
            inputs_dict["input_ids"],
            attention_mask=inputs_dict["attention_mask"],
            labels=labels,
            return_dict=True,
        )
        self.assertEqual(
            result.logits.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, config.num_labels],
        )
        self.assertIsNotNone(result.loss)

    @unittest.skip("Hunyuan does not expose output_attentions/output_hidden_states in its forward API")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Hunyuan does not expose output_attentions/output_hidden_states in its forward API")
    def test_hidden_states_output(self):
        pass
