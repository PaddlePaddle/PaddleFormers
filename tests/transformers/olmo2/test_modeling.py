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

import tempfile
import unittest

import paddle

from paddleformers.transformers import Olmo2Config, Olmo2ForCausalLM, Olmo2Model
from tests.testing_utils import gpu_device_initializer, slow
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class Olmo2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=10,
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_input_mask=True,
        use_labels=False,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        return_dict=False,
    ):
        self.parent: Olmo2ModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.return_dict = return_dict

    def get_config(self):
        return Olmo2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=True,
            attention_dropout=self.attention_dropout,
            _attn_implementation="eager",
            fuse_rms_norm=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length], dtype="int64")

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        config = self.get_config()
        return config, input_ids, input_mask, None, token_labels, None

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _, _, _ = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}

    def create_and_check_model(self, config, input_ids, input_mask, *args):
        model = Olmo2Model(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config, input_ids, input_mask, *args):
        model = Olmo2Model(config)
        model.eval()
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length], dtype="int64")
        result_2d = model(input_ids, attention_mask=attn_mask_2d)[0]
        batch, seq_length = input_ids.shape
        causal_mask = paddle.tril(paddle.ones((batch, seq_length, seq_length), dtype=attn_mask_2d.dtype))
        attn_mask_3d = causal_mask & attn_mask_2d.unsqueeze(-1)
        result_3d = model(input_ids, attention_mask=attn_mask_3d)[0]
        attn_mask_4d = attn_mask_3d.unsqueeze(1)
        result_4d = model(input_ids, attention_mask=attn_mask_4d)[0]
        result_no_mask = model(input_ids, attention_mask=None)[0]

        self.parent.assertTrue(
            (result_2d[attn_mask_2d.astype("bool")] == result_3d[attn_mask_2d.astype("bool")]).all()
        )
        self.parent.assertTrue(
            (result_2d[attn_mask_2d.astype("bool")] == result_4d[attn_mask_2d.astype("bool")]).all()
        )
        self.parent.assertTrue(
            (result_2d[attn_mask_2d.astype("bool")] == result_no_mask[attn_mask_2d.astype("bool")]).all()
        )

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = Olmo2ForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_gqa_model(self, config, input_ids, input_mask, *args):
        config.num_key_value_heads = self.num_attention_heads
        model = Olmo2ForCausalLM(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = Olmo2ForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        position_ids = paddle.arange(input_ids.shape[1], dtype="int64").expand(input_ids.shape)
        result_position_id = model(
            input_ids,
            attention_mask=input_mask,
            position_ids=position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


class Olmo2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Olmo2Model
    return_dict = False
    use_labels = False
    test_resize_embeddings = False
    has_attentions = False

    all_model_classes = (Olmo2Model, Olmo2ForCausalLM)
    all_generative_model_classes = {Olmo2ForCausalLM: (Olmo2Model, "olmo2")}

    @gpu_device_initializer(log_prefix="Olmo2ModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = Olmo2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Olmo2Config, hidden_size=24, vocab_size=128)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        input_ids = input_ids[:2, : input_ids.shape[-1] // 2].clone()
        attention_mask = attention_mask[:2, : attention_mask.shape[-1] // 2].unsqueeze([1, 2])
        attention_mask = attention_mask * attention_mask.transpose([0, 1, 3, 2])
        max_length = 3

        if config.eos_token_id or config.pad_token_id:
            config["pad_token_id"] = config["eos_token_id"]
        return config, input_ids, attention_mask, max_length

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_model_attention_mask(self):
        self.model_tester.create_and_check_model_attention_mask(*self.model_tester.prepare_config_and_inputs())

    def test_model_position_ids(self):
        self.model_tester.check_model_position_ids(*self.model_tester.prepare_config_and_inputs())

    def test_generate_without_input_ids(self):
        pass

    def test_olmo2_lm_head_model(self):
        self.model_tester.create_and_check_lm_head_model(*self.model_tester.prepare_config_and_inputs())

    def test_olmo2_gqa_model(self):
        self.model_tester.create_and_check_gqa_model(*self.model_tester.prepare_config_and_inputs())


class Olmo2ModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = Olmo2Model

    @gpu_device_initializer(log_prefix="Olmo2ModelIntegrationTest")
    def setUp(self):
        pass

    @slow
    def test_model_from_hf_tiny_random(self):
        model = Olmo2ForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
            download_hub="huggingface",
            convert_from_hf=True,
            load_checkpoint_format="",
        )
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        with paddle.no_grad():
            outputs = model(input_ids, return_dict=True)
        self.assertEqual(list(outputs.logits.shape), [1, 4, model.config.vocab_size])

    @slow
    def test_inference_from_local_pretrained(self):
        config = Olmo2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_cache=False,
            attention_dropout=0.0,
            _attn_implementation="eager",
            fuse_rms_norm=False,
            pad_token_id=0,
            eos_token_id=2,
        )
        model = Olmo2Model(config).eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")

        with paddle.no_grad():
            ref_output = model(input_ids)[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, save_to_hf=False, save_checkpoint_format="")
            reloaded = Olmo2Model.from_pretrained(tmp_dir, convert_from_hf=False, load_checkpoint_format="")
            reloaded.eval()
            with paddle.no_grad():
                new_output = reloaded(input_ids)[0]

        self.assertEqual(list(new_output.shape), [1, 4, config.hidden_size])
        self.assertTrue(paddle.allclose(ref_output, new_output, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
