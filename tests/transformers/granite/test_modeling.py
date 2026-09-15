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

import tempfile
import unittest

import numpy as np
import paddle

from paddleformers.transformers import GraniteConfig, GraniteForCausalLM, GraniteModel
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class GraniteModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        embedding_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        attention_multiplier=0.5,
        hidden_act="silu",
        dtype="float32",
        batch_size: int = 2,
        seq_length: int = 10,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        is_training=True,
        use_input_mask: bool = True,
        use_labels: bool = False,
        return_dict: bool = False,
    ):
        self.parent: GraniteModelTest = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.hidden_act = hidden_act
        self.dtype = dtype

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.is_training = is_training
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

    def get_config(self) -> GraniteConfig:
        return GraniteConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.seq_length + 10,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=True,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            embedding_multiplier=self.embedding_multiplier,
            logits_scaling=self.logits_scaling,
            residual_multiplier=self.residual_multiplier,
            attention_multiplier=self.attention_multiplier,
            dtype=self.dtype,
            _attn_implementation="eager",
        )

    def create_and_check_model(
        self, config: GraniteConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GraniteModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: GraniteConfig, input_ids):
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

    def create_and_check_model_past_large_inputs(
        self,
        config: GraniteConfig,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = GraniteModel(config)
        model.eval()

        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.return_dict)
        past_key_values = outputs.past_key_values if self.return_dict else outputs[1]

        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size, dtype=input_ids.dtype)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2, dtype=input_mask.dtype)
        next_input_ids = paddle.cat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.cat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True, return_dict=self.return_dict
        )
        output_from_no_past = outputs.hidden_states[0] if self.return_dict else outputs[2][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=self.return_dict,
        )
        output_from_past = outputs.hidden_states[0] if self.return_dict else outputs[2][0]

        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertEqual(output_from_past_slice.shape[1], next_tokens.shape[1])
        self.parent.assertTrue(paddle.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

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
        model = GraniteForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            attention_mask=input_mask,
            use_cache=True,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = GraniteForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            attention_mask=input_mask,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
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


class GraniteModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = GraniteModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (GraniteModel, GraniteForCausalLM)
    all_generative_model_classes = {GraniteForCausalLM: (GraniteModel, "granite")}

    @gpu_device_initializer(log_prefix="GraniteModelTest")
    def setUp(self):
        super().setUp()
        self.model_tester = GraniteModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_past_large_inputs(*config_and_inputs)

    def test_granite_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_attention_outputs(self):
        pass

    def test_generate_without_input_ids(self):
        pass


class GraniteModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = GraniteModel


class GraniteCompatibilityTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="GraniteCompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import GraniteConfig as HFGraniteConfig
        from transformers import GraniteForCausalLM as HFGraniteForCausalLM

        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = HFGraniteConfig(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            attention_multiplier=0.5,
            embedding_multiplier=1.0,
            logits_scaling=1.0,
            residual_multiplier=1.0,
            tie_word_embeddings=True,
        )
        model = HFGraniteForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_granite_converter(self):
        input_ids = np.random.randint(10, 100, [1, 20])

        import torch
        from transformers import GraniteForCausalLM as HFGraniteForCausalLM

        torch_model = HFGraniteForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
        torch_model.eval()
        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        paddle_model = GraniteForCausalLM.from_pretrained(
            self.torch_model_path, dtype="float32", load_checkpoint_format="flex_checkpoint"
        )
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_granite_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_ids = np.random.randint(10, 100, [1, 20])

            import torch
            from transformers import GraniteForCausalLM as HFGraniteForCausalLM

            torch_model = HFGraniteForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            paddle_model = GraniteForCausalLM.from_pretrained(
                tempdir, dtype="float32", load_checkpoint_format="flex_checkpoint"
            )
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )
