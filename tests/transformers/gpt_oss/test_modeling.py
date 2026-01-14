# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from parameterized import parameterized

from paddleformers.transformers import GptOssConfig, GptOssForCausalLM, GptOssModel
from tests.testing_utils import require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ModelTesterPretrainedMixin,
    ids_tensor,
    random_attention_mask,
)


class GptOssModelTester:
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
        self.parent: GptOssModelTest = parent
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

    def get_config(self) -> GptOssConfig:
        return GptOssConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            masked_softmax_fusion=self.masked_softmax_fusion,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            attention_softmax_in_fp32=self.attention_softmax_in_fp32,
            pretraining_tp=self.pretraining_tp,
            dtype=self.dtype,
            slow_but_exact=self.slow_but_exact,
            activation_function=self.activation_function,
        )

    def create_and_check_model(
        self, config: GptOssConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GptOssModel(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(
        self, config: GptOssConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GptOssModel(config)
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
        # Assert non-padding tokens have the same logits with different attention_mask shape
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_3d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_4d[attn_mask_2d]).all())
        self.parent.assertTrue((result_2d[attn_mask_2d] == result_no_attention_mask[attn_mask_2d]).all())

    def create_and_check_model_past_large_inputs(
        self,
        config: GptOssConfig,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = GptOssModel(config)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True, return_dict=self.return_dict)
        past_key_values = outputs.past_key_values if self.return_dict else outputs[2]

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), self.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = paddle.cat([input_ids, next_tokens], axis=-1)
        next_attention_mask = paddle.cat([input_mask, next_mask], axis=-1)

        outputs = model(
            next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True, return_dict=self.return_dict
        )

        output_from_no_past = outputs[2][0]

        outputs = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=self.return_dict,
        )

        output_from_past = outputs[2][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
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
        model = GptOssForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
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
        model = GptOssForCausalLM(config)
        model.eval()

        result_no_position_id = model(
            input_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
        result_position_id = model(
            input_ids,
            position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())

    def create_and_check_gqa_model(self, config, input_ids, input_mask, *args):
        model = GptOssForCausalLM(config)
        config.num_key_value_heads = 8  # gqa
        config.apply_rope_fusion = True
        model.eval()

        result = model(
            input_ids,
            use_cache=True,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])


class GptOssModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = GptOssModel
    return_dict = False
    use_labels = False

    all_model_classes = (GptOssModel, GptOssForCausalLM)
    all_generative_model_classes = {GptOssForCausalLM: (GptOssModel, "GptOss")}

    def setUp(self):
        super().setUp()

        self.model_tester = GptOssModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GptOssConfig, vocab_size=256, hidden_size=24)

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
        # pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_attention_mask(*config_and_inputs)
        # pass

    def test_model_position_ids(self):
        # pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_GptOss_lm_head_model(self):
        # pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_GptOss_gqa_model(self):
        # pass
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gqa_model(*config_and_inputs)

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


class GptOssModelIntegrationTest(ModelTesterPretrainedMixin, unittest.TestCase):
    base_model_class = GptOssModel

    def test_inference_no_attention(self):
        model = GptOssModel.from_pretrained(
            "PaddleFormers/tiny-random-gptoss", download_hub="aistudio", convert_from_hf=True
        )
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 16]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor(
            [
                [
                    [-1.29414248, -2.16081142, 1.12413347],
                    [-0.15411879, 0.49254864, 0.03585127],
                    [-0.96914506, -0.94086194, 0.72755915],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))

    def test_inference_with_attention(self):
        model = GptOssModel.from_pretrained(
            "PaddleFormers/tiny-random-gptoss", download_hub="aistudio", convert_from_hf=True
        )
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 16]
        self.assertEqual(output.shape, expected_shape)
        expected_slice = paddle.to_tensor(
            [
                [
                    [-1.29414248, -2.16081142, 1.12413347],
                    [-0.15411879, 0.49254864, 0.03585127],
                    [-0.96914506, -0.94086194, 0.72755915],
                ]
            ]
        )
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))


class GptOssCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import GptOssConfig, GptOssForCausalLM

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = GptOssConfig(hidden_size=16, num_hidden_layers=8, num_attention_heads=8)
        model = GptOssForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_GptOss_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the paddle model
        from paddleformers.transformers import GptOssModel

        paddle_model = GptOssModel.from_pretrained(self.torch_model_path, convert_from_hf=True, dtype="float32")
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        # 3. forward the torch  model
        import torch
        from transformers import GptOssModel

        torch_model = GptOssModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
        torch_model.eval()
        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                rtol=1e2,
            )
        )

    @require_package("transformers", "torch")
    def test_GptOss_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import GptOssModel

            torch_model = GptOssModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import GptOssModel

            paddle_model = GptOssModel.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
            paddle_model.eval()
            paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    rtol=1e2,
                )
            )

    @parameterized.expand([("GptOssModel",), ("GptOssForCausalLM",)])
    @require_package("transformers", "torch")
    def test_GptOss_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            import transformers

            torch_model_class = getattr(transformers, pytorch_class_name)
            torch_model = torch_model_class.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()

            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 3. forward the paddle model
            from paddleformers import transformers

            paddle_model_class = getattr(transformers, class_name)
            paddle_model = paddle_model_class.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
            paddle_model.eval()

            if class_name == "GptOssModel":
                paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]
            else:
                paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=True).logits

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e2,
                )
            )


if __name__ == "__main__":
    unittest.main()
