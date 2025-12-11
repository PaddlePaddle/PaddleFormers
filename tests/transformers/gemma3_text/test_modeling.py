# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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

from paddleformers.transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Gemma3TextModel,
)
from tests.testing_utils import require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class Gemma3TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=262208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        layer_types=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_scaling=None,
        rope_local_base_freq=10000.0,
        use_bidirectional_attention=False,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
    ):
        self.parent: Gemma3TextModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.rope_scaling = rope_scaling
        self.rope_local_base_freq = rope_local_base_freq
        self.use_bidirectional_attention = use_bidirectional_attention

        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

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

    def get_config(self) -> Gemma3TextConfig:
        return Gemma3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_activation=self.hidden_activation,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            sliding_window=self.sliding_window,
            layer_types=self.layer_types,
            final_logit_softcapping=self.final_logit_softcapping,
            attn_logit_softcapping=self.attn_logit_softcapping,
            rope_scaling=self.rope_scaling,
            rope_local_base_freq=self.rope_local_base_freq,
            use_bidirectional_attention=self.use_bidirectional_attention,
        )

    def create_and_check_model(
        self, config: Gemma3TextConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Gemma3TextModel(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(
        self, config: Gemma3TextConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Gemma3TextModel(config)
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
        config: Gemma3TextConfig,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Gemma3TextModel(config)
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
        model = Gemma3ForCausalLM(config)
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
        model = Gemma3ForCausalLM(config)
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
            position_ids=position_ids,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())

    def create_and_check_gqa_model(self, config, input_ids, input_mask, *args):
        model = Gemma3ForCausalLM(config)
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

    def create_and_check_tp(self, config, input_ids, input_mask, *args):
        config.tensor_model_parallel_size = 2

        # check num_key_value_heads
        config.num_key_value_heads = 1
        with self.parent.assertRaises(AssertionError):
            Gemma3ForCausalLM(config)

        # check num_attention_heads
        config.num_key_value_heads = 4
        config.num_attention_heads = 1
        with self.parent.assertRaises(AssertionError):
            Gemma3ForCausalLM(config)

    def create_and_check_fuse_attn(self, config, input_ids, input_mask, *args):
        config.fuse_attention_qkv = True
        config.fuse_attention_ffn = True
        model = Gemma3ForCausalLM(config)
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


class Gemma3TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Gemma3TextModel
    return_dict = False
    use_labels = False

    all_model_classes = (Gemma3TextModel, Gemma3ForCausalLM)
    all_generative_model_classes = {Gemma3ForCausalLM: {Gemma3TextModel, "Gemma3"}}

    def setUp(self):
        super().setUp()
        self.model_tester = Gemma3TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma3TextConfig, vocab_size=256, hidden_size=24)

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

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_gemma3_text_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gemma3_text_gqa_model(self):
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

    def test_gemma3_text_tp(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_tp(*config_and_inputs)

    def test_gemma3_text_fuse_attn(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_fuse_attn(*config_and_inputs)

    def test_gemma3_text_generate(self):
        config = Gemma3TextConfig(
            hidden_size=16, intermediate_size=1120, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2
        )
        model = Gemma3ForCausalLM(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3]], dtype="int64")
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
        )
        assert output[0].shape == [1, 2]


class Gemma3TextIntegrationTest(unittest.TestCase):
    base_model_class = Gemma3TextModel
    test_dtype = "float32"  # "bfloat16"

    def test_inference_no_attention(self):
        model = Gemma3TextModel.from_pretrained(
            "PaddleFormers/tiny-random-gemma3", download_hub="aistudio", convert_from_hf=True, dtype=self.test_dtype
        )
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        with paddle.no_grad():
            output = model(input_ids)[0]
        expected_shape = [1, 11, 16]
        self.assertEqual(output.shape, expected_shape)
        expected_slice_bf16 = paddle.to_tensor(
            [
                [
                    [-1.24218750, -1.01562500, 0.68750000],
                    [0.32617188, -0.24609375, 1.25000000],
                    [1.10156250, 0.29687500, 0.88671875],
                ]
            ]
        )
        expected_slice_fp32 = paddle.to_tensor(
            [
                [
                    [-1.25233459, -1.01471460, 0.69251710],
                    [0.32604450, -0.25053313, 1.26085544],
                    [0.98726571, 0.30734059, 0.91449308],
                ]
            ]
        )
        expected_slice = expected_slice_fp32 if self.test_dtype == "float32" else expected_slice_bf16
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))

    def test_inference_with_attention(self):
        model = Gemma3TextModel.from_pretrained(
            "PaddleFormers/tiny-random-gemma3", download_hub="aistudio", convert_from_hf=True, dtype=self.test_dtype
        )
        model.eval()
        input_ids = paddle.to_tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = paddle.to_tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with paddle.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = [1, 11, 16]
        self.assertEqual(output.shape, expected_shape)
        expected_slice_bf16 = paddle.to_tensor(
            [
                [
                    [-1.26562500, -1.28125000, 1.30468750],
                    [0.39257812, -0.23437500, 0.94921875],
                    [0.84765625, -0.00598145, 1.53125000],
                ]
            ]
        )
        expected_slice_fp32 = paddle.to_tensor(
            [
                [
                    [-1.27054501, -1.26936519, 1.29382658],
                    [0.37663761, -0.25405365, 0.95409876],
                    [0.81471157, -0.01011910, 1.53275037],
                ]
            ]
        )
        expected_slice = expected_slice_fp32 if self.test_dtype == "float32" else expected_slice_bf16
        self.assertTrue(paddle.allclose(output[:, 1:4, 1:4].cast(paddle.float32), expected_slice, atol=1e-4))


class Gemma3TextCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = Gemma3TextConfig(
            hidden_size=16, intermediate_size=1120, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2
        )
        model = Gemma3ForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Gemma3Text_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the paddle model
        from paddleformers.transformers import Gemma3TextModel

        paddle_model = Gemma3TextModel.from_pretrained(self.torch_model_path, convert_from_hf=True, dtype="float32")
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        # 3. forward the torch model
        import torch
        from transformers import Gemma3ForCausalLM

        torch_model = Gemma3ForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).model
        torch_model.eval()
        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @require_package("transformers", "torch")
    def test_Gemma3_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import Gemma3ForCausalLM

            torch_model = Gemma3ForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).model
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import Gemma3TextModel

            paddle_model = Gemma3TextModel.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
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

    @parameterized.expand([("Gemma3TextModel",), ("Gemma3ForCausalLM",)])
    @require_package("transformers", "torch")
    def test_Gemma3_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            import transformers

            if pytorch_class_name == "Gemma3TextModel":
                torch_model_class = getattr(transformers, "Gemma3ForCausalLM")
                torch_model = torch_model_class.from_pretrained(self.torch_model_path, torch_dtype=torch.float32).model
            else:
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

            if class_name == "Gemma3TextModel":
                paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]
            else:
                paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=True).logits

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )


if __name__ == "__main__":
    unittest.main()
