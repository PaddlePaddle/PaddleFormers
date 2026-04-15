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

from paddleformers.transformers import (
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
    Qwen3NextModel,
)
from tests.testing_utils import gpu_device_initializer, require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class Qwen3NextModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=32,
        num_experts=16,
        intermediate_size=64,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        pretraining_tp=1,  # TP rank used when training with megatron
        dtype="bfloat16",
        slow_but_exact=False,
        batch_size: int = 2,
        seq_length: int = 10,
        type_sequence_label_size=2,
        activation_function="silu",
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        use_input_mask: bool = False,
        use_labels: bool = False,
        return_dict=False,
    ):
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

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

    def get_config(self) -> Qwen3NextConfig:
        model_args = self.__dict__.copy()
        model_args.pop("parent")
        return Qwen3NextConfig(**model_args)

    def create_and_check_model(
        self, config: Qwen3NextConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Qwen3NextModel(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: Qwen3NextConfig, input_ids):
        model = Qwen3NextModel(config)
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

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.add_cross_attention = True
        model = Qwen3NextModel(config)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = Qwen3NextForCausalLM(config=config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])

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
        model = Qwen3NextForCausalLM(config)
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
        model = Qwen3NextForCausalLM(config)
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


class Qwen3NextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Qwen3NextModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Qwen3NextModel, Qwen3NextForCausalLM)
    all_generative_model_classes = {Qwen3NextForCausalLM: (Qwen3NextModel, "qwen3_next")}

    @gpu_device_initializer(log_prefix="Qwen3NextModelTest")
    def setUp(self):
        super().setUp()

        self.model_tester = Qwen3NextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3NextConfig, vocab_size=256, hidden_size=24)

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
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict["input_ids"])

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_model_decoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_save_load(self):
        for model_class in self.all_model_classes:
            # test from_pretrained
            model1 = model_class.from_pretrained(
                "PaddleFormers/tiny-random-qwen3next",
                download_hub="aistudio",
                load_checkpoint_format="flex_checkpoint",
                num_nextn_predict_layers=0,
            )
            model_state_1 = model1.state_dict()

            # test save_pretrained
            with tempfile.TemporaryDirectory() as tmpdirname:
                model1.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")
                model2 = model_class.from_pretrained(
                    tmpdirname,
                    convert_from_hf=True,
                    load_checkpoint_format="flex_checkpoint",
                    num_nextn_predict_layers=0,
                )
                model_state_2 = model2.state_dict()

                for k, v in model_state_2.items():
                    md52 = v._md5sum()
                    md51 = model_state_1[k]._md5sum()
                    assert md51 == md52


class Qwen3NextIntegrationTest(unittest.TestCase):
    @gpu_device_initializer(log_prefix="Qwen3NextIntegrationTest")
    def setUp(self):
        pass

    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen3NextForCausalLM.from_pretrained(
            "PaddleFormers/tiny-random-qwen3next",
            dtype="float32",
            load_checkpoint_format="flex_checkpoint",
        )
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model(input_ids, return_dict=True).logits

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[-0.00205501, 0.00027839, 0.00424480, -0.00688356, 0.00100611, -0.00839691, 0.00667181, 0.00160779]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([2.87202048, 0.26276401, -0.80719441, 0.73548907, 1.77654934,
                                           0.55275565, -0.30633882, -0.14590700, -0.50525594, 1.51723337,
                                           2.42696047, -0.14693625, -1.74664390, 0.68826008, -1.45081413,
                                           -0.49273738, 1.14591551, -1.71560407, -1.54047251, 0.56033224,
                                           0.99398780, -0.45625472, 0.46429783, -0.91559821, -0.71507078,
                                           -1.16813612, 0.36878633, -3.33972144, 1.14574468, -0.63741481])  # fmt: skip
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2))


class Qwen3NextGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "PaddleFormers/tiny-random-qwen3next"


class Qwen3NextCompatibilityTest:
    @gpu_device_initializer(log_prefix="Qwen3NextCompatibilityTest")
    def setUp(self):
        pass

    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = Qwen3NextConfig(
            hidden_size=16,
            intermediate_size=384,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            moe_intermediate_size=192,
            num_experts_per_tok=2,
            num_experts=8,
        )
        model = Qwen3NextForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Qwen3Next_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the torch model
        import torch
        from transformers import Qwen3NextForCausalLM

        torch_model = Qwen3NextForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
        torch_model.eval()
        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        # 3. forward the paddle model
        from paddleformers.transformers import Qwen3NextForCausalLM

        paddle_model = Qwen3NextForCausalLM.from_pretrained(
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
    def test_Qwen3Next_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            import torch
            from transformers import Qwen3NextForCausalLM

            torch_model = Qwen3NextForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import Qwen3NextForCausalLM

            paddle_model = Qwen3NextForCausalLM.from_pretrained(
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
