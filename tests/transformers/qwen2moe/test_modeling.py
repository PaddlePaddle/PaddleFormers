# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import tempfile
import unittest

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.transformers import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen2MoeModel,
)
from tests.testing_utils import require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    GenerationD2STestMixin,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class Qwen2MoeModelTester:
    def __init__(
        self,
        parent,
        vocab_size=151936,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        bos_token_id=151463,
        pad_token_id=151463,
        eos_token_id=151463,
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
        activation_function="gelu",
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        use_input_mask: bool = False,
        use_labels: bool = False,
        return_dict=False,
    ):
        self.parent: Qwen2MoeModelTest = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
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

    def get_config(self) -> Qwen2MoeConfig:
        return Qwen2MoeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            masked_softmax_fusion=self.masked_softmax_fusion,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
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
        self, config: Qwen2MoeConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Qwen2MoeModel(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: Qwen2MoeConfig, input_ids):
        model = Qwen2MoeModel(config)
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
        model = Qwen2MoeModel(config)
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
        model = Qwen2MoeForCausalLM(config=config)
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
        model = Qwen2MoeForCausalLM(config)
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
        model = Qwen2MoeForCausalLM(config)
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


class Qwen2MoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Qwen2MoeModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Qwen2MoeModel, Qwen2MoeForCausalLM)
    all_generative_model_classes = {Qwen2MoeForCausalLM: (Qwen2MoeModel, "qwen2_moe")}

    def setUp(self):
        super().setUp()

        self.model_tester = Qwen2MoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen2MoeConfig, vocab_size=256, hidden_size=24)

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


class Qwen2MoeIntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen2MoeForCausalLM.from_pretrained(
            "PaddleFormers/tiny-random-qwen2moe", dtype="float32", convert_from_hf=True
        )
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model(input_ids, return_dict=True).logits

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[0.00013129, 0.00055262, 0.00041001, 0.00103886, 0.00101127, 0.00109748, 0.00117971, 0.001671118]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([-0.05819187, -0.22854444, -0.01670399, -0.21067668, 0.09893159,
                                           0.07734174, -0.20733158, -0.07557553, -0.15745537, -0.15629001,
                                           0.17131621, 0.02966851, 0.20745607, 0.18703115, 0.04797143,
                                           -0.05834797, -0.49455544, 0.00927463, 0.36364549, -0.11451467,
                                           0.58765817, -0.16567171, 0.44204327, 0.35513058, 0.14218493,
                                           0.00553618, 0.15461002, -0.20002352, -0.05449944, -0.10040712])  # fmt: skip
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-3, rtol=1e-3))


class Qwen2MoeGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "PaddleFormers/tiny-random-qwen2moe"


class Qwen2MoeCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = Qwen2MoeConfig(
            hidden_size=16,
            intermediate_size=304,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=512,
            num_experts_per_tok=2,
            num_experts=4,
        )
        model = Qwen2MoeForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Qwen2Moe_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the paddle model
        from paddleformers.transformers import Qwen2MoeModel

        paddle_model = Qwen2MoeModel.from_pretrained(self.torch_model_path, convert_from_hf=True, dtype="float32")
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        # 3. forward the torch  model
        try:
            sys.modules["torch"] = sys.modules["torch_save"]
        except:
            pass
        try:
            del sys.modules["transformers"]
        except:
            pass
        import torch
        from transformers import Qwen2MoeModel

        torch_model = Qwen2MoeModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
        torch_model.eval()
        torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

        self.assertTrue(
            np.allclose(
                paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                atol=1e-2,
                rtol=1e-2,
            )
        )
        sys.modules["torch"] = None
        try:
            del sys.modules["transformers"]
        except:
            pass

    @require_package("transformers", "torch")
    def test_Qwen2Moe_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            try:
                sys.modules["torch"] = sys.modules["torch_save"]
            except:
                pass
            try:
                del sys.modules["transformers"]
            except:
                pass
            import torch
            from transformers import Qwen2MoeForCausalLM

            torch_model = Qwen2MoeForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model with fc
            from paddleformers.transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

            paddle_model = Qwen2MoeForCausalLM.from_pretrained(
                tempdir, convert_from_hf=True, dtype="float32", load_checkpoint_format="flex_checkpoint"
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

            # 3. fuse qkv/ffn with fc
            model_config = Qwen2MoeConfig.from_pretrained(tempdir)
            model_config.fuse_attention_qkv = True
            model_config.fuse_attention_ffn = True
            paddle_model_fused = Qwen2MoeForCausalLM.from_pretrained(
                tempdir,
                config=model_config,
                convert_from_hf=True,
                dtype="float32",
                load_checkpoint_format="flex_checkpoint",
            )
            paddle_model_fused.eval()
            paddle_fused_logit = paddle_model_fused(paddle.to_tensor(input_ids))[0]

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    paddle_fused_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )
            sys.modules["torch"] = None
            try:
                del sys.modules["transformers"]
            except:
                pass

    @parameterized.expand([("Qwen2MoeModel",), ("Qwen2MoeForCausalLM",)])
    @require_package("transformers", "torch")
    def test_Qwen2Moe_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
        pytorch_class_name = pytorch_class_name or class_name
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch model
            try:
                sys.modules["torch"] = sys.modules["torch_save"]
            except:
                pass
            try:
                del sys.modules["transformers"]
            except:
                pass
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

            if class_name == "Qwen2MoeModel":
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
            sys.modules["torch"] = None
            try:
                del sys.modules["transformers"]
            except:
                pass
