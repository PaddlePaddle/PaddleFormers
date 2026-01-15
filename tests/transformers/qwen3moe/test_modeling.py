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

import copy
import inspect
import tempfile
import unittest

import numpy as np
import paddle
from parameterized import parameterized

from paddleformers.trainer.trainer_utils import set_random_seed
from paddleformers.transformers import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
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


class Qwen3MoeModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
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
        self.parent: Qwen3MoeModelTest = parent
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
        self.pad_token_id = pad_token_id
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

    def get_config(self) -> Qwen3MoeConfig:
        return Qwen3MoeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            masked_softmax_fusion=self.masked_softmax_fusion,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
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
        self, config: Qwen3MoeConfig, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Qwen3MoeModel(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: Qwen3MoeConfig, input_ids):
        model = Qwen3MoeModel(config)
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
        model = Qwen3MoeModel(config)
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
        model = Qwen3MoeForCausalLM(config=config)
        model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")
        data = {
            "input_ids": input_ids,
            "position_ids": None,
            "attention_mask": input_mask,
            "labels": token_labels,
        }
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            model.eval()
            result = model(data)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

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
        model = Qwen3MoeForCausalLM(config)
        data = {
            "input_ids": input_ids,
            "position_ids": None,
            "attention_mask": None,
            "labels": input_ids if self.parent.use_labels else None,
        }
        model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")

        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            model.eval()
            result = model(data)

        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = Qwen3MoeForCausalLM(config)
        model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")
        data = {
            "input_ids": input_ids,
            "position_ids": None,
            "attention_mask": None,
            "labels": input_ids if self.parent.use_labels else None,
        }
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            model.eval()
            result_no_position_id = model(data)

        batch_size, seq_len = input_ids.shape
        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))

        data = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
            "labels": input_ids if self.parent.use_labels else None,
        }
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            result_position_id = model(data)
        if self.parent.use_labels:
            self.parent.assertTrue((result_position_id[1] == result_no_position_id[1]).all())
        else:
            self.parent.assertTrue((result_position_id[0] == result_no_position_id[0]).all())


class Qwen3MoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = Qwen3MoeModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (Qwen3MoeModel, Qwen3MoeForCausalLM)
    all_generative_model_classes = {Qwen3MoeForCausalLM: (Qwen3MoeModel, "qwen3_moe")}

    def setUp(self):
        super().setUp()
        set_random_seed(42)
        self.model_tester = Qwen3MoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3MoeConfig, vocab_size=256, hidden_size=24)

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

    # def test_save_load(self):
    #     for model_class in self.all_model_classes:
    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #             model = model_class(config)
    #             model.save_pretrained(tmpdirname, save_checkpoint_format="flex_checkpoint")

    #             model1 = model_class.from_pretrained(tmpdirname, convert_from_hf=True)

    #             model2 = model_class.from_pretrained(tmpdirname, load_checkpoint_format="flex_checkpoint")

    #             model_state_1 = model1.state_dict()
    #             model_state_2 = model2.state_dict()

    #             for k, v in model_state_1.items():
    #                 md51 = v._md5sum()
    #                 md52 = model_state_2[k]._md5sum()
    #                 assert md51 == md52

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["input_ids", "input"]
            assert arg_names[:1][0] in expected_arg_names

    def test_for_missed_attribute(self):
        if not self.test_model_compatibility_keys:
            self.skipTest(f"Do not test model_compatibility_keys on {self.base_model_class}")
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if not model_class.constructed_from_pretrained_config():
                continue

            model = self._make_model_instance(config, model_class)

            all_maps: dict = copy.deepcopy(model_class.config_class.attribute_map)

            for old_attribute, new_attribute in all_maps.items():
                print("old_attribute", old_attribute, "new_attribute", new_attribute)
                if old_attribute == "num_classes":
                    continue
                old_value = getattr(model.config, old_attribute)
                new_value = getattr(model.config, new_attribute)

                # eg: dropout can be an instance of nn.Dropout, so we should check it attribute
                if type(new_value) != type(old_value):
                    continue

                self.assertEqual(old_value, new_value)

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


class Qwen3MoeIntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        config = Qwen3MoeConfig.from_pretrained("PaddleFormers/tiny-random-qwen3moev2")
        config.fuse_attention_ffn = (True,)
        config.fuse_attention_qkv = (True,)
        model = Qwen3MoeForCausalLM.from_pretrained(
            "PaddleFormers/tiny-random-qwen3moev2",
            config=config,
            dtype="float32",
            convert_from_hf=True,
            load_checkpoint_format="flex_checkpoint",
        )
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model({"input_ids": input_ids})

        print("out.mean(-1) is ", out.mean(-1))

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[0.00170604, 0.00594666, 0.00417538, 0.00733661, 0.00917683, 0.00863101, 0.00907406, 0.00700793]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor([1.19751632, 1.76759684, 1.42320514, -3.55444431, 0.54329103,
                                           -0.24107473, -2.48883653, 0.09119778, 0.10803542, 0.95290345,
                                           0.08615199, 0.75243753, 0.67679799, -0.49227887, -0.11838460,
                                           -1.38586426, -1.02522457, -0.34655067, 0.00249448, 0.01345686,
                                           -1.25499344, -2.20100021, 1.13552403, -1.18407190, -1.93378878,
                                           -0.31357813, -2.56630087, 0.80468446, 0.56240237, -0.04839380])  # fmt: skip
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2))


class Qwen3MoeGenerationD2STest(GenerationD2STestMixin, unittest.TestCase):
    internal_testing_model = "PaddleFormers/tiny-random-qwen3moev2"


class Qwen3MoeCompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=384,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            moe_intermediate_size=192,
            num_experts_per_tok=2,
            num_experts=8,
        )
        model = Qwen3MoeForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_Qwen3Moe_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the paddle model
        from paddleformers.transformers import Qwen3MoeModel

        paddle_model = Qwen3MoeModel.from_pretrained(
            self.torch_model_path, convert_from_hf=True, dtype="float32", load_checkpoint_format=""
        )
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        # 3. forward the torch  model
        import torch
        from transformers import Qwen3MoeModel

        torch_model = Qwen3MoeModel.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
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

    @require_package("transformers", "torch")
    def test_Qwen3Moe_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import Qwen3MoeForCausalLM

            torch_model = Qwen3MoeForCausalLM.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model with fc
            from paddleformers.transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

            paddle_model = Qwen3MoeForCausalLM.from_pretrained(
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
            model_config = Qwen3MoeConfig.from_pretrained(tempdir)
            model_config.fuse_attention_qkv = True
            model_config.fuse_attention_ffn = True
            paddle_model_fused = Qwen3MoeForCausalLM.from_pretrained(
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

    @parameterized.expand([("Qwen3MoeModel",), ("Qwen3MoeForCausalLM",)])
    @require_package("transformers", "torch")
    def test_Qwen3Moe_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
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
            paddle_model = paddle_model_class.from_pretrained(
                tempdir, convert_from_hf=True, dtype="float32", load_checkpoint_format=""
            )
            paddle_model.eval()

            if class_name == "Qwen3MoeModel":
                paddle_logit = paddle_model(paddle.to_tensor(input_ids), return_dict=False)[0]
            else:
                paddle_logit = paddle_model({"input_ids": paddle.to_tensor(input_ids)})

            self.assertTrue(
                np.allclose(
                    paddle_logit.detach().cpu().reshape([-1])[:9].astype("float32").numpy(),
                    torch_logit.detach().cpu().reshape([-1])[:9].float().numpy(),
                    atol=1e-2,
                    rtol=1e-2,
                )
            )
