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

from paddleformers.transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
)
from tests.testing_utils import require_package
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_generation_utils import GenerationTesterMixin
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class DeepseekV3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=64,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        num_hidden_layers=2,
        hidden_size=128,
        vocab_size=1000,
        num_attention_heads=8,
        num_key_value_heads=8,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        moe_intermediate_size=256,
        intermediate_size=512,
        max_position_embeddings=512,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        aux_loss_alpha=0.001,
        first_k_dense_replace=1,
        hidden_act="silu",
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }

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
        config.seq_length = input_ids.shape[1]
        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self) -> DeepseekV3Config:
        return DeepseekV3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            aux_loss_alpha=self.aux_loss_alpha,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_probs_dropout_prob,
            rope_scaling=self.rope_scaling,
        )

    def create_and_check_model(
        self, config: DeepseekV3Config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DeepseekV3Model(config)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_model_attention_mask(self, config: DeepseekV3Config, input_ids):
        model = DeepseekV3Model(config)
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
        model = DeepseekV3Model(config)
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
        model = DeepseekV3ForCausalLM(config=config)
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
        config.seq_length = input_ids.shape[1]
        return config, inputs_dict

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, *args):
        model = DeepseekV3ForCausalLM(config)
        model.eval()

        result = model(
            input_ids,
            use_cache=False,
            labels=input_ids if self.parent.use_labels else None,
            return_dict=self.parent.return_dict,
        )
        if self.parent.use_labels:
            self.parent.assertIsInstance(result[0].item(), float)
            self.parent.assertEqual(result[1].shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, input_ids, input_mask, *args):
        model = DeepseekV3ForCausalLM(config)
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


class DeepseekV3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    base_model_class = DeepseekV3Model
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (DeepseekV3Model, DeepseekV3ForCausalLM)
    all_generative_model_classes = {DeepseekV3ForCausalLM: (DeepseekV3Model, "deepseek_v3")}

    def setUp(self):
        super().setUp()

        self.model_tester = DeepseekV3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekV3Config, vocab_size=256, hidden_size=24)

    def _get_input_ids_and_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_ids = inputs_dict[self.input_name]
        attention_mask = paddle.ones_like(input_ids, dtype=paddle.int64)

        max_batch_size = 2
        input_ids = input_ids[:max_batch_size, :]
        attention_mask = attention_mask[:max_batch_size, :]
        max_length = input_ids.shape[-1]
        return config, input_ids, attention_mask, max_length

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_attention_mask(self):
        pass

    def test_model_position_ids(self):
        config, input_ids, input_mask, _, _, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(config, input_ids, input_mask)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_DeepseekV3_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

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


class DeepseekV3IntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = DeepseekV3ForCausalLM.from_pretrained(
            "PaddleFormers/tiny-random-deepseek-v3",
            dtype="float32",
            download_hub="aistudio",
            convert_from_hf=True,
            seq_length=len(input_ids),
        )
        model.config.seq_length = len(input_ids)
        input_ids = paddle.to_tensor([input_ids])
        with paddle.no_grad():
            out = model(input_ids, return_dict=True).logits

        # Expected mean on dim = -1
        EXPECTED_MEAN = paddle.to_tensor(
            [[-0.00238119, 0.00045322, -0.00630159, 0.00018909, -0.00675243, -0.00595381, -0.00273710, 0.00166127]]
        )
        self.assertTrue(paddle.allclose(out.mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3))

        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = paddle.to_tensor(
            [
                -1.78225815,
                -1.78054917,
                -0.21763977,
                0.95293641,
                1.47887754,
                -0.67339337,
                1.67287385,
                -1.51305628,
                2.61438489,
                -0.72594410,
                0.44677153,
                2.14733338,
                0.94409180,
                0.21087888,
                0.49765375,
                2.02336526,
                -3.16481948,
                0.89196634,
                0.33974639,
                0.14313932,
                1.56812930,
                0.69366109,
                1.50022423,
                -0.34549955,
                -0.20486489,
                1.90403473,
                -2.12635422,
                1.04159653,
                -0.09439243,
                -1.98977375,
            ]
        )
        self.assertTrue(paddle.allclose(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2))


class DeepseekV3CompatibilityTest(unittest.TestCase):
    @classmethod
    @require_package("transformers", "torch")
    def setUpClass(cls) -> None:
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

        # when python application is done, `TemporaryDirectory` will be free
        cls.torch_model_path = tempfile.TemporaryDirectory().name
        config = DeepseekV3Config(
            hidden_size=128,
            intermediate_size=384,
            num_hidden_layers=4,
            qk_head_dim=32,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            v_head_dim=16,
            moe_intermediate_size=192,
            num_experts_per_tok=2,
            num_experts=8,
            seq_length=20,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
        model = DeepseekV3ForCausalLM(config)
        model.save_pretrained(cls.torch_model_path)

    @require_package("transformers", "torch")
    def test_DeepseekV3_converter(self):
        # 1. create common input
        input_ids = np.random.randint(100, 200, [1, 20])

        # 2. forward the paddle model
        from paddleformers.transformers import DeepseekV3Model

        paddle_model = DeepseekV3Model.from_pretrained(self.torch_model_path, convert_from_hf=True, dtype="float32")
        paddle_model.eval()
        paddle_logit = paddle_model(paddle.to_tensor(input_ids))[0]

        # 3. forward the torch  model
        import torch
        from transformers import DeepseekV3Model

        torch_model = DeepseekV3Model.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
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
    def test_DeepseekV3_converter_from_local_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # 1. create common input
            input_ids = np.random.randint(100, 200, [1, 20])

            # 2. forward the torch  model
            import torch
            from transformers import DeepseekV3Model

            torch_model = DeepseekV3Model.from_pretrained(self.torch_model_path, torch_dtype=torch.float32)
            torch_model.eval()
            torch_model.save_pretrained(tempdir)
            torch_logit = torch_model(torch.tensor(input_ids), return_dict=False)[0]

            # 2. forward the paddle model
            from paddleformers.transformers import DeepseekV3Model

            paddle_model = DeepseekV3Model.from_pretrained(tempdir, convert_from_hf=True, dtype="float32")
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

    @parameterized.expand([("DeepseekV3Model",), ("DeepseekV3ForCausalLM",)])
    @require_package("transformers", "torch")
    def test_DeepseekV3_classes_from_local_dir(self, class_name, pytorch_class_name: str | None = None):
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

            if class_name == "DeepseekV3Model":
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
