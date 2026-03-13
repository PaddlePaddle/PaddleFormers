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

import unittest

import numpy as np
import paddle

from paddleformers.trainer import set_random_seed
from paddleformers.transformers import KimiK2Config, KimiK2ForCausalLM
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class KimiK2ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_nextn_predict_layers=0,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=4,
        ep_size=1,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=None,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        aux_loss_alpha=0.001,
        seq_aux=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,
        dtype="bfloat16",
        batch_size: int = 2,
        seq_length: int = 10,
        type_sequence_label_size=2,
        activation_function="silu",
        num_labels=3,
        num_choices=4,
        scope=None,
        dropout=0.56,
        use_labels: bool = False,
    ):
        self.parent: KimiK2ModelTest = parent
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        self.dtype = dtype

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.type_sequence_label_size = type_sequence_label_size
        self.activation_function = activation_function
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.dropout = dropout

        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        # Return dict format for PipelineLayer compatibility
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict, sequence_labels, token_labels, choice_labels

    def get_config(self) -> KimiK2Config:
        return KimiK2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_nextn_predict_layers=self.num_nextn_predict_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            ep_size=self.ep_size,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            topk_method=self.topk_method,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_layer_freq=self.moe_layer_freq,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            scoring_func=self.scoring_func,
            aux_loss_alpha=self.aux_loss_alpha,
            seq_aux=self.seq_aux,
            rms_norm_eps=self.layer_norm_epsilon,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            pretraining_tp=self.pretraining_tp,
            dtype=self.dtype,
            hidden_act=self.activation_function,
            fp32_residual_connection=False,
        )

    def create_and_check_model_attention_mask(self, config: KimiK2Config, inputs_dict):
        model = KimiK2ForCausalLM(config)
        model.eval()
        input_ids = inputs_dict["input_ids"]
        attn_mask_2d = random_attention_mask([self.batch_size, self.seq_length])
        # Use dict input for PipelineLayer compatibility
        inputs_dict_2d = {"input_ids": input_ids, "attention_mask": attn_mask_2d}

        result_2d = model(inputs_dict_2d)
        # Check output shapes are correct
        self.parent.assertEqual(result_2d.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_for_causal_lm(
        self,
        config,
        inputs_dict,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = KimiK2ForCausalLM(config=config)
        model.eval()
        input_ids = inputs_dict["input_ids"]
        seq_len = input_ids.shape[-1]
        # Create causal attention mask
        causal_mask = paddle.tril(paddle.ones((seq_len, seq_len), dtype=paddle.int64))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        # Use dict input for PipelineLayer compatibility
        inputs_dict_with_labels = dict(inputs_dict)
        inputs_dict_with_labels["attention_mask"] = causal_mask
        inputs_dict_with_labels["labels"] = token_labels
        result = model(inputs_dict_with_labels)
        self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            inputs_dict,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        return config, inputs_dict

    def create_and_check_lm_head_model(self, config, inputs_dict, *args):
        model = KimiK2ForCausalLM(config)
        model.eval()
        input_ids = inputs_dict["input_ids"]
        seq_len = input_ids.shape[-1]
        # Create causal attention mask
        causal_mask = paddle.tril(paddle.ones((seq_len, seq_len), dtype=paddle.int64))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        # Use dict input for PipelineLayer compatibility
        inputs_dict_with_cache = dict(inputs_dict)
        inputs_dict_with_cache["attention_mask"] = causal_mask
        inputs_dict_with_cache["use_cache"] = True
        inputs_dict_with_cache["labels"] = input_ids if self.parent.use_labels else None
        result = model(inputs_dict_with_cache)
        if self.parent.use_labels:
            self.parent.assertIsInstance(result.item(), float)
            self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])
        else:
            self.parent.assertEqual(result.shape, [self.batch_size, self.seq_length, self.vocab_size])

    def check_model_position_ids(self, config, inputs_dict, *args):
        model = KimiK2ForCausalLM(config)
        model.eval()
        input_ids = inputs_dict["input_ids"]
        batch_size, seq_len = input_ids.shape
        # Create causal attention mask
        causal_mask = paddle.tril(paddle.ones((seq_len, seq_len), dtype=paddle.int64))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        # Use dict input for PipelineLayer compatibility
        inputs_dict_no_pos = dict(inputs_dict)
        inputs_dict_no_pos["attention_mask"] = causal_mask
        inputs_dict_no_pos["labels"] = input_ids if self.parent.use_labels else None
        result_no_position_id = model(inputs_dict_no_pos)

        position_ids = paddle.arange(seq_len).expand((batch_size, seq_len))
        inputs_dict_with_pos = dict(inputs_dict)
        inputs_dict_with_pos["attention_mask"] = causal_mask
        inputs_dict_with_pos["position_ids"] = position_ids
        inputs_dict_with_pos["labels"] = input_ids if self.parent.use_labels else None
        result_position_id = model(inputs_dict_with_pos)

        self.parent.assertTrue((result_position_id == result_no_position_id).all())


class KimiK2ModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = KimiK2ForCausalLM
    use_labels = False
    use_test_model_name_list = False
    input_name = "input_ids"

    all_model_classes = (KimiK2ForCausalLM,)
    all_generative_model_classes = {KimiK2ForCausalLM: (KimiK2ForCausalLM, "kimi_k2")}

    @gpu_device_initializer(log_prefix="KimiK2ModelTest")
    def setUp(self):
        super().setUp()
        set_random_seed(seed_=42)
        # paddle.set_default_dtype("bfloat16")
        self.model_tester = KimiK2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=KimiK2Config, vocab_size=256, hidden_size=24)

    def test_determinism(self):
        """Override test_determinism to use dict input for PipelineLayer compatibility."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                # Use dict input for PipelineLayer compatibility
                first = model(inputs_dict)[0]
                second = model(inputs_dict)[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

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
        # attention_mask or attn_mask_startend_row_indices can not be None at the same time
        pass

    def test_model_attention_mask(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model_attention_mask(config, input_dict)

    def test_model_position_ids(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_model_position_ids(*config_and_inputs)

    def test_generate_without_input_ids(self):
        # this requires 4-D attention mask logic, which is not supported yet
        pass

    def test_model_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_save_load(self):
        pass
        # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # input_ids = inputs_dict["input_ids"]
        # seq_len = input_ids.shape[-1]
        # # Create causal attention mask
        # causal_mask = paddle.tril(paddle.ones((seq_len, seq_len), dtype=paddle.int64))
        # causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # for model_class in self.all_model_classes:
        #     model = self._make_model_instance(config, model_class)
        #     model.eval()

        #     # Get output from original model
        #     with paddle.no_grad():
        #         inputs_dict_with_mask = dict(inputs_dict)
        #         inputs_dict_with_mask["attention_mask"] = causal_mask
        #         original_output = model(inputs_dict_with_mask)

        #     # Save and reload model
        #     with tempfile.TemporaryDirectory() as tmp_dir:
        #         model.save_pretrained(tmp_dir)
        #         loaded_model = model_class.from_pretrained(tmp_dir)
        #         loaded_model.eval()

        #         # Get output from loaded model
        #         with paddle.no_grad():
        #             loaded_output = loaded_model(inputs_dict_with_mask)

        #     # Verify outputs are close
        #     def check_outputs_equal(first, second):
        #         out_1 = first.numpy()
        #         out_2 = second.numpy()
        #         max_diff = np.amax(np.abs(out_1 - out_2))
        #         self.assertLessEqual(max_diff, 1e-5)

        #     if isinstance(original_output, tuple) and isinstance(loaded_output, tuple):
        #         for tensor1, tensor2 in zip(original_output, loaded_output):
        #             check_outputs_equal(tensor1, tensor2)
        #     else:
        #         check_outputs_equal(original_output, loaded_output)

    def test_for_missed_attribute(self):
        pass

    def test_forward_signature(self):
        pass

    def test_resize_tokens_embeddings(self):
        pass
