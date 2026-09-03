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
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddleformers.nn.norm import RMSNorm
from paddleformers.transformers import MiniMaxConfig, MiniMaxForCausalLM, MiniMaxModel
from paddleformers.transformers.auto.modeling import AutoModelForCausalLM
from paddleformers.transformers.minimax.modeling import MiniMaxLightningAttention
from tests.testing_utils import gpu_device_initializer
from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)


class MiniMaxModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        block_size=4,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.parent: MiniMaxModelTest = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.block_size = block_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype=paddle.int64)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()
        return config, input_ids, input_mask

    def get_config(self) -> MiniMaxConfig:
        return MiniMaxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            layer_types=["full_attention", "linear_attention"],
            block_size=self.block_size,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
            use_cache=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

    def create_and_check_model(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxModel(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result[0].shape, [self.batch_size, self.seq_length, self.hidden_size])

    def create_and_check_for_causal_lm(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxForCausalLM(config)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=True)
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(result.loss)

    def create_and_check_training_step(self, config: MiniMaxConfig, input_ids, input_mask):
        model = MiniMaxForCausalLM(config)
        model.train()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids, return_dict=True)
        result.loss.backward()
        self.parent.assertEqual(result.logits.shape, [self.batch_size, self.seq_length, self.vocab_size])
        self.parent.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def create_and_check_auto_model(self, config: MiniMaxConfig):
        model = AutoModelForCausalLM.from_config(config)
        self.parent.assertIsInstance(model, MiniMaxForCausalLM)

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": input_mask}


class MiniMaxModelTest(ModelTesterMixin, unittest.TestCase):
    base_model_class = MiniMaxModel
    return_dict = False
    use_labels = False
    use_test_model_name_list = False

    all_model_classes = (MiniMaxModel, MiniMaxForCausalLM)
    all_generative_model_classes = {MiniMaxForCausalLM: (MiniMaxModel, "minimax")}

    @gpu_device_initializer(log_prefix="MiniMaxModelTest")
    def setUp(self):
        super().setUp()

        self.model_tester = MiniMaxModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniMaxConfig, vocab_size=256, hidden_size=24)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_training_step(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_training_step(*config_and_inputs)

    def test_auto_model_for_causal_lm(self):
        config = self.model_tester.get_config()
        self.model_tester.create_and_check_auto_model(config)

    def test_uses_general_norm_and_linear_router(self):
        config = self.model_tester.get_config()
        model = MiniMaxForCausalLM(config)

        self.assertIsInstance(model.model.norm, RMSNorm)
        for layer in model.model.layers:
            self.assertIsInstance(layer.input_layernorm, RMSNorm)
            self.assertIsInstance(layer.post_attention_layernorm, RMSNorm)
            if isinstance(layer.self_attn, MiniMaxLightningAttention):
                self.assertIsInstance(layer.self_attn.norm, RMSNorm)
            self.assertIsInstance(layer.block_sparse_moe.gate, paddle.nn.Linear)

        self.assertIn("model.layers.0.block_sparse_moe.gate.weight", model.state_dict())

    def test_output_hidden_states_uses_config_default(self):
        config, input_ids, input_mask = self.model_tester.prepare_config_and_inputs()
        config.output_hidden_states = True
        model = MiniMaxForCausalLM(config)
        model.eval()

        outputs = model(input_ids, attention_mask=input_mask, return_dict=True)
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)

    def test_full_layer_recompute_path(self):
        config, input_ids, input_mask = self.model_tester.prepare_config_and_inputs()
        config.recompute_granularity = "full"
        config.recompute_method = "uniform"
        config.recompute_num_layers = 1
        config.recompute_use_reentrant = False
        model = MiniMaxModel(config)
        model.train()

        def run_forward(function, *args, **kwargs):
            kwargs.pop("use_reentrant", None)
            return function(*args, **kwargs)

        with patch("paddleformers.transformers.minimax.modeling.recompute", side_effect=run_forward) as mock_recompute:
            outputs = model(input_ids, attention_mask=input_mask, return_dict=True)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            [self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size],
        )
        self.assertEqual(mock_recompute.call_count, config.num_hidden_layers)

    def test_transpose_weight_keys_match_minimax_modules(self):
        self.assertNotIn("gate_up_proj", MiniMaxForCausalLM.transpose_weight_keys)
        self.assertNotIn("down_proj", MiniMaxForCausalLM.transpose_weight_keys)

    def test_lm_head_aoa_keeps_hf_layout(self):
        config = self.model_tester.get_config()
        config.tie_word_embeddings = False

        forward_statements = MiniMaxForCausalLM._gen_aoa_config(config)["aoa_statements"]
        inverse_statements = MiniMaxForCausalLM._gen_inv_aoa_config(config)["aoa_statements"]

        self.assertIn("lm_head.weight -> lm_head.weight", forward_statements)
        self.assertIn("lm_head.weight -> lm_head.weight", inverse_statements)
        self.assertNotIn("lm_head.weight^T -> lm_head.weight", forward_statements)
        self.assertNotIn("lm_head.weight^T -> lm_head.weight", inverse_statements)

    def test_default_lora_targets_cover_all_projection_types(self):
        from paddleformers.cli.utils import get_lora_target_modules
        from paddleformers.peft import LoRAConfig, LoRAModel

        config = self.model_tester.get_config()
        model = MiniMaxForCausalLM(config)
        target_modules = get_lora_target_modules(model)
        self.assertEqual(
            target_modules,
            [
                ".*q_proj.*",
                ".*k_proj.*",
                ".*v_proj.*",
                ".*o_proj.*",
                ".*qkv_proj.*",
                ".*out_proj.*",
                ".*output_gate.*",
                ".*block_sparse_moe.experts.*w1.*",
                ".*block_sparse_moe.experts.*w2.*",
                ".*block_sparse_moe.experts.*w3.*",
            ],
        )

        lora_model = LoRAModel(
            model,
            LoRAConfig(
                target_modules=target_modules,
                r=4,
                lora_alpha=8,
                merge_weights=False,
                dtype="float32",
            ),
        )
        injected_layers = {
            name for name, layer in lora_model.model.named_sublayers() if type(layer).__name__ == "LoRALinear"
        }
        expected_attention_layers = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.1.self_attn.qkv_proj",
            "model.layers.1.self_attn.out_proj",
            "model.layers.1.self_attn.output_gate",
        }
        expected_expert_layers = {
            f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.{projection}"
            for layer_idx in range(config.num_hidden_layers)
            for expert_idx in range(config.num_local_experts)
            for projection in ("w1", "w2", "w3")
        }
        self.assertSetEqual(injected_layers, expected_attention_layers | expected_expert_layers)
        self.assertNotIn("model.layers.0.block_sparse_moe.gate", injected_layers)

    def test_linear_attention_accepts_single_segment_row_indices(self):
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = MiniMaxModel(config)
        model.eval()

        row_indices = paddle.full(
            [self.model_tester.batch_size, 1, self.model_tester.seq_length, 1],
            self.model_tester.seq_length,
            dtype=paddle.int32,
        )
        expected = model(input_ids)[0]
        actual = model(input_ids, attn_mask_startend_row_indices=row_indices)[0]
        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_linear_attention_packing_matches_independent_sequences(self):
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = MiniMaxModel(config)
        model.eval()
        first_length = 3
        second_length = self.model_tester.seq_length - first_length
        position_ids = paddle.to_tensor(
            [list(range(first_length)) + list(range(second_length))] * self.model_tester.batch_size,
            dtype=paddle.int64,
        )
        row_indices = paddle.to_tensor(
            [
                [
                    [[first_length] for _ in range(first_length)]
                    + [[self.model_tester.seq_length] for _ in range(second_length)]
                ]
                for _ in range(self.model_tester.batch_size)
            ],
            dtype=paddle.int32,
        )

        packed_output = model(
            input_ids,
            position_ids=position_ids,
            attn_mask_startend_row_indices=row_indices,
        )[0]
        first_output = model(input_ids[:, :first_length])[0]
        second_output = model(input_ids[:, first_length:])[0]
        expected = paddle.cat([first_output, second_output], axis=1)
        paddle.testing.assert_close(packed_output, expected, rtol=1e-5, atol=1e-5)

    def test_linear_attention_right_padding_preserves_valid_prefix(self):
        config = self.model_tester.get_config()
        layer = MiniMaxLightningAttention(config, layer_idx=1)
        layer.eval()
        valid_length = 5
        hidden_states = paddle.randn([1, self.model_tester.seq_length, config.hidden_size])
        position_ids = paddle.to_tensor([[0, 1, 2, 3, 4, 0, 0]], dtype=paddle.int64)
        row_indices = paddle.to_tensor(
            [
                [
                    [[valid_length] for _ in range(valid_length)]
                    + [[key_idx] for key_idx in range(valid_length, self.model_tester.seq_length)]
                ]
            ],
            dtype=paddle.int32,
        )

        actual = layer(
            hidden_states,
            position_ids=position_ids,
            attn_mask_startend_row_indices=row_indices,
        )[0]
        expected_valid = layer(hidden_states[:, :valid_length])[0]

        paddle.testing.assert_close(
            actual[:, :valid_length],
            expected_valid,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(
            paddle.max(paddle.abs(actual[:, valid_length:])).item(),
            0.0,
        )

    def test_linear_attention_branched_mask_matches_dense_reference(self):
        config = self.model_tester.get_config()
        layer = MiniMaxLightningAttention(config, layer_idx=1)
        layer.eval()
        hidden_states = paddle.randn([1, 8, config.hidden_size])
        position_ids = paddle.to_tensor([[0, 1, 2, 3, 4, 2, 3, 4]], dtype=paddle.int64)

        attention_mask = paddle.tril(paddle.ones([1, 1, 8, 8], dtype=paddle.bool))
        attention_mask[:, :, 5:, 2:5] = False
        actual = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        qkv_states = layer.act_fn(layer.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape([1, 8, config.num_attention_heads, 3 * config.head_dim])
        query_states, key_states, value_states = paddle.split(qkv_states, num_or_sections=3, axis=-1)
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        logical_distance = position_ids[:, :, None] - position_ids[:, None, :]
        decay = paddle.exp(
            -layer.slope_rate.unsqueeze(0) * logical_distance.astype(layer.slope_rate.dtype).unsqueeze(1)
        )
        decay = paddle.where(attention_mask, decay, paddle.zeros_like(decay))
        weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * decay
        expected = paddle.matmul(weights, value_states)
        expected = expected.transpose([0, 2, 1, 3]).reshape([1, 8, -1])
        expected = layer.norm(expected)
        expected = F.sigmoid(layer.output_gate(hidden_states)).astype(expected.dtype) * expected
        expected = layer.out_proj(expected)

        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_linear_attention_dpo_row_indices_match_explicit_reference(self):
        config = self.model_tester.get_config()
        layer = MiniMaxLightningAttention(config, layer_idx=1)
        layer.eval()
        valid_length = 8
        hidden_states = paddle.randn([1, 10, config.hidden_size])
        position_ids = paddle.to_tensor([[0, 1, 2, 3, 4, 2, 3, 4, 0, 0]], dtype=paddle.int64)
        end_indices = [8, 8, 8, 5, 5, 8, 8, 8, 8, 9]
        row_indices = paddle.to_tensor([[list(zip(end_indices))]], dtype=paddle.int32)

        actual = layer(
            hidden_states,
            position_ids=position_ids,
            attn_mask_startend_row_indices=row_indices,
        )[0]

        qkv_states = layer.act_fn(layer.qkv_proj(hidden_states))
        qkv_states = qkv_states.reshape([1, 10, config.num_attention_heads, 3 * config.head_dim])
        query_states, key_states, value_states = paddle.split(qkv_states, num_or_sections=3, axis=-1)
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        attention_mask = paddle.zeros([1, 1, 10, 10], dtype=paddle.bool)
        for key_idx, end_idx in enumerate(end_indices[:valid_length]):
            attention_mask[:, :, key_idx:end_idx, key_idx] = True
        logical_distance = position_ids[:, :, None] - position_ids[:, None, :]
        decay = paddle.exp(
            -layer.slope_rate.unsqueeze(0) * logical_distance.astype(layer.slope_rate.dtype).unsqueeze(1)
        )
        decay = paddle.where(attention_mask, decay, paddle.zeros_like(decay))
        weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * decay
        expected = paddle.matmul(weights, value_states)
        expected = expected.transpose([0, 2, 1, 3]).reshape([1, 10, -1])
        expected = layer.norm(expected)
        expected = F.sigmoid(layer.output_gate(hidden_states)).astype(expected.dtype) * expected
        expected = layer.out_proj(expected)

        paddle.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        actual.mean().backward()
        self.assertIsNotNone(layer.qkv_proj.weight.grad)
        self.assertTrue(paddle.isfinite(layer.qkv_proj.weight.grad).all().item())

    def test_causal_lm_passes_loss_mask_to_criterion(self):
        class RecordingCriterion(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.received_loss_mask = None

            def forward(self, logits, labels, loss_mask=None):
                self.received_loss_mask = loss_mask
                return paddle.zeros([], dtype=logits.dtype), None

        config, input_ids, input_mask = self.model_tester.prepare_config_and_inputs()
        model = MiniMaxForCausalLM(config)
        model.criterion = RecordingCriterion()
        loss_mask = paddle.ones_like(input_ids, dtype=paddle.float32)

        model(input_ids, attention_mask=input_mask, labels=input_ids, loss_mask=loss_mask)

        self.assertIs(model.criterion.received_loss_mask, loss_mask)

    def test_causal_lm_rejects_router_logits_output(self):
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = MiniMaxForCausalLM(config)

        with self.assertRaisesRegex(NotImplementedError, "does not support output_router_logits"):
            model(input_ids, output_router_logits=True)
