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

import unittest
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddleformers.cli.utils.llm_utils import get_lora_target_modules
from paddleformers.transformers.auto.configuration import CONFIG_MAPPING
from paddleformers.transformers.auto.modeling import AutoModel, AutoModelForCausalLM
from paddleformers.transformers.telechat.configuration import TelechatConfig
from paddleformers.transformers.telechat.modeling import (
    TelechatAttention,
    TelechatForCausalLM,
    TelechatModel,
    TelechatRotaryEmbedding,
)


class TelechatModelTest(unittest.TestCase):
    def setUp(self):
        paddle.seed(42)
        self.config = TelechatConfig(vocab_size=64, hidden_size=64, ffn_hidden_size=128, n_layer=2, n_head=4)
        self.input_ids = paddle.randint(0, 64, shape=[2, 5], dtype="int64")

    def test_rotary_embedding_preserves_fp32_dtype(self):
        rotary_emb = TelechatRotaryEmbedding(self.config)
        x = paddle.randn([2, 5, self.config.hidden_size], dtype="float32")
        position_ids = paddle.arange(5, dtype="int64").unsqueeze(0).tile([2, 1])
        cos, sin = rotary_emb(x, position_ids)
        self.assertEqual(cos.dtype, paddle.float32)
        self.assertEqual(sin.dtype, paddle.float32)

    def test_attention_uses_local_heads_for_tensor_parallel_config(self):
        config = TelechatConfig(
            vocab_size=64,
            hidden_size=64,
            ffn_hidden_size=128,
            n_layer=2,
            n_head=4,
            tensor_model_parallel_size=2,
        )

        def create_linear(in_features, out_features, **kwargs):
            if kwargs["tp_plan"] == "colwise":
                out_features //= config.tensor_model_parallel_size
            else:
                in_features //= config.tensor_model_parallel_size
            return paddle.nn.Linear(in_features, out_features, bias_attr=kwargs["has_bias"])

        with patch("paddleformers.transformers.telechat.modeling.GeneralLinear.create", side_effect=create_linear):
            attention = TelechatAttention(config, layer_idx=0)
        hidden_states = paddle.randn([2, 5, config.hidden_size])
        position_ids = paddle.arange(5, dtype="int64").unsqueeze(0).tile([2, 1])
        position_embeddings = TelechatRotaryEmbedding(config)(hidden_states, position_ids)
        output = attention(hidden_states, position_embeddings=position_embeddings)

        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(list(output.shape), [2, 5, 64])

    def test_lm_head_gathers_tensor_parallel_output(self):
        config = TelechatConfig(vocab_size=97, hidden_size=64, ffn_hidden_size=128, n_layer=2, n_head=4)
        with patch(
            "paddleformers.transformers.telechat.modeling.GeneralLinear.create", return_value=paddle.nn.Identity()
        ) as create:
            TelechatForCausalLM(config)

        lm_head_calls = [
            call
            for call in create.call_args_list
            if call.args[1] == config.vocab_size and call.kwargs.get("tp_plan") == "colwise"
        ]
        self.assertTrue(any(call.kwargs.get("gather_output") is True for call in lm_head_calls))

    def test_forward_shape_and_auto_config(self):
        self.assertIs(CONFIG_MAPPING["telechat"], TelechatConfig)
        model = TelechatModel(self.config)
        result = model(self.input_ids, return_dict=True)
        self.assertEqual(list(result.last_hidden_state.shape), [2, 5, 64])

    def test_loss_mask_and_empty_mask(self):
        model = TelechatForCausalLM(self.config)
        labels = self.input_ids.clone()
        loss_mask = paddle.ones_like(labels)
        loss_mask[:, 2:4] = 0
        output = model(self.input_ids, labels=labels, loss_mask=loss_mask, return_dict=True)
        self.assertEqual(list(output.logits.shape), [2, 5, 64])
        self.assertTrue(bool(paddle.isfinite(output.loss).item()))

        shift_logits = output.logits[:, :-1].reshape([-1, self.config.vocab_size])
        shift_labels = labels[:, 1:].reshape([-1])
        valid = loss_mask[:, 1:].reshape([-1]).astype("bool")
        expected_loss = F.cross_entropy(shift_logits[valid], shift_labels[valid])
        self.assertTrue(bool(paddle.allclose(output.loss, expected_loss)))

        empty_output = model(self.input_ids, labels=labels, loss_mask=paddle.zeros_like(labels), return_dict=True)
        self.assertEqual(float(empty_output.loss), 0.0)

    def test_auto_model_routing(self):
        self.assertIsInstance(AutoModel.from_config(self.config), TelechatModel)
        causal_lm_config = TelechatConfig(**self.config.to_dict())
        causal_lm_config.architectures = ["TelechatForCausalLM"]
        causal_lm_config.dtype = "float32"
        self.assertIsInstance(AutoModelForCausalLM.from_config(causal_lm_config), TelechatForCausalLM)

    def test_lora_target_modules(self):
        self.assertEqual(
            get_lora_target_modules(TelechatForCausalLM(self.config)),
            [
                ".*query.*",
                ".*key_value.*",
                ".*dense.*",
                ".*gate_proj.*",
                ".*up_proj.*",
                ".*down_proj.*",
            ],
        )

    def test_cached_decode_matches_full_forward(self):
        model = TelechatForCausalLM(self.config)
        model.eval()
        prompt = self.input_ids[:, :3]
        next_token = self.input_ids[:, 3:4]
        cached = model(prompt, use_cache=True, return_dict=True)
        decode = model(next_token, past_key_values=cached.past_key_values, use_cache=True, return_dict=True)
        full = model(paddle.concat([prompt, next_token], axis=-1), use_cache=False, return_dict=True)
        self.assertTrue(bool(paddle.allclose(decode.logits[:, -1], full.logits[:, -1], atol=1e-5)))

    def test_generate(self):
        model = TelechatForCausalLM(self.config)
        model.eval()
        generated = model.generate(self.input_ids[:, :3], max_new_tokens=2, decode_strategy="greedy_search")
        if isinstance(generated, tuple):
            generated = generated[0]
        self.assertEqual(list(generated.shape), [2, 2])


if __name__ == "__main__":
    unittest.main()
