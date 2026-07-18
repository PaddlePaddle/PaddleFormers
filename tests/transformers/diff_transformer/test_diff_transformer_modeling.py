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

import paddle

from paddleformers.transformers.diff_transformer.configuration import (
    DiffTransformerConfig,
)
from paddleformers.transformers.diff_transformer.modeling import (
    DiffTransformerForCausalLM,
    DiffTransformerModel,
)


class DiffTransformerModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.vocab_size = 32000
        self.hidden_size = 32
        self.num_hidden_layers = 1
        self.num_attention_heads = 2
        self.batch_size = 2
        self.seq_length = 8

    def get_config(self):
        return DiffTransformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=64,
            max_position_embeddings=128,
        )

    def prepare_inputs(self):
        input_ids = paddle.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        return input_ids

    def check_model(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = DiffTransformerModel(config)
        model.eval()
        output = model(input_ids)
        self.parent.assertIsNotNone(output)

    def check_causal_lm(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = DiffTransformerForCausalLM(config)
        model.eval()
        output = model(input_ids)
        self.parent.assertIsNotNone(output)

    def check_loss(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = DiffTransformerForCausalLM(config)
        model.train()
        loss, logits = model(input_ids, labels=input_ids)
        self.parent.assertIsInstance(loss.item(), float)

    def check_backward(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = DiffTransformerForCausalLM(config)
        loss, _ = model(input_ids, labels=input_ids)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                self.parent.assertIsNotNone(p.grad)


class DiffTransformerTest(unittest.TestCase):
    def setUp(self):
        self.tester = DiffTransformerModelTester(self)

    def test_model_forward(self):
        self.tester.check_model()

    def test_causal_lm_forward(self):
        self.tester.check_causal_lm()

    def test_loss_computation(self):
        self.tester.check_loss()

    def test_backward_pass(self):
        self.tester.check_backward()

    def test_public_imports(self):
        from paddleformers.transformers import (
            DiffTransformerConfig,
            DiffTransformerForCausalLM,
        )

        self.assertIsNotNone(DiffTransformerConfig)
        self.assertIsNotNone(DiffTransformerForCausalLM)

    def test_auto_module_does_not_export_diff_transformer(self):
        import paddleformers.transformers.auto as auto

        self.assertNotIn("DiffTransformerConfig", auto.__all__)
        with self.assertRaises(AttributeError):
            auto.DiffTransformerConfig

    def test_head_dim_must_match_hidden_size(self):
        with self.assertRaisesRegex(ValueError, "head_dim \* num_attention_heads"):
            DiffTransformerConfig(hidden_size=32, num_attention_heads=2, head_dim=8)

    def test_causal_lm_does_not_attend_to_future_tokens(self):
        config = self.tester.get_config()
        model = DiffTransformerForCausalLM(config)
        model.eval()
        input_ids = self.tester.prepare_inputs()
        modified_input_ids = input_ids.clone()
        modified_input_ids[:, -1] = (modified_input_ids[:, -1] + 1) % config.vocab_size
        attention_mask = paddle.ones_like(input_ids)

        with paddle.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
            modified_logits = model(modified_input_ids, attention_mask=attention_mask)

        paddle.testing.assert_close(logits[:, :-1], modified_logits[:, :-1])


if __name__ == "__main__":
    unittest.main()
