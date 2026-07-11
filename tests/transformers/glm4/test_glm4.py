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

import paddle

from paddleformers.transformers.glm4.configuration import Glm4Config
from paddleformers.transformers.glm4.modeling import Glm4ForCausalLM, Glm4Model


class Glm4ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 8
        # 使用较小的网络参数，保证单元测试秒级运行，且不会爆内存
        self.vocab_size = 1000
        self.hidden_size = 32
        self.intermediate_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 8  # GLM4 中 head_dim * num_attention_heads == hidden_size

    def get_config(self):
        return Glm4Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=128,
            use_cache=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=[2],
        )

    def prepare_inputs(self):
        # 保证 input_ids 在 vocab_size 范围内
        input_ids = paddle.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        return input_ids

    def check_model(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = Glm4Model(config)
        model.eval()
        # Glm4Model 默认 return_dict=True，返回字典
        output = model(input_ids)
        self.parent.assertIsNotNone(output)
        self.parent.assertIn("last_hidden_state", output)
        # 校验 hidden state 的形状 [batch_size, seq_len, hidden_size]
        self.parent.assertEqual(
            output["last_hidden_state"].shape,
            [self.batch_size, self.seq_length, self.hidden_size],
        )

    def check_causal_lm(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = Glm4ForCausalLM(config)
        model.eval()
        # Glm4ForCausalLM 默认返回 CausalLMOutputWithPast 对象
        output = model(input_ids)
        self.parent.assertIsNotNone(output.logits)
        # 校验 logits 的形状 [batch_size, seq_len, vocab_size]
        self.parent.assertEqual(
            output.logits.shape,
            [self.batch_size, self.seq_length, self.vocab_size],
        )

    def check_loss(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = Glm4ForCausalLM(config)
        model.train()
        # 传入 labels 触发内部的交叉熵 Loss 计算
        output = model(input_ids, labels=input_ids)
        self.parent.assertIsNotNone(output.loss)
        self.parent.assertIsInstance(output.loss.item(), float)

    def check_backward(self):
        config = self.get_config()
        input_ids = self.prepare_inputs()
        model = Glm4ForCausalLM(config)
        model.train()
        # 反向传播前清理可能残留的梯度
        model.clear_gradients()
        output = model(input_ids, labels=input_ids)
        loss = output.loss
        loss.backward()
        # 验证所有需要训练的参数是否都成功计算了梯度
        for name, p in model.named_parameters():
            if not p.stop_gradient:
                self.parent.assertIsNotNone(p.grad, msg=f"Parameter {name} has no gradient.")


class Glm4Test(unittest.TestCase):
    def setUp(self):
        self.tester = Glm4ModelTester(self)

    def test_model_forward(self):
        self.tester.check_model()

    def test_causal_lm_forward(self):
        self.tester.check_causal_lm()

    def test_loss_computation(self):
        self.tester.check_loss()

    def test_backward_pass(self):
        self.tester.check_backward()


if __name__ == "__main__":
    unittest.main()
