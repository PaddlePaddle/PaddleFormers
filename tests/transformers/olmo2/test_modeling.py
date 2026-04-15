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

import paddle

from paddleformers.transformers.olmo2 import Olmo2Config, Olmo2ForCausalLM, Olmo2Model


class Olmo2ModelTest(unittest.TestCase):
    def get_config(self):
        return Olmo2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            use_cache=False,
            attention_dropout=0.0,
            _attn_implementation="eager",
            fuse_rms_norm=False,
        )

    def test_model_forward(self):
        config = self.get_config()
        model = Olmo2Model(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype="int64")

        with paddle.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)

        self.assertEqual(outputs.last_hidden_state.shape, [2, 4, config.hidden_size])

    def test_lm_forward_and_loss(self):
        config = self.get_config()
        model = Olmo2ForCausalLM(config)
        model.eval()
        input_ids = paddle.to_tensor([[1, 5, 7, 9], [3, 4, 8, 12]], dtype="int64")
        labels = paddle.to_tensor([[5, 7, 9, 11], [4, 8, 12, 16]], dtype="int64")

        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)

        self.assertEqual(outputs.logits.shape, [2, 4, config.vocab_size])
        self.assertEqual(outputs.loss.shape, [])


if __name__ == "__main__":
    unittest.main()
