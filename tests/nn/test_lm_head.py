# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddleformers.nn.lm_head import LMHead
from paddleformers.transformers import LlamaConfig


class TestLMHead(unittest.TestCase):
    def test_initialization_default(self):
        # Test default initialization
        config = LlamaConfig()
        lm_head = LMHead(config)

        # Check weight shape and attributes
        self.assertEqual(lm_head.weight.shape, [config.vocab_size, config.hidden_size])
        self.assertFalse(lm_head.weight.is_distributed)
        self.assertIsNone(lm_head.bias)
        self.assertFalse(lm_head.vocab_parallel)

    def test_initialization_with_tie_word_embeddings(self):
        # Test initialization with tied embeddings
        config = LlamaConfig()
        config.tie_word_embeddings = True
        lm_head = LMHead(config)

        self.assertEqual(lm_head.weight.shape, [config.vocab_size, config.hidden_size])

    def test_forward_normal(self):
        # Test normal forward pass
        config = LlamaConfig()
        test_input = paddle.randn([1, 10, config.hidden_size])
        lm_head = LMHead(config)
        lm_head(test_input)

    def test_forward_fused_head_loss(self):
        # Test forward with recompute loss flag
        config = LlamaConfig()
        config.use_fused_head_and_loss_fn = True
        lm_head = LMHead(config)
        test_input = paddle.randn([1, 10, config.hidden_size])

        output = lm_head(test_input)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)
        self.assertEqual(output[0].shape, test_input.shape)
        self.assertEqual(output[1].shape, lm_head.weight.shape)
        self.assertIs(output[2], lm_head.bias)
        self.assertEqual(output[3], True)


if __name__ == "__main__":
    unittest.main()
