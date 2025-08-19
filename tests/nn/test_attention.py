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

from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS


class TestAttentionInterface(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 32
        self.num_heads = 8
        self.embed_dim = 128
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.training = True
        self.query = paddle.randn([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype="float16")
        self.key = paddle.randn([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype="float16")
        self.value = paddle.randn([self.batch_size, self.seq_len, self.num_heads, self.head_dim], dtype="float16")

    def test_forward_calls_correct_function(self):
        eager_interface = ALL_ATTENTION_FUNCTIONS["eager"]

        eager_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
        )
        sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        sdpa_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
        )
        flashmask_interface = ALL_ATTENTION_FUNCTIONS["flashmask"]
        flashmask_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
        )


if __name__ == "__main__":
    unittest.main()
