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

import numpy as np
import paddle

from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS


class TestAttentionInterface(unittest.TestCase):
    def gen_random_flashmask(bz, num_head, seqlen, has_end, causal):
        mask_num = 1
        if not causal:
            mask_num *= 2
        if has_end:
            mask_num *= 2
        m = np.random.randint(0, seqlen, (bz, num_head, seqlen, mask_num))
        diag = np.arange(seqlen).reshape((1, 1, seqlen))
        m[:, :, :, 0] = np.maximum(diag + 1, m[:, :, :, 0])
        if not causal:
            if has_end:
                raise NotImplementedError
            else:
                m[:, :, :, 1] = np.minimum(diag, m[:, :, :, 1])
        else:
            if has_end:
                m[:, :, :, 1] = m[:, :, :, 0] + 1
                m[:, :, :, 1] = np.maximum(m[:, :, :, 0], m[:, :, :, 1])

        return paddle.to_tensor(m, dtype="int32")

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
        self.sink = paddle.randn([self.num_heads], dtype="float16")
        self.startend_row_indices = self.gen_random_flashmask(
            self.batch_size, self.num_heads, self.seq_len, has_end=False, causal=False
        )

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
        sdpa_interface(
            self,
            self.query,
            self.key,
            self.value,
            sink=self.sink,
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
        flashmask_interface(
            self,
            self.query,
            self.key,
            self.value,
            scaling=self.scaling,
            attn_mask_start_row_indices=self.startend_row_indices,
            sink=self.sink,
        )


if __name__ == "__main__":
    unittest.main()
