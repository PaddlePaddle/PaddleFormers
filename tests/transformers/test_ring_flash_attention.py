# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
import paddle
from paddle.nn.functional.flash_attention import scaled_dot_product_attention

from paddleformers.transformers.ring_flash_attention import (
    RingFlashAttention,
    get_chunk_id,
)


class TestRingFlashAttention(unittest.TestCase):
    def setUp(self):
        paddle.distributed.init_parallel_env()
        self.group = paddle.distributed.new_group(range(paddle.distributed.get_world_size()), backend="nccl")
        self.degree = self.group.world_size
        self.rank = self.group.rank

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        self.test_id = 0

    def generate_full_data(self, batch_size, seq_len, num_head, head_dim):
        query = paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.bfloat16)
        key = paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.bfloat16)
        value = paddle.randn([batch_size, seq_len, num_head, head_dim], dtype=paddle.bfloat16)

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        return query, key, value

    def split_belanced_data(self, input):
        sliced_datas = paddle.split(input, num_or_sections=self.degree * 2, axis=1)
        sliced_data0, sliced_data1 = sliced_datas[self.rank], sliced_datas[self.degree * 2 - 1 - self.rank]
        return paddle.cat([sliced_data0, sliced_data1], axis=1).detach()

    def single_test(self, bsz, seq_len_per_device, head_num, head_dim, is_causal, use_mask):
        if self.degree < 2:
            return
        query, key, value = self.generate_full_data(bsz, seq_len_per_device * self.degree, head_num, head_dim)

        local_query = self.split_belanced_data(query)
        local_key = self.split_belanced_data(key)
        local_value = self.split_belanced_data(value)

        local_query.stop_gradient = False
        local_key.stop_gradient = False
        local_value.stop_gradient = False

        if use_mask:
            mask_shape = (bsz, 1, query.shape[1], query.shape[1])
            mask = np.random.random(mask_shape)
            attn_mask = paddle.to_tensor(mask, place=query.place, dtype=query.dtype)
            attn_mask = paddle.ones(mask_shape).to(query.dtype)
            attn_mask_list = paddle.split(attn_mask, axis=2, num_or_sections=self.degree * 2)
            first_chunk_id, second_chunk_id = get_chunk_id(self.rank, self.degree)
            local_attn_mask = paddle.cat([attn_mask_list[first_chunk_id], attn_mask_list[second_chunk_id]], axis=2)
        else:
            attn_mask = None
            local_attn_mask = None

        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            local_out = RingFlashAttention.apply(
                local_query, local_key, local_value, self.group, is_causal=is_causal, attn_mask=local_attn_mask
            )
            ref_out = scaled_dot_product_attention(query, key, value, is_causal=is_causal, attn_mask=attn_mask)

        local_out.backward()
        ref_out.backward()

        ref_local_query_grad = self.split_belanced_data(query.grad)
        ref_local_key_grad = self.split_belanced_data(key.grad)
        ref_local_value_grad = self.split_belanced_data(value.grad)

        ref_local_out = self.split_belanced_data(ref_out)
        rtol = 1e-02
        atol = 1e-02
        np.testing.assert_allclose(
            local_out.to("float32").numpy(), ref_local_out.to("float32").numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            local_query.grad.to("float32").numpy(), ref_local_query_grad.to("float32").numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            local_key.grad.to("float32").numpy(), ref_local_key_grad.to("float32").numpy(), rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(
            local_value.grad.to("float32").numpy(), ref_local_value_grad.to("float32").numpy(), rtol=rtol, atol=atol
        )

        print(f"Test {self.test_id} passed!")
        self.test_id += 1

    def test_normal_flash_attention(self):
        self.single_test(2, 1024, 2, 128, False, False)

    def test_masked_flash_attention(self):
        self.single_test(2, 1024, 2, 128, False, True)

    def test_casual_flash_attention(self):
        self.single_test(2, 1024, 2, 128, True, False)


if __name__ == "__main__":
    unittest.main()
