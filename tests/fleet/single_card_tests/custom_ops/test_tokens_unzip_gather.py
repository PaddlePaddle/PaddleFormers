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

import itertools
import unittest

import numpy as np
import paddle
from paddle.nn.functional import moe_permute
from paddlefleet_ops import tokens_unzip_gather


def fabricate_dispatch_result(
    seqlen,
    token_length,
    topk,
    num_experts,
    data_type="bfloat16",
    broadcast_ratio=0.5,
    using_ue8m0_scale=False,
):
    """Helper function to generate test data."""
    hidden_states = paddle.randn([seqlen, token_length]).astype(data_type)

    scale = paddle.empty([0])
    if data_type == "float8_e4m3fn":
        if using_ue8m0_scale:
            scale_cols = (token_length + 127) // 128
            scale = paddle.randn([seqlen, scale_cols], dtype="float32").astype(paddle.int32)
        else:
            scale_cols = (token_length + 127) // 128
            scale = paddle.randn([seqlen, scale_cols], dtype="float32")

    # Calculate expert counts with normal distribution
    expected_experts = max(1, min(broadcast_ratio * num_experts, topk))
    std_dev = max(1, expected_experts / 6)
    experts_count = paddle.normal(expected_experts, std_dev, [seqlen])
    experts_count = paddle.clip(paddle.round(experts_count), 1, min(topk, num_experts))
    experts_count = paddle.cast(experts_count, "int32")

    # Preallocate results
    expert_routemap_topk = paddle.full([seqlen, topk], -1, dtype="int32")
    expert_prob_topk = paddle.zeros([seqlen, topk], dtype="float32")

    # Batch generate expert indices and probabilities
    for i in range(seqlen):
        count = experts_count[i].item()
        indices = paddle.randperm(num_experts)[:count]
        expert_routemap_topk[i, :count] = indices
        prob_value = 1.0 / count
        expert_prob_topk[i, :count] = paddle.full([count], prob_value, dtype=data_type)

    # Calculate expert token counts
    valid_indices = expert_routemap_topk.reshape([-1])
    valid_mask = valid_indices >= 0
    valid_experts = valid_indices[valid_mask]
    tokens_per_expert = paddle.histogram(valid_experts, bins=num_experts, min=0, max=num_experts - 1)
    tokens_per_expert = paddle.cast(tokens_per_expert, "int32")
    tokens_per_expert = list(tokens_per_expert)

    return (
        hidden_states,
        scale,
        expert_routemap_topk,
        expert_prob_topk,
        tokens_per_expert,
    )


class TestTokensUnzipGatherUE8M0Scale(unittest.TestCase):
    def test_tokens_unzip_gather(self):
        SEQLEN = 16384
        TOKEN_LEN = 7168
        DTYPES = ["float8_e4m3fn"]
        EXPERT_NUMS = [16]
        TOPKS = [8]
        # Generate test data
        for dt, expert_num, topk in itertools.product(DTYPES, EXPERT_NUMS, TOPKS):
            with self.subTest(dtype=dt, expert_num=expert_num, topk=topk):
                (
                    hidden_states,
                    scale,
                    expert_routemap_topk,
                    expert_prob_topk,
                    tokens_per_expert,
                ) = fabricate_dispatch_result(
                    SEQLEN,
                    TOKEN_LEN,
                    topk,
                    expert_num,
                    data_type=dt,
                    broadcast_ratio=0.5,
                    using_ue8m0_scale=True,
                )
                # Generate float32 scale, only cast the scale to float32, don't change shape
                scale_fp32 = paddle.cast(scale, "float32")

                # Using Permute get rowmap
                (_, zipped_expertwise_rowmap, _, _,) = moe_permute(
                    hidden_states,
                    scale_fp32,
                    expert_routemap_topk,
                    expert_prob_topk,
                    num_experts=expert_num,
                    tokens_per_expert=tokens_per_expert,
                    padding_alignment=128,
                )

                for expert_id in range(expert_num):
                    # Test tokens_unzip_gather with int32 (four ue8m0) scale
                    (x_unzipped, x_scale_unzipped, index_unzipped) = tokens_unzip_gather(
                        hidden_states,
                        scale,
                        zipped_expertwise_rowmap,
                        expert_id,
                        tokens_per_expert,
                        128,
                    )
                    x_scale_unzipped_np = x_scale_unzipped.numpy()
                    # Test tokens_unzip_gather with float32 scale
                    (x_unzipped, x_scale_fp32_unzipped, index_unzipped_fp32) = tokens_unzip_gather(
                        hidden_states,
                        scale_fp32,
                        zipped_expertwise_rowmap,
                        expert_id,
                        tokens_per_expert,
                        128,
                    )
                    # Verify the result of scale is the same
                    self.assertTrue(
                        np.allclose(
                            x_scale_unzipped_np,
                            x_scale_fp32_unzipped.astype("int32").numpy(),
                        )
                    )
                    index_unzipped_np = index_unzipped.numpy()
                    x_unzipped_np = x_unzipped.numpy()
                    check_rows = min(len(index_unzipped_np), 5)
                    for i in range(check_rows):
                        index = index_unzipped_np[i]
                        self.assertTrue(np.allclose(x_unzipped_np[i], hidden_states.numpy()[index]))


if __name__ == "__main__":
    unittest.main()
