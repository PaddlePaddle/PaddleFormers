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

import numpy as np
import paddle
from paddlefleet_ops import router_metadata


def expert_parallel_TC_topk_router_metadata(topk_router_indices: paddle.Tensor, expert_frequency_offset, K: int):
    invalid_tokens = paddle.sum(topk_router_indices == -1)
    s_scatter_idx = paddle.argsort(topk_router_indices.reshape([-1])).astype("int64")
    expert_frequency_offset = paddle.concat(
        [
            paddle.zeros([1], dtype=expert_frequency_offset.dtype),
            expert_frequency_offset,
        ]
    )

    num_activated_expert_per_token_offset = (topk_router_indices > -1).sum(axis=-1)
    num_activated_expert_per_token_offset = paddle.concat(
        [
            paddle.to_tensor([0], dtype=num_activated_expert_per_token_offset.dtype),
            num_activated_expert_per_token_offset,
        ]
    )
    num_activated_expert_per_token_offset = num_activated_expert_per_token_offset.cumsum(0).astype("int64")

    x_gather_idx = s_scatter_idx // K

    topk_router_indices_valid = topk_router_indices[topk_router_indices >= 0]
    s_scatter_idx_valid = paddle.argsort(topk_router_indices_valid.reshape([-1])).astype("int64")
    s_reverse_scatter_idx_valid = paddle.empty_like(s_scatter_idx_valid)
    s_reverse_scatter_idx_valid[s_scatter_idx_valid] = paddle.arange(
        s_scatter_idx_valid.shape[0], dtype=s_scatter_idx_valid.dtype
    )

    return (
        expert_frequency_offset,
        x_gather_idx[invalid_tokens:],
        s_scatter_idx_valid,
        s_reverse_scatter_idx_valid,
        num_activated_expert_per_token_offset,
    )


class TestRouterMetadataOp(unittest.TestCase):
    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        np.random.seed(2026)
        paddle.seed(2026)

    def run_test_case(self, num_tokens, k, n_expert):
        topk_router_indices_np = np.random.randint(-1, n_expert, size=(num_tokens, k)).astype("int64")

        expert_frequency_offset_np = np.random.randint(0, 20, size=(n_expert,)).astype("int64")

        topk_router_indices = paddle.to_tensor(topk_router_indices_np, place="gpu")
        expert_frequency_offset = paddle.to_tensor(expert_frequency_offset_np, place="gpu")

        ref_out = expert_parallel_TC_topk_router_metadata(topk_router_indices, expert_frequency_offset, k)

        custom_out = router_metadata(topk_router_indices, expert_frequency_offset, k)

        self.assertEqual(len(ref_out), len(custom_out), "Number of outputs mismatch")
        for i in range(len(ref_out)):
            ref = ref_out[i].cpu().numpy()
            custom = custom_out[i].cpu().numpy()
            self.assertEqual(ref.shape, custom.shape, f"Shape mismatch at output {i}")
            np.testing.assert_allclose(
                ref,
                custom,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Output mismatch at index {i}",
            )

    def test_router_metadata(self):
        test_cases = [
            (8, 2, 6),
            (8, 3, 8),
            (6, 2, 4),
            (1024, 2, 8),
        ]
        for num_tokens, k, n_expert in test_cases:
            with self.subTest(num_tokens=num_tokens, k=k, n_expert=n_expert):
                self.run_test_case(num_tokens, k, n_expert)


if __name__ == "__main__":
    unittest.main()
