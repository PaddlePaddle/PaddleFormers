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
from paddle.base import core
from paddlefleet_ops import fuse_stack_fp8_quant, fuse_stack_transpose_fp8_quant

from paddleformers.fleet.triton_ops import fuse_stack_ue8m0_scale_transpose


class TestFuseStackUe8m0ScaleTransposeTriton(unittest.TestCase):
    def _skip_if_not_sm10(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("UE8M0 scale path requires SM10+ (Blackwell)")

    def _assert_matches_fused_transpose(self, num_experts, m, k, seed):
        paddle.seed(seed)
        inputs = [paddle.randn([m, k], dtype=paddle.bfloat16) for _ in range(num_experts)]

        out, scale = fuse_stack_fp8_quant(
            inputs,
            False,
            True,
            False,
        )
        transpose_out, transpose_scale = fuse_stack_transpose_fp8_quant(
            inputs,
            False,
            True,
            False,
        )
        converted_scale = fuse_stack_ue8m0_scale_transpose(scale, num_experts, m, k)

        expected_out = out.reshape([num_experts, m, k]).transpose([0, 2, 1]).reshape([-1, m])
        np.testing.assert_allclose(
            expected_out.numpy(),
            transpose_out.numpy(),
            atol=0,
            rtol=0,
            err_msg=f"output mismatch for num_experts={num_experts}, m={m}, k={k}",
        )
        np.testing.assert_allclose(
            converted_scale.numpy(),
            transpose_scale.numpy(),
            atol=0,
            rtol=0,
            err_msg=f"scale mismatch for num_experts={num_experts}, m={m}, k={k}",
        )

    def test_matches_fuse_stack_transpose_ue8m0_scale(self):
        self._skip_if_not_sm10()
        for seed, (num_experts, m, k) in enumerate(
            [
                (1, 512, 512),
                (2, 512, 1024),
                (3, 1024, 512),
                (4, 1024, 1536),
                (5, 1536, 1024),
            ],
            start=2026,
        ):
            with self.subTest(num_experts=num_experts, m=m, k=k):
                self._assert_matches_fused_transpose(num_experts, m, k, seed)

    def test_zero_size(self):
        self._skip_if_not_sm10()
        test_cases = [
            (0, 512, 512, [0, 1], [0, 1]),
            (2, 0, 512, [0, 1], [1024, 0]),
            (2, 512, 0, [1024, 0], [0, 1]),
        ]
        for num_experts, m, k, scale_shape, expected_shape in test_cases:
            with self.subTest(num_experts=num_experts, m=m, k=k):
                scale = paddle.empty(scale_shape, dtype=paddle.int32)
                converted_scale = fuse_stack_ue8m0_scale_transpose(scale, num_experts, m, k)
                self.assertEqual(list(converted_scale.shape), expected_shape)

    def test_large_logical_tensor_exceeds_int32(self):
        self._skip_if_not_sm10()
        num_experts, m, k = 1, 512, 4 * 1024 * 1024
        self.assertGreater(num_experts * m * k, np.iinfo(np.int32).max)

        num_k_groups = k // 512
        scale = paddle.arange(m * num_k_groups, dtype=paddle.int32).reshape([m, num_k_groups])
        converted_scale = fuse_stack_ue8m0_scale_transpose(scale, num_experts, m, k)

        self.assertEqual(list(converted_scale.shape), [k, 1])
        sample_rows = paddle.to_tensor([0, 1, 127, 128, 511, 512, k - 1], dtype=paddle.int64)
        actual = paddle.gather(converted_scale, sample_rows, axis=0).numpy().reshape([-1])
        packed_scale = scale.numpy()
        expected = []
        for row in sample_rows.numpy():
            k_block = row // 128
            k_block_group = k_block // 4
            k_block_inner = k_block % 4
            packed = 0
            for m_block in range(4):
                value = packed_scale[m_block * 128, k_block_group]
                exp = (value >> (k_block_inner * 8)) & 0xFF
                packed |= exp << (m_block * 8)
            expected.append(packed)
        np.testing.assert_array_equal(actual, np.array(expected, dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
