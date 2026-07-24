#!/usr/bin/env python3

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

from paddlefleet_ops import deep_gemm


class TestDeepGemmUsage(unittest.TestCase):
    def test_deep_gemm_usage(self):
        paddle.set_device("gpu")

        # 简单的测试参数
        total_seq_len = 1024
        input_hidden_size = 256
        output_hidden_size = 128
        num_batches = 4

        # 创建简单的批次大小（平均分配）
        batch_sizes = [total_seq_len // num_batches] * num_batches
        batch_sizes[-1] = total_seq_len - sum(
            batch_sizes[:-1]
        )  # 调整最后一个批次

        # 创建测试数据
        lhs = paddle.randn(
            [total_seq_len, input_hidden_size], dtype="bfloat16", device="cuda"
        )
        rhs = paddle.randn(
            [num_batches, input_hidden_size, output_hidden_size],
            dtype="bfloat16",
            device="cuda",
        )

        # 处理rhs数据格式以适配DeepGEMM
        rhs_deepseek = rhs.transpose([0, 2, 1])

        # 创建输出tensor
        out_tensor = paddle.zeros(
            [total_seq_len, output_hidden_size], dtype="bfloat16", device="cuda"
        )

        # 生成indices
        indices = []
        for idx, count in enumerate(batch_sizes):
            indices.extend([idx] * count)
        indices = paddle.to_tensor(indices, dtype="int32", place="cuda")

        print("Running DeepGEMM...")
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            lhs, rhs_deepseek, out_tensor, indices
        )
        paddle.device.cuda.synchronize()

        print("DeepGEMM executed successfully!")


if __name__ == "__main__":
    unittest.main()
