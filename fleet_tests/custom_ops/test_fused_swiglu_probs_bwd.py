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

"""
数值正确性单测：fused_swiglu_probs_bwd

用 paddle.incubate.nn.functional.fused_swiglu_weighted_bwd 作为 reference，
对比自定义 CUDA kernel 的 do1、probs_grad、o2_s 输出。
同时覆盖 inplace=True / False 两种模式。

两者数学完全一致，接口差异仅在 probs shape：
  - fused_swiglu_weighted_bwd:  unzipped_probs=[N, 1], probs_grad=[N, 1]
  - fused_swiglu_probs_bwd:     unzipped_probs=[N],    probs_grad=[N]

运行方式：
    PYTHONPATH=src:$PYTHONPATH python fleet_tests/test_fused_swiglu_probs_bwd.py
"""

import unittest

import numpy as np
import paddle
import paddle.incubate.nn.functional as incubate_F
from paddlefleet_ops import fused_swiglu_probs_bwd


class TestFusedSwigluProbsBwd(unittest.TestCase):
    """数值正确性测试：自定义 kernel vs paddle 内置实现。"""

    def _run_case(self, N, H, inplace):
        paddle.seed(42)
        o1 = paddle.randn([N, H * 2], dtype="bfloat16")
        do2_s = paddle.randn([N, H], dtype="bfloat16")
        unzipped_probs = paddle.rand([N], dtype="float32")

        # reference: paddle 内置实现，probs 需要 [N, 1]
        do1_ref, pg_ref, o2s_ref = incubate_F.fused_swiglu_weighted_bwd(
            o1, do2_s, unzipped_probs.unsqueeze(-1)
        )
        pg_ref = pg_ref.squeeze(-1)  # [N, 1] -> [N]

        # kernel under test
        do1, probs_grad, o2_s = fused_swiglu_probs_bwd(
            o1, do2_s, unzipped_probs, inplace
        )
        paddle.device.synchronize()

        # 两个 kernel 计算路径一致，结果应 bit-identical
        np.testing.assert_array_equal(
            do1.numpy(),
            do1_ref.numpy(),
            err_msg=f"do1 mismatch (N={N}, H={H}, inplace={inplace})",
        )
        np.testing.assert_array_equal(
            probs_grad.numpy(),
            pg_ref.numpy(),
            err_msg=f"probs_grad mismatch (N={N}, H={H}, inplace={inplace})",
        )
        np.testing.assert_array_equal(
            o2_s.numpy(),
            o2s_ref.numpy(),
            err_msg=f"o2_s mismatch (N={N}, H={H}, inplace={inplace})",
        )

    # ---------- inplace=False ----------
    def test_basic_outofplace(self):
        self._run_case(N=64, H=256, inplace=False)

    def test_large_outofplace(self):
        self._run_case(N=512, H=2048, inplace=False)

    def test_non_vec4_outofplace(self):
        """H 不能被 4 整除，走非向量化分支。"""
        self._run_case(N=32, H=255, inplace=False)

    # ---------- inplace=True ----------
    def test_basic_inplace(self):
        self._run_case(N=64, H=256, inplace=True)

    def test_large_inplace(self):
        self._run_case(N=512, H=2048, inplace=True)

    def test_non_vec4_inplace(self):
        self._run_case(N=32, H=255, inplace=True)

    # ---------- inplace data_ptr 验证 ----------
    def test_inplace_shares_buffer(self):
        """inplace=True 时 do1 应与 o1 共用 GPU buffer。"""
        o1 = paddle.randn([64, 512], dtype="bfloat16")
        do2_s = paddle.randn([64, 256], dtype="bfloat16")
        probs = paddle.rand([64], dtype="float32")
        o1_ptr = o1.data_ptr()

        do1, _, _ = fused_swiglu_probs_bwd(o1, do2_s, probs, True)
        self.assertEqual(
            do1.data_ptr(), o1_ptr, "inplace=True: do1 应复用 o1 buffer"
        )

    def test_outofplace_new_buffer(self):
        """inplace=False 时 do1 应该是独立的新 buffer。"""
        o1 = paddle.randn([64, 512], dtype="bfloat16")
        do2_s = paddle.randn([64, 256], dtype="bfloat16")
        probs = paddle.rand([64], dtype="float32")
        o1_ptr = o1.data_ptr()

        do1, _, _ = fused_swiglu_probs_bwd(o1, do2_s, probs, False)
        self.assertNotEqual(
            do1.data_ptr(), o1_ptr, "inplace=False: do1 应为独立 buffer"
        )

    # ---------- 边界 case ----------
    def test_single_token(self):
        self._run_case(N=1, H=128, inplace=False)

    def test_single_token_inplace(self):
        self._run_case(N=1, H=128, inplace=True)


if __name__ == "__main__":
    unittest.main()
