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
Tests for fused_stack_quant / fused_stack_quant_without_cache.
fused_stack_quant / fused_stack_quant_without_cache 的单元测试。

Covers / 覆盖:
  1. Quantize -> dequant round-trip accuracy (blockwise scale)
     量化 -> 反量化 round-trip 精度验证（blockwise scale）
  2. Cache-hit path (_get_fp8_weight_and_scale all branches)
     缓存命中路径（_get_fp8_weight_and_scale 全分支覆盖）
  3. use_ue8m0=True dequant round-trip (Blackwell SM10+ only)
     use_ue8m0=True 量化反量化精度验证（仅 Blackwell SM10+）

Scale layout / Scale 布局:
  ue8m0=False:
    - fuse_stack_fp8_quant:            scale [num_experts*M/128, K/128]  float32
    - fuse_stack_transpose_fp8_quant:  scale [num_experts*K/128, M/128]  float32
  ue8m0=True (4 个 uint8 e8m0 exponent pack 成 1 个 int32, 需要 dim >= 512):
    - fuse_stack_fp8_quant:            scale [num_experts*M, K/128/4]   int32
    - fuse_stack_transpose_fp8_quant:  scale [num_experts*K, M/128/4]   int32
"""

import unittest

import numpy as np
import paddle
from paddle.base import core

from paddleformers.fleet.transformer.moe.fp8_utils import (
    fused_stack_quant,
    fused_stack_quant_without_cache,
)

TILE = 128  # FP8 quantization tile size
NUM_EXPERTS = 4
N, K = 512, 256  # must be multiples of TILE


def _make_weight_list(
    num_experts=NUM_EXPERTS, shape=(N, K), dtype=paddle.bfloat16
):
    return [paddle.randn(shape, dtype=dtype) for _ in range(num_experts)]


def _blockwise_dequant_per_expert(
    stacked_w, stacked_s, num_experts, rows_per_expert
):
    """
    Split stacked FP8 weight and blockwise scale per expert, dequant each
    via fused_act_dequant (requires scale shape [M, K/128]).

    stacked_s shape: [num_experts * rows_per_expert / TILE, cols / TILE]
    Per expert slice: [rows_per_expert / TILE, cols / TILE]
    Expand to:        [rows_per_expert, cols / TILE]  via repeat_interleave
    """
    scale_rows_per_expert = rows_per_expert // TILE
    results = []
    for i in range(num_experts):
        w_i = stacked_w[i * rows_per_expert : (i + 1) * rows_per_expert]
        s_i = stacked_s[
            i * scale_rows_per_expert : (i + 1) * scale_rows_per_expert
        ]
        # fused_act_dequant expects scale shape [M, K/128], not [M/128, K/128]
        s_expanded = s_i.repeat_interleave(TILE, axis=0)
        dq_i = paddle.incubate.nn.functional.fused_act_dequant(w_i, s_expanded)
        results.append(dq_i)
    return results


# ---------- 1. dequant round-trip accuracy ----------


class TestDequantAccuracy(unittest.TestCase):
    """
    Quantize -> split per expert -> dequant -> compare with original BF16.
    FP8 E4M3 has ~3 bit mantissa; 5% rtol is generous.
    """

    rtol = 0.05
    atol = 0.25

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        paddle.seed(42)

    def test_nontranspose(self):
        weights = _make_weight_list()
        w_fp8, scale = fused_stack_quant(weights, transpose=False)
        dequant_list = _blockwise_dequant_per_expert(
            w_fp8, scale, NUM_EXPERTS, N
        )
        for idx, (orig, dq) in enumerate(zip(weights, dequant_list)):
            np.testing.assert_allclose(
                orig.astype(paddle.float32).numpy(),
                dq.astype(paddle.float32).numpy(),
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Expert {idx} non-transpose dequant mismatch",
            )

    def test_transpose(self):
        weights = _make_weight_list()
        w_fp8, scale = fused_stack_quant(weights, transpose=True)
        dequant_list = _blockwise_dequant_per_expert(
            w_fp8, scale, NUM_EXPERTS, K
        )
        for idx, (orig, dq) in enumerate(zip(weights, dequant_list)):
            np.testing.assert_allclose(
                orig.astype(paddle.float32).numpy(),
                dq.T.astype(paddle.float32).numpy(),
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Expert {idx} transpose dequant mismatch",
            )


# ---------- 2. cache behavior ----------


class TestCacheHit(unittest.TestCase):
    """
    _get_fp8_weight_and_scale has 3 branches:

    Scenario A: only fp8_weight_stacked (no transpose cache)
      A1: transpose=False → return fp8_weight_stacked directly
      A2: transpose=True  → on-the-fly reshape+transpose fp8_weight_stacked

    Scenario B: both fp8_weight_stacked and fp8_weight_stacked_transpose
      B1: transpose=False → return fp8_weight_stacked directly  (same as A1)
      B2: transpose=True  → return fp8_weight_stacked_transpose directly
    """

    rtol = 0.05
    atol = 0.25

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        paddle.seed(0)
        self.weights = _make_weight_list()
        self.w_nt, self.s_nt = fused_stack_quant_without_cache(
            self.weights, transpose=False
        )
        self.w_t, self.s_t = fused_stack_quant_without_cache(
            self.weights, transpose=True
        )

    def _attach_cache(self, with_transpose_cache):
        """Attach pre-computed cache attributes to weights[0]."""
        self.weights[0].fp8_weight_stacked = self.w_nt
        self.weights[0].fp8_scale_stacked = self.s_nt
        if with_transpose_cache:
            self.weights[0].fp8_weight_stacked_transpose = self.w_t
            self.weights[0].fp8_scale_stacked_transpose = self.s_t

    def _assert_dequant_close(self, w_fp8, scale, transpose):
        rows = K if transpose else N
        dequant_list = _blockwise_dequant_per_expert(
            w_fp8, scale, NUM_EXPERTS, rows
        )
        for idx, (orig, dq) in enumerate(zip(self.weights, dequant_list)):
            expected = orig.astype(paddle.float32).numpy()
            actual = (
                dq.T.astype(paddle.float32).numpy()
                if transpose
                else dq.astype(paddle.float32).numpy()
            )
            np.testing.assert_allclose(
                expected,
                actual,
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Expert {idx} dequant mismatch",
            )

    # -- A1: only non-transpose cache, query transpose=False
    def test_only_nt_cache_query_nontranspose(self):
        self._attach_cache(with_transpose_cache=False)
        w, s = fused_stack_quant(self.weights, transpose=False)
        self.assertIs(w, self.w_nt)
        self.assertIs(s, self.s_nt)

    # -- A2: only non-transpose cache, query transpose=True → on-the-fly transpose
    def test_only_nt_cache_query_transpose(self):
        self._attach_cache(with_transpose_cache=False)
        w, s = fused_stack_quant(self.weights, transpose=True)
        # Not identity (new tensors from on-the-fly transpose), verify by dequant
        self._assert_dequant_close(w, s, transpose=True)

    # -- B1: both caches, query transpose=False
    def test_both_caches_query_nontranspose(self):
        self._attach_cache(with_transpose_cache=True)
        w, s = fused_stack_quant(self.weights, transpose=False)
        self.assertIs(w, self.w_nt)
        self.assertIs(s, self.s_nt)

    # -- B2: both caches, query transpose=True
    def test_both_caches_query_transpose(self):
        self._attach_cache(with_transpose_cache=True)
        w, s = fused_stack_quant(self.weights, transpose=True)
        self.assertIs(w, self.w_t)
        self.assertIs(s, self.s_t)


# ---------- 3. use_ue8m0 (Blackwell SM10+ only) ----------


class TestUe8m0(unittest.TestCase):
    """
    use_ue8m0=True path (Blackwell SM10+ only).

    ue8m0 scale 是 int32（4 个 uint8 e8m0 exponent pack 成 1 个 int32），
    列维 shape = dim/128/4，所以 dim 必须 >= 512 才不为空。
    这里用 (512, 512) 保证两个方向的 scale 都非空。
    """

    # ue8m0 需要 M >= 512 且 K >= 512，否则 scale 某维为 0
    UE_N, UE_K = 512, 512
    rtol = 0.05
    atol = 0.25

    def _skip_if_not_sm10(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CUDA required")
        arch = paddle.device.cuda.get_device_capability()[0]
        if arch < 10:
            self.skipTest("use_ue8m0=True requires SM10+ (Blackwell)")

    def test_nontranspose(self):
        self._skip_if_not_sm10()
        paddle.seed(0)
        weights = _make_weight_list(shape=(self.UE_N, self.UE_K))
        w, scale = fused_stack_quant(weights, transpose=False, use_ue8m0=True)
        # ue8m0 scale is per-row int32, can directly pass to fused_act_dequant
        for i in range(NUM_EXPERTS):
            w_i = w[i * self.UE_N : (i + 1) * self.UE_N]
            s_i = scale[i * self.UE_N : (i + 1) * self.UE_N]
            dq_i = paddle.incubate.nn.functional.fused_act_dequant(w_i, s_i)
            np.testing.assert_allclose(
                weights[i].astype(paddle.float32).numpy(),
                dq_i.astype(paddle.float32).numpy(),
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Expert {i} ue8m0 non-transpose dequant mismatch",
            )

    def test_transpose(self):
        self._skip_if_not_sm10()
        paddle.seed(1)
        weights = _make_weight_list(shape=(self.UE_N, self.UE_K))
        w, scale = fused_stack_quant(weights, transpose=True, use_ue8m0=True)
        for i in range(NUM_EXPERTS):
            w_i = w[i * self.UE_K : (i + 1) * self.UE_K]
            s_i = scale[i * self.UE_K : (i + 1) * self.UE_K]
            dq_i = paddle.incubate.nn.functional.fused_act_dequant(w_i, s_i)
            np.testing.assert_allclose(
                weights[i].astype(paddle.float32).numpy(),
                dq_i.T.astype(paddle.float32).numpy(),
                rtol=self.rtol,
                atol=self.atol,
                err_msg=f"Expert {i} ue8m0 transpose dequant mismatch",
            )


if __name__ == "__main__":
    unittest.main()
