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
import paddle.incubate.nn.functional as F

paddle.enable_compat()

from paddlefleet_ops import deep_gemm, fuse_weighted_swiglu_fp8_quant
from paddlefleet_ops.deep_gemm.testing import get_arch_major


class TestSPAQ(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float16", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def dequantize_fp8_to_bf16(
        self, fp8_tensor: paddle.Tensor, scale: paddle.Tensor
    ) -> paddle.Tensor:
        """Helper function to dequantize fp8 tensor to bf16"""
        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
        # Handle non-aligned cases by truncating
        expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
        return fp8_tensor.astype("float32") * expanded_scale

    def run_fused_op_test(self, M, N, K):
        x = paddle.clip(
            paddle.randn([M, K * 2]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([M, 1]).astype("float32")
        weight_x = paddle.clip(
            paddle.randn([N, K]).astype("bfloat16"), min=-50, max=50
        )

        x.stop_gradient = False
        prob.stop_gradient = False
        weight_x.stop_gradient = False

        golden_res = F.swiglu(x) * prob

        fp8_x_out_ref, fp32_x_scale_ref = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=False, use_ue8m0=False
        )

        dequantized_res = self.dequantize_fp8_to_bf16(
            fp8_x_out_ref, fp32_x_scale_ref
        )

        golden_np = golden_res.astype("float32").numpy()
        fused_np = dequantized_res.numpy()

        np.testing.assert_allclose(
            golden_np,
            fused_np,
            rtol=0.01,
            atol=1,
        )

    def run_fused_op_ue8m0_test(self, M, N, K):
        x = paddle.clip(
            paddle.randn([M, K * 2]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([M, 1]).astype("float32")

        weight_x = paddle.clip(
            paddle.randn([N, K]).astype("bfloat16"), min=-50, max=50
        )

        fp8_weight, fp32_weight_scale = (
            paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                weight_x,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
                using_pow2_scale=True,
            )
        )

        x.stop_gradient = False
        prob.stop_gradient = False
        weight_x.stop_gradient = False

        # spaq test with using_pow2_scaling=True, use_ue8m0=False
        fp8_x_out_ref, fp32_x_scale_ref = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )
        out_ref = paddle.empty([M, N], dtype="bfloat16")
        deep_gemm.fp8_gemm_nt(
            (fp8_x_out_ref, fp32_x_scale_ref),
            (fp8_weight, fp32_weight_scale),
            out_ref,
        )

        # spaq test with using_pow2_scaling=True, use_ue8m0=True
        fp8_x_out, ue8m0_x_scale = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=True
        )
        out = paddle.empty([M, N], dtype="bfloat16")
        deep_gemm.fp8_gemm_nt(
            (fp8_x_out, ue8m0_x_scale), (fp8_weight, fp32_weight_scale), out
        )

        np.testing.assert_allclose(
            out_ref,
            out,
        )

    def test_spaq_0(self):
        self.run_fused_op_test(128 * 3, 128 * 4, 2048)

    def test_spaq_1(self):
        self.run_fused_op_test(128 * 10, 128 * 20, 4096)

    def test_spaq_ue8m0_0(self):
        if get_arch_major() == 10:
            self.run_fused_op_ue8m0_test(128 * 3, 128 * 4, 2048)

    def test_spaq_ue8m0_1(self):
        if get_arch_major() == 10:
            self.run_fused_op_ue8m0_test(128 * 10, 128 * 20, 4096)


class TestSPAQClamp(unittest.TestCase):
    """Independent unit test for the clamp variant
    (``fuse_weighted_swiglu_fp8_quant_clamp``).

    The non-clamp path is already covered by ``TestSPAQ``; the clamp path
    was previously only exercised indirectly through ``fp8_utils.py``
    dispatch tests. Here we apply the same clamp the kernel does on the
    golden side and verify the dequantized output of the fused op matches.
    """

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        try:
            from paddlefleet_ops import (  # noqa: F401
                fuse_weighted_swiglu_fp8_quant_clamp,
            )
        except ImportError:
            self.skipTest("fuse_weighted_swiglu_fp8_quant_clamp not built")

        np.random.seed(42)
        paddle.seed(42)

    @staticmethod
    def _dequantize_fp8_to_bf16(
        fp8_tensor: paddle.Tensor, scale: paddle.Tensor
    ) -> paddle.Tensor:
        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
        expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
        return fp8_tensor.astype("float32") * expanded_scale

    def _run(self, M, K, clamp_value):
        from paddlefleet_ops import fuse_weighted_swiglu_fp8_quant_clamp

        x = paddle.clip(
            paddle.randn([M, K * 2]).astype("bfloat16") * 10,
            min=-50,
            max=50,
        )
        prob = paddle.randn([M, 1]).astype("float32")

        # Golden mirrors the kernel exactly: bf16 -> fp32, clamp, swiglu in fp32.
        # The kernel reads each bf16 element, casts to fp32, then clamps gate
        # via fminf and value via fmax/fmin, then computes silu(gate)*value
        # in fp32. We avoid the fp32->bf16->fp32 round-trip in the middle
        # (which would introduce bf16 quantization error vs the kernel).
        gate_f32, value_f32 = paddle.chunk(x.astype("float32"), 2, axis=-1)
        gate_clamped = paddle.minimum(
            gate_f32, paddle.full_like(gate_f32, clamp_value)
        )
        value_clamped = paddle.clip(value_f32, -clamp_value, clamp_value)
        silu_gate = paddle.nn.functional.silu(gate_clamped)
        golden_res = silu_gate * value_clamped * prob

        fp8_out, fp32_scale = fuse_weighted_swiglu_fp8_quant_clamp(
            x,
            prob,
            using_pow2_scaling=False,
            use_ue8m0=False,
            clamp_value=float(clamp_value),
        )

        dequantized_res = self._dequantize_fp8_to_bf16(fp8_out, fp32_scale)

        np.testing.assert_allclose(
            golden_res.astype("float32").numpy(),
            dequantized_res.numpy(),
            # Looser rtol than the non-clamp TestSPAQ: the clamp kernel uses
            # fast-math intrinsics (__expf, __frcp_rn) for SiLU, whose error
            # envelope vs paddle.nn.functional.silu is ~6% on saturated inputs;
            # combined with fp8 e4m3 quantization that's the realistic bound.
            rtol=0.07,
            atol=1,
        )

    def test_clamp_small(self):
        self._run(M=128 * 3, K=2048, clamp_value=7.0)

    def test_clamp_large(self):
        self._run(M=128 * 10, K=4096, clamp_value=3.0)

    def test_clamp_int32_overflow_numel(self):
        """Verify the int64-offset path of fuse_weighted_swiglu_fp8_quant_clamp.

        Picks shape such that M * (K*2) > 2**31, exercising the VEC8 dispatch
        with int64 offsets in dispatch_fused_spaq.
        """
        # M=65536, K=16392 -> input numel = 65536 * 32784 = 2,148,925,440 > 2**31
        # Input bf16 = ~4.3 GB, output fp8 ~1.07 GB, plus scales ~ small.
        try:
            free_bytes, _ = paddle.device.cuda.mem_get_info()
        except (AttributeError, Exception):
            try:
                import pynvml

                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                free_bytes = pynvml.nvmlDeviceGetMemoryInfo(h).free
            except Exception:
                self.skipTest("cannot query GPU memory")
        if free_bytes < 30 * (1 << 30):
            self.skipTest(
                f"need >=30GB free GPU mem, have {free_bytes / 1e9:.1f}GB"
            )
        paddle.device.cuda.empty_cache()
        # Use moderate clamp_value that exercises clamp branches without
        # saturating every value.
        self._run(M=65536, K=16392, clamp_value=5.0)


class TestSPAQScalarFallback(unittest.TestCase):
    """Tests for the scalar fallback (cols % 8 != 0) FusedSPAQKernel.

    Specifically validates the row-loop fix that handles rows > gridDim.y cap
    (65535) and the int64 indexing for large numel.
    """

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")
        np.random.seed(42)
        paddle.seed(42)

    @staticmethod
    def _dequantize_fp8_to_bf16(fp8_tensor, scale):
        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
        expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
        return fp8_tensor.astype("float32") * expanded_scale

    def test_scalar_fallback_rows_over_65535(self):
        """rows > 65535 with cols % 8 != 0 forces FusedSPAQKernel row loop."""
        # cols = 12 (cols % 8 == 4 != 0), so dispatch falls into scalar path.
        # M = 70000 exceeds gridDim.y cap (65535); without row-loop fix the
        # last (70000 - 65535) = 4465 rows would be silently zeroed.
        M, cols = 70000, 12
        K = cols // 2  # K = 6, so input shape is [M, 2*K] = [M, 12]

        x = paddle.clip(
            paddle.randn([M, 2 * K]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([M, 1]).astype("float32")

        gate_f32, value_f32 = paddle.chunk(x.astype("float32"), 2, axis=-1)
        silu_gate = paddle.nn.functional.silu(gate_f32)
        golden = silu_gate * value_f32 * prob

        fp8_out, fp32_scale = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=False, use_ue8m0=False
        )
        deq = self._dequantize_fp8_to_bf16(fp8_out, fp32_scale)

        # Spot-check rows past the gridDim.y boundary (65535) — these are
        # exactly the rows the un-fixed kernel would leave as zeros.
        for r in (0, 65535, 65536, 69999):
            np.testing.assert_allclose(
                golden[r].astype("float32").numpy(),
                deq[r].numpy(),
                rtol=0.1,
                atol=1.0,
                err_msg=f"scalar-fallback row {r} mismatch",
            )


if __name__ == "__main__":
    unittest.main()
