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
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

from paddleformers.fleet.fusions.fused_swiglu_scale import (
    fused_swiglu_scale_forward,
)


class TestFusedSwiGLUScale(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float16", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def get_reference_impl(self, x, scale):
        """
        Paddle native implementation of SwiGLU + Scale as ground truth.
        """
        # x shape: [B, 2*H] -> chunk -> gate, val
        gate, val = paddle.chunk(x, chunks=2, axis=-1)

        # SwiGLU: silu(gate) * val
        # F.silu(x) = x * sigmoid(x)
        swiglu_out = F.silu(gate) * val

        # Scale: swiglu_out * scale (Broadcast multiplication)
        out = swiglu_out * scale
        return out

    def run_fused_op_test(self, batch_size, hidden_size, dtype_x, dtype_scale):
        # 1. Construct input data
        shape_x = (batch_size, 2 * hidden_size)
        shape_scale = (batch_size, 1)

        # Use a reasonable range to avoid numerical instability
        x_np = np.random.normal(0, 1.0, shape_x).astype("float32")
        scale_np = np.random.uniform(0.5, 1.5, shape_scale).astype("float32")

        # 2. Create Reference Tensors
        if dtype_x == "bfloat16":
            x_ref = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_ref = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_ref = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_ref = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_ref.stop_gradient = False
        scale_ref.stop_gradient = False

        # 3. Create Custom Op Tensors (Independent copies)
        if dtype_x == "bfloat16":
            x_custom = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_custom = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_custom = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_custom = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_custom.stop_gradient = False
        scale_custom.stop_gradient = False

        # 4. Forward Pass
        # Reference implementation
        out_ref = self.get_reference_impl(x_ref, scale_ref)

        # Custom Op implementation
        # Note: The C++ op might return a list [Tensor], handle it if necessary
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        # 5. Backward Pass
        # Create random output gradient
        grad_np = np.random.random(out_ref.shape).astype("float32")
        if out_ref.dtype == paddle.bfloat16:
            out_grad = paddle.to_tensor(grad_np).astype("bfloat16")
        else:
            out_grad = paddle.to_tensor(grad_np, dtype=out_ref.dtype)

        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])

        # 6. Verification
        # Set tolerance: BF16 has lower precision, requiring larger tolerance
        if "bfloat16" in [dtype_x, dtype_scale]:
            rtol, atol = 2e-2, 2e-2
            # Relax tolerance for BF16 gradient accumulation with large shape (Reduction dim > 1024)
            # This accounts for the accumulation error difference between FP32 (Fused) and BF16 (Ref)
            if hidden_size > 1024:
                rtol, atol = 0.1, 0.1
        else:
            rtol, atol = 1e-4, 1e-4

        # Verify Forward Output
        np.testing.assert_allclose(
            out_custom.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Forward output mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Input Gradient (dX)
        np.testing.assert_allclose(
            x_custom.grad.astype("float32").numpy(),
            x_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient X mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Scale Gradient (dScale)
        np.testing.assert_allclose(
            scale_custom.grad.astype("float32").numpy(),
            scale_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient Scale mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

    def test_fused_swiglu_fp32(self):
        self.run_fused_op_test(32, 64, "float32", "float32")

    def test_fused_swiglu_bf16(self):
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(32, 64, "bfloat16", "bfloat16")

    def test_fused_swiglu_mixed_precision(self):
        # Test mixed precision: Input=BF16, Scale=FP32
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(16, 128, "bfloat16", "float32")

    def test_fused_swiglu_large_shape(self):
        # Test large shape to ensure no index overflow or memory alignment issues
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(4, 4096, "bfloat16", "float32")

    def test_fused_swiglu_int32_overflow_numel(self):
        """Forward/backward when numel of x > 2**31.

        Exercises the int64-offset path in VectorizedFusedSwiGLUFwd/Bwd.
        Skipped on hosts without enough free GPU memory.
        """
        if not core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.skipTest("bf16 not supported")

        # [rows, 2*hidden] bf16 with rows * 2 * hidden > 2**31.
        # hidden must be divisible by VEC_SIZE=8 for the bf16 kernel.
        # 65536 * 2 * 16392 = 2,148,925,440 > 2,147,483,648.
        rows, hidden = 65536, 16392
        bytes_per_elem = 2
        x_bytes = rows * 2 * hidden * bytes_per_elem
        # x + out + d_x + d_out + scale + tmp ~ 4-5x x_bytes
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
        if free_bytes < 40 * (1 << 30) or free_bytes < x_bytes * 5:
            self.skipTest(
                f"need >=40GB free GPU mem, have {free_bytes / 1e9:.1f}GB"
            )

        paddle.device.cuda.empty_cache()
        # Use small fp32 host arrays then cast on device to limit host RAM.
        x_ref = paddle.randn([rows, 2 * hidden], dtype="float32").astype(
            "bfloat16"
        )
        scale_ref = paddle.uniform([rows, 1], dtype="float32", min=0.5, max=1.5)
        x_ref.stop_gradient = False
        scale_ref.stop_gradient = False

        x_custom = x_ref.detach().clone()
        scale_custom = scale_ref.detach().clone()
        x_custom.stop_gradient = False
        scale_custom.stop_gradient = False

        out_ref = self.get_reference_impl(x_ref, scale_ref)
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        sample_rows = [0, rows // 2, rows - 1]
        for r in sample_rows:
            np.testing.assert_allclose(
                out_custom[r].astype("float32").numpy(),
                out_ref[r].astype("float32").numpy(),
                rtol=0.1,
                atol=0.1,
                err_msg=f"fwd mismatch at row {r}",
            )

        out_grad = paddle.ones_like(out_ref)
        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])
        for r in sample_rows:
            np.testing.assert_allclose(
                x_custom.grad[r].astype("float32").numpy(),
                x_ref.grad[r].astype("float32").numpy(),
                rtol=0.1,
                atol=0.1,
                err_msg=f"dX mismatch at row {r}",
            )

    def _run_grid_stride_test(self, rows, hidden, dtype):
        """Drive rows > kMaxSwiGLUGridSize (65535) with a tiny hidden so the
        grid-stride row loop in VectorizedFusedSwiGLUFwd/Bwd fires multiple
        times per block. Memory footprint is ~ rows * 2*hidden * sizeof(dtype).
        """
        if dtype == "bfloat16" and not core.is_bfloat16_supported(
            base.CUDAPlace(0)
        ):
            self.skipTest("bf16 not supported")

        x_np = np.random.normal(0, 1.0, (rows, 2 * hidden)).astype("float32")
        scale_np = np.random.uniform(0.5, 1.5, (rows, 1)).astype("float32")

        def _to(t):
            if dtype == "bfloat16":
                return paddle.to_tensor(t).astype("bfloat16")
            return paddle.to_tensor(t, dtype=dtype)

        x_ref, scale_ref = _to(x_np), _to(scale_np)
        x_custom, scale_custom = _to(x_np), _to(scale_np)
        for t in (x_ref, scale_ref, x_custom, scale_custom):
            t.stop_gradient = False

        out_ref = self.get_reference_impl(x_ref, scale_ref)
        ret = fused_swiglu_scale_forward(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        rtol, atol = (5e-2, 5e-2) if dtype == "bfloat16" else (1e-4, 1e-4)
        np.testing.assert_allclose(
            out_custom.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride fwd mismatch (rows={rows}, dtype={dtype})",
        )

        out_grad = paddle.ones_like(out_ref)
        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])
        np.testing.assert_allclose(
            x_custom.grad.astype("float32").numpy(),
            x_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride dX mismatch (rows={rows}, dtype={dtype})",
        )
        np.testing.assert_allclose(
            scale_custom.grad.astype("float32").numpy(),
            scale_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"grid-stride dScale mismatch (rows={rows}, dtype={dtype})",
        )

    def test_fused_swiglu_grid_stride_loop_bf16(self):
        # rows > kMaxSwiGLUGridSize (65535) -> each block iterates the row
        # loop at least twice. hidden=8 is the bf16 VEC_SIZE so launch is
        # 1 thread per row's hidden lane; tiny memory footprint.
        self._run_grid_stride_test(rows=65536, hidden=8, dtype="bfloat16")

    def test_fused_swiglu_grid_stride_loop_fp32(self):
        # fp32 VEC_SIZE=4; rows just above the cap forces 2-iteration loop
        # only on the first 1 (=65536-65535) blocks while the rest run once.
        self._run_grid_stride_test(rows=65536, hidden=8, dtype="float32")

    def test_fused_swiglu_grid_stride_loop_far_above_cap(self):
        # rows ~ 3x cap so every block iterates the row loop multiple times,
        # and the trailing __syncthreads() between iterations is exercised
        # heavily on the bwd shared_sum reduction.
        self._run_grid_stride_test(rows=200000, hidden=8, dtype="bfloat16")

    def test_fused_swiglu_clamp_grid_stride_loop(self):
        """Same grid-stride coverage but for the clamp + weighted bwd path
        (VectorizedFusedSwiGLUWeightedBwd kernel)."""
        if not core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.skipTest("bf16 not supported")
        try:
            from paddlefleet_ops import fused_swiglu_weighted_clamp_bwd
        except ImportError:
            self.skipTest(
                "fused_swiglu_weighted_clamp_bwd not in installed extension"
            )

        rows, hidden = 65536, 8
        x_np = np.random.normal(0, 1.0, (rows, 2 * hidden)).astype("float32")
        probs_np = np.random.uniform(0.1, 0.9, (rows,)).astype("float32")
        dout_np = np.random.normal(0, 1.0, (rows, hidden)).astype("float32")

        x = paddle.to_tensor(x_np).astype("bfloat16")
        probs = paddle.to_tensor(probs_np).astype("bfloat16")
        dout = paddle.to_tensor(dout_np).astype("bfloat16")

        clamp_value = 7.0
        dx, dprobs, out = fused_swiglu_weighted_clamp_bwd(
            x, probs, dout, clamp_value
        )

        # Reference: clamp(g, cv); clamp(v, -cv, cv); silu(g_eff)*v_eff
        cv = clamp_value
        gate, val = paddle.chunk(x.astype("float32"), 2, axis=-1)
        g_eff = paddle.minimum(gate, paddle.full_like(gate, cv))
        v_eff = paddle.clip(val, -cv, cv)
        silu_g = F.silu(g_eff)
        swiglu = silu_g * v_eff
        out_ref = (swiglu * probs.astype("float32").unsqueeze(-1)).astype(
            "bfloat16"
        )
        # dprobs[row] = sum_h(dout * silu(clamp(g)) * clamp(v))
        dprobs_ref = paddle.sum(
            dout.astype("float32") * swiglu, axis=-1, keepdim=True
        ).astype("bfloat16")

        # The point of this test is exercising the kernel's row grid-stride
        # loop and the shared_sum cross-iteration __syncthreads in the
        # dprobs reduction. Verifying out alone would not catch a broken
        # cross-iteration shared_sum sync, since out is computed without
        # any block-wide reduction.
        self.assertEqual(out.shape, [rows, hidden])
        self.assertEqual(dx.shape, [rows, 2 * hidden])
        self.assertEqual(dprobs.shape, [rows, 1])
        np.testing.assert_allclose(
            out.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=5e-2,
            atol=5e-2,
            err_msg="weighted-clamp grid-stride fwd output mismatch",
        )
        np.testing.assert_allclose(
            dprobs.astype("float32").numpy(),
            dprobs_ref.astype("float32").numpy(),
            rtol=5e-2,
            atol=5e-2,
            err_msg="weighted-clamp grid-stride dprobs mismatch",
        )


if __name__ == "__main__":
    unittest.main()
