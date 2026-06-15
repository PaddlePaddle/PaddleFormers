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
"""Tests for clamped swiglu alignment changes.

Covers: clamped_bias_swiglu / clamped_bias_swiglu_back, fused_swiglu_scale
CPU clamp forward/backward, BiasSwiGLUFunction / SwiGLUFunction /
WeightedSwiGLUFunction with clamp_value, weighted_bias_swiglu_impl /
bias_swiglu_impl with clamp_value, d_scale verification, edge cases.
"""

import os
import sys
import unittest
from unittest import mock

# Walk up to find the repo root.
_test_file = os.path.abspath(__file__)
_repo_root = _test_file
for _ in range(10):
    _repo_root = os.path.dirname(_repo_root)
    if os.path.isdir(os.path.join(_repo_root, "src", "paddleformers.fleet")):
        break
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))
for _mod in list(sys.modules.keys()):
    if _mod == "paddleformers.fleet" or _mod.startswith("paddleformers.fleet."):
        del sys.modules[_mod]

import numpy as np
import paddle
import paddle.nn.functional as F

from paddleformers.fleet.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_back,
    bias_swiglu_impl,
    clamped_bias_swiglu,
    clamped_bias_swiglu_back,
    clamped_swiglu,
    clamped_swiglu_back,
    clamped_weighted_swiglu,
    clamped_weighted_swiglu_back,
    weighted_bias_swiglu_impl,
)
from paddleformers.fleet.fusions.fused_swiglu_scale import (
    fused_swiglu_scale_backward,
    fused_swiglu_scale_forward,
)

# paddle may have re-imported; clear again
for _mod in list(sys.modules.keys()):
    if _mod == "paddleformers.fleet" or _mod.startswith("paddleformers.fleet."):
        pass  # already cleared; paddleformers.fleet from src/ is now in sys.modules

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_cuda():
    return mock.patch.object(paddle, "is_compiled_with_cuda", return_value=False)


def _ref_clamped_swiglu(x, clamp_value):
    x_fp32 = x.cast(paddle.float32)
    hidden = x.shape[-1] // 2
    gate = paddle.clip(x_fp32[..., :hidden], max=clamp_value)
    val = paddle.clip(x_fp32[..., hidden:], min=-clamp_value, max=clamp_value)
    return (F.silu(gate) * val).cast(x.dtype)


# ---------------------------------------------------------------------------
# clamped_bias_swiglu / clamped_bias_swiglu_back
# ---------------------------------------------------------------------------


class TestClampedBiasSwiGLU(unittest.TestCase):
    def test_forward_no_clamp_effect(self):
        """Large clamp_value → same result as bias_swiglu."""
        x, bias = paddle.randn([4, 16]), paddle.randn([4, 16])
        out_ref = bias_swiglu(x, bias)
        out = clamped_bias_swiglu(x, bias, clamp_value=100.0)
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(out_ref.numpy(), out.numpy(), rtol=1e-5, atol=1e-5)

    def test_forward_saturated_different(self):
        """Small clamp_value clips gate/value → output differs from non-clamp."""
        x = paddle.full([2, 8], 5.0)
        bias = paddle.zeros([2, 8])
        out_ref = bias_swiglu(x, bias)
        out = clamped_bias_swiglu(x, bias, clamp_value=0.5)
        self.assertEqual(out.shape, [2, 4])
        self.assertFalse(bool((out_ref.numpy() == out.numpy()).all().item()))

    def test_backward_nonclamp_and_saturated(self):
        """Large clamp: grad ≈ non-clamp. Small clamp: saturated → zero grad."""
        g = paddle.randn([4, 8])
        y, bias = paddle.randn([4, 16]), paddle.randn([4, 16])
        grad_large = clamped_bias_swiglu_back(g, y, bias, clamp_value=100.0)
        grad_ref = bias_swiglu_back(g, y, bias)
        np.testing.assert_allclose(grad_ref.numpy(), grad_large.numpy(), rtol=1e-5, atol=1e-5)
        g2 = paddle.randn([2, 4])
        y2 = paddle.full([2, 8], 5.0)
        grad0 = clamped_bias_swiglu_back(g2, y2, paddle.zeros([2, 8]), clamp_value=0.5)
        self.assertTrue(bool((grad0.abs().sum() == 0).item()))

    def test_e2e_autograd(self):
        x = paddle.randn([4, 16])
        bias = paddle.randn([4, 16])
        x.stop_gradient = False
        bias.stop_gradient = False
        out = clamped_bias_swiglu(x, bias, clamp_value=2.0)
        out.sum().backward()
        self.assertEqual(out.shape, [4, 8])
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertEqual(bias.grad.shape, [4, 16])

    def test_shapes_and_dtype(self):
        """2D only, dtype preserved (float32)."""
        x, bias = paddle.randn([8, 32]), paddle.randn([8, 32])
        out = clamped_bias_swiglu(x, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [8, 16])
        self.assertEqual(out.dtype, paddle.float32)


# ---------------------------------------------------------------------------
# fused_swiglu_scale CPU clamp backward
# ---------------------------------------------------------------------------


class TestFusedSwiGLUScaleBackwardClampCPU(unittest.TestCase):
    def test_basic_shapes_and_order(self):
        """d_x shape, [d_gate, d_val] order, d_scale keepdim."""
        with _no_cuda():
            dx, ds = fused_swiglu_scale_backward(
                paddle.randn([4, 16]),
                paddle.ones([4, 1]),
                paddle.randn([4, 8]),
                clamp_value=2.0,
            )
            self.assertEqual(dx.shape, [4, 16])
            self.assertEqual(dx[..., :8].shape, [4, 8])
            self.assertEqual(dx[..., 8:].shape, [4, 8])
            self.assertEqual(ds.shape, [4, 1])

    def test_1d_scale_d_scale_keepdim(self):
        """1D scale -> d_scale still [B, 1] matching Megatron keepdim."""
        with _no_cuda():
            dx, ds = fused_swiglu_scale_backward(
                paddle.randn([4, 16]),
                paddle.ones([4]),
                paddle.randn([4, 8]),
                clamp_value=2.0,
            )
            self.assertEqual(dx.shape, [4, 16])
            self.assertEqual(ds.shape, [4, 1])

    def test_saturation_masking(self):
        """Fully saturated → d_x=0. Partial → gradients masked. Vs non-clamp."""
        with _no_cuda():
            dx, _ = fused_swiglu_scale_backward(
                paddle.full([2, 8], 10.0),
                paddle.ones([2, 1]),
                paddle.randn([2, 4]),
                clamp_value=0.5,
            )
            self.assertTrue(bool((dx.abs().sum() < 1e-10).item()))
            dx_nc, _ = fused_swiglu_scale_backward(
                paddle.full([2, 8], 10.0),
                paddle.ones([2, 1]),
                paddle.randn([2, 4]),
                clamp_value=None,
            )
            self.assertFalse(bool((dx_nc.abs().sum() < 1e-10).item()))
            # Partial saturation
            x = paddle.concat(
                [
                    paddle.full([2, 2], 0.3),
                    paddle.full([2, 2], 10.0),
                    paddle.full([2, 2], 0.3),
                    paddle.full([2, 2], 10.0),
                ],
                axis=-1,
            )
            dx2, _ = fused_swiglu_scale_backward(x, paddle.ones([2, 1]), paddle.randn([2, 4]), clamp_value=1.0)
            self.assertFalse(bool((dx2[..., :2].abs().sum() == 0).item()))

    def test_d_scale_numeric(self):
        """d_scale matches reference: sum(swiglu_val * out_grad), keepdim."""
        with _no_cuda():
            paddle.seed(42)
            x, sg = paddle.randn([4, 16]), paddle.randn([4, 8])
            _, ds = fused_swiglu_scale_backward(x, paddle.ones([4, 1]), sg, clamp_value=3.0)
            h = x.shape[-1] // 2
            xf = x.cast(paddle.float32)
            g = paddle.clip(xf[..., :h], max=3.0)
            v = paddle.clip(xf[..., h:], min=-3.0, max=3.0)
            sw = (F.silu(g) * v).cast(x.dtype)
            ref = paddle.sum(sw * sg.cast(paddle.float32), axis=-1, keepdim=True)
            np.testing.assert_allclose(ds.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# fused_swiglu_scale CPU clamp forward
# ---------------------------------------------------------------------------


class TestFusedSwiGLUScaleForwardClampCPU(unittest.TestCase):
    def test_vs_reference_and_shapes(self):
        """Matches reference, correct shape, dtype, 1D scale broadcast."""
        with _no_cuda():
            paddle.seed(42)
            x = paddle.randn([4, 16])
            out = fused_swiglu_scale_forward(x, paddle.ones([4, 1]), clamp_value=2.0)
            ref = _ref_clamped_swiglu(x, 2.0) * paddle.ones([4, 1])
            np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
            self.assertEqual(out.shape, [4, 8])
            self.assertEqual(out.dtype, paddle.float32)
            out1d = fused_swiglu_scale_forward(x, paddle.ones([4]), clamp_value=2.0)
            self.assertEqual(out1d.shape, [4, 8])

    def test_saturated_differs_from_nonclamp(self):
        """Small clamp on saturated inputs → different from non-clamp."""
        with _no_cuda():
            x = paddle.full([2, 8], 3.0)
            out_nc = fused_swiglu_scale_forward(x, paddle.ones([2, 1]))
            out_c = fused_swiglu_scale_forward(x, paddle.ones([2, 1]), clamp_value=0.5)
            self.assertFalse(bool((out_nc.numpy() == out_c.numpy()).all().item()))
            out_large = fused_swiglu_scale_forward(x, paddle.ones([2, 1]), clamp_value=20.0)
            self.assertFalse(bool((out_c.numpy() == out_large.numpy()).all().item()))


# ---------------------------------------------------------------------------
# BiasSwiGLUFunction / SwiGLUFunction with clamp_value
# ---------------------------------------------------------------------------


class TestBiasSwiGLUFunctionClamp(unittest.TestCase):
    def _fwd_bwd(self, clamp_value, fp8=False):
        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = BiasSwiGLUFunction.apply(x, bias, fp8, False, clamp_value=clamp_value)
        out.sum().backward()
        self.assertEqual(out.shape, [4, 8])
        self.assertIsNotNone(x.grad)

    def test_clamp_and_fp8(self):
        self._fwd_bwd(clamp_value=2.0)
        self._fwd_bwd(clamp_value=1.0, fp8=True)

    def test_no_clamp_and_zero(self):
        for cv in (None, 0.0):
            with self.subTest(clamp_value=cv):
                self._fwd_bwd(clamp_value=cv)


class TestSwiGLUFunctionClamp(unittest.TestCase):
    def test_with_and_without_clamp(self):
        for cv in (2.0, None):
            with self.subTest(clamp_value=cv):
                x = paddle.randn([4, 16]).astype("float32")
                x.stop_gradient = False
                out = SwiGLUFunction.apply(x, False, False, clamp_value=cv)
                out.sum().backward()
                self.assertEqual(out.shape, [4, 8])
                self.assertIsNotNone(x.grad)


# ---------------------------------------------------------------------------
# weighted_bias_swiglu_impl with clamp_value
# ---------------------------------------------------------------------------


class TestWeightedBiasSwiGLUImplClamp(unittest.TestCase):
    def _fwd_bwd(self, x, w, clamp_value=None, fp8=False):
        x.stop_gradient = False
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(x, None, w, fp8_input_store=fp8, clamp_value=clamp_value)
        out.sum().backward()
        self.assertEqual(out.shape, (*x.shape[:-1], x.shape[-1] // 2))
        return x.grad, w.grad

    def test_2d_clamp_and_nonclamp(self):
        for cv in (2.0, None):
            with self.subTest(clamp_value=cv):
                dx, dw = self._fwd_bwd(
                    paddle.randn([4, 16]).astype("float32"),
                    paddle.randn([4, 1]).astype("float32"),
                    clamp_value=cv,
                )
                self.assertEqual(dx.shape, [4, 16])
                self.assertEqual(dw.shape, [4, 1])

    def test_3d_clamp_and_nonclamp_and_b1(self):
        """3D flattens weights before PyLayer (clamp, non-clamp, B=1 edge)."""
        for cv, S, B in ((1.0, 2, 3), (None, 2, 3), (1.0, 4, 1)):
            with self.subTest(clamp_value=cv, S=S, B=B):
                H = 16
                x = paddle.randn([S, B, H]).astype("float32")
                w = paddle.full([S, B, 1], 0.5, dtype="float32")
                dx, dw = self._fwd_bwd(x, w, clamp_value=cv)
                self.assertEqual(dx.shape, [S, B, H])
                self.assertEqual(dw.shape, [S, B, 1])

    def test_fp8_input_store(self):
        dx, dw = self._fwd_bwd(
            paddle.randn([4, 16]).astype("float32"),
            paddle.randn([4, 1]).astype("float32"),
            clamp_value=1.0,
            fp8=True,
        )
        self.assertIsNotNone(dx)
        self.assertIsNotNone(dw)

    def test_bias_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(
                paddle.randn([4, 16]),
                paddle.randn([4, 16]),
                paddle.randn([4, 1]),
                clamp_value=2.0,
            )


# ---------------------------------------------------------------------------
# d_scale numerical correctness
# ---------------------------------------------------------------------------


class TestDScaleAlignment(unittest.TestCase):
    @staticmethod
    def _dscale_ref(x, scale, out_grad, clamp_value=None):
        sd = scale.dtype
        if clamp_value is not None:
            h = x.shape[-1] // 2
            g, v = paddle.chunk(x.cast(paddle.float32), 2, axis=-1)
            g = paddle.clip(g, max=clamp_value)
            v = paddle.clip(v, min=-clamp_value, max=clamp_value)
            sw = (F.silu(g) * v).cast(x.dtype)
        else:
            sw = F.swiglu(x)
        keepdim = clamp_value is not None
        return paddle.sum(sw * out_grad.cast(sd), axis=-1, keepdim=keepdim).cast(sd)

    def _check(self, x, scale, out_grad, clamp_value=None):
        with _no_cuda():
            _, ds = fused_swiglu_scale_backward(x, scale, out_grad, clamp_value=clamp_value)
            ref = self._dscale_ref(x, scale, out_grad, clamp_value)
            np.testing.assert_allclose(ds.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)
            expected_shape = [x.shape[0], 1] if clamp_value is not None else [x.shape[0]]
            self.assertEqual(ds.shape, expected_shape)
            self.assertEqual(ds.dtype, scale.dtype)

    def test_no_clamp_fp32(self):
        paddle.seed(1)
        self._check(
            paddle.randn([4, 16], dtype=paddle.float32),
            paddle.ones([4, 1], dtype=paddle.float32),
            paddle.randn([4, 8], dtype=paddle.float32),
        )

    def test_with_clamp_fp32_and_bf16_scale(self):
        paddle.seed(1)
        self._check(
            paddle.randn([4, 16], dtype=paddle.float32),
            paddle.full([4, 1], 0.5, dtype=paddle.float32),
            paddle.randn([4, 8], dtype=paddle.float32),
            clamp_value=2.0,
        )
        self._check(
            paddle.randn([4, 16], dtype=paddle.float32),
            paddle.ones([4, 1], dtype=paddle.bfloat16),
            paddle.randn([4, 8], dtype=paddle.float32),
            clamp_value=2.0,
        )


# ---------------------------------------------------------------------------
# bias_swiglu_impl with clamp_value
# ---------------------------------------------------------------------------


class TestBiasSwiGLUImplClamp(unittest.TestCase):
    def _fwd_bwd(self, x, bias=None, clamp_value=None):
        x.stop_gradient = False
        if bias is not None:
            bias.stop_gradient = False
        out = bias_swiglu_impl(x, bias, clamp_value=clamp_value)
        out.sum().backward()
        self.assertEqual(out.shape, (*x.shape[:-1], x.shape[-1] // 2))
        self.assertIsNotNone(x.grad)

    def test_2d_with_bias_and_no_bias(self):
        for bias in (paddle.randn([4, 16]).astype("float32"), None):
            self._fwd_bwd(
                paddle.randn([4, 16]).astype("float32"),
                bias=bias,
                clamp_value=2.0,
            )

    def test_batch8(self):
        self._fwd_bwd(
            paddle.randn([8, 16]).astype("float32"),
            bias=paddle.randn([8, 16]).astype("float32"),
            clamp_value=2.0,
        )

    def test_no_clamp(self):
        self._fwd_bwd(paddle.randn([4, 16]).astype("float32"), clamp_value=None)


# ---------------------------------------------------------------------------
# WeightedSwiGLUFunction PyLayer with clamp_value
# ---------------------------------------------------------------------------


class TestWeightedSwiGLUFunctionClamp(unittest.TestCase):
    def test_clamp_and_nonclamp(self):
        for cv in (1.0, None):
            with self.subTest(clamp_value=cv):
                x = paddle.randn([4, 16]).astype("float32")
                w = paddle.ones([4, 1]).astype("float32")
                x.stop_gradient = False
                w.stop_gradient = False
                out = WeightedSwiGLUFunction.apply(x, w, False, clamp_value=cv)
                out.sum().backward()
                self.assertEqual(out.shape, [4, 8])
                self.assertIsNotNone(x.grad)
                self.assertIsNotNone(w.grad)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestClampEdgeCases(unittest.TestCase):
    def test_clamp_value_zero_or_negative(self):
        """clamp_value ≤ 0 falls through to non-clamp path."""
        with _no_cuda():
            x = paddle.randn([4, 16])
            s = paddle.ones([4, 1])
            out0 = fused_swiglu_scale_forward(x, s, clamp_value=0.0)
            out_nc = fused_swiglu_scale_forward(x, s)
            np.testing.assert_allclose(out0.numpy(), out_nc.numpy(), rtol=1e-5, atol=1e-5)
        x2 = paddle.randn([4, 16]).astype("float32")
        x2.stop_gradient = False
        out = bias_swiglu_impl(x2, None, clamp_value=-1)
        out.sum().backward()
        self.assertEqual(out.shape, [4, 8])

    def test_invalid_ndim(self):
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(paddle.randn([4]), None, clamp_value=2.0)

    def test_zero_rows(self):
        """All zero-row paths return correct shapes without crash."""
        with _no_cuda():
            out = fused_swiglu_scale_forward(paddle.empty([0, 16]), paddle.empty([0, 1]), clamp_value=2.0)
            self.assertEqual(out.shape, [0, 8])
            dx, ds = fused_swiglu_scale_backward(
                paddle.empty([0, 16]),
                paddle.empty([0, 1]),
                paddle.empty([0, 8]),
                clamp_value=2.0,
            )
            self.assertEqual(dx.shape, [0, 16])
            self.assertEqual(ds.shape, [0, 1])
        # clamped functions
        y, cv, b = paddle.empty([0, 16]), 2.0, paddle.empty([0, 16])
        for fwd, bwd, bias_val in (
            (clamped_swiglu, clamped_swiglu_back, None),
            (clamped_bias_swiglu, clamped_bias_swiglu_back, b),
        ):
            g = paddle.empty([0, 8])
            if bias_val is None:
                self.assertEqual(fwd(y, cv).shape, [0, 8])
                self.assertEqual(bwd(g, y, cv).shape, [0, 16])
            else:
                self.assertEqual(fwd(y, bias_val, cv).shape, [0, 8])
                self.assertEqual(bwd(g, y, bias_val, cv).shape, [0, 16])
        # impl layers
        self.assertEqual(
            bias_swiglu_impl(paddle.empty([0, 16]), None, clamp_value=2.0).shape,
            [0, 8],
        )
        self.assertEqual(
            weighted_bias_swiglu_impl(
                paddle.empty([0, 16]),
                None,
                paddle.empty([0, 1]),
                clamp_value=2.0,
            ).shape,
            [0, 8],
        )
        # PyLayer zero rows
        x0 = paddle.empty([0, 16]).astype("float32")
        b0 = paddle.empty([0, 16]).astype("float32")
        w0 = paddle.empty([0, 1]).astype("float32")
        self.assertEqual(
            BiasSwiGLUFunction.apply(x0, b0, False, False, clamp_value=2.0).shape,
            [0, 8],
        )
        self.assertEqual(
            WeightedSwiGLUFunction.apply(x0, w0, False, clamp_value=2.0).shape,
            [0, 8],
        )


# ---------------------------------------------------------------------------
# Large tensor (int32-overflow numel > 2**31)
# ---------------------------------------------------------------------------


class TestClampLargeTensor(unittest.TestCase):
    @staticmethod
    def _skip_if_oom(rows, cols):
        est_gib = (rows * cols * 2) / (1024**3)
        if paddle.is_compiled_with_cuda():
            try:
                probe = paddle.empty([rows, cols], dtype=paddle.bfloat16)
                del probe
                paddle.device.synchronize()
                paddle.device.cuda.empty_cache()
            except Exception:
                raise unittest.SkipTest(f"GPU OOM for large tensor ({est_gib:.1f} GiB)")
        elif est_gib > 24:
            raise unittest.SkipTest(f"Large tensor ({est_gib:.1f} GiB) skipped on CPU")

    def test_clamped_swiglu_fwd_bwd_large(self):
        rows, hidden2 = 2**24, 136  # ~2.28B elements > 2**31
        self._skip_if_oom(rows, hidden2)
        x = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
        out = clamped_swiglu(x, clamp_value=3.0)
        self.assertEqual(out.shape, [rows, hidden2 // 2])
        g = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
        grad = clamped_swiglu_back(g, x, clamp_value=3.0)
        self.assertEqual(grad.shape, [rows, hidden2])

    def test_clamped_weighted_swiglu_large(self):
        rows, hidden2 = 2**24, 136
        self._skip_if_oom(rows, hidden2)
        y = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
        w = paddle.randn([rows, 1], dtype=paddle.bfloat16)
        out = clamped_weighted_swiglu(y, w, clamp_value=3.0)
        self.assertEqual(out.shape, [rows, hidden2 // 2])
        g = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
        d_y, d_w = clamped_weighted_swiglu_back(g, y, w, clamp_value=3.0)
        self.assertEqual(d_y.shape, [rows, hidden2])
        self.assertEqual(d_w.shape, [rows, 1])

    def test_fused_swiglu_scale_backward_large_both(self):
        """Non-clamp and clamp backward with large tensors."""
        rows, hidden2 = 2**24, 136
        self._skip_if_oom(rows, hidden2)
        with _no_cuda():
            x = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
            sc = paddle.ones([rows, 1], dtype=paddle.bfloat16)
            sg = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
            for cv in (None, 3.0):
                dx, ds = fused_swiglu_scale_backward(x, sc, sg, clamp_value=cv)
                self.assertEqual(dx.shape, [rows, hidden2])
                expected_ds = [rows, 1] if cv is not None else [rows]
                self.assertEqual(ds.shape, expected_ds)


if __name__ == "__main__":
    unittest.main()
