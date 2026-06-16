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

"""Unit tests for fused mHC cuTile kernels vs native implementations.

Each test compares the fused kernel's forward output AND backward gradients
against a pure-Paddle differentiable reference to catch numerical drift
introduced by kernel fusion.
"""

import math
import unittest

import numpy as np
import paddle
from paddle import Tensor

from paddleformers.fleet.fusions.fused_mhc_kernels import is_cutile_available
from paddleformers.fleet.transformer.hyper_connection import (
    native_h_aggregate,
    native_h_post_bda,
    native_proj_rms,
    native_sinkhorn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPE = "float32"
FWD_ATOL, FWD_RTOL = 2e-5, 2e-5
BWD_ATOL, BWD_RTOL = 5e-5, 5e-5
# Relaxed tolerances for bfloat16
BF16_FWD_ATOL, BF16_FWD_RTOL = 2e-2, 2e-2
BF16_BWD_ATOL, BF16_BWD_RTOL = 5e-2, 5e-2
COSINE_SIM_THRESH = 0.999
# Relaxed tolerances for TF32 matmul kernels (10-bit mantissa)
TF32_FWD_ATOL, TF32_FWD_RTOL = 1e-3, 1e-3
TF32_BWD_ATOL, TF32_BWD_RTOL = 2e-3, 2e-3
# E2E fused pipeline accumulates TF32 error across multiple kernels
E2E_FUSED_FWD_ATOL, E2E_FUSED_FWD_RTOL = 1e-2, 1e-2
RAND_LO, RAND_HI = -0.1, 0.1
# Relaxed tolerances for large-shape tests (accumulated fp error over more elements)
LARGE_FWD_ATOL, LARGE_FWD_RTOL = 1e-4, 1e-4
LARGE_BWD_ATOL, LARGE_BWD_RTOL = 5e-4, 5e-4
LARGE_TF32_FWD_ATOL, LARGE_TF32_FWD_RTOL = 5e-3, 5e-3
LARGE_TF32_BWD_ATOL, LARGE_TF32_BWD_RTOL = 1e-2, 1e-2
LARGE_COSINE_SIM_THRESH = 0.998


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _requires_cutile(test_func):
    """Decorator to skip tests when cuTile is not available."""
    return unittest.skipUnless(
        is_cutile_available(), "cuTile (cuda.tile) not installed"
    )(test_func)


def _rand(*shape, dtype=DTYPE, requires_grad=False):
    """Uniform in [RAND_LO, RAND_HI] to keep magnitudes small for bf16 stability."""
    t = paddle.uniform(shape=list(shape), min=RAND_LO, max=RAND_HI, dtype=dtype)
    if requires_grad:
        t.stop_gradient = False
    return t


def _assert_close(a, b, atol, rtol, msg=""):
    """Assert tensors are close with both absolute and relative tolerance."""
    a_np = a.numpy()
    b_np = b.numpy()
    max_abs_diff = np.max(np.abs(a_np - b_np))
    np.testing.assert_allclose(
        a_np,
        b_np,
        atol=atol,
        rtol=rtol,
        err_msg=f"{msg} (max_abs_diff={max_abs_diff:.6e})",
    )


def _assert_cosine_similar(
    a: Tensor, b: Tensor, threshold: float, msg: str = ""
):
    """Assert that flattened tensors have cosine similarity >= threshold."""
    a_flat = a.flatten().astype("float32")
    b_flat = b.flatten().astype("float32")
    dot = paddle.sum(a_flat * b_flat)
    norm_a = paddle.norm(a_flat)
    norm_b = paddle.norm(b_flat)
    sim = (dot / (norm_a * norm_b + 1e-12)).item()
    max_abs_diff = paddle.max(paddle.abs(a_flat - b_flat)).item()
    assert sim >= threshold, (
        f"{msg}: cosine similarity {sim:.6f} < {threshold} "
        f"(max_abs_diff={max_abs_diff:.6e})"
    )


def _assert_not_all_zero(t: Tensor, msg: str = ""):
    """Assert tensor is not all zeros (sanity check for gradients)."""
    assert paddle.any(t != 0).item(), f"{msg}: tensor is all zeros"


# ---------------------------------------------------------------------------
# Pure-Paddle differentiable references (used by both fwd AND bwd tests)
# These bypass any autograd.PyLayer wrappers, testing the math directly.
# ---------------------------------------------------------------------------


def _ref_sinkhorn(logits: Tensor, num_iters: int, eps: float = 1e-6) -> Tensor:
    """Pure Paddle differentiable sinkhorn (no custom backward)."""
    row_max = logits.max(axis=-1, keepdim=True)
    M = paddle.exp(logits - row_max)
    for _ in range(num_iters):
        M = M / M.sum(axis=-1, keepdim=True).clip(min=eps)
        M = M / M.sum(axis=-2, keepdim=True).clip(min=eps)
    return M


def _ref_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Pure Paddle differentiable h_aggregate."""
    return (x * h_pre.unsqueeze(-1)).sum(axis=2)


def _ref_h_post_bda(h_res, orig_res, h_post, x, bias):
    """Pure Paddle differentiable h_post_bda."""
    s, b, n, C = orig_res.shape
    mixed = paddle.bmm(
        h_res.reshape([s * b, n, n]).transpose([0, 2, 1]),
        orig_res.reshape([s * b, n, C]),
    ).reshape([s, b, n, C])
    x_exp = h_post.unsqueeze(-1) * x.unsqueeze(2)
    out = x_exp + mixed
    if bias is not None:
        out = out + h_post.unsqueeze(-1) * bias.reshape([1, 1, 1, C])
    return out


def _ref_proj_rms(x: Tensor, weight: Tensor, eps: float = 1e-6):
    """Pure Paddle differentiable proj_rms."""
    proj = paddle.matmul(x, weight.t())
    norm = x.norm(axis=-1, keepdim=True)
    K = x.shape[-1]
    r = 1.0 / (norm / math.sqrt(K) + eps)
    return proj, r


# ============================================================================
# Sinkhorn Tests
# ============================================================================


class TestNativeSinkhorn(unittest.TestCase):
    """Tests for the native SinkhornKnopp implementation."""

    def setUp(self):
        paddle.set_device("gpu")

    def _run_fwd_bwd(self, s, b, n, iters, eps=1e-6):
        """Run native vs reference and check forward + backward."""
        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        # -- native path (PyLayer) --
        inp_f = data.clone()
        inp_f.stop_gradient = False
        out_f = native_sinkhorn(inp_f, iters, eps)
        paddle.autograd.backward([out_f], [grad_out])
        grad_f = inp_f.grad.clone()

        # -- pure reference (differentiable) --
        inp_r = data.clone()
        inp_r.stop_gradient = False
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        paddle.autograd.backward([out_r], [grad_out])
        grad_r = inp_r.grad.clone()

        _assert_close(out_f, out_r, FWD_ATOL, FWD_RTOL, "sinkhorn fwd")
        _assert_close(grad_f, grad_r, BWD_ATOL, BWD_RTOL, "sinkhorn bwd")
        _assert_cosine_similar(
            grad_f, grad_r, COSINE_SIM_THRESH, "sinkhorn grad cosine"
        )
        _assert_not_all_zero(grad_f, "sinkhorn grad")

    def test_shape_8192_1_4_iters5(self):
        self._run_fwd_bwd(8192, 1, 4, 5)

    def test_doubly_stochastic(self):
        """Output should be approximately doubly stochastic."""
        logits = _rand(4, 6, 6)
        out = native_sinkhorn(logits, 20, 1e-6)
        out_np = out.numpy()
        np.testing.assert_allclose(out_np.sum(axis=-1), 1.0, atol=1e-4)
        np.testing.assert_allclose(out_np.sum(axis=-2), 1.0, atol=1e-4)

    def test_output_non_negative(self):
        """Doubly stochastic matrix elements should be non-negative."""
        logits = _rand(3, 4, 4)
        out = native_sinkhorn(logits, 10, 1e-6)
        assert (out.numpy() >= 0).all(), "Sinkhorn output has negative values"


class TestFusedSinkhorn(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    @_requires_cutile
    def test_fwd_bwd_shape_8192_1_4_iters5(self):
        self._run_fwd_bwd(8192, 1, 4, 5)

    @_requires_cutile
    def test_fwd_bwd_large_shape_int32_overflow(self):
        """Large N_batch to stress int32 offset boundaries."""
        # N_batch=262144, n=4 => tensor [262144, 4, 4], 16MB fp32 (small but tests large pid offsets)
        self._run_fwd_bwd(262144, 1, 4, 5)

    def _run_fwd_bwd(self, s, b, n, iters, eps=1e-6):
        from paddleformers.fleet.fusions.fused_mhc_kernels import fused_sinkhorn

        data = _rand(s, b, n, n)
        grad_out = _rand(s, b, n, n)

        # Use relaxed tolerances for large shapes
        is_large = s * b * n * n > 2**20
        fwd_atol = LARGE_FWD_ATOL if is_large else FWD_ATOL
        fwd_rtol = LARGE_FWD_RTOL if is_large else FWD_RTOL
        bwd_atol = LARGE_BWD_ATOL if is_large else BWD_ATOL
        bwd_rtol = LARGE_BWD_RTOL if is_large else BWD_RTOL
        cosine_thresh = (
            LARGE_COSINE_SIM_THRESH if is_large else COSINE_SIM_THRESH
        )

        # -- fused path --
        inp_f = data.clone()
        inp_f.stop_gradient = False
        out_f = fused_sinkhorn(inp_f, iters, eps)
        paddle.autograd.backward([out_f], [grad_out])
        grad_f = inp_f.grad.clone()

        # -- reference path --
        inp_r = data.clone()
        inp_r.stop_gradient = False
        out_r = _ref_sinkhorn(inp_r, iters, eps)
        paddle.autograd.backward([out_r], [grad_out])
        grad_r = inp_r.grad.clone()

        _assert_close(out_f, out_r, fwd_atol, fwd_rtol, "fused sinkhorn fwd")
        _assert_close(grad_f, grad_r, bwd_atol, bwd_rtol, "fused sinkhorn bwd")
        _assert_cosine_similar(
            grad_f, grad_r, cosine_thresh, "fused sinkhorn grad cosine"
        )


# ============================================================================
# H_aggregate Tests
# ============================================================================


class TestNativeHAggregate(unittest.TestCase):
    """Tests for native_h_aggregate."""

    def setUp(self):
        paddle.set_device("gpu")

    def _run_fwd_bwd(self, s, b, n, C):
        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)
        grad_out = _rand(s, b, C)

        xf = x_data.clone()
        xf.stop_gradient = False
        hf = h_data.clone()
        hf.stop_gradient = False
        of = native_h_aggregate(xf, hf)
        paddle.autograd.backward([of], [grad_out])

        xr = x_data.clone()
        xr.stop_gradient = False
        hr = h_data.clone()
        hr.stop_gradient = False
        oref = _ref_h_aggregate(xr, hr)
        paddle.autograd.backward([oref], [grad_out])

        _assert_close(of, oref, FWD_ATOL, FWD_RTOL, "h_aggregate fwd")
        _assert_close(
            xf.grad, xr.grad, BWD_ATOL, BWD_RTOL, "h_aggregate grad_x"
        )
        _assert_close(
            hf.grad, hr.grad, BWD_ATOL, BWD_RTOL, "h_aggregate grad_h"
        )
        _assert_cosine_similar(
            xf.grad, xr.grad, COSINE_SIM_THRESH, "h_aggregate grad_x cosine"
        )
        _assert_not_all_zero(xf.grad, "h_aggregate grad_x")
        _assert_not_all_zero(hf.grad, "h_aggregate grad_h")

    def test_shape_8192_1_4_4096(self):
        self._run_fwd_bwd(8192, 1, 4, 4096)

    def test_output_shape(self):
        """Output should be [s, b, C] when input is [s, b, n, C]."""
        x = _rand(2, 3, 4, 64)
        h = _rand(2, 3, 4)
        out = native_h_aggregate(x, h)
        self.assertEqual(list(out.shape), [2, 3, 64])


class TestFusedHAggregate(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    @_requires_cutile
    def test_fwd_bwd_shape_8192_1_4_4096(self):
        self._run_fwd_bwd(8192, 1, 4, 4096)

    @_requires_cutile
    def test_fwd_bwd_large_shape_int32_overflow(self):
        """Large sb*C to stress int32 offset boundaries."""
        # sb=65536, n=4, C=8192 => x tensor [65536, 4, 8192] ~8GB fp32
        self._run_fwd_bwd(65536, 1, 4, 8192)

    def _run_fwd_bwd(self, s, b, n, C):
        from paddleformers.fleet.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
        )

        x_data = _rand(s, b, n, C)
        h_data = _rand(s, b, n)
        grad_out = _rand(s, b, C)

        # Use relaxed tolerances for large shapes
        is_large = s * b * n * C > 2**20
        fwd_atol = LARGE_FWD_ATOL if is_large else FWD_ATOL
        fwd_rtol = LARGE_FWD_RTOL if is_large else FWD_RTOL
        bwd_atol = LARGE_BWD_ATOL if is_large else BWD_ATOL
        bwd_rtol = LARGE_BWD_RTOL if is_large else BWD_RTOL
        cosine_thresh = (
            LARGE_COSINE_SIM_THRESH if is_large else COSINE_SIM_THRESH
        )

        # -- fused --
        xf = x_data.clone()
        xf.stop_gradient = False
        hf = h_data.clone()
        hf.stop_gradient = False
        of = fused_h_aggregate(xf, hf)
        paddle.autograd.backward([of], [grad_out])

        # -- reference --
        xr = x_data.clone()
        xr.stop_gradient = False
        hr = h_data.clone()
        hr.stop_gradient = False
        oref = _ref_h_aggregate(xr, hr)
        paddle.autograd.backward([oref], [grad_out])

        _assert_close(of, oref, fwd_atol, fwd_rtol, "fused h_aggregate fwd")
        _assert_close(
            xf.grad, xr.grad, bwd_atol, bwd_rtol, "fused h_aggregate grad_x"
        )
        _assert_close(
            hf.grad, hr.grad, bwd_atol, bwd_rtol, "fused h_aggregate grad_h"
        )
        _assert_cosine_similar(
            xf.grad, xr.grad, cosine_thresh, "fused h_aggregate grad_x cosine"
        )


# ============================================================================
# H_post BDA Tests
# ============================================================================


class TestNativeHPostBDA(unittest.TestCase):
    """Tests for native_h_post_bda."""

    def setUp(self):
        paddle.set_device("gpu")

    def _run_fwd_bwd(self, s, b, n, C, with_bias):
        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None
        grad_out = _rand(s, b, n, C)

        def _make_inputs():
            hr = hr_data.clone()
            hr.stop_gradient = False
            orig = orig_data.clone()
            orig.stop_gradient = False
            hp = hp_data.clone()
            hp.stop_gradient = False
            x = x_data.clone()
            x.stop_gradient = False
            bi = bias_data.clone() if with_bias else None
            if bi is not None:
                bi.stop_gradient = False
            return hr, orig, hp, x, bi

        hr_f, orig_f, hp_f, x_f, bi_f = _make_inputs()
        out_f = native_h_post_bda(hr_f, orig_f, hp_f, x_f, bi_f)
        paddle.autograd.backward([out_f], [grad_out])

        hr_r, orig_r, hp_r, x_r, bi_r = _make_inputs()
        out_r = _ref_h_post_bda(hr_r, orig_r, hp_r, x_r, bi_r)
        paddle.autograd.backward([out_r], [grad_out])

        _assert_close(out_f, out_r, FWD_ATOL, FWD_RTOL, "h_post_bda fwd")
        for name, gf, gr in [
            ("h_res", hr_f.grad, hr_r.grad),
            ("orig_res", orig_f.grad, orig_r.grad),
            ("h_post", hp_f.grad, hp_r.grad),
            ("x", x_f.grad, x_r.grad),
        ]:
            _assert_close(gf, gr, BWD_ATOL, BWD_RTOL, f"h_post_bda bwd {name}")
            _assert_cosine_similar(
                gf, gr, COSINE_SIM_THRESH, f"h_post_bda bwd {name} cosine"
            )
            _assert_not_all_zero(gf, f"h_post_bda bwd {name}")
        if with_bias:
            _assert_close(
                bi_f.grad, bi_r.grad, BWD_ATOL, BWD_RTOL, "h_post_bda bwd bias"
            )

    def test_no_bias_8192_1_4_4096(self):
        self._run_fwd_bwd(8192, 1, 4, 4096, with_bias=False)

    def test_with_bias_8192_1_4_4096(self):
        self._run_fwd_bwd(8192, 1, 4, 4096, with_bias=True)


class TestFusedHPostBDA(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    @_requires_cutile
    def test_no_bias_8192_1_4_1280(self):
        self._run_fwd_bwd(8192, 1, 4, 1280, with_bias=False)

    @_requires_cutile
    def test_with_bias_8192_1_4_1280(self):
        self._run_fwd_bwd(8192, 1, 4, 1280, with_bias=True)

    @_requires_cutile
    def test_large_shape_int32_overflow(self):
        """Large sb*n*C to stress int32 offset boundaries."""
        # sb=65536, n=4, C=8192 => orig_residual [65536, 4, 8192] ~8GB fp32
        self._run_fwd_bwd(65536, 1, 4, 8192, with_bias=True)

    def _run_fwd_bwd(self, s, b, n, C, with_bias):
        from paddleformers.fleet.fusions.fused_mhc_kernels import (
            fused_h_post_bda,
        )

        hr_data = _rand(s, b, n, n)
        orig_data = _rand(s, b, n, C)
        hp_data = _rand(s, b, n)
        x_data = _rand(s, b, C)
        bias_data = _rand(C) if with_bias else None
        grad_out = _rand(s, b, n, C)

        # Use relaxed tolerances for large shapes
        is_large = s * b * n * C > 2**20
        fwd_atol = LARGE_FWD_ATOL if is_large else FWD_ATOL
        fwd_rtol = LARGE_FWD_RTOL if is_large else FWD_RTOL
        bwd_atol = LARGE_BWD_ATOL if is_large else BWD_ATOL
        bwd_rtol = LARGE_BWD_RTOL if is_large else BWD_RTOL
        cosine_thresh = (
            LARGE_COSINE_SIM_THRESH if is_large else COSINE_SIM_THRESH
        )

        def _make_inputs():
            hr = hr_data.clone()
            hr.stop_gradient = False
            orig = orig_data.clone()
            orig.stop_gradient = False
            hp = hp_data.clone()
            hp.stop_gradient = False
            x = x_data.clone()
            x.stop_gradient = False
            bi = bias_data.clone() if with_bias else None
            if bi is not None:
                bi.stop_gradient = False
            return hr, orig, hp, x, bi

        # -- fused path --
        hr_f, orig_f, hp_f, x_f, bi_f = _make_inputs()
        out_f = fused_h_post_bda(hr_f, orig_f, hp_f, x_f, bi_f)
        paddle.autograd.backward([out_f], [grad_out])

        # -- reference path --
        hr_r, orig_r, hp_r, x_r, bi_r = _make_inputs()
        out_r = _ref_h_post_bda(hr_r, orig_r, hp_r, x_r, bi_r)
        paddle.autograd.backward([out_r], [grad_out])

        _assert_close(out_f, out_r, fwd_atol, fwd_rtol, "fused h_post_bda fwd")
        for name, gf, gr in [
            ("h_res", hr_f.grad, hr_r.grad),
            ("orig_res", orig_f.grad, orig_r.grad),
            ("h_post", hp_f.grad, hp_r.grad),
            ("x", x_f.grad, x_r.grad),
        ]:
            _assert_close(
                gf, gr, bwd_atol, bwd_rtol, f"fused h_post_bda bwd {name}"
            )
            _assert_cosine_similar(
                gf, gr, cosine_thresh, f"fused h_post_bda bwd {name} cosine"
            )
        if with_bias:
            _assert_close(
                bi_f.grad,
                bi_r.grad,
                bwd_atol,
                bwd_rtol,
                "fused h_post_bda bwd bias",
            )


# ============================================================================
# Proj RMS Tests
# ============================================================================


class TestNativeProjRms(unittest.TestCase):
    """Tests for native_proj_rms."""

    def setUp(self):
        paddle.set_device("gpu")

    def _run_fwd_bwd(self, M, N, K, eps=1e-6):
        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        # native_proj_rms expects weight shape [K, N] (same as nn.Linear)
        xf = x_data.clone()
        xf.stop_gradient = False
        wf = w_data.t().clone()
        wf.stop_gradient = False
        proj_f, r_f = native_proj_rms(xf, wf, eps)
        loss_f = (proj_f * grad_proj + r_f * grad_r).sum()
        loss_f.backward()

        # _ref_proj_rms expects weight shape [N, K] (does x @ weight.t())
        xr = x_data.clone()
        xr.stop_gradient = False
        wr = w_data.clone()
        wr.stop_gradient = False
        proj_r, r_r = _ref_proj_rms(xr, wr, eps)
        loss_r = (proj_r * grad_proj + r_r * grad_r).sum()
        loss_r.backward()

        _assert_close(proj_f, proj_r, FWD_ATOL, FWD_RTOL, "proj_rms proj fwd")
        _assert_close(r_f, r_r, FWD_ATOL, FWD_RTOL, "proj_rms r fwd")
        _assert_close(xf.grad, xr.grad, BWD_ATOL, BWD_RTOL, "proj_rms bwd x")
        _assert_close(
            wf.grad, wr.grad.t(), BWD_ATOL, BWD_RTOL, "proj_rms bwd weight"
        )
        _assert_cosine_similar(
            xf.grad, xr.grad, COSINE_SIM_THRESH, "proj_rms grad_x cosine"
        )
        _assert_not_all_zero(xf.grad, "proj_rms grad_x")
        _assert_not_all_zero(wf.grad, "proj_rms grad_w")

    def test_shape_8192_24_16384(self):
        self._run_fwd_bwd(8192, 24, 16384)

    def test_r_positive(self):
        """r = 1/(norm/sqrt(K)+eps) should always be positive."""
        x = _rand(32, 128)
        w = _rand(128, 16)  # native_proj_rms expects [K, N]
        _, r = native_proj_rms(x, w, 1e-6)
        assert (r.numpy() > 0).all(), "proj_rms r has non-positive values"


class TestFusedProjRms(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    @_requires_cutile
    def test_fwd_bwd_8192_24_16384(self):
        self._run_fwd_bwd(8192, 24, 16384)

    @_requires_cutile
    def test_fwd_bwd_large_shape_int32_overflow(self):
        """M*K*elem_size > 2^31 to verify no int32 offset overflow."""
        # M=131072, K=8192, bf16 => 131072*8192*2 = 2GB (boundary)
        # M=262144, K=8192, bf16 => 262144*8192*2 = 4GB (exceeds int32)
        self._run_fwd_bwd(262144, 24, 8192)

    @_requires_cutile
    def test_large_tensor_boundary_correctness(self):
        """Verify last-tile data is read/written correctly (catches address truncation)."""
        from paddleformers.fleet.fusions.fused_mhc_kernels import fused_proj_rms

        M, K, N = 262144, 8192, 24
        x = paddle.zeros([M, K], dtype="float32")
        x[-1, -1] = 1.0  # highest address element
        w = paddle.zeros([K, N], dtype="float32")
        w[-1, 0] = 1.0  # so proj[-1, 0] should be 1.0
        proj, r = fused_proj_rms(x, w, 1e-6)
        assert abs(proj[-1, 0].item() - 1.0) < 1e-3, (
            f"Last element incorrect: {proj[-1, 0].item()}, possible address overflow"
        )
        assert proj[0, 0].item() == 0.0, "First element should be zero"

    def _run_fwd_bwd(self, M, N, K, eps=1e-6):
        from paddleformers.fleet.fusions.fused_mhc_kernels import fused_proj_rms

        x_data = _rand(M, K)
        w_data = _rand(N, K)
        grad_proj = _rand(M, N)
        grad_r = _rand(M, 1)

        # Use relaxed tolerances for large shapes
        is_large = M * K > 2**20
        tf32_fwd_atol = LARGE_TF32_FWD_ATOL if is_large else TF32_FWD_ATOL
        tf32_fwd_rtol = LARGE_TF32_FWD_RTOL if is_large else TF32_FWD_RTOL
        tf32_bwd_atol = LARGE_TF32_BWD_ATOL if is_large else TF32_BWD_ATOL
        tf32_bwd_rtol = LARGE_TF32_BWD_RTOL if is_large else TF32_BWD_RTOL
        fwd_atol = LARGE_FWD_ATOL if is_large else FWD_ATOL
        fwd_rtol = LARGE_FWD_RTOL if is_large else FWD_RTOL
        cosine_thresh = (
            LARGE_COSINE_SIM_THRESH if is_large else COSINE_SIM_THRESH
        )

        # -- fused (expects weight [K, N]) --
        xf = x_data.clone()
        xf.stop_gradient = False
        wf = w_data.t().clone()
        wf.stop_gradient = False
        proj_f, r_f = fused_proj_rms(xf, wf, eps)
        (proj_f * grad_proj + r_f * grad_r).sum().backward()

        # -- reference (expects weight [N, K], does x @ weight.t()) --
        xr = x_data.clone()
        xr.stop_gradient = False
        wr = w_data.clone()
        wr.stop_gradient = False
        proj_r, r_r = _ref_proj_rms(xr, wr, eps)
        (proj_r * grad_proj + r_r * grad_r).sum().backward()

        _assert_close(
            proj_f,
            proj_r,
            tf32_fwd_atol,
            tf32_fwd_rtol,
            "fused proj_rms proj fwd",
        )
        _assert_close(r_f, r_r, fwd_atol, fwd_rtol, "fused proj_rms r fwd")
        _assert_close(
            xf.grad,
            xr.grad,
            tf32_bwd_atol,
            tf32_bwd_rtol,
            "fused proj_rms bwd x",
        )
        _assert_close(
            wf.grad,
            wr.grad.t(),
            tf32_bwd_atol,
            tf32_bwd_rtol,
            "fused proj_rms bwd weight",
        )
        _assert_cosine_similar(
            xf.grad, xr.grad, cosine_thresh, "fused proj_rms grad_x cosine"
        )
        _assert_cosine_similar(
            wf.grad, wr.grad.t(), cosine_thresh, "fused proj_rms grad_w cosine"
        )


# ============================================================================
# End-to-end pipeline (all four kernels chained)
# ============================================================================


class TestEndToEndNative(unittest.TestCase):
    """Full mHC pipeline using native modules.

    proj_rms -> compute_h -> sinkhorn -> aggregate -> h_post_bda.
    Compares the native modules against inline Paddle reference.
    """

    def setUp(self):
        paddle.set_device("gpu")

    def test_full_pipeline_fwd_bwd(self):
        s, b, n, C = 8192, 1, 4, 4096
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_native_modules():
            hs = hs_data.clone()
            hs.stop_gradient = False
            w = w_data.t().clone()
            w.stop_gradient = False

            x_2d = hs.reshape([s * b, n * C])
            proj, r = native_proj_rms(x_2d, w, eps)
            proj = proj.reshape([s, b, -1])
            r = r.reshape([s, b, 1])

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = native_sinkhorn(
                h_res_logits.reshape([s, b, n, n]), sinkhorn_iters, eps
            )

            aggregated = native_h_aggregate(hs.reshape([s, b, n, C]), h_pre)

            output = native_h_post_bda(
                h_res,
                hs.reshape([s, b, n, C]),
                h_post,
                layer_out_data,
                layer_bias_data,
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        def _run_inline_ref():
            hs = hs_data.clone()
            hs.stop_gradient = False
            w = w_data.clone()
            w.stop_gradient = False

            x_2d = hs.reshape([s * b, n * C])
            proj, r = _ref_proj_rms(x_2d, w, eps)
            proj = proj.reshape([s, b, -1])
            r = r.reshape([s, b, 1])

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = _ref_sinkhorn(
                h_res_logits.reshape([s, b, n, n]), sinkhorn_iters, eps
            )

            aggregated = _ref_h_aggregate(hs.reshape([s, b, n, C]), h_pre)

            output = _ref_h_post_bda(
                h_res,
                hs.reshape([s, b, n, C]),
                h_post,
                layer_out_data,
                layer_bias_data,
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_m, agg_m, grad_m = _run_native_modules()
        out_r, agg_r, grad_r = _run_inline_ref()

        _assert_close(agg_m, agg_r, FWD_ATOL, FWD_RTOL, "E2E aggregated output")
        _assert_close(out_m, out_r, FWD_ATOL, FWD_RTOL, "E2E h_post_bda output")
        _assert_cosine_similar(
            grad_m, grad_r, COSINE_SIM_THRESH, "E2E hidden_states grad"
        )
        _assert_not_all_zero(grad_m, "E2E grad")


class TestEndToEndFused(unittest.TestCase):
    """Full mHC pipeline using fused cuTile kernels (requires cuTile)."""

    def setUp(self):
        paddle.set_device("gpu")

    @_requires_cutile
    def test_full_pipeline_fwd_bwd(self):
        from paddleformers.fleet.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

        s, b, n, C = 8192, 1, 4, 4096
        eps = 1e-6
        sinkhorn_iters = 5

        hs_data = _rand(s, b, n * C)
        w_data = _rand(n * n + 2 * n, n * C)
        layer_out_data = _rand(s, b, C)
        layer_bias_data = _rand(C)

        def _run_fused():
            hs = hs_data.clone()
            hs.stop_gradient = False
            w = w_data.t().clone()
            w.stop_gradient = False

            x_2d = hs.reshape([s * b, n * C])
            proj, r = fused_proj_rms(x_2d, w, eps)
            proj = proj.reshape([s, b, -1])
            r = r.reshape([s, b, 1])

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = fused_sinkhorn(
                h_res_logits.reshape([s, b, n, n]), sinkhorn_iters, eps
            )

            aggregated = fused_h_aggregate(hs.reshape([s, b, n, C]), h_pre)

            output = fused_h_post_bda(
                h_res,
                hs.reshape([s, b, n, C]),
                h_post,
                layer_out_data,
                layer_bias_data,
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        def _run_ref():
            hs = hs_data.clone()
            hs.stop_gradient = False
            w = w_data.clone()
            w.stop_gradient = False

            x_2d = hs.reshape([s * b, n * C])
            proj, r = _ref_proj_rms(x_2d, w, eps)
            proj = proj.reshape([s, b, -1])
            r = r.reshape([s, b, 1])

            h = r * proj
            h_pre = h[..., :n].sigmoid()
            h_post = h[..., n : 2 * n].sigmoid() * 2
            h_res_logits = h[..., 2 * n :]
            h_res = _ref_sinkhorn(
                h_res_logits.reshape([s, b, n, n]), sinkhorn_iters, eps
            )

            aggregated = _ref_h_aggregate(hs.reshape([s, b, n, C]), h_pre)

            output = _ref_h_post_bda(
                h_res,
                hs.reshape([s, b, n, C]),
                h_post,
                layer_out_data,
                layer_bias_data,
            )

            loss = output.sum() + aggregated.sum()
            loss.backward()
            return output.detach(), aggregated.detach(), hs.grad.clone()

        out_f, agg_f, grad_f = _run_fused()
        out_r, agg_r, grad_r = _run_ref()

        _assert_close(
            agg_f,
            agg_r,
            E2E_FUSED_FWD_ATOL,
            E2E_FUSED_FWD_RTOL,
            "E2E fused aggregated output",
        )
        _assert_close(
            out_f,
            out_r,
            E2E_FUSED_FWD_ATOL,
            E2E_FUSED_FWD_RTOL,
            "E2E fused h_post_bda output",
        )
        _assert_cosine_similar(
            grad_f, grad_r, COSINE_SIM_THRESH, "E2E fused hidden_states grad"
        )


# ============================================================================
# Error handling
# ============================================================================


class TestFusedMHCErrorHandling(unittest.TestCase):
    """Test error handling when cuTile is not available."""

    def test_no_cutile_raises(self):
        """If cuTile unavailable, fused ops should raise RuntimeError."""
        if is_cutile_available():
            self.skipTest("cuTile is available, cannot test error path")

        from paddleformers.fleet.fusions.fused_mhc_kernels import (
            fused_h_aggregate,
            fused_h_post_bda,
            fused_proj_rms,
            fused_sinkhorn,
        )

        with self.assertRaises(RuntimeError):
            fused_sinkhorn(paddle.randn([2, 4, 4]), 5)
        with self.assertRaises(RuntimeError):
            fused_h_aggregate(
                paddle.randn([2, 2, 4, 64]), paddle.randn([2, 2, 4])
            )
        with self.assertRaises(RuntimeError):
            fused_h_post_bda(
                paddle.randn([2, 2, 4, 4]),
                paddle.randn([2, 2, 4, 64]),
                paddle.randn([2, 2, 4]),
                paddle.randn([2, 2, 64]),
                None,
            )
        with self.assertRaises(RuntimeError):
            fused_proj_rms(paddle.randn([32, 128]), paddle.randn([128, 16]))


if __name__ == "__main__":
    unittest.main()
