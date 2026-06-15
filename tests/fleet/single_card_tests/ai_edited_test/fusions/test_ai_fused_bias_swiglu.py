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
import os
import sys

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

import unittest
from unittest.mock import MagicMock, patch

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
    clamped_swiglu,
    clamped_swiglu_back,
    clamped_weighted_swiglu,
    clamped_weighted_swiglu_back,
    swiglu,
    swiglu_back,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
    weighted_swiglu_back,
)


class TestSwiGLUForward(unittest.TestCase):
    """Forward computations for swiglu / bias_swiglu / weighted_swiglu."""

    def test_swiglu_forward(self):
        out = swiglu(paddle.randn([4, 16]))
        self.assertEqual(out.shape, [4, 8])

    def test_bias_swiglu_forward(self):
        out = bias_swiglu(paddle.randn([4, 16]), paddle.randn([16]))
        self.assertEqual(out.shape, [4, 8])

    def test_weighted_swiglu_dtype_preserved(self):
        x = paddle.randn([2, 8], dtype=paddle.float32)
        w = paddle.randn([2, 1], dtype=paddle.float32)
        out = weighted_swiglu(x, w)
        self.assertEqual(out.shape, [2, 4])
        self.assertEqual(out.dtype, paddle.float32)


class TestSwiGLUBackwardNativeGrad(unittest.TestCase):
    """*_back functions are now thin wrappers over paddle._C_ops.swiglu_grad
    and must match it numerically (no NotImplementedError fallback)."""

    def test_swiglu_back_matches_native_grad(self):
        paddle.seed(0)
        y = paddle.randn([2, 8])
        g = paddle.randn([2, 4])
        expected, _ = paddle._C_ops.swiglu_grad(y, None, g)
        np.testing.assert_allclose(
            swiglu_back(g, y).numpy(),
            expected.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_bias_swiglu_back_matches_native_grad(self):
        paddle.seed(0)
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        expected, _ = paddle._C_ops.swiglu_grad(y + bias, None, g)
        np.testing.assert_allclose(
            bias_swiglu_back(g, y, bias).numpy(),
            expected.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_weighted_swiglu_back_matches_native_grad(self):
        paddle.seed(0)
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        w = paddle.randn([2, 1])
        ig, wg = weighted_swiglu_back(g, y, w)
        expected_ig, _ = paddle._C_ops.swiglu_grad(y, None, g * w)
        expected_wg = paddle.sum(swiglu(y) * g, axis=-1, keepdim=True)
        self.assertEqual(ig.shape, [2, 8])
        self.assertEqual(wg.shape, [2, 1])
        np.testing.assert_allclose(ig.numpy(), expected_ig.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(wg.numpy(), expected_wg.numpy(), rtol=1e-5, atol=1e-5)


class TestSwigluBackShapes(unittest.TestCase):
    """Sanity-check output shapes from the live (native-grad backed) ops."""

    def test_swiglu_back_shape(self):
        out = swiglu_back(paddle.randn([2, 4]), paddle.randn([2, 8]))
        self.assertEqual(out.shape, [2, 8])

    def test_weighted_swiglu_back_shapes(self):
        ig, wg = weighted_swiglu_back(
            paddle.randn([2, 4]),
            paddle.randn([2, 8]),
            paddle.randn([2, 1]),
        )
        self.assertEqual(ig.shape, [2, 8])
        self.assertEqual(wg.shape, [2, 1])


class TestPyLayerForward(unittest.TestCase):
    """Forward dispatch through PyLayer.apply for each variant."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_bias_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.bias_swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                BiasSwiGLUFunction.apply(paddle.randn([2, 8]), paddle.randn([8]), False, False)
            except NotImplementedError:
                pass

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                SwiGLUFunction.apply(paddle.randn([2, 8]), False, False)
            except NotImplementedError:
                pass

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_weighted_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.weighted_swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                WeightedSwiGLUFunction.apply(paddle.randn([2, 8]), paddle.randn([2, 1]), False)
            except NotImplementedError:
                pass


class TestPyLayerBackwardReturnCount(unittest.TestCase):
    """Verify that each PyLayer.backward returns the correct number of values.

    Paddle's PyLayer requires backward to return exactly as many values as
    there are tensor inputs that require gradients in forward. Non-tensor
    arguments (bool, float) do not count. Returning the wrong number causes
    a runtime ValueError during backward.
    """

    def test_swiglu_function_backward_returns_1(self):
        """SwiGLUFunction.forward(input, fp8_input_store, cpu_offload_input)
        has 1 tensor input (input) => backward must return 1 value."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        result = SwiGLUFunction.apply(x, False, False)
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [4, 16])

    def test_weighted_swiglu_function_backward_returns_2(self):
        """WeightedSwiGLUFunction.forward(input, weights, fp8_input_store)
        has 2 tensor inputs (input, weights) => backward must return 2."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        result = WeightedSwiGLUFunction.apply(x, w, False)
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertEqual(w.grad.shape, [4, 1])

    def test_clamped_weighted_swiglu_function_backward_returns_2(self):
        """WeightedSwiGLUFunction.apply(input, weights,
        fp8_input_store, clamp_value) has 2 tensor inputs (input, weights)
        => backward must return 2 values."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        result = WeightedSwiGLUFunction.apply(x, w, False, 2.0)
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertEqual(w.grad.shape, [4, 1])

    def test_bias_swiglu_function_backward_returns_2(self):
        """BiasSwiGLUFunction.forward(input, bias, fp8_input_store,
        cpu_offload_input) has 2 tensor inputs (input, bias)
        => backward must return 2 values."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        b = paddle.randn([16]).astype("float32")
        b.stop_gradient = False
        result = BiasSwiGLUFunction.apply(x, b, False, False)
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(b.grad)
        self.assertEqual(x.grad.shape, [4, 16])


class TestImplShapes(unittest.TestCase):
    """bias_swiglu_impl / weighted_bias_swiglu_impl shape & branch coverage."""

    def test_bias_swiglu_impl_2d_with_bias(self):
        out = bias_swiglu_impl(paddle.randn([4, 16]), paddle.randn([16]))
        self.assertEqual(out.shape, [4, 8])

    def test_bias_swiglu_impl_3d_no_bias(self):
        # Covers both 3D-reshape path and bias-None branch.
        out = bias_swiglu_impl(paddle.randn([2, 4, 16]), None)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_weighted_bias_swiglu_impl_no_bias(self):
        out = weighted_bias_swiglu_impl(paddle.randn([4, 16]), None, paddle.randn([4, 1]))
        self.assertEqual(out.shape, [4, 8])

    def test_weighted_bias_swiglu_impl_with_bias_raises(self):
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(
                paddle.randn([4, 16]),
                paddle.randn([16]),
                paddle.randn([4, 1]),
            )


class TestClampedSwiGLU(unittest.TestCase):
    """Tests for clamped_swiglu / clamped_weighted_swiglu and their backwards."""

    def test_forward_clamp_effect(self):
        """Output is bounded when clamp_value is small AND shape is correct."""
        clamp_value = 0.1
        y = paddle.full([4, 16], fill_value=100.0)
        out = clamped_swiglu(y, clamp_value=clamp_value)
        self.assertEqual(out.shape, [4, 8])
        max_possible = float(F.silu(paddle.to_tensor(clamp_value)).numpy()) * clamp_value
        self.assertTrue(
            float(out.abs().max().numpy()) <= max_possible + 1e-5,
        )

    def test_large_clamp_equals_standard_swiglu(self):
        """With huge clamp_value, clamped_swiglu must match standard swiglu."""
        y = paddle.randn([4, 16])
        np.testing.assert_allclose(
            clamped_swiglu(y, clamp_value=1e9).cast("float32").numpy(),
            swiglu(y).cast("float32").numpy(),
            atol=1e-4,
        )

    def test_clamped_swiglu_back_zero_at_saturation(self):
        """Backward shape matches AND saturated inputs produce zero grad."""
        y = paddle.full([2, 8], fill_value=100.0)
        g = paddle.ones([2, 4])
        grad = clamped_swiglu_back(g, y, clamp_value=1.0)
        self.assertEqual(grad.shape, [2, 8])
        np.testing.assert_allclose(grad.numpy(), np.zeros_like(grad.numpy()), atol=1e-6)

    def test_clamped_weighted_swiglu_fwd_bwd(self):
        """clamped_weighted_swiglu forward + backward shape coverage."""
        y = paddle.randn([4, 16])
        w = paddle.randn([4, 1])
        out = clamped_weighted_swiglu(y, w, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        grad_y, grad_w = clamped_weighted_swiglu_back(paddle.randn([4, 8]), y, w, clamp_value=2.0)
        self.assertEqual(grad_y.shape, [4, 16])
        self.assertEqual(grad_w.shape, [4, 1])

    def test_weighted_bias_swiglu_impl_clamp_e2e(self):
        """End-to-end fwd+bwd through weighted_bias_swiglu_impl."""
        from paddleformers.fleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        inp = paddle.randn([4, 16])
        inp.stop_gradient = False
        w = paddle.randn([4, 1])
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(inp, None, w, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        grads = paddle.grad([out.sum()], [inp, w])
        self.assertEqual(grads[0].shape, [4, 16])
        self.assertEqual(grads[1].shape, [4, 1])
        self.assertFalse(bool(paddle.isnan(out).any().numpy()))

    def test_clamped_weighted_pylayer_fwd_bwd(self):
        """WeightedSwiGLUFunction PyLayer with clamp_value + backward."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        result = WeightedSwiGLUFunction.apply(x, w, False, 1.0)
        self.assertEqual(result.shape, [4, 8])
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_pylayer_without_clamp(self):
        """Non-clamp PyLayer dispatches to weighted_swiglu_back."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        mock_back = MagicMock(return_value=(paddle.randn([4, 16]), paddle.randn([4, 1])))
        with patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.weighted_swiglu_back",
            mock_back,
        ):
            result = WeightedSwiGLUFunction.apply(x, w, False)
            self.assertEqual(result.shape, [4, 8])
            result.sum().backward()
            mock_back.assert_called_once()

    # ------------------------------------------------------------------
    # Large-tensor coverage: numel must exceed int32 range (2**31) so that
    # any internal indexing/stride computation that still uses int32 will
    # overflow and surface here. Sized just over 2**31 elements in bfloat16
    # to keep peak GPU memory tractable (~30 GB transient incl. fp32 cast
    # and chunked copies). Skipped on CPU-only runs and when the GPU has
    # insufficient total memory.
    # ------------------------------------------------------------------
    @unittest.skipUnless(
        paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
        "int32-overflow large-tensor test requires a CUDA device",
    )
    def test_clamped_swiglu_int32_overflow_numel(self):
        """fwd+bwd on a tensor whose numel > 2**31, to catch int32 indexing overflow."""
        # 65536 * 32770 = 2,147,614,720 > 2**31 = 2,147,483,648
        rows, hidden2 = 65536, 32770
        self.assertGreater(rows * hidden2, 2**31)

        # Require ~40 GB of total GPU memory as a conservative gate; the
        # internal fp32 cast + two fp32 chunks dominate transient usage.
        props = paddle.device.cuda.get_device_properties(0)
        if props.total_memory < 40 * (1024**3):
            self.skipTest(
                f"need >=40GB GPU memory for int32-overflow test, " f"have {props.total_memory / 1024**3:.1f}GB"
            )

        prev_device = paddle.get_device()
        paddle.set_device("gpu")
        try:
            y = paddle.randn([rows, hidden2], dtype="bfloat16")
            out = clamped_swiglu(y, clamp_value=2.0)
            self.assertEqual(out.shape, [rows, hidden2 // 2])
            self.assertEqual(out.dtype, paddle.bfloat16)
            # Spot-check finiteness on a slice to avoid an extra full-tensor
            # reduction allocation; clamp guarantees a bounded range so a
            # NaN/Inf would indicate an indexing/overflow bug, not numerics.
            self.assertTrue(bool(paddle.isfinite(out[:1024].cast("float32")).all().numpy()))
            self.assertTrue(bool(paddle.isfinite(out[-1024:].cast("float32")).all().numpy()))
            del out

            g = paddle.randn([rows, hidden2 // 2], dtype="bfloat16")
            grad = clamped_swiglu_back(g, y, clamp_value=2.0)
            self.assertEqual(grad.shape, [rows, hidden2])
            self.assertTrue(bool(paddle.isfinite(grad[:1024].cast("float32")).all().numpy()))
            self.assertTrue(bool(paddle.isfinite(grad[-1024:].cast("float32")).all().numpy()))
        finally:
            paddle.device.cuda.empty_cache()
            paddle.set_device(prev_device)

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
        "int32-overflow large-tensor test requires a CUDA device",
    )
    def test_clamped_weighted_swiglu_int32_overflow_numel(self):
        """Weighted clamp variant fwd+bwd with numel > 2**31."""
        rows, hidden2 = 65536, 32770
        self.assertGreater(rows * hidden2, 2**31)

        props = paddle.device.cuda.get_device_properties(0)
        if props.total_memory < 40 * (1024**3):
            self.skipTest(
                f"need >=40GB GPU memory for int32-overflow test, " f"have {props.total_memory / 1024**3:.1f}GB"
            )

        prev_device = paddle.get_device()
        paddle.set_device("gpu")
        try:
            y = paddle.randn([rows, hidden2], dtype="bfloat16")
            w = paddle.randn([rows, 1], dtype="bfloat16")
            out = clamped_weighted_swiglu(y, w, clamp_value=2.0)
            self.assertEqual(out.shape, [rows, hidden2 // 2])
            del out

            grad_y, grad_w = clamped_weighted_swiglu_back(
                paddle.randn([rows, hidden2 // 2], dtype="bfloat16"),
                y,
                w,
                clamp_value=2.0,
            )
            self.assertEqual(grad_y.shape, [rows, hidden2])
            self.assertEqual(grad_w.shape, [rows, 1])
            self.assertTrue(bool(paddle.isfinite(grad_y[:1024].cast("float32")).all().numpy()))
            self.assertTrue(bool(paddle.isfinite(grad_w.cast("float32")).all().numpy()))
        finally:
            paddle.device.cuda.empty_cache()
            paddle.set_device(prev_device)

    # ------------------------------------------------------------------
    # 0-size tensor coverage: empty rows ([0, H]) is the realistic case
    # (e.g. an MoE expert that received no tokens this step). We also
    # exercise [N, 0] and [0, 0] to ensure chunk/clip/silu compose without
    # asserting on a non-empty axis.
    # ------------------------------------------------------------------
    def test_clamped_swiglu_zero_size_forward(self):
        """clamped_swiglu accepts 0-size inputs and returns correctly-halved shape."""
        for shape, expected in (
            ([0, 16], [0, 8]),
            ([4, 0], [4, 0]),
            ([0, 0], [0, 0]),
        ):
            y = paddle.zeros(shape)
            out = clamped_swiglu(y, clamp_value=1.0)
            self.assertEqual(out.shape, expected)
            self.assertEqual(out.dtype, y.dtype)

    def test_clamped_swiglu_zero_size_backward(self):
        """clamped_swiglu_back accepts 0-size inputs and returns input-shaped grad."""
        for shape in ([0, 16], [4, 0]):
            y = paddle.zeros(shape)
            g = paddle.zeros([shape[0], shape[1] // 2])
            grad = clamped_swiglu_back(g, y, clamp_value=1.0)
            self.assertEqual(grad.shape, shape)

    def test_clamped_weighted_swiglu_zero_size_fwd_bwd(self):
        """Empty-row case for the weighted clamp variant (e.g. MoE expert with 0 tokens)."""
        y = paddle.zeros([0, 16])
        w = paddle.zeros([0, 1])
        out = clamped_weighted_swiglu(y, w, clamp_value=1.0)
        self.assertEqual(out.shape, [0, 8])
        grad_y, grad_w = clamped_weighted_swiglu_back(paddle.zeros([0, 8]), y, w, clamp_value=1.0)
        self.assertEqual(grad_y.shape, [0, 16])
        self.assertEqual(grad_w.shape, [0, 1])

    def test_clamped_weighted_pylayer_zero_size(self):
        """WeightedSwiGLUFunction PyLayer with clamp_value on 0 rows."""
        x = paddle.zeros([0, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.zeros([0, 1]).astype("float32")
        w.stop_gradient = False
        out = WeightedSwiGLUFunction.apply(x, w, False, 1.0)
        self.assertEqual(out.shape, [0, 8])
        out.sum().backward()
        self.assertEqual(x.grad.shape, [0, 16])
        self.assertEqual(w.grad.shape, [0, 1])


if __name__ == "__main__":
    unittest.main()
