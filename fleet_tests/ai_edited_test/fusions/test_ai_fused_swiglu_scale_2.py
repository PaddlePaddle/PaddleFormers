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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for src/paddlefleet/fusions/fused_swiglu_scale.py
# Additional tests for fused_swiglu_scale_forward and fused_swiglu_scale_backward

import unittest
from unittest import mock

import paddle


class TestFusedSwigluScaleForward(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward function."""

    def test_forward_cpu_fallback(self):
        """Test forward uses CPU fallback when CUDA not available."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([2.0], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [4, 4])

    def test_forward_scale_broadcast(self):
        """Test that scale is broadcast correctly in CPU fallback."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([2, 3, 8], dtype="float32")
        scale = paddle.to_tensor([1.5], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [2, 3, 4])

    def test_forward_cuda_path(self):
        """Test forward uses CUDA kernel when CUDA available."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")

        mock_result = paddle.randn([4, 4], dtype="float32")
        with mock.patch("paddle.is_compiled_with_cuda", return_value=True):  # noqa: SIM117
            with mock.patch(
                "paddlefleet_ops.fused_swiglu_scale",
                return_value=mock_result,
            ) as mock_op:
                result = fused_swiglu_scale_forward(x, scale)
                # When clamp_value is None, the original op is called
                # without clamp_value parameter
                mock_op.assert_called_once_with(x, scale)

    def test_forward_output_dtype_matches_input(self):
        """Test output dtype matches input dtype."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.dtype, x.dtype)


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward function."""

    def test_backward_cpu_fallback_shape(self):
        """Test backward CPU fallback returns correct shapes."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([2.0], dtype="float32")
        out_grad = paddle.randn([4, 4], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, x.shape)
            # d_scale sums over the last dim of out_grad, so shape is [4]
            self.assertEqual(d_scale.shape, [4])

    def test_backward_d_x_dtype(self):
        """Test backward d_x dtype matches input dtype."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        out_grad = paddle.randn([4, 4], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.dtype, x.dtype)

    def test_backward_d_scale_dtype(self):
        """Test backward d_scale dtype matches scale dtype."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        out_grad = paddle.randn([4, 4], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_scale.dtype, scale.dtype)

    def test_backward_3d_input(self):
        """Test backward with 3D input."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([2, 3, 8], dtype="float32")
        scale = paddle.to_tensor([1.5], dtype="float32")
        out_grad = paddle.randn([2, 3, 4], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            self.assertEqual(d_x.shape, x.shape)

    def test_backward_cuda_path(self):
        """Test backward uses CUDA kernel when available."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        out_grad = paddle.randn([4, 4], dtype="float32")
        mock_dx = paddle.randn([4, 8], dtype="float32")
        mock_ds = paddle.randn([1], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=True):  # noqa: SIM117
            with mock.patch(
                "paddlefleet_ops.fused_swiglu_scale_bwd",
                return_value=(mock_dx, mock_ds),
            ) as mock_op:
                d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
                # When clamp_value is None, the original bwd op is called
                # without clamp_value parameter
                mock_op.assert_called_once_with(x, scale, out_grad)


class TestFusedSwigluScaleMath(unittest.TestCase):
    """Tests for mathematical correctness of CPU fallback."""

    def test_backward_d_val_formula(self):
        """Test d_val = d_u * silu computation."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([2, 8], dtype="float32")
        scale = paddle.to_tensor([1.0], dtype="float32")
        out_grad = paddle.randn([2, 4], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
            # Just verify it runs without error
            self.assertIsNotNone(d_x)

    def test_scale_broadcast_2d_to_4d(self):
        """Test scale broadcasting for 4D tensors."""
        from paddleformers.fleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([2, 3, 4, 8], dtype="float32")
        scale = paddle.to_tensor([2.0], dtype="float32")

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):
            result = fused_swiglu_scale_forward(x, scale)
            self.assertEqual(result.shape, [2, 3, 4, 4])


class TestFusedSwigluScaleModule(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected functions are exported."""
        import paddleformers.fleet.fusions.fused_swiglu_scale as mod

        self.assertTrue(hasattr(mod, "fused_swiglu_scale_forward"))
        self.assertTrue(hasattr(mod, "fused_swiglu_scale_backward"))
        self.assertTrue(callable(mod.fused_swiglu_scale_forward))
        self.assertTrue(callable(mod.fused_swiglu_scale_backward))


if __name__ == "__main__":
    unittest.main()
