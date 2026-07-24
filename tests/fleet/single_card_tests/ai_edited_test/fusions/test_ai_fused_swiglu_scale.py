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

import unittest
from unittest.mock import patch

import paddle

from paddleformers.fleet.fusions.fused_swiglu_scale import (
    fused_swiglu_scale_backward,
    fused_swiglu_scale_forward,
)


class TestFusedSwigluScaleForward(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward function."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_cpu_fallback(self, mock_cuda):
        """Test CPU fallback computes swiglu * scale."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor([1.0, 2.0])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [2, 4])
        self.assertIsNotNone(result)

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_scale_broadcast_1d(self, mock_cuda):
        """Test that 1D scale is broadcast correctly."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor([2.0])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [2, 4])

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_scale_broadcast_0d(self, mock_cuda):
        """Test that 0D scale works correctly."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor(2.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [2, 4])

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_dtype_preserved(self, mock_cuda):
        """Test output dtype matches input dtype."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor(1.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.dtype, paddle.float32)

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_half_dtype(self, mock_cuda):
        """Test forward with float16 input."""
        x = paddle.randn([2, 8], dtype=paddle.float16)
        scale = paddle.to_tensor(1.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.dtype, paddle.float16)


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward function."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_backward_returns_dx_and_dscale(self, mock_cuda):
        """Test backward returns (d_x, d_scale)."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor([1.0, 2.0])
        out_grad = paddle.randn([2, 4], dtype=paddle.float32)
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [2, 8])
        self.assertEqual(d_scale.shape, [2])

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_backward_0d_scale(self, mock_cuda):
        """Test backward with 0D scale."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor(2.0)
        out_grad = paddle.randn([2, 4], dtype=paddle.float32)
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [2, 8])
        # d_scale is summed over the last axis of out_grad, so shape is [2]
        self.assertEqual(d_scale.shape, [2])

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_backward_dtype_matches(self, mock_cuda):
        """Test backward output dtype matches input dtype."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor(1.0, dtype=paddle.float32)
        out_grad = paddle.randn([2, 4], dtype=paddle.float32)
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.dtype, paddle.float32)
        self.assertEqual(d_scale.dtype, paddle.float32)

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_backward_scale_grad_dtype(self, mock_cuda):
        """Test d_scale dtype matches scale dtype."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        scale = paddle.to_tensor(1.0, dtype=paddle.float32)
        out_grad = paddle.randn([2, 4], dtype=paddle.float32)
        _, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_scale.dtype, paddle.float32)


class TestFusedSwigluScaleForwardCuda(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward with CUDA."""

    @patch(
        "paddleformers.fleet.fusions.fused_swiglu_scale.paddle.is_compiled_with_cuda",
        return_value=True,
    )
    def test_cuda_path_calls_ops(self, mock_cuda):
        """Test CUDA path calls fused_swiglu_scale from ops."""
        x = paddle.randn([2, 8])
        scale = paddle.to_tensor(1.0)
        with patch(
            "paddlefleet_ops.fused_swiglu_scale",
            return_value=paddle.randn([2, 4]),
        ) as mock_op:
            result = fused_swiglu_scale_forward(x, scale)
            mock_op.assert_called_once()


class TestFusedSwigluScaleBackwardCuda(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward with CUDA."""

    @patch(
        "paddleformers.fleet.fusions.fused_swiglu_scale.paddle.is_compiled_with_cuda",
        return_value=True,
    )
    def test_cuda_path_calls_ops(self, mock_cuda):
        """Test CUDA path calls fused_swiglu_scale_bwd from ops."""
        x = paddle.randn([2, 8])
        scale = paddle.to_tensor(1.0)
        out_grad = paddle.randn([2, 4])
        with patch(
            "paddlefleet_ops.fused_swiglu_scale_bwd",
            return_value=(paddle.randn([2, 8]), paddle.to_tensor(1.0)),
        ) as mock_op:
            result = fused_swiglu_scale_backward(x, scale, out_grad)
            mock_op.assert_called_once()


if __name__ == "__main__":
    unittest.main()
