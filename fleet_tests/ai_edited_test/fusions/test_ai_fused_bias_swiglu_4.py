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

# Tests for src/paddlefleet/fusions/fused_bias_swiglu.py
# Focus: numeric equivalence of *_back wrappers against native
# paddle._C_ops.swiglu_grad, plus the cpu_offload_input branch of the
# PyLayer forward methods. PyLayer / Impl shape + apply paths are already
# covered via real .apply() calls in test_ai_fused_bias_swiglu.py
# (TestPyLayerBackwardReturnCount / TestImplShapes), so they are not
# duplicated here.

import unittest
from unittest import mock

import numpy as np
import paddle


class TestSwigluBackward(unittest.TestCase):
    """swiglu_back must match native paddle._C_ops.swiglu_grad."""

    def test_swiglu_back_matches_native_grad(self):
        from paddleformers.fleet.fusions.fused_bias_swiglu import swiglu_back

        paddle.seed(0)
        y = paddle.randn([2, 8])
        g = paddle.randn([2, 4])
        expected, _ = paddle._C_ops.swiglu_grad(y, None, g)
        out = swiglu_back(g, y)
        self.assertEqual(out.shape, y.shape)
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
        )


class TestBiasSwigluBack(unittest.TestCase):
    """bias_swiglu_back(g, y, bias) must equal native swiglu_grad(y+bias, g)."""

    def test_bias_swiglu_back_matches_native_grad_with_bias(self):
        from paddleformers.fleet.fusions.fused_bias_swiglu import bias_swiglu_back

        paddle.seed(0)
        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        bias = paddle.randn([8])
        expected, _ = paddle._C_ops.swiglu_grad(y + bias, None, g)
        out = bias_swiglu_back(g, y, bias)
        self.assertEqual(out.shape, y.shape)
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
        )


class TestWeightedSwigluBack(unittest.TestCase):
    """weighted_swiglu_back: shape + numeric equivalence to ref.

    input_grad == swiglu_grad(y, g*w),
    weights_grad == sum(swiglu(y) * g, axis=-1, keepdim=True).
    """

    def test_weighted_swiglu_back_shapes_and_values(self):
        from paddleformers.fleet.fusions.fused_bias_swiglu import (
            swiglu,
            weighted_swiglu_back,
        )

        paddle.seed(0)
        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])

        input_grad, weights_grad = weighted_swiglu_back(g, y, weights)
        self.assertEqual(input_grad.shape, y.shape)
        self.assertEqual(weights_grad.shape, [4, 1])

        expected_input_grad, _ = paddle._C_ops.swiglu_grad(y, None, g * weights)
        expected_weights_grad = paddle.sum(swiglu(y) * g, axis=-1, keepdim=True)
        np.testing.assert_allclose(
            input_grad.numpy(),
            expected_input_grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            weights_grad.numpy(),
            expected_weights_grad.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


class TestCpuOffloadInputBranch(unittest.TestCase):
    """Cover the cpu_offload_input=True branch of BiasSwiGLU/SwiGLU PyLayers."""

    def test_bias_swiglu_function_forward_cpu_offload(self):
        from paddleformers.fleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.bias_swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            BiasSwiGLUFunction.forward(mock_ctx, inp, bias, False, True)

    def test_swiglu_function_forward_cpu_offload(self):
        from paddleformers.fleet.fusions.fused_bias_swiglu import SwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        with mock.patch(
            "paddleformers.fleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            SwiGLUFunction.forward(mock_ctx, inp, False, True)


if __name__ == "__main__":
    unittest.main()
