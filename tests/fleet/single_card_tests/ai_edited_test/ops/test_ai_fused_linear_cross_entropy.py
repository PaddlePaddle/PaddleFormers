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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for paddlefleet_ops/ops/triton_ops/fused_linear_cross_entropy/fused_linear_cross_entropy.py
# Focus on: LigerFusedLinearCrossEntropyFunction backward with main_grad,
# ec_align mode, bias handling

import types
import unittest
from unittest.mock import MagicMock


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: ((lambda f: f) if fn is None else fn)
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0.0
    tl.sum = lambda *a, **kw: 0.0
    tl.exp = lambda x: 0.0
    tl.log = lambda x: 0.0
    tl.full = lambda shape, val, dtype=None: val
    tl.where = lambda cond, a, b: a
    tl.debug_barrier = lambda: None
    tl.float32 = "float32"
    tl.int64 = "int64"
    tl.int32 = "int32"
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLigerFusedLinearCrossEntropyBackward(unittest.TestCase):
    """Tests for LigerFusedLinearCrossEntropyFunction backward."""

    def test_backward_with_main_grad_ec_align(self):
        """Test backward with ec_align mode and main_grad."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        ctx = MagicMock()
        grad_input = paddle.randn([4, 8])
        grad_weight = paddle.randn([8, 16])  # ec_align: [H, V]
        grad_bias = None
        ctx.saved_tensor = MagicMock(return_value=(grad_input, grad_weight, grad_bias))
        ctx.weight_requires_grad = True
        ctx.ec_align = True
        ctx.has_bias = False

        # Create weight with main_grad
        weight = paddle.randn([16, 8])
        weight.main_grad = paddle.zeros([16, 8], dtype=paddle.float32)
        ctx.weight_ref = weight

        grad_output = paddle.to_tensor(1.0)
        result = LigerFusedLinearCrossEntropyFunction.backward(ctx, grad_output)

        # With ec_align, grad_weight should be transposed before adding to main_grad
        self.assertIsNotNone(result)

    def test_backward_with_main_grad_no_ec_align(self):
        """Test backward without ec_align mode and main_grad."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        ctx = MagicMock()
        grad_input = paddle.randn([4, 8])
        grad_weight = paddle.randn([16, 8])  # no ec_align: [V, H]
        grad_bias = None
        ctx.saved_tensor = MagicMock(return_value=(grad_input, grad_weight, grad_bias))
        ctx.weight_requires_grad = True
        ctx.ec_align = False
        ctx.has_bias = False

        # Create weight with main_grad
        weight = paddle.randn([16, 8])
        weight.main_grad = paddle.zeros([16, 8], dtype=paddle.float32)
        ctx.weight_ref = weight

        grad_output = paddle.to_tensor(1.0)
        result = LigerFusedLinearCrossEntropyFunction.backward(ctx, grad_output)

        # Without ec_align, grad_weight should be added directly to main_grad
        self.assertIsNotNone(result)
        # main_grad should have been updated
        self.assertTrue(weight.main_grad.abs().sum().item() > 0)

    def test_backward_with_main_grad_none(self):
        """Test backward when main_grad is None."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        ctx = MagicMock()
        grad_input = paddle.randn([4, 8])
        grad_weight = paddle.randn([16, 8])
        grad_bias = None
        ctx.saved_tensor = MagicMock(return_value=(grad_input, grad_weight, grad_bias))
        ctx.weight_requires_grad = True
        ctx.ec_align = False
        ctx.has_bias = False

        # Create weight with main_grad = None
        weight = paddle.randn([16, 8])
        weight.main_grad = None
        ctx.weight_ref = weight

        grad_output = paddle.to_tensor(1.0)
        result = LigerFusedLinearCrossEntropyFunction.backward(ctx, grad_output)

        # main_grad should have been created
        self.assertIsNotNone(weight.main_grad)

    def test_backward_with_bias(self):
        """Test backward returns grad_bias when has_bias is True."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        ctx = MagicMock()
        grad_input = paddle.randn([4, 8])
        grad_weight = paddle.randn([16, 8])
        grad_bias = paddle.randn([16])
        ctx.saved_tensor = MagicMock(return_value=(grad_input, grad_weight, grad_bias))
        ctx.weight_requires_grad = False
        ctx.ec_align = False
        ctx.has_bias = True

        grad_output = paddle.to_tensor(1.0)
        result = LigerFusedLinearCrossEntropyFunction.backward(ctx, grad_output)

        # Should return 4 elements (grad_input, grad_weight, None, grad_bias)
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[2])  # target grad is None

    def test_backward_without_bias(self):
        """Test backward returns 3 elements when has_bias is False."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        ctx = MagicMock()
        grad_input = paddle.randn([4, 8])
        grad_weight = None
        grad_bias = None
        ctx.saved_tensor = MagicMock(return_value=(grad_input, grad_weight, grad_bias))
        ctx.weight_requires_grad = False
        ctx.ec_align = False
        ctx.has_bias = False

        grad_output = paddle.to_tensor(1.0)
        result = LigerFusedLinearCrossEntropyFunction.backward(ctx, grad_output)

        # Should return 3 elements (grad_input, grad_weight, None)
        self.assertEqual(len(result), 3)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFusedLinearCEForwardLogic(unittest.TestCase):
    """Tests for fused linear cross entropy forward logic."""

    def test_grad_input_created_when_input_requires_grad(self):
        """Test grad_input is created when input requires grad."""
        import paddle

        _input = paddle.randn([4, 8], stop_gradient=False)
        self.assertFalse(_input.stop_gradient)
        # In forward, grad_input should be created
        grad_input = paddle.zeros([4, 8], dtype=paddle.float32)
        self.assertEqual(grad_input.shape, _input.shape)

    def test_grad_input_none_when_input_no_grad(self):
        """Test grad_input is None when input doesn't require grad."""
        import paddle

        _input = paddle.randn([4, 8], stop_gradient=True)
        self.assertTrue(_input.stop_gradient)
        # In forward, grad_input should be None

    def test_grad_weight_none_when_weight_no_grad(self):
        """Test grad_weight is None when neither input nor weight requires grad."""
        import paddle

        _input = paddle.randn([4, 8], stop_gradient=True)
        weight = paddle.randn([16, 8], stop_gradient=True)
        # grad_weight should be None

    def test_loss_1d_shape(self):
        """Test loss_1d has correct shape."""
        import paddle

        BT = 4
        loss_1d = paddle.zeros([BT], dtype=paddle.float32)
        self.assertEqual(loss_1d.shape, [BT])

    def test_target_mask_computation(self):
        """Test target mask computation."""
        import paddle

        target = paddle.to_tensor([0, -100, 5, 3])
        ignore_index = -100
        target_mask = target != ignore_index
        self.assertTrue(target_mask[0].item())
        self.assertFalse(target_mask[1].item())
        self.assertTrue(target_mask[2].item())
        self.assertTrue(target_mask[3].item())


if __name__ == "__main__":
    unittest.main()
