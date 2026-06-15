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

import unittest

import numpy as np
import paddle


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is not available")
class TestLigerCrossEntropyKernelGPU(unittest.TestCase):
    """Tests for liger_cross_entropy_kernel on GPU."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.seed(42)
        np.random.seed(42)
        paddle.enable_compat(scope={"triton"}, silent=True)

    def tearDown(self):
        """Clean up after tests."""
        paddle.disable_compat()

    def test_kernel_basic_forward(self):
        """Test basic forward pass of liger_cross_entropy_kernel."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        BT, V = 4, 16
        BLOCK_SIZE = min(32768, triton.next_power_of_2(V))

        logits = paddle.randn([BT, V], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])
        loss_1d = paddle.zeros([BT], dtype=paddle.float32)

        logits = logits.contiguous()
        target = target.contiguous()
        loss_1d = loss_1d.contiguous()

        n_non_ignore = BT
        ignore_index = -100

        liger_cross_entropy_kernel[(BT,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            reduction="none",
            HAS_GRADIENTS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertEqual(loss_1d.shape[0], BT)
        self.assertTrue(paddle.all(paddle.isfinite(loss_1d)))

    def test_kernel_with_gradients(self):
        """Test kernel with gradient computation."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        BT, V = 4, 16
        BLOCK_SIZE = min(32768, triton.next_power_of_2(V))

        logits = paddle.randn([BT, V], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])
        loss_1d = paddle.zeros([BT], dtype=paddle.float32)

        logits = logits.contiguous()
        target = target.contiguous()
        loss_1d = loss_1d.contiguous()

        n_non_ignore = BT
        ignore_index = -100

        liger_cross_entropy_kernel[(BT,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            reduction="none",
            HAS_GRADIENTS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertTrue(paddle.all(paddle.isfinite(loss_1d)))

    def test_kernel_with_ignore_index(self):
        """Test kernel with ignore_index."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        BT, V = 4, 16
        BLOCK_SIZE = min(32768, triton.next_power_of_2(V))

        logits = paddle.randn([BT, V], dtype=paddle.float32)
        target = paddle.to_tensor([0, -100, 5, 3])
        loss_1d = paddle.zeros([BT], dtype=paddle.float32)

        logits = logits.contiguous()
        target = target.contiguous()
        loss_1d = loss_1d.contiguous()

        n_non_ignore = 3
        ignore_index = -100

        liger_cross_entropy_kernel[(BT,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            reduction="none",
            HAS_GRADIENTS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertTrue(paddle.all(paddle.isfinite(loss_1d)))

    def test_kernel_mean_reduction(self):
        """Test kernel with mean reduction."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        BT, V = 4, 16
        BLOCK_SIZE = min(32768, triton.next_power_of_2(V))

        logits = paddle.randn([BT, V], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])
        loss_1d = paddle.zeros([BT], dtype=paddle.float32)

        logits = logits.contiguous()
        target = target.contiguous()
        loss_1d = loss_1d.contiguous()

        n_non_ignore = BT
        ignore_index = -100

        liger_cross_entropy_kernel[(BT,)](
            X_ptr=logits,
            X_stride=logits.stride(-2),
            Y_ptr=target,
            Y_stride=target.stride(-1),
            loss_ptr=loss_1d,
            loss_stride=loss_1d.stride(-1),
            n_cols=V,
            n_non_ignore=n_non_ignore,
            ignore_index=ignore_index,
            reduction="mean",
            HAS_GRADIENTS=True,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertTrue(paddle.all(paddle.isfinite(loss_1d)))


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is not available")
class TestElementMulKernelGPU(unittest.TestCase):
    """Tests for element_mul_kernel on GPU."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.seed(42)
        np.random.seed(42)
        paddle.enable_compat(scope={"triton"}, silent=True)

    def tearDown(self):
        """Clean up after tests."""
        paddle.disable_compat()

    def test_element_mul_kernel_basic(self):
        """Test basic element_mul_kernel execution."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils import (
            element_mul_kernel,
        )

        BT, H = 4, 8
        BLOCK_SIZE = min(32768, triton.next_power_of_2(H))

        X = paddle.randn([BT, H], dtype=paddle.float32)
        grad_output = paddle.to_tensor(2.0)

        X = X.contiguous()
        grad_output = grad_output.contiguous()
        original_X = X.clone()

        element_mul_kernel[(BT,)](
            X,
            X.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        expected = original_X * 2.0
        self.assertTrue(paddle.allclose(X, expected, atol=1e-5))

    def test_element_mul_kernel_with_ones(self):
        """Test element_mul_kernel with grad_output=1.0."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils import (
            element_mul_kernel,
        )

        BT, H = 4, 8
        BLOCK_SIZE = min(32768, triton.next_power_of_2(H))

        X = paddle.randn([BT, H], dtype=paddle.float32)
        grad_output = paddle.to_tensor(1.0)

        X = X.contiguous()
        grad_output = grad_output.contiguous()
        original_X = X.clone()

        element_mul_kernel[(BT,)](
            X,
            X.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertTrue(paddle.allclose(X, original_X, atol=1e-5))

    def test_element_mul_kernel_with_zero(self):
        """Test element_mul_kernel with grad_output=0.0."""
        import triton

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.utils import (
            element_mul_kernel,
        )

        BT, H = 4, 8
        BLOCK_SIZE = min(32768, triton.next_power_of_2(H))

        X = paddle.randn([BT, H], dtype=paddle.float32)
        grad_output = paddle.to_tensor(0.0)

        X = X.contiguous()
        grad_output = grad_output.contiguous()

        element_mul_kernel[(BT,)](
            X,
            X.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

        self.assertTrue(paddle.allclose(X, paddle.zeros_like(X), atol=1e-5))


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is not available")
class TestFusedLinearCrossEntropyForwardEdgeCases(unittest.TestCase):
    """Tests for edge cases in fused_linear_cross_entropy_forward."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.seed(42)
        np.random.seed(42)
        paddle.enable_compat(scope={"triton"}, silent=True)

    def tearDown(self):
        """Clean up after tests."""
        paddle.disable_compat()

    def test_forward_input_requires_grad_weight_no_grad(self):
        """Test forward when input requires grad but weight doesn't."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = True
        target = paddle.randint(0, V, [BT])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=None,
            ignore_index=-100,
            reduction="mean",
            num_chunks=1,
            ec_align=False,
        )

        self.assertIsNotNone(grad_input)
        self.assertIsNone(grad_weight)

    def test_forward_with_bias(self):
        """Test forward with bias."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        bias = paddle.randn([V], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=-100,
            reduction="mean",
            num_chunks=1,
            ec_align=False,
        )

        self.assertIsNotNone(grad_bias)

    def test_forward_ec_align_mode(self):
        """Test forward with ec_align mode."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=None,
            ignore_index=-100,
            reduction="mean",
            num_chunks=1,
            ec_align=True,
        )

        self.assertIsNotNone(grad_weight)
        self.assertEqual(grad_weight.shape, [H, V])

    def test_forward_sum_reduction(self):
        """Test forward with sum reduction."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=None,
            ignore_index=-100,
            reduction="sum",
            num_chunks=1,
            ec_align=False,
        )

        self.assertIsNotNone(loss)

    def test_forward_with_ignore_index(self):
        """Test forward with ignore_index."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.to_tensor([0, -100, 5, 3])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=None,
            ignore_index=-100,
            reduction="mean",
            num_chunks=1,
            ec_align=False,
        )

        self.assertIsNotNone(loss)

    def test_forward_multiple_chunks(self):
        """Test forward with multiple chunks."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        BT, H, V = 16, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=None,
            ignore_index=-100,
            reduction="mean",
            num_chunks=4,
            ec_align=False,
        )

        self.assertIsNotNone(loss)
        self.assertIsNotNone(grad_input)


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is not available")
class TestFusedLinearCrossEntropyBackwardEdgeCases(unittest.TestCase):
    """Tests for edge cases in fused_linear_cross_entropy_backward."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.seed(42)
        np.random.seed(42)
        paddle.enable_compat(scope={"triton"}, silent=True)

    def tearDown(self):
        """Clean up after tests."""
        paddle.disable_compat()

    def test_backward_with_grad_output_not_one(self):
        """Test backward when grad_output is not 1.0."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        BT, H, V = 4, 8, 16
        grad_output = paddle.to_tensor(0.5)
        grad_input = paddle.randn([BT, H], dtype=paddle.float32)
        grad_weight = paddle.randn([V, H], dtype=paddle.float32)
        grad_bias = paddle.randn([V], dtype=paddle.float32)

        original_grad_input = grad_input.clone()

        result_gi, result_gw, result_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )

        self.assertTrue(paddle.allclose(result_gi, original_grad_input * 0.5, atol=1e-5))

    def test_backward_with_grad_output_vector(self):
        """Test backward when grad_output is a vector."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        BT, H, V = 4, 8, 16
        grad_output = paddle.to_tensor([0.5] * BT)
        grad_input = paddle.randn([BT, H], dtype=paddle.float32)
        grad_weight = paddle.randn([V, H], dtype=paddle.float32)
        grad_bias = paddle.randn([V], dtype=paddle.float32)

        original_grad_input = grad_input.clone()

        result_gi, result_gw, result_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )

        self.assertIsNotNone(result_gi)

    def test_backward_with_none_grad_weight(self):
        """Test backward when grad_weight is None."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        BT, H = 4, 8
        grad_output = paddle.to_tensor(0.5)
        grad_input = paddle.randn([BT, H], dtype=paddle.float32)
        grad_weight = None
        grad_bias = None

        result_gi, result_gw, result_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )

        self.assertIsNone(result_gw)
        self.assertIsNone(result_gb)


@unittest.skipIf(not paddle.is_compiled_with_cuda(), "CUDA is not available")
class TestLigerFusedLinearCrossEntropyFunctionGPU(unittest.TestCase):
    """Tests for LigerFusedLinearCrossEntropyFunction on GPU."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.seed(42)
        np.random.seed(42)
        paddle.enable_compat(scope={"triton"}, silent=True)

    def tearDown(self):
        """Clean up after tests."""
        paddle.disable_compat()

    def test_apply_basic(self):
        """Test basic apply of LigerFusedLinearCrossEntropyFunction."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "mean",
            1,
            False,
        )

        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(_input.grad)

    def test_apply_with_main_grad(self):
        """Test apply with main_grad on weight."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        weight.main_grad = paddle.zeros([V, H], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "mean",
            1,
            False,
        )

        loss.backward()
        self.assertTrue(weight.main_grad.abs().sum().item() > 0)

    def test_apply_with_main_grad_ec_align(self):
        """Test apply with main_grad and ec_align mode."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        weight.main_grad = paddle.zeros([V, H], dtype=paddle.float32)
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "mean",
            1,
            True,
        )

        loss.backward()
        self.assertTrue(weight.main_grad.abs().sum().item() > 0)

    def test_apply_with_bias(self):
        """Test apply with bias."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        bias = paddle.randn([V], dtype=paddle.float32)
        bias.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            bias,
            -100,
            "mean",
            1,
            False,
        )

        loss.backward()
        self.assertIsNotNone(bias.grad)

    def test_apply_with_backward_hook(self):
        """Test apply with _apply_backward_hook on weight."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        weight.main_grad = paddle.zeros([V, H], dtype=paddle.float32)

        hook_called = [False]

        def hook_fn():
            hook_called[0] = True

        weight._apply_backward_hook = hook_fn
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "mean",
            1,
            False,
        )

        loss.backward()
        self.assertTrue(hook_called[0])

    def test_apply_main_grad_none(self):
        """Test apply when main_grad is None."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        weight.main_grad = None
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "mean",
            1,
            False,
        )

        loss.backward()
        self.assertIsNotNone(weight.main_grad)

    def test_apply_sum_reduction(self):
        """Test apply with sum reduction."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "sum",
            1,
            False,
        )

        self.assertIsNotNone(loss)
        loss.backward()

    def test_apply_none_reduction(self):
        """Test apply with none reduction."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        BT, H, V = 4, 8, 16
        _input = paddle.randn([BT, H], dtype=paddle.float32)
        _input.stop_gradient = False
        weight = paddle.randn([V, H], dtype=paddle.float32)
        weight.stop_gradient = False
        target = paddle.randint(0, V, [BT])

        loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            weight,
            target,
            None,
            -100,
            "none",
            1,
            False,
        )

        self.assertEqual(loss.shape[0], BT)
        loss.sum().backward()


if __name__ == "__main__":
    unittest.main()
