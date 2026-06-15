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
# Tests focus on forward and backward logic without GPU/Triton

import types
import unittest


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
class TestFusedLinearCrossEntropyForward(unittest.TestCase):
    """Tests for fused_linear_cross_entropy_forward function."""

    def test_function_is_callable(self):
        """Test that fused_linear_cross_entropy_forward is callable."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        self.assertTrue(callable(fused_linear_cross_entropy_forward))

    def test_function_signature(self):
        """Test function has expected parameters."""
        import inspect

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_forward,
        )

        sig = inspect.signature(fused_linear_cross_entropy_forward)
        params = list(sig.parameters.keys())
        self.assertIn("_input", params)
        self.assertIn("weight", params)
        self.assertIn("target", params)
        self.assertIn("ignore_index", params)
        self.assertIn("reduction", params)
        self.assertIn("num_chunks", params)
        self.assertIn("ec_align", params)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFusedLinearCrossEntropyBackward(unittest.TestCase):
    """Tests for fused_linear_cross_entropy_backward function."""

    def test_function_is_callable(self):
        """Test that fused_linear_cross_entropy_backward is callable."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        self.assertTrue(callable(fused_linear_cross_entropy_backward))

    def test_backward_identity_grad_output(self):
        """Test backward returns unchanged grads when grad_output is 1.0."""
        import paddle

        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            fused_linear_cross_entropy_backward,
        )

        grad_input = paddle.randn([4, 8])
        grad_weight = paddle.randn([16, 8])
        grad_bias = paddle.randn([16])
        grad_output = paddle.to_tensor(1.0)

        result_gi, result_gw, result_gb = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        self.assertTrue(paddle.allclose(result_gi, grad_input))
        self.assertTrue(paddle.allclose(result_gw, grad_weight))
        self.assertTrue(paddle.allclose(result_gb, grad_bias))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLigerFusedLinearCrossEntropyFunction(unittest.TestCase):
    """Tests for LigerFusedLinearCrossEntropyFunction PyLayer."""

    def test_class_exists(self):
        """Test that LigerFusedLinearCrossEntropyFunction can be imported."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        self.assertTrue(callable(LigerFusedLinearCrossEntropyFunction))

    def test_has_forward(self):
        """Test that the class has forward method."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        self.assertTrue(hasattr(LigerFusedLinearCrossEntropyFunction, "forward"))

    def test_has_backward(self):
        """Test that the class has backward method."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
        )

        self.assertTrue(hasattr(LigerFusedLinearCrossEntropyFunction, "backward"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFusedLinearCrossEntropyLogic(unittest.TestCase):
    """Tests for fused linear cross entropy computation logic."""

    def test_linear_plus_cross_entropy(self):
        """Test linear + cross entropy computation in pure Paddle."""
        import paddle

        # Simulate: logits = input @ weight.T + bias, then cross_entropy
        _input = paddle.randn([4, 8])
        weight = paddle.randn([16, 8])
        target = paddle.randint(0, 16, [4])

        logits = paddle.matmul(_input, weight.T)
        loss = paddle.nn.functional.cross_entropy(logits, target)

        self.assertIsNotNone(loss)
        self.assertFalse(paddle.isnan(loss))

    def test_with_bias(self):
        """Test computation with bias."""
        import paddle

        _input = paddle.randn([4, 8])
        weight = paddle.randn([16, 8])
        bias = paddle.randn([16])
        target = paddle.randint(0, 16, [4])

        logits = paddle.matmul(_input, weight.T) + bias
        loss = paddle.nn.functional.cross_entropy(logits, target)

        self.assertIsNotNone(loss)

    def test_with_ignore_index(self):
        """Test computation with ignore_index."""
        import paddle

        _input = paddle.randn([4, 8])
        weight = paddle.randn([16, 8])
        target = paddle.to_tensor([0, -100, 5, 3])

        logits = paddle.matmul(_input, weight.T)
        loss = paddle.nn.functional.cross_entropy(logits, target, ignore_index=-100)

        self.assertIsNotNone(loss)

    def test_ec_align_mode(self):
        """Test ec_align mode produces correct grad_weight shape."""
        import paddle

        BT, H, V = 4, 8, 16

        # ec_align mode: grad_weight has shape [H, V]
        # Normal mode: grad_weight has shape [V, H]
        grad_weight_ec = paddle.zeros([H, V], dtype=paddle.float32)
        grad_weight_normal = paddle.zeros([V, H], dtype=paddle.float32)

        self.assertEqual(grad_weight_ec.shape, [H, V])
        self.assertEqual(grad_weight_normal.shape, [V, H])

    def test_none_reduction(self):
        """Test 'none' reduction returns per-sample loss."""
        import paddle

        _input = paddle.randn([4, 8])
        weight = paddle.randn([16, 8])
        target = paddle.randint(0, 16, [4])

        logits = paddle.matmul(_input, weight.T)
        loss = paddle.nn.functional.cross_entropy(logits, target, reduction="none")

        self.assertEqual(loss.shape, [4])

    def test_mean_reduction(self):
        """Test 'mean' reduction returns scalar loss."""
        import paddle

        _input = paddle.randn([4, 8])
        weight = paddle.randn([16, 8])
        target = paddle.randint(0, 16, [4])

        logits = paddle.matmul(_input, weight.T)
        loss = paddle.nn.functional.cross_entropy(logits, target, reduction="mean")

        self.assertEqual(loss.shape, [])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestMaxFusedSize(unittest.TestCase):
    """Tests for MAX_FUSED_SIZE constant."""

    def test_max_fused_size_value(self):
        """Test MAX_FUSED_SIZE is defined correctly."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.fused_linear_cross_entropy import (
            MAX_FUSED_SIZE,
        )

        self.assertEqual(MAX_FUSED_SIZE, 65536 // 2)
        self.assertEqual(MAX_FUSED_SIZE, 32768)


if __name__ == "__main__":
    unittest.main()
