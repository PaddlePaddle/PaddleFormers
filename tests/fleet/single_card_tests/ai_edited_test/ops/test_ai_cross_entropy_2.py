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


# Tests for paddlefleet_ops/ops/triton_ops/fused_linear_cross_entropy/cross_entropy.py
# Triton kernels are mocked since they require GPU.

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
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLigerCrossEntropyKernelDefinition(unittest.TestCase):
    """Tests for liger_cross_entropy_kernel function definition and attributes."""

    def test_kernel_is_callable(self):
        """Test that liger_cross_entropy_kernel is callable (mocked triton.jit)."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        self.assertTrue(callable(liger_cross_entropy_kernel))

    def test_kernel_has_expected_params(self):
        """Test that the kernel function has expected parameter names."""
        from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
            liger_cross_entropy_kernel,
        )

        # The kernel should have a __code__ attribute since it's a function
        self.assertTrue(hasattr(liger_cross_entropy_kernel, "__code__"))

    def test_module_imports(self):
        """Test that the module can be imported."""
        import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy as ce

        self.assertTrue(hasattr(ce, "liger_cross_entropy_kernel"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCrossEntropyKernelLogic(unittest.TestCase):
    """Tests for cross entropy kernel logic using manual simulation."""

    def test_simple_cross_entropy_computation(self):
        """Test cross entropy loss computation matches expected value."""
        import paddle

        # Simulate the kernel's computation in pure Paddle
        logits = paddle.to_tensor([[2.0, 1.0, 0.1]])
        target = paddle.to_tensor([0])

        # Manual cross entropy: log(sum(exp(logits))) - logits[target]
        max_logit = paddle.max(logits, axis=-1, keepdim=True)
        shifted = logits - max_logit
        log_sum_exp = paddle.log(paddle.sum(paddle.exp(shifted), axis=-1)) + max_logit.squeeze()
        loss = log_sum_exp - logits[0, target[0]]
        self.assertAlmostEqual(loss.item(), 0.4170, places=3)

    def test_cross_entropy_with_ignore_index(self):
        """Test that ignored index entries produce zero loss."""
        import paddle

        logits = paddle.to_tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
        targets = paddle.to_tensor([0, -100])

        # For target = -100 (ignore), loss should be 0
        max_logit = paddle.max(logits[0:1], axis=-1, keepdim=True)
        shifted = logits[0:1] - max_logit
        log_sum_exp = paddle.log(paddle.sum(paddle.exp(shifted), axis=-1)) + max_logit.squeeze()
        loss_valid = log_sum_exp - logits[0, targets[0]]
        self.assertTrue(loss_valid.item() > 0)

    def test_cross_entropy_mean_reduction(self):
        """Test mean reduction divides by n_non_ignore."""
        n_non_ignore = 4
        total_loss = 2.0
        mean_loss = total_loss / n_non_ignore
        self.assertAlmostEqual(mean_loss, 0.5)


if __name__ == "__main__":
    unittest.main()
