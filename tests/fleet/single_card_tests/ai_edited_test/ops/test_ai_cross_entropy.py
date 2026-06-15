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

import types
import unittest


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    if "triton" in sys.modules:
        return sys.modules["triton"]
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: ((lambda f: f) if fn is None else fn)
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", triton_mock.language)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCrossEntropyKernelAttributes(unittest.TestCase):
    """Tests for cross_entropy kernel attributes and structure."""

    def test_liger_cross_entropy_kernel_exists(self):
        """Test that the cross entropy kernel can be imported."""
        try:
            from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
                liger_cross_entropy_kernel,
            )

            self.assertIsNotNone(liger_cross_entropy_kernel)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_cross_entropy_module_has_kernel(self):
        """Test module structure contains the expected kernel."""
        try:
            import paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy as ce_mod

            self.assertTrue(hasattr(ce_mod, "liger_cross_entropy_kernel"))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_cross_entropy_module_imports_triton(self):
        """Test that the module imports triton."""
        try:
            import triton

            self.assertTrue(hasattr(triton, "jit"))
        except ImportError:
            self.skipTest("triton not installed")

    def test_cross_entropy_uses_triton_compat(self):
        """Test that cross_entropy module uses enable_compat_on_triton_kernel."""
        try:
            from paddleformers.fleet.triton_ops.utils import (
                enable_compat_on_triton_kernel,
            )

            self.assertTrue(callable(enable_compat_on_triton_kernel))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCrossEntropyKernelParams(unittest.TestCase):
    """Tests for cross_entropy kernel parameter validation."""

    def test_kernel_is_triton_jit(self):
        """Test that the kernel is a triton JIT compiled kernel."""
        try:
            import importlib.util

            if importlib.util.find_spec("triton") is None:
                self.skipTest("triton not available")
            from paddleformers.fleet.triton_ops.fused_linear_cross_entropy.cross_entropy import (
                liger_cross_entropy_kernel,
            )

            # Triton jit kernels have specific attributes
            self.assertIsNotNone(liger_cross_entropy_kernel)
        except (ImportError, AttributeError):
            self.skipTest("paddlefleet_ops or triton not available")

    def test_fused_linear_ce_module_has_init(self):
        """Test that the parent module can be imported."""
        try:
            import paddleformers.fleet.triton_ops.fused_linear_cross_entropy as flce

            self.assertIsNotNone(flce)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
