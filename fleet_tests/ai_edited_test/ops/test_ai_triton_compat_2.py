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


# Tests for paddlefleet_ops/ops/triton_ops/triton_compat.py (ops package version)
# This is the triton_compat in the ops package, different from the src/paddlefleet one

import types
import unittest
from unittest.mock import patch


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    if "triton" in sys.modules:
        return sys.modules["triton"]
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: (
        (lambda f: f) if fn is None else fn
    )
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0.0
    tl.store = lambda *a, **kw: None
    tl.int64 = "int64"
    sys.modules["triton"] = triton_mock
    sys.modules["triton.language"] = tl
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.triton_compat  # noqa: F401

    _TRITON_COMPAT_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _TRITON_COMPAT_AVAILABLE = False


@unittest.skipUnless(
    _TRITON_COMPAT_AVAILABLE, "paddlefleet_ops triton_compat not available"
)
class TestOpsTritonCompatModule(unittest.TestCase):
    """Tests for ops triton_compat module structure."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        import paddleformers.fleet.triton_ops.triton_compat as tc

        self.assertTrue(hasattr(tc, "_is_package_installed"))
        self.assertTrue(hasattr(tc, "enable_compat_on_triton_kernel"))

    def test_is_package_installed_cached(self):
        """Test _is_package_installed is cached."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        r1 = _is_package_installed("paddle")
        r2 = _is_package_installed("paddle")
        self.assertEqual(r1, r2)

    def test_installed_package(self):
        """Test _is_package_installed returns True for paddle."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        self.assertTrue(_is_package_installed("paddle"))

    def test_not_installed_package(self):
        """Test _is_package_installed returns False for fake package."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        self.assertFalse(_is_package_installed("nonexistent_package_xyz_12345"))


@unittest.skipUnless(
    _TRITON_COMPAT_AVAILABLE, "paddlefleet_ops triton_compat not available"
)
class TestOpsEnableCompatOnTritonKernel(unittest.TestCase):
    """Tests for enable_compat_on_triton_kernel in ops package."""

    def test_returns_kernel_when_torch_not_installed(self):
        """Test kernel returned as-is when torch not installed."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            enable_compat_on_triton_kernel,
        )

        def dummy_kernel():
            pass

        with patch(
            "paddleformers.fleet.triton_ops.triton_compat._is_package_installed",
            return_value=False,
        ):
            result = enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIs(result, dummy_kernel)

    def test_returns_kernel_when_no_cuda(self):
        """Test kernel returned as-is when no CUDA."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            enable_compat_on_triton_kernel,
        )

        def dummy_kernel():
            pass

        with (
            patch(
                "paddleformers.fleet.triton_ops.triton_compat._is_package_installed",
                return_value=True,
            ),
            patch("paddle.is_compiled_with_cuda", return_value=False),
        ):
            result = enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIs(result, dummy_kernel)


@unittest.skipUnless(
    _TRITON_COMPAT_AVAILABLE, "paddlefleet_ops triton_compat not available"
)
class TestOpsSwapDriverGuard(unittest.TestCase):
    """Tests for _swap_driver_guard in ops package."""

    def test_wraps_function(self):
        """Test _swap_driver_guard wraps a function."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _swap_driver_guard,
        )

        def dummy_fn():
            return 42

        wrapped = _swap_driver_guard(dummy_fn)
        self.assertTrue(callable(wrapped))


if __name__ == "__main__":
    unittest.main()
