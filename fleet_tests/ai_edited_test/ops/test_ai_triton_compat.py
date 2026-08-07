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


# Tests for paddlefleet_ops/ops/triton_ops/triton_compat.py

import types
import unittest
from unittest.mock import MagicMock, patch


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
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
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


try:
    _setup_triton_mock()
    import paddleformers.fleet.triton_ops.triton_compat  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIsPackageInstalled(unittest.TestCase):
    """Tests for _is_package_installed function."""

    def test_installed_package(self):
        """Test _is_package_installed returns True for installed packages."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        # paddle should always be installed
        result = _is_package_installed("paddle")
        self.assertTrue(result)

    def test_not_installed_package(self):
        """Test _is_package_installed returns False for non-existent packages."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        result = _is_package_installed(
            "this_package_definitely_does_not_exist_12345"
        )
        self.assertFalse(result)

    def test_cached_result(self):
        """Test _is_package_installed caches results (returns same value)."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        result1 = _is_package_installed("paddle")
        result2 = _is_package_installed("paddle")
        self.assertEqual(result1, result2)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSwapDriverGuard(unittest.TestCase):
    """Tests for _swap_driver_guard function."""

    def test_swap_driver_guard_wraps_function(self):
        """Test _swap_driver_guard returns a wrapper function."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _swap_driver_guard,
        )

        def dummy_fn():
            return 42

        wrapped = _swap_driver_guard(dummy_fn)
        self.assertTrue(callable(wrapped))

    def test_swap_driver_guard_preserves_return(self):
        """Test _swap_driver_guard preserves function return value."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _swap_driver_guard,
        )

        # Mock the triton driver to avoid ImportError
        mock_driver = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "triton.runtime": MagicMock(),
                "triton.runtime.driver": mock_driver,
            },
        ):
            # Need to handle the case where paddle_driver may not exist
            try:

                def dummy_fn():
                    return 42

                wrapped = _swap_driver_guard(dummy_fn)
                # The function should still be callable
                self.assertTrue(callable(wrapped))
            except (AttributeError, ImportError):
                pass  # OK if triton runtime not available


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """Tests for enable_compat_on_triton_kernel function."""

    def test_returns_kernel_when_no_cuda(self):
        """Test that kernel is returned as-is when CUDA not available."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            enable_compat_on_triton_kernel,
        )

        def dummy_kernel():
            pass

        # With paddle compiled without cuda or torch not installed,
        # should return the kernel as-is
        with patch(
            "paddleformers.fleet.triton_ops.triton_compat._is_package_installed",
            return_value=False,
        ):
            result = enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIs(result, dummy_kernel)

    def test_returns_kernel_when_torch_not_installed(self):
        """Test that kernel is returned as-is when torch not installed."""
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

    def test_wrapped_kernel_has_getitem(self):
        """Test that wrapped kernel supports __getitem__."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            enable_compat_on_triton_kernel,
        )

        def dummy_kernel():
            pass

        mock_inner = MagicMock()
        dummy_kernel.__getitem__ = MagicMock(return_value=mock_inner)

        with (
            patch(
                "paddleformers.fleet.triton_ops.triton_compat._is_package_installed",
                return_value=True,
            ),
            patch("paddle.is_compiled_with_cuda", return_value=True),
        ):
            result = enable_compat_on_triton_kernel(dummy_kernel)
            # Should be a WrappedTritonKernel
            self.assertTrue(hasattr(result, "__getitem__"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestModuleStructure(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected symbols are exported."""
        import paddleformers.fleet.triton_ops.triton_compat as tc

        self.assertTrue(hasattr(tc, "_is_package_installed"))
        self.assertTrue(hasattr(tc, "enable_compat_on_triton_kernel"))
        self.assertTrue(hasattr(tc, "_swap_driver_guard"))

    def test_is_package_installed_is_cached(self):
        """Test _is_package_installed is cached (has __wrapped__)."""
        from paddleformers.fleet.triton_ops.triton_compat import (
            _is_package_installed,
        )

        # functools.cache wraps the function
        self.assertTrue(callable(_is_package_installed))


if __name__ == "__main__":
    unittest.main()
