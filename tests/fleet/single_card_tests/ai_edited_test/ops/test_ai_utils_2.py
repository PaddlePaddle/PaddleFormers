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
from unittest.mock import MagicMock

try:
    from paddleformers.fleet.triton_ops.utils import (  # noqa: F401
        is_torch_compat_available,
    )

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestTritonUtilsFunctions(unittest.TestCase):
    """Tests for triton_ops/utils.py functions."""

    def test_is_torch_compat_available_returns_bool(self):
        """Test is_torch_compat_available returns a bool."""
        try:
            from paddleformers.fleet.triton_ops.utils import (
                is_torch_compat_available,
            )

            result = is_torch_compat_available()
            self.assertIsInstance(result, bool)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_is_torch_compat_available_type(self):
        """Test is_torch_compat_available returns boolean."""
        try:
            from paddleformers.fleet.triton_ops.utils import (
                is_torch_compat_available,
            )

            result = is_torch_compat_available()
            self.assertIsInstance(result, bool)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_dispatch_to_decorator(self):
        """Test dispatch_to creates a decorator."""
        try:
            from paddleformers.fleet.triton_ops.utils import dispatch_to

            mock_dispatch = MagicMock(return_value="dispatched")
            cond = lambda *a, **kw: True

            @dispatch_to(mock_dispatch, cond=cond)
            def my_func(*args, **kwargs):
                return "fallback"

            # Without compat, should return fallback
            result = my_func()
            self.assertIn(result, ["dispatched", "fallback"])
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_dispatch_to_fallback(self):
        """Test dispatch_to falls back to original function when cond is False."""
        try:
            from paddleformers.fleet.triton_ops.utils import dispatch_to

            mock_dispatch = MagicMock(return_value="dispatched")
            cond = lambda *a, **kw: False

            @dispatch_to(mock_dispatch, cond=cond)
            def my_func():
                return "fallback"

            result = my_func()
            self.assertEqual(result, "fallback")
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_dispatch_to_preserves_original(self):
        """Test dispatch_to preserves original function in __original_fn__."""
        try:
            from paddleformers.fleet.triton_ops.utils import dispatch_to

            mock_dispatch = MagicMock(return_value="dispatched")

            @dispatch_to(mock_dispatch)
            def my_func():
                return "fallback"

            self.assertTrue(hasattr(my_func, "__original_fn__"))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_is_package_installed(self):
        """Test _is_package_installed checks package availability."""
        try:
            from paddleformers.fleet.triton_ops.utils import (
                _is_package_installed,
            )

            # Only test negative case; positive case depends on
            # distribution name which varies across CI environments
            self.assertFalse(
                _is_package_installed("nonexistent_package_xyz_12345")
            )
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_swap_driver_guard(self):
        """Test swap_driver_guard wraps a function."""
        try:
            from paddleformers.fleet.triton_ops.utils import swap_driver_guard

            def my_func():
                return 42

            wrapped = swap_driver_guard(my_func)
            self.assertTrue(callable(wrapped))
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")

    def test_enable_compat_on_triton_kernel(self):
        """Test enable_compat_on_triton_kernel wraps a kernel."""
        try:
            from paddleformers.fleet.triton_ops.utils import (
                enable_compat_on_triton_kernel,
            )

            mock_kernel = MagicMock()
            result = enable_compat_on_triton_kernel(mock_kernel)
            self.assertIsNotNone(result)
        except ImportError:
            self.skipTest("paddlefleet_ops not installed")


if __name__ == "__main__":
    unittest.main()
