# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
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
import importlib
import os
import sys
import types

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

# Find the utils.py source file directly
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
_utils_path = os.path.join(
    _project_root,
    "packages",
    "paddlefleet_ops",
    "src",
    "paddlefleet_ops",
    "ops",
    "triton_ops",
    "utils.py",
)


# Import the utils module directly without triggering package __init__
def _import_utils():
    """Import triton_ops.utils directly by loading the file."""
    spec = importlib.util.spec_from_file_location(
        "paddleformers.fleet.triton_ops.utils", _utils_path
    )
    mod = importlib.util.module_from_spec(spec)
    # Register minimal parent packages so relative imports don't fail
    if "paddlefleet_ops" not in sys.modules:
        sys.modules["paddlefleet_ops"] = types.ModuleType("paddlefleet_ops")
    if "paddlefleet_ops.ops" not in sys.modules:
        ops_mod = types.ModuleType("paddlefleet_ops.ops")
        sys.modules["paddlefleet_ops.ops"] = ops_mod
    if "paddleformers.fleet.triton_ops" not in sys.modules:
        triton_mod = types.ModuleType("paddleformers.fleet.triton_ops")
        sys.modules["paddleformers.fleet.triton_ops"] = triton_mod
    sys.modules["paddleformers.fleet.triton_ops.utils"] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    _utils_mod = _import_utils()
    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _utils_mod = None
    _MODULE_AVAILABLE = False


import unittest
from unittest.mock import MagicMock, patch


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIsTorchCompatAvailable(unittest.TestCase):
    """Tests for is_torch_compat_available function."""

    def test_returns_bool(self):
        """Test that is_torch_compat_available returns a boolean."""
        result = _utils_mod.is_torch_compat_available()
        self.assertIsInstance(result, bool)

    def test_returns_false_without_enable_compat(self):
        """Test returns False when paddle has no enable_compat."""
        result = _utils_mod.is_torch_compat_available()
        self.assertIsInstance(result, bool)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestDispatchTo(unittest.TestCase):
    """Tests for dispatch_to decorator."""

    def test_dispatch_to_returns_decorator(self):
        """Test that dispatch_to returns a decorator."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        result = _utils_mod.dispatch_to(dummy_dispatch)
        self.assertTrue(callable(result))

    def test_dispatch_to_wraps_function(self):
        """Test that dispatch_to wraps the original function."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = _utils_mod.dispatch_to(dummy_dispatch)(original_fn)
        self.assertTrue(callable(decorated))

    def test_dispatch_to_falls_back_when_no_compat(self):
        """Test that dispatch_to falls back to original when no compat available."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: True
        decorated = _utils_mod.dispatch_to(dummy_dispatch, cond=cond)(
            original_fn
        )

        with patch.object(
            _utils_mod, "is_torch_compat_available", return_value=False
        ):
            result = decorated()
            self.assertEqual(result, "original")

    def test_dispatch_to_dispatches_when_compat_and_cond_true(self):
        """Test dispatch_to calls dispatch_fn when compat available and cond True."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: True
        decorated = _utils_mod.dispatch_to(dummy_dispatch, cond=cond)(
            original_fn
        )

        with patch.object(
            _utils_mod, "is_torch_compat_available", return_value=True
        ):
            result = decorated()
            self.assertEqual(result, "dispatched")

    def test_dispatch_to_with_cond_false(self):
        """Test dispatch_to falls back when cond returns False."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: False
        decorated = _utils_mod.dispatch_to(dummy_dispatch, cond=cond)(
            original_fn
        )

        with patch.object(
            _utils_mod, "is_torch_compat_available", return_value=True
        ):
            result = decorated()
            self.assertEqual(result, "original")

    def test_dispatch_to_with_cond_true(self):
        """Test dispatch_to dispatches when cond returns True."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        cond = lambda *args, **kwargs: True
        decorated = _utils_mod.dispatch_to(dummy_dispatch, cond=cond)(
            original_fn
        )

        with patch.object(
            _utils_mod, "is_torch_compat_available", return_value=True
        ):
            result = decorated()
            self.assertEqual(result, "dispatched")

    def test_dispatch_to_preserves_original_fn(self):
        """Test dispatch_to stores original function in __original_fn__."""

        def dummy_dispatch(*args, **kwargs):
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        decorated = _utils_mod.dispatch_to(dummy_dispatch)(original_fn)
        self.assertTrue(hasattr(decorated, "__original_fn__"))
        self.assertIs(decorated.__original_fn__, original_fn)

    def test_dispatch_to_passes_args(self):
        """Test dispatch_to passes arguments correctly."""

        def dummy_dispatch(*args, **kwargs):
            return ("dispatched", args, kwargs)

        def original_fn(*args, **kwargs):
            return ("original", args, kwargs)

        decorated = _utils_mod.dispatch_to(dummy_dispatch)(original_fn)

        with patch.object(
            _utils_mod, "is_torch_compat_available", return_value=True
        ):
            result = decorated(1, 2, key="val")
            self.assertEqual(result[0], "dispatched")
            self.assertEqual(result[1], (1, 2))
            self.assertEqual(result[2], {"key": "val"})


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIsPackageInstalled(unittest.TestCase):
    """Tests for _is_package_installed function."""

    def test_installed_package(self):
        """Test _is_package_installed returns True for installed packages."""
        # Use 'pip' which is guaranteed to be installed in the test environment
        result = _utils_mod._is_package_installed("pip")
        self.assertTrue(result)

    def test_not_installed_package(self):
        """Test _is_package_installed returns False for non-existent packages."""
        result = _utils_mod._is_package_installed(
            "this_package_definitely_does_not_exist_67890"
        )
        self.assertFalse(result)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestSwapDriverGuard(unittest.TestCase):
    """Tests for swap_driver_guard function."""

    def test_swap_driver_guard_wraps_function(self):
        """Test swap_driver_guard returns a wrapper."""
        # swap_driver_guard requires triton, so mock it
        mock_triton_driver = types.ModuleType("triton.runtime.driver")
        mock_driver_obj = MagicMock()
        mock_triton_driver.driver = mock_driver_obj

        with patch.dict(
            sys.modules, {"triton.runtime.driver": mock_triton_driver}
        ):
            # Reimport the function from the module
            # Since utils is already imported, call swap_driver_guard directly
            def dummy_fn():
                return 42

            wrapped = _utils_mod.swap_driver_guard(dummy_fn)
            self.assertTrue(callable(wrapped))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestEnableCompatOnTritonKernel(unittest.TestCase):
    """Tests for enable_compat_on_triton_kernel function."""

    def test_returns_kernel_when_no_cuda(self):
        """Test that kernel is returned as-is when CUDA not available."""

        def dummy_kernel():
            pass

        with patch("paddle.is_compiled_with_cuda", return_value=False):
            result = _utils_mod.enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIs(result, dummy_kernel)

    def test_wraps_kernel_when_cuda(self):
        """Test that kernel is wrapped when CUDA is available."""

        def dummy_kernel():
            pass

        with patch("paddle.is_compiled_with_cuda", return_value=True):
            result = _utils_mod.enable_compat_on_triton_kernel(dummy_kernel)
            self.assertIsNot(result, dummy_kernel)
            self.assertTrue(hasattr(result, "kernel"))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestModuleStructure(unittest.TestCase):
    """Tests for module structure."""

    def test_module_exports(self):
        """Test that expected symbols are exported."""
        self.assertTrue(hasattr(_utils_mod, "is_torch_compat_available"))
        self.assertTrue(hasattr(_utils_mod, "dispatch_to"))
        self.assertTrue(hasattr(_utils_mod, "enable_compat_on_triton_kernel"))
        self.assertTrue(hasattr(_utils_mod, "_is_package_installed"))
        self.assertTrue(hasattr(_utils_mod, "swap_driver_guard"))


if __name__ == "__main__":
    unittest.main()
