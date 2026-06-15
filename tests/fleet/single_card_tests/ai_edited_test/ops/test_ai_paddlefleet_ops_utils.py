# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from pathlib import Path
from unittest.mock import MagicMock, patch

from paddlefleet_ops.utils import (
    HardwareIncompatibleBlocker,
    ModuleContext,
    clean_module_namespace,
    get_cuda_version,
    get_nvshmem_host_lib_path,
    import_custom_ops,
    patch_module_namespace,
)


class TestImportCustomOps(unittest.TestCase):
    """Test import_custom_ops function."""

    def test_import_custom_ops_valid(self):
        """Test importing from a valid module."""
        global_ns = {}
        import_custom_ops(
            package="paddleformers.fleet",
            module_name=".ops",
            global_ns=global_ns,
        )
        self.assertIsInstance(global_ns, dict)

    def test_import_custom_ops_invalid(self):
        """Test importing from invalid module does not raise."""
        global_ns = {}
        import_custom_ops(
            package="nonexistent_pkg",
            module_name=".fake_module",
            global_ns=global_ns,
        )
        self.assertIsInstance(global_ns, dict)


class TestModuleContext(unittest.TestCase):
    """Test ModuleContext context manager."""

    def setUp(self):
        self.temp_path = "/tmp/test_module_context"

    def test_stash_and_restore(self):
        """Test that modules are stashed and restored."""
        # Add a dummy module to sys.modules
        test_module_name = "_test_module_for_context"
        sys.modules[test_module_name] = MagicMock()

        with ModuleContext([test_module_name], Path(self.temp_path)):
            # Module should be removed from sys.modules
            self.assertNotIn(test_module_name, sys.modules)

        # Module should be restored
        self.assertIn(test_module_name, sys.modules)
        # Clean up
        del sys.modules[test_module_name]

    def test_path_management(self):
        """Test that path is added and removed."""
        test_module_name = "_test_module_for_path"
        original_path = sys.path.copy()

        with ModuleContext([test_module_name], Path(self.temp_path)):
            self.assertIn(self.temp_path, sys.path)

        self.assertNotIn(self.temp_path, sys.path)
        # Restore original path
        sys.path[:] = original_path

    def test_stash_submodules(self):
        """Test that submodules are also stashed."""
        parent = "_test_parent_mod"
        child = "_test_parent_mod.child"
        sys.modules[parent] = MagicMock()
        sys.modules[child] = MagicMock()

        with ModuleContext([parent], Path(self.temp_path)):
            self.assertNotIn(parent, sys.modules)
            self.assertNotIn(child, sys.modules)

        self.assertIn(parent, sys.modules)
        self.assertIn(child, sys.modules)
        del sys.modules[parent]
        del sys.modules[child]

    def test_empty_module_names(self):
        """Test with empty module names list."""
        with ModuleContext([], Path(self.temp_path)):
            pass  # Should not raise


class TestPatchModuleNamespace(unittest.TestCase):
    """Test patch_module_namespace function."""

    def test_patch_single_module(self):
        """Test patching a single module."""
        test_mod = "_test_patch_mod"
        sys.modules[test_mod] = MagicMock()

        patch_module_namespace(test_mod, "paddlefleet_ops.")

        self.assertNotIn(test_mod, sys.modules)
        self.assertIn("paddlefleet_ops." + test_mod, sys.modules)

        # Clean up
        del sys.modules["paddlefleet_ops." + test_mod]

    def test_patch_with_submodules(self):
        """Test patching a module with submodules."""
        parent = "_test_patch_parent"
        child = "_test_patch_parent.sub"
        sys.modules[parent] = MagicMock()
        sys.modules[child] = MagicMock()

        patch_module_namespace(parent, "new_prefix.")

        self.assertNotIn(parent, sys.modules)
        self.assertNotIn(child, sys.modules)
        self.assertIn("new_prefix." + parent, sys.modules)
        self.assertIn("new_prefix." + child, sys.modules)

        # Clean up
        del sys.modules["new_prefix." + parent]
        del sys.modules["new_prefix." + child]


class TestCleanModuleNamespace(unittest.TestCase):
    """Test clean_module_namespace function."""

    def test_clean_existing_module(self):
        """Test cleaning an existing module."""
        test_mod = "_test_clean_mod"
        sys.modules[test_mod] = MagicMock()

        clean_module_namespace(test_mod)
        self.assertNotIn(test_mod, sys.modules)

    def test_clean_nonexistent_module(self):
        """Test cleaning a nonexistent module does not raise."""
        clean_module_namespace("_test_nonexistent_mod_12345")


class TestHardwareIncompatibleBlocker(unittest.TestCase):
    """Test HardwareIncompatibleBlocker meta path finder."""

    def test_blocked_module_raises(self):
        """Test that blocked module raises RuntimeError."""
        error_messages = {"paddlefleet_ops.blocked_lib": "not supported"}
        blocker = HardwareIncompatibleBlocker(error_messages)

        with self.assertRaises(RuntimeError) as ctx:
            blocker.find_spec("paddlefleet_ops.blocked_lib", None, None)
        self.assertIn("not supported", str(ctx.exception))

    def test_blocked_submodule_raises(self):
        """Test that blocked submodule raises RuntimeError."""
        error_messages = {"paddlefleet_ops.blocked_lib": "not supported"}
        blocker = HardwareIncompatibleBlocker(error_messages)

        with self.assertRaises(RuntimeError):
            blocker.find_spec("paddlefleet_ops.blocked_lib.submodule", None, None)

    def test_allowed_module_passes(self):
        """Test that non-blocked module returns None (no spec)."""
        error_messages = {"paddlefleet_ops.blocked_lib": "not supported"}
        blocker = HardwareIncompatibleBlocker(error_messages)

        result = blocker.find_spec("paddlefleet_ops.safe_lib", None, None)
        self.assertIsNone(result)


class TestGetNvshmemHostLibPath(unittest.TestCase):
    """Test get_nvshmem_host_lib_path function."""

    def test_not_found_raises(self):
        """Test FileNotFoundError when lib is not found."""
        with self.assertRaises(FileNotFoundError):
            get_nvshmem_host_lib_path("/nonexistent/path")


class TestGetCudaVersion(unittest.TestCase):
    """Test get_cuda_version function."""

    @patch("shutil.which", return_value=None)
    def test_nvcc_not_found(self, mock_which):
        """Test FileNotFoundError when nvcc is not found."""
        with self.assertRaises(FileNotFoundError):
            get_cuda_version()

    @patch("shutil.which", return_value="/usr/bin/nvcc")
    @patch(
        "subprocess.run",
        return_value=MagicMock(
            stdout="nvcc: NVIDIA (R) Cuda compiler driver\n" "Cuda compilation tools, release 12.4, V12.4.131"
        ),
    )
    def test_cuda_version_parsed(self, mock_run, mock_which):
        """Test correct parsing of CUDA version."""
        result = get_cuda_version()
        self.assertEqual(result, (12, 4))

    @patch("shutil.which", return_value="/usr/bin/nvcc")
    @patch(
        "subprocess.run",
        return_value=MagicMock(stdout="no version info here"),
    )
    def test_unparsable_version_raises(self, mock_run, mock_which):
        """Test ValueError when version string cannot be parsed."""
        with self.assertRaises(ValueError):
            get_cuda_version()


if __name__ == "__main__":
    unittest.main()
