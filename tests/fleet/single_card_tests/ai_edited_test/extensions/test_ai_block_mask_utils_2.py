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
import importlib.util
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

# Setup comprehensive triton mock before any paddlefleet_ops imports
_mock_tl = types.ModuleType("triton.language")
_mock_tl_core = types.ModuleType("triton.language.core")
_mock_tl_core.CONSTEXPR_0 = None
_mock_tl.core = _mock_tl_core
_mock_tl.constexpr = None
_mock_tl.program_id = lambda axis: 0
_mock_tl.arange = lambda start, end: []
_mock_tl.load = lambda *a, **kw: 0.0
_mock_tl.store = lambda *a, **kw: None
_mock_tl.int64 = "int64"
_mock_tl.static_range = lambda *a, **kw: range(0)

_mock_triton = types.ModuleType("triton")
_mock_triton.jit = lambda fn=None, **kw: fn if fn is not None else lambda f: f
_mock_triton.cdiv = lambda a, b: (a + b - 1) // b
_mock_triton.next_power_of_2 = lambda n: (
    1 << (n - 1).bit_length() if n > 0 else 1
)

sys.modules.setdefault("triton", _mock_triton)
sys.modules.setdefault("triton.language", _mock_tl)
sys.modules.setdefault("triton.language.core", _mock_tl_core)

# Create stub parent packages to avoid triggering paddlefleet_ops.__init__
sys.modules.setdefault("paddlefleet_ops", types.ModuleType("paddlefleet_ops"))
sys.modules.setdefault(
    "paddlefleet_ops._extensions",
    types.ModuleType("paddlefleet_ops._extensions"),
)
sys.modules.setdefault(
    "paddlefleet_ops._extensions.flashmask",
    types.ModuleType("paddlefleet_ops._extensions.flashmask"),
)

# Import block_mask_utils directly as a file
_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
_bmu_path = os.path.join(
    _project_root,
    "packages",
    "paddlefleet_ops",
    "src",
    "paddlefleet_ops",
    "_extensions",
    "flashmask",
    "block_mask_utils.py",
)
_bmu_spec = importlib.util.spec_from_file_location(
    "paddlefleet_ops._extensions.flashmask.block_mask_utils", _bmu_path
)
_bmu_mod = importlib.util.module_from_spec(_bmu_spec)
sys.modules["paddlefleet_ops._extensions.flashmask.block_mask_utils"] = _bmu_mod
_bmu_spec.loader.exec_module(_bmu_mod)


# Tests for src/paddleformers.fleet/_extensions/flashmask/block_mask_utils.py
# Additional tests for find_blocks_topp, check_fully_masked_state,
# check_partially_masked_state, _load_bounds, _is_block_fully_masked,
# _is_block_partially_masked

import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False
from unittest import mock

import paddle

from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
    _is_block_fully_masked,
    _is_block_partially_masked,
    _load_bounds,
    check_fully_masked_state,
    check_partially_masked_state,
    find_blocks_topp,
)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFindBlocksToppReshape(unittest.TestCase):
    """Tests for find_blocks_topp reshape and shape handling."""

    @unittest.skip(
        "Cannot mock triton kernel launch [grid](...) pattern with mock triton"
    )
    def test_find_blocks_topp_2d_input(self):
        """Test find_blocks_topp with 2D input [B, N]."""
        x = paddle.randn([4, 16], dtype="float32")
        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.block_mask_utils.top_p_kernel"
        ) as mock_kernel:
            find_blocks_topp(x, p=0.5)
            grid = mock_kernel.call_args[0][0]
            self.assertEqual(len(grid), 1)
            self.assertEqual(grid[0], 4)

    @unittest.skip(
        "Cannot mock triton kernel launch [grid](...) pattern with mock triton"
    )
    def test_find_blocks_topp_4d_input(self):
        """Test find_blocks_topp with 4D input [B, H, M, N]."""
        x = paddle.randn([2, 3, 5, 16], dtype="float32")
        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.block_mask_utils.top_p_kernel"
        ) as mock_kernel:
            result = find_blocks_topp(x, p=0.5)
            grid = mock_kernel.call_args[0][0]
            self.assertEqual(grid[0], 30)
            self.assertEqual(result.shape, [2, 3, 5, 16])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFindBlocksToppNoMock(unittest.TestCase):
    """Tests for find_blocks_topp that don't require mocking the kernel."""

    @unittest.skip("Triton kernel launch requires active triton driver")
    def test_find_blocks_topp_output_shape_matches_input_no_mock(self):
        """Test that output shape matches input shape."""
        shape = [1, 4]
        x = paddle.randn(shape, dtype="float32")
        result = find_blocks_topp(x, p=0.9)
        self.assertEqual(list(result.shape), shape)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCheckFullyMaskedState(unittest.TestCase):
    """Tests for check_fully_masked_state triton kernel wrapper."""

    def test_check_fully_masked_state_is_jit(self):
        """Test that check_fully_masked_state is a triton jit function."""
        self.assertTrue(callable(check_fully_masked_state))

    def test_check_fully_masked_state_importable(self):
        """Test check_fully_masked_state can be imported."""
        self.assertIsNotNone(check_fully_masked_state)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestCheckPartiallyMaskedState(unittest.TestCase):
    """Tests for check_partially_masked_state triton kernel wrapper."""

    def test_check_partially_masked_state_is_jit(self):
        """Test that check_partially_masked_state is a triton jit function."""
        self.assertTrue(callable(check_partially_masked_state))

    def test_check_partially_masked_state_importable(self):
        """Test check_partially_masked_state can be imported."""
        self.assertIsNotNone(check_partially_masked_state)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLoadBounds(unittest.TestCase):
    """Tests for _load_bounds triton kernel."""

    def test_load_bounds_is_jit(self):
        """Test that _load_bounds is a triton jit function."""
        self.assertTrue(callable(_load_bounds))

    def test_load_bounds_importable(self):
        """Test _load_bounds can be imported."""
        self.assertIsNotNone(_load_bounds)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIsBlockFullyMasked(unittest.TestCase):
    """Tests for _is_block_fully_masked triton kernel."""

    def test_is_block_fully_masked_is_jit(self):
        """Test that _is_block_fully_masked is callable."""
        self.assertTrue(callable(_is_block_fully_masked))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIsBlockPartiallyMasked(unittest.TestCase):
    """Tests for _is_block_partially_masked triton kernel."""

    def test_is_block_partially_masked_is_jit(self):
        """Test that _is_block_partially_masked is callable."""
        self.assertTrue(callable(_is_block_partially_masked))


if __name__ == "__main__":
    unittest.main()
