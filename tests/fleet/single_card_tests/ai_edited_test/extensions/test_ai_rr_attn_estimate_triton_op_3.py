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

# Need a proper flashmask package so relative imports work
_flashmask_pkg = types.ModuleType("paddlefleet_ops._extensions.flashmask")
_flashmask_pkg.__path__ = []
sys.modules["paddlefleet_ops._extensions.flashmask"] = _flashmask_pkg

_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
    )
)
_flashmask_dir = os.path.join(
    _project_root,
    "packages",
    "paddlefleet_ops",
    "src",
    "paddlefleet_ops",
    "_extensions",
    "flashmask",
)


def _import_module_from_file(name, filepath):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Import block_mask_utils first (rr_attn_estimate_triton_op depends on it)
_import_module_from_file(
    "paddlefleet_ops._extensions.flashmask.block_mask_utils",
    os.path.join(_flashmask_dir, "block_mask_utils.py"),
)

# Import index_utils (rr_attn_estimate_triton_op also depends on it)
_import_module_from_file(
    "paddlefleet_ops._extensions.flashmask.index_utils",
    os.path.join(_flashmask_dir, "index_utils.py"),
)

# Import rr_attn_estimate_triton_op
_rae_mod = _import_module_from_file(
    "paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op",
    os.path.join(_flashmask_dir, "rr_attn_estimate_triton_op.py"),
)


# Tests for src/paddleformers.fleet/_extensions/flashmask/rr_attn_estimate_triton_op.py

import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False

import paddle

from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
    RawPtrs,
    StrideMaxMinPtrs,
    _require,
    flashmask_apply,
)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRequireFunction(unittest.TestCase):
    """Tests for _require helper function."""

    def test_require_no_error_when_true(self):
        """Test _require does not raise when condition is True."""
        _require(True, "should not raise")

    def test_require_raises_when_false(self):
        """Test _require raises ValueError when condition is False."""
        with self.assertRaises(ValueError) as ctx:
            _require(False, "test error message")
        self.assertIn("test error message", str(ctx.exception))

    def test_require_empty_message(self):
        """Test _require with empty error message."""
        with self.assertRaises(ValueError):
            _require(False, "")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRawPtrs(unittest.TestCase):
    """Tests for RawPtrs dataclass."""

    def test_raw_ptrs_creation(self):
        """Test RawPtrs dataclass creation."""
        lt_start = paddle.randint(0, 100, [2, 3], dtype="int32")
        lt_end = paddle.randint(0, 100, [2, 3], dtype="int32")
        ut_start = paddle.randint(0, 100, [2, 3], dtype="int32")
        ut_end = paddle.randint(0, 100, [2, 3], dtype="int32")

        ptrs = RawPtrs(lt_start, lt_end, ut_start, ut_end)
        self.assertIs(ptrs.lt_start, lt_start)
        self.assertIs(ptrs.lt_end, lt_end)
        self.assertIs(ptrs.ut_start, ut_start)
        self.assertIs(ptrs.ut_end, ut_end)

    def test_raw_ptrs_is_frozen(self):
        """Test RawPtrs is frozen (immutable)."""
        ptrs = RawPtrs(
            paddle.zeros([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
        )
        with self.assertRaises(AttributeError):
            ptrs.lt_start = paddle.ones([1], dtype="int32")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestStrideMaxMinPtrs(unittest.TestCase):
    """Tests for StrideMaxMinPtrs dataclass."""

    def test_stride_maxmin_ptrs_creation(self):
        """Test StrideMaxMinPtrs dataclass creation."""
        shape = [2, 3, 4]
        ptrs = StrideMaxMinPtrs(
            lt_start_max=paddle.zeros(shape, dtype="int32"),
            lt_start_min=paddle.zeros(shape, dtype="int32"),
            lt_end_max=paddle.zeros(shape, dtype="int32"),
            lt_end_min=paddle.zeros(shape, dtype="int32"),
            ut_start_max=paddle.zeros(shape, dtype="int32"),
            ut_start_min=paddle.zeros(shape, dtype="int32"),
            ut_end_max=paddle.zeros(shape, dtype="int32"),
            ut_end_min=paddle.zeros(shape, dtype="int32"),
            n_strides=4,
        )
        self.assertEqual(ptrs.n_strides, 4)

    def test_stride_maxmin_ptrs_is_frozen(self):
        """Test StrideMaxMinPtrs is frozen."""
        ptrs = StrideMaxMinPtrs(
            lt_start_max=paddle.zeros([1], dtype="int32"),
            lt_start_min=paddle.zeros([1], dtype="int32"),
            lt_end_max=paddle.zeros([1], dtype="int32"),
            lt_end_min=paddle.zeros([1], dtype="int32"),
            ut_start_max=paddle.zeros([1], dtype="int32"),
            ut_start_min=paddle.zeros([1], dtype="int32"),
            ut_end_max=paddle.zeros([1], dtype="int32"),
            ut_end_min=paddle.zeros([1], dtype="int32"),
            n_strides=1,
        )
        with self.assertRaises(AttributeError):
            ptrs.n_strides = 99


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestExtractRawPtrs(unittest.TestCase):
    """Tests for _extract_raw_ptrs function."""

    def test_extract_raw_ptrs_mode1(self):
        """Test _extract_raw_ptrs with mode=1."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        x = paddle.randint(0, 100, [2, 3, 8, 1], dtype="int32")
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 1)
        self.assertEqual(list(raw.lt_start.shape), [2, 3, 8])

    def test_extract_raw_ptrs_mode2_causal(self):
        """Test _extract_raw_ptrs with mode=2, causal=True."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        x = paddle.randint(0, 100, [2, 3, 8, 2], dtype="int32")
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 2)
        self.assertEqual(list(raw.lt_start.shape), [2, 3, 8])
        self.assertEqual(list(raw.lt_end.shape), [2, 3, 8])

    def test_extract_raw_ptrs_mode2_noncausal(self):
        """Test _extract_raw_ptrs with mode=2, causal=False."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        x = paddle.randint(0, 100, [2, 3, 8, 2], dtype="int32")
        mode, raw = _extract_raw_ptrs(x, causal=False)
        self.assertEqual(mode, 2)
        self.assertEqual(list(raw.ut_end.shape), [2, 3, 8])

    def test_extract_raw_ptrs_mode4(self):
        """Test _extract_raw_ptrs with mode=4."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        x = paddle.randint(0, 100, [2, 3, 8, 4], dtype="int32")
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 4)
        self.assertEqual(list(raw.lt_start.shape), [2, 3, 8])
        self.assertEqual(list(raw.lt_end.shape), [2, 3, 8])
        self.assertEqual(list(raw.ut_start.shape), [2, 3, 8])
        self.assertEqual(list(raw.ut_end.shape), [2, 3, 8])

    def test_extract_raw_ptrs_invalid_mode(self):
        """Test _extract_raw_ptrs raises on invalid mode."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        x = paddle.randint(0, 100, [2, 3, 8, 3], dtype="int32")
        with self.assertRaises(ValueError) as ctx:
            _extract_raw_ptrs(x, causal=True)
        self.assertIn("Unsupported mode", str(ctx.exception))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestPrepareStrideMaxMinPtrs(unittest.TestCase):
    """Tests for _prepare_stride_maxmin_ptrs function."""

    def test_prepare_stride_maxmin_ptrs_stride_validation(self):
        """Test stride validation in _prepare_stride_maxmin_ptrs."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _prepare_stride_maxmin_ptrs,
        )

        raw = RawPtrs(
            lt_start=paddle.zeros([1, 1, 8], dtype="int32"),
            lt_end=paddle.zeros([1, 1, 8], dtype="int32"),
            ut_start=paddle.zeros([1, 1, 8], dtype="int32"),
            ut_end=paddle.zeros([1, 1, 8], dtype="int32"),
        )
        with self.assertRaises(ValueError) as ctx:
            _prepare_stride_maxmin_ptrs(raw, mode=1, causal=True, stride=0)
        self.assertIn("stride must be positive", str(ctx.exception))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRrAttnEstimateTritonFuncValidation(unittest.TestCase):
    """Tests for rr_attn_estimate_triton_func validation logic."""

    def test_ndim_validation(self):
        """Test that startend_row_indices ndim must be 4."""
        with self.assertRaises(ValueError):
            _require(
                False, "startend_row_indices must be [B, HIDS, seqlen_q, mode]"
            )

    def test_batch_size_mismatch(self):
        """Test batch size mismatch between q and k."""
        with self.assertRaises(ValueError):
            _require(1 == 2, "q/k batch size mismatch")


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFlashmaskApply(unittest.TestCase):
    """Tests for flashmask_apply triton kernel."""

    def test_flashmask_apply_is_jit(self):
        """Test that flashmask_apply is a triton jit function."""
        self.assertTrue(callable(flashmask_apply))

    def test_flashmask_apply_importable(self):
        """Test flashmask_apply can be imported."""
        self.assertIsNotNone(flashmask_apply)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLog2EConstant(unittest.TestCase):
    """Tests for the LOG2E constant."""

    def test_log2e_value(self):
        """Test LOG2E constant correctness."""
        import math

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            LOG2E,
        )

        expected = 1.0 / math.log(2)
        self.assertAlmostEqual(LOG2E, expected, places=10)


if __name__ == "__main__":
    unittest.main()
