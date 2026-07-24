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


# Extra tests for paddlefleet_ops/_extensions/flashmask/block_mask_utils.py
# Focus on: find_blocks_topp, _extract_raw_ptrs, _prepare_stride_maxmin_ptrs

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False
from unittest.mock import MagicMock


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
    tl.load = lambda *a, **kw: 0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0
    tl.min = lambda *a, **kw: 0
    tl.sum = lambda *a, **kw: 0
    tl.where = lambda cond, a, b: a
    tl.full = lambda shape, val, dtype=None: val
    tl.zeros = lambda shape, dtype=None: 0
    tl.broadcast_to = lambda x, shape: x
    tl.reshape = lambda x, shape: x
    tl.int32 = "int32"
    tl.int64 = "int64"
    tl.int8 = "int8"
    tl.int1 = "int1"
    tl.float32 = "float32"
    tl.core = MagicMock()
    tl.core.CONSTEXPR_0 = 0
    tl.static_assert = lambda cond, msg=None: None
    tl.math = MagicMock()
    tl.math.exp2 = lambda x: x
    tl.cumsum = lambda x, axis=0: x
    tl.arange = lambda start, end: list(range(end))
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b

    # Mock libdevice
    libdevice = types.ModuleType("triton.language.extra")
    libdevice2 = types.ModuleType("triton.language.extra.cuda")
    libdevice2.exp = lambda x: x
    libdevice2.div_rn = lambda a, b: a
    sys.modules.setdefault("triton.language.extra", libdevice)
    sys.modules.setdefault("triton.language.extra.cuda", libdevice2)
    sys.modules.setdefault("triton.language.extra.cuda.libdevice", libdevice2)
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


_setup_triton_mock()


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestBlockMaskUtilsModule(unittest.TestCase):
    """Tests for block_mask_utils module structure."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        import paddlefleet_ops._extensions.flashmask.block_mask_utils as bmu

        self.assertTrue(hasattr(bmu, "find_blocks_topp"))

    def test_find_blocks_topp_callable(self):
        """Test that find_blocks_topp is callable."""
        from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
            find_blocks_topp,
        )

        self.assertTrue(callable(find_blocks_topp))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestExtractRawPtrs(unittest.TestCase):
    """Tests for _extract_raw_ptrs function."""

    def test_mode_1(self):
        """Test _extract_raw_ptrs with mode=1."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        # mode=1: only lt_start
        indices = paddle.randint(0, 10, [1, 1, 8, 1])
        mode, raw = _extract_raw_ptrs(indices, causal=True)
        self.assertEqual(mode, 1)
        self.assertIsNotNone(raw.lt_start)

    def test_mode_2_causal(self):
        """Test _extract_raw_ptrs with mode=2, causal=True."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        indices = paddle.randint(0, 10, [1, 1, 8, 2])
        mode, raw = _extract_raw_ptrs(indices, causal=True)
        self.assertEqual(mode, 2)
        self.assertIsNotNone(raw.lt_start)
        self.assertIsNotNone(raw.lt_end)

    def test_mode_2_non_causal(self):
        """Test _extract_raw_ptrs with mode=2, causal=False."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        indices = paddle.randint(0, 10, [1, 1, 8, 2])
        mode, raw = _extract_raw_ptrs(indices, causal=False)
        self.assertEqual(mode, 2)
        self.assertIsNotNone(raw.lt_start)
        self.assertIsNotNone(raw.ut_end)

    def test_mode_4(self):
        """Test _extract_raw_ptrs with mode=4."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        indices = paddle.randint(0, 10, [1, 1, 8, 4])
        mode, raw = _extract_raw_ptrs(indices, causal=True)
        self.assertEqual(mode, 4)
        self.assertIsNotNone(raw.lt_start)
        self.assertIsNotNone(raw.lt_end)
        self.assertIsNotNone(raw.ut_start)
        self.assertIsNotNone(raw.ut_end)

    def test_invalid_mode_raises(self):
        """Test _extract_raw_ptrs raises for invalid mode."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _extract_raw_ptrs,
        )

        indices = paddle.randint(0, 10, [1, 1, 8, 3])
        with self.assertRaises(ValueError):
            _extract_raw_ptrs(indices, causal=True)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRequireHelper(unittest.TestCase):
    """Tests for _require helper function."""

    def test_require_true_does_not_raise(self):
        """Test _require with True condition does not raise."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _require,
        )

        _require(True, "should not raise")

    def test_require_false_raises(self):
        """Test _require with False condition raises ValueError."""
        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            _require,
        )

        with self.assertRaises(ValueError) as ctx:
            _require(False, "error message")
        self.assertIn("error message", str(ctx.exception))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRrAttnEstimateValidation(unittest.TestCase):
    """Tests for rr_attn_estimate_triton_func input validation."""

    def test_ndim_validation(self):
        """Test that startend_row_indices must be 4D."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            rr_attn_estimate_triton_func,
        )

        q = paddle.randn([1, 8, 4, 16])
        k = paddle.randn([1, 8, 4, 16])
        indices = paddle.randint(0, 10, [1, 4, 8])

        with self.assertRaises(ValueError):
            rr_attn_estimate_triton_func(q, k, indices)

    def test_batch_size_mismatch(self):
        """Test that q and k must have same batch size."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            rr_attn_estimate_triton_func,
        )

        q = paddle.randn([1, 8, 4, 16])
        k = paddle.randn([2, 8, 4, 16])
        indices = paddle.randint(0, 10, [1, 1, 8, 2])

        with self.assertRaises(ValueError):
            rr_attn_estimate_triton_func(q, k, indices)

    def test_stride_must_be_positive(self):
        """Test that stride must be positive."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            rr_attn_estimate_triton_func,
        )

        q = paddle.randn([1, 8, 4, 16])
        k = paddle.randn([1, 8, 4, 16])
        indices = paddle.randint(0, 10, [1, 1, 8, 2])

        with self.assertRaises(ValueError):
            rr_attn_estimate_triton_func(q, k, indices, stride=0)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRawPtrsDataclass(unittest.TestCase):
    """Tests for RawPtrs dataclass."""

    def test_raw_ptrs_creation(self):
        """Test RawPtrs can be created with all fields."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            RawPtrs,
        )

        raw = RawPtrs(
            lt_start=paddle.zeros([1]),
            lt_end=paddle.zeros([1]),
            ut_start=paddle.zeros([1]),
            ut_end=paddle.zeros([1]),
        )
        self.assertIsNotNone(raw.lt_start)

    def test_raw_ptrs_frozen(self):
        """Test RawPtrs is frozen (immutable)."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            RawPtrs,
        )

        raw = RawPtrs(
            lt_start=paddle.zeros([1]),
            lt_end=paddle.zeros([1]),
            ut_start=paddle.zeros([1]),
            ut_end=paddle.zeros([1]),
        )
        with self.assertRaises(AttributeError):
            raw.lt_start = paddle.zeros([2])


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestStrideMaxMinPtrsDataclass(unittest.TestCase):
    """Tests for StrideMaxMinPtrs dataclass."""

    def test_creation(self):
        """Test StrideMaxMinPtrs can be created with all fields."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            StrideMaxMinPtrs,
        )

        smp = StrideMaxMinPtrs(
            lt_start_max=paddle.zeros([1]),
            lt_start_min=paddle.zeros([1]),
            lt_end_max=paddle.zeros([1]),
            lt_end_min=paddle.zeros([1]),
            ut_start_max=paddle.zeros([1]),
            ut_start_min=paddle.zeros([1]),
            ut_end_max=paddle.zeros([1]),
            ut_end_min=paddle.zeros([1]),
            n_strides=4,
        )
        self.assertEqual(smp.n_strides, 4)

    def test_frozen(self):
        """Test StrideMaxMinPtrs is frozen."""
        import paddle

        from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
            StrideMaxMinPtrs,
        )

        smp = StrideMaxMinPtrs(
            lt_start_max=paddle.zeros([1]),
            lt_start_min=paddle.zeros([1]),
            lt_end_max=paddle.zeros([1]),
            lt_end_min=paddle.zeros([1]),
            ut_start_max=paddle.zeros([1]),
            ut_start_min=paddle.zeros([1]),
            ut_end_max=paddle.zeros([1]),
            ut_end_min=paddle.zeros([1]),
            n_strides=4,
        )
        with self.assertRaises(AttributeError):
            smp.n_strides = 8


if __name__ == "__main__":
    unittest.main()
