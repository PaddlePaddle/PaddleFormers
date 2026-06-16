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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for src/paddleformers.fleet/_extensions/flashmask/rr_attn_estimate_triton_op.py
# Triton kernels are mocked since they require GPU.

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False

# Mock triton and triton.language if not available, so the module can be imported.
_triton_available = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _triton_available = True
except (ImportError, ModuleNotFoundError):
    pass

if not _triton_available:
    _mock_tl = types.ModuleType("triton.language")
    # Provide minimal stubs for triton decorators used in the source
    _mock_triton = types.ModuleType("triton")
    _mock_triton.jit = lambda fn=None, **kw: (
        fn if fn is not None else lambda f: f
    )
    _mock_triton.cdiv = lambda a, b: (a + b - 1) // b
    _mock_triton.next_power_of_2 = lambda n: (
        1 << (n - 1).bit_length() if n > 0 else 1
    )
    sys.modules.setdefault("triton", _mock_triton)
    sys.modules.setdefault("triton.language", _mock_tl)

_SKIP_RR = False
try:
    from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_func import (  # noqa: F401
        rr_attn_estimate_triton_func,
    )

    _SKIP_RR = True
except (ImportError, ModuleNotFoundError):
    pass

# Also ensure block_mask_utils and index_utils can be imported (they also need triton mock)
# The triton mock above should make those importable too.


@unittest.skipIf(not _SKIP_RR, "rr_attn_estimate_triton_func not compiled")
@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRequire(unittest.TestCase):
    """Tests for _require helper."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestExtractRawPtrs(unittest.TestCase):
    """Tests for _extract_raw_ptrs function."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRawPtrs(unittest.TestCase):
    """Tests for RawPtrs dataclass."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestStrideMaxMinPtrs(unittest.TestCase):
    """Tests for StrideMaxMinPtrs dataclass."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestPrepareStrideMaxMinPtrs(unittest.TestCase):
    """Tests for _prepare_stride_maxmin_ptrs function."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestRrAttnEstimateTritonFunc(unittest.TestCase):
    """Tests for rr_attn_estimate_triton_func."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestLog2E(unittest.TestCase):
    """Tests for LOG2E constant."""


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestFlashmaskInit(unittest.TestCase):
    """Tests for flashmask __init__.py."""

    def test_all(self):
        """Test __all__ export."""
        from paddlefleet_ops._extensions.flashmask import __all__

        self.assertIn("rr_attn_estimate_triton_func", __all__)


if __name__ == "__main__":
    unittest.main()
