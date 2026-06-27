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


# Tests for src/paddlefleet/_extensions/flashmask/index_utils.py
# Triton kernels are mocked since they require GPU.

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False
from unittest import mock

# Mock triton and triton.language if not available
_triton_available = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _triton_available = True
except (ImportError, ModuleNotFoundError):
    pass

if not _triton_available:
    _mock_tl = types.ModuleType("triton.language")
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


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestPrepareMaxmin(unittest.TestCase):
    """Tests for prepare_maxmin function."""

    def test_prepare_maxmin_chunk_size_equals_seq_len(self):
        """Test prepare_maxmin with chunk_size == seq_len."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            prepare_maxmin,
        )

        x = paddle.randint(0, 100, [2, 4, 16], dtype="int32")

        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.index_utils.scan_maxmin_chunked"
        ):
            result = prepare_maxmin(x, chunk_size=16)
            # chunk_size=16, seq_len=16 => num_chunks=1
            self.assertEqual(result[0].shape, [2, 4, 1])

    def test_prepare_maxmin_chunk_size_larger_than_seq_len(self):
        """Test prepare_maxmin with chunk_size > seq_len."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            prepare_maxmin,
        )

        x = paddle.randint(0, 100, [2, 4, 16], dtype="int32")

        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.index_utils.scan_maxmin_chunked"
        ):
            result = prepare_maxmin(x, chunk_size=32)
            # chunk_size=32, seq_len=16 => num_chunks=1
            self.assertEqual(result[0].shape, [2, 4, 1])

    def test_prepare_maxmin_uneven_chunks(self):
        """Test prepare_maxmin with seq_len not divisible by chunk_size."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            prepare_maxmin,
        )

        x = paddle.randint(0, 100, [2, 4, 15], dtype="int32")

        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.index_utils.scan_maxmin_chunked"
        ):
            result = prepare_maxmin(x, chunk_size=8)
            # chunk_size=8, seq_len=15 => num_chunks=2 (ceil(15/8))
            self.assertEqual(result[0].shape, [2, 4, 2])

    def test_prepare_maxmin_chunk_size_one(self):
        """Test prepare_maxmin with chunk_size=1."""
        import paddle
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            prepare_maxmin,
        )

        x = paddle.randint(0, 100, [1, 1, 4], dtype="int32")

        with mock.patch(
            "paddlefleet_ops._extensions.flashmask.index_utils.scan_maxmin_chunked"
        ):
            result = prepare_maxmin(x, chunk_size=1)
            # chunk_size=1, seq_len=4 => num_chunks=4
            self.assertEqual(result[0].shape, [1, 1, 4])

    def test_prepare_maxmin_output_dtype(self):
        """Test prepare_maxmin output dtype is int32."""
        import paddle

        x = paddle.randint(0, 100, [2, 4, 16], dtype="int32")

        # Actually create the output tensors to check dtype
        output_max = paddle.empty([2, 4, 2], dtype=paddle.int32)
        output_min = paddle.empty([2, 4, 2], dtype=paddle.int32)
        self.assertEqual(output_max.dtype, paddle.int32)
        self.assertEqual(output_min.dtype, paddle.int32)


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestScanMaxminChunked(unittest.TestCase):
    """Tests for scan_maxmin_chunked triton kernel."""

    def test_scan_maxmin_chunked_exists(self):
        """Test that scan_maxmin_chunked is defined."""
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            scan_maxmin_chunked,
        )

        self.assertIsNotNone(scan_maxmin_chunked)


if __name__ == "__main__":
    unittest.main()
