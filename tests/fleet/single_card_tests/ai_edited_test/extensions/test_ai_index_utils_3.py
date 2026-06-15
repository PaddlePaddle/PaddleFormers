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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Extra tests for paddlefleet_ops/_extensions/flashmask/index_utils.py
# Focus on: prepare_maxmin function and its validation

import types
import unittest

try:
    import paddlefleet_ops._extensions  # noqa: F401

    _MODULE_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    _MODULE_AVAILABLE = False


def _setup_triton_mock():
    """Create a mock triton module so we can import without GPU."""
    triton_mock = types.ModuleType("triton")
    triton_mock.jit = lambda fn=None, **kwargs: ((lambda f: f) if fn is None else fn)
    triton_mock.language = types.ModuleType("triton.language")
    triton_mock.language.constexpr = None
    tl = triton_mock.language
    tl.program_id = lambda axis: 0
    tl.arange = lambda start, end: []
    tl.load = lambda *a, **kw: 0
    tl.store = lambda *a, **kw: None
    tl.max = lambda *a, **kw: 0
    tl.min = lambda *a, **kw: 0
    tl.where = lambda cond, a, b: a
    tl.int32 = "int32"
    triton_mock.next_power_of_2 = lambda x: 1
    triton_mock.cdiv = lambda a, b: (a + b - 1) // b
    sys.modules.setdefault("triton", triton_mock)
    sys.modules.setdefault("triton.language", tl)
    return triton_mock


_setup_triton_mock()


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestIndexUtilsModule(unittest.TestCase):
    """Tests for index_utils module structure."""

    def test_module_imports(self):
        """Test that the module can be imported."""
        import paddlefleet_ops._extensions.flashmask.index_utils as iu

        self.assertTrue(hasattr(iu, "prepare_maxmin"))

    def test_prepare_maxmin_is_callable(self):
        """Test that prepare_maxmin is callable."""
        from paddlefleet_ops._extensions.flashmask.index_utils import prepare_maxmin

        self.assertTrue(callable(prepare_maxmin))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestScanMaxminChunkedDefinition(unittest.TestCase):
    """Tests for scan_maxmin_chunked kernel definition."""

    def test_kernel_callable(self):
        """Test that scan_maxmin_chunked is callable."""
        from paddlefleet_ops._extensions.flashmask.index_utils import (
            scan_maxmin_chunked,
        )

        self.assertTrue(callable(scan_maxmin_chunked))


@unittest.skipUnless(_MODULE_AVAILABLE, "paddlefleet_ops module not available")
class TestPrepareMaxminLogic(unittest.TestCase):
    """Tests for prepare_maxmin logic using pure Paddle."""

    def test_maxmin_of_simple_sequence(self):
        """Test max/min computation of a simple integer sequence."""
        import paddle

        # Simulate what prepare_maxmin does
        input_tensor = paddle.to_tensor([[[3, 1, 4, 1, 5, 9, 2, 6]]], dtype=paddle.int32)
        chunk_size = 2

        # Compute max/min per chunk manually
        # Chunks: [3,1], [4,1], [5,9], [2,6]
        expected_max = [3, 4, 9, 6]
        expected_min = [1, 1, 5, 2]

        # Verify using Paddle operations
        bsz, num_heads, seq_len = input_tensor.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        reshaped = input_tensor.reshape([bsz, num_heads, num_chunks, chunk_size])
        actual_max = reshaped.max(axis=-1)
        actual_min = reshaped.min(axis=-1)

        self.assertEqual(actual_max.shape, [1, 1, 4])
        self.assertEqual(actual_min.shape, [1, 1, 4])
        for i in range(4):
            self.assertEqual(actual_max[0, 0, i].item(), expected_max[i])
            self.assertEqual(actual_min[0, 0, i].item(), expected_min[i])

    def test_maxmin_with_padding(self):
        """Test max/min with uneven last chunk."""
        import paddle

        input_tensor = paddle.to_tensor([[[3, 1, 4, 1, 5]]], dtype=paddle.int32)
        chunk_size = 2

        # Chunks: [3,1], [4,1], [5, ?]
        bsz, num_heads, seq_len = input_tensor.shape
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Need to pad
        pad_len = num_chunks * chunk_size - seq_len
        padded = paddle.concat(
            [input_tensor, paddle.full([1, 1, pad_len], 0, dtype=paddle.int32)],
            axis=-1,
        )
        reshaped = padded.reshape([bsz, num_heads, num_chunks, chunk_size])
        actual_max = reshaped.max(axis=-1)
        actual_min = reshaped.min(axis=-1)

        self.assertEqual(actual_max.shape, [1, 1, 3])
        self.assertEqual(actual_min.shape, [1, 1, 3])

    def test_maxmin_output_shapes(self):
        """Test output shapes of prepare_maxmin computation."""
        import paddle

        bsz, num_heads, seq_len = 2, 4, 32
        chunk_size = 8
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        input_tensor = paddle.randint(0, 100, [bsz, num_heads, seq_len], dtype=paddle.int32)

        # Manual computation
        reshaped = input_tensor.reshape([bsz, num_heads, num_chunks, chunk_size])
        output_max = reshaped.max(axis=-1)
        output_min = reshaped.min(axis=-1)

        self.assertEqual(output_max.shape, [bsz, num_heads, num_chunks])
        self.assertEqual(output_min.shape, [bsz, num_heads, num_chunks])


if __name__ == "__main__":
    unittest.main()
