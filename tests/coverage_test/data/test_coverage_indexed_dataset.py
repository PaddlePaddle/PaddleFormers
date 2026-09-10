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
import struct
import tempfile
import unittest

import numpy as np

import paddleformers.data.indexed_dataset as _idx_mod
from paddleformers.data.indexed_dataset import (
    code,
    create_doc_idx,
    data_file_path,
    dtypes,
    get_available_dataset_impl,
    index_file_path,
    make_dataset,
    read_longs,
    read_shorts,
    write_longs,
    write_shorts,
)

# Access module-level function with dunder prefix via getattr to avoid class name mangling
_best_fitting_dtype = getattr(_idx_mod, "__best_fitting_dtype")


class TestBestFittingDtype(unittest.TestCase):
    """Tests for __best_fitting_dtype function."""

    def test_small_vocab_returns_uint16(self):
        """Test that small vocab_size returns uint16."""
        self.assertEqual(_best_fitting_dtype(vocab_size=100), np.uint16)

    def test_large_vocab_returns_int32(self):
        """Test that large vocab_size returns int32."""
        self.assertEqual(_best_fitting_dtype(vocab_size=100000), np.int32)

    def test_none_returns_int32(self):
        """Test that None vocab_size returns int32."""
        self.assertEqual(_best_fitting_dtype(vocab_size=None), np.int32)

    def test_boundary_returns_uint16(self):
        """Test boundary value 65499 returns uint16."""
        self.assertEqual(_best_fitting_dtype(vocab_size=65499), np.uint16)

    def test_boundary_returns_int32(self):
        """Test boundary value 65500 returns int32."""
        self.assertEqual(_best_fitting_dtype(vocab_size=65500), np.int32)


class TestGetAvailableDatasetImpl(unittest.TestCase):
    """Tests for get_available_dataset_impl function."""

    def test_returns_list(self):
        """Test that get_available_dataset_impl returns a list."""
        result = get_available_dataset_impl()
        self.assertIsInstance(result, list)

    def test_contains_lazy(self):
        """Test that 'lazy' is in available implementations."""
        self.assertIn("lazy", get_available_dataset_impl())

    def test_contains_mmap(self):
        """Test that 'mmap' is in available implementations."""
        self.assertIn("mmap", get_available_dataset_impl())


class TestDtypes(unittest.TestCase):
    """Tests for dtypes and code functions."""

    def test_dtypes_dict_has_expected_keys(self):
        """Test that dtypes dict has expected keys."""
        self.assertIn(1, dtypes)
        self.assertIn(7, dtypes)

    def test_code_uint8(self):
        """Test code function returns correct code for uint8."""
        self.assertEqual(code(np.uint8), 1)

    def test_code_float32(self):
        """Test code function returns correct code for float32."""
        self.assertEqual(code(np.float32), 7)

    def test_code_int64(self):
        """Test code function returns correct code for int64."""
        self.assertEqual(code(np.int64), 5)

    def test_code_unknown_raises(self):
        """Test code function raises ValueError for unknown dtype."""
        with self.assertRaises(ValueError):
            code(np.complex64)


class TestFilePaths(unittest.TestCase):
    """Tests for file path helper functions."""

    def test_index_file_path(self):
        """Test index_file_path appends .idx suffix."""
        self.assertEqual(index_file_path("/tmp/data"), "/tmp/data.idx")

    def test_data_file_path(self):
        """Test data_file_path appends .bin suffix."""
        self.assertEqual(data_file_path("/tmp/data"), "/tmp/data.bin")


class TestCreateDocIdx(unittest.TestCase):
    """Tests for create_doc_idx function."""

    def test_no_zeros(self):
        """Test create_doc_idx with no zero sizes."""
        sizes = [5, 3, 7]
        result = create_doc_idx(sizes)
        self.assertEqual(result, [0])

    def test_with_zeros(self):
        """Test create_doc_idx with zero sizes (document boundaries)."""
        sizes = [5, 0, 3, 0, 7]
        result = create_doc_idx(sizes)
        self.assertEqual(result, [0, 2, 4])

    def test_all_zeros(self):
        """Test create_doc_idx with all zero sizes."""
        sizes = [0, 0, 0]
        result = create_doc_idx(sizes)
        self.assertEqual(result, [0, 1, 2, 3])


class TestReadWriteLongs(unittest.TestCase):
    """Tests for read_longs and write_longs functions."""

    def test_write_and_read_longs(self):
        """Test writing and reading long integers."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
            with open(path, "wb") as f:
                write_longs(f, data)
            with open(path, "rb") as f:
                result = read_longs(f, 5)
            np.testing.assert_array_equal(result, data)
        finally:
            os.unlink(path)


class TestReadWriteShorts(unittest.TestCase):
    """Tests for read_shorts and write_shorts functions."""

    def test_write_and_read_shorts(self):
        """Test writing and reading short integers."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            data = np.array([10, 20, 30], dtype=np.int32)
            with open(path, "wb") as f:
                write_shorts(f, data)
            with open(path, "rb") as f:
                result = read_shorts(f, 3)
            np.testing.assert_array_equal(result, data)
        finally:
            os.unlink(path)


class TestMakeDataset(unittest.TestCase):
    """Tests for make_dataset function."""

    def test_nonexistent_path_returns_none(self):
        """Test make_dataset returns None for nonexistent path."""
        result = make_dataset("/nonexistent/path/12345", "lazy")
        self.assertIsNone(result)

    def test_invalid_impl_returns_none(self):
        """Test make_dataset returns None for invalid implementation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .idx and .bin files so IndexedDataset.exists passes
            idx_path = os.path.join(tmpdir, "test.idx")
            bin_path = os.path.join(tmpdir, "test.bin")
            with open(idx_path, "wb") as f:
                f.write(b"TNTIDX\x00\x00")
                f.write(struct.pack("<Q", 1))  # version
                f.write(struct.pack("<QQ", code(np.int32), 4))  # code, element_size
                f.write(struct.pack("<QQ", 0, 0))  # len, s
                f.write(struct.pack("<Q", 0))  # doc_count
                write_longs(f, np.array([0], dtype=np.int64))  # dim_offsets
                write_longs(f, np.array([0], dtype=np.int64))  # data_offsets
            with open(bin_path, "wb"):
                pass

            result = make_dataset(os.path.join(tmpdir, "test"), "invalid_impl")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
