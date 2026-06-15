# Copyright (c) 2026 PaddleFaddle Authors. All Rights Reserved.
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

import importlib.util
import unittest

import paddle


def _load_fp8_utils():
    """Load fp8 utils module directly from source to avoid deep_gemm import."""
    spec = importlib.util.spec_from_file_location(
        "fp8_utils",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "src",
            "paddleformers.fleet",
            "fp8",
            "utils.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.is_fp8_tensor


def _make_fp8_tensor(shape, dtype):
    """Create an fp8 tensor by casting from float32 (randn doesn't support fp8)."""
    t = paddle.randn(shape, dtype=paddle.float32)
    return t.cast(dtype)


class TestIsFp8Tensor(unittest.TestCase):
    """Tests for is_fp8_tensor in fp8/utils.py."""

    def test_non_tuple_returns_false(self):
        """Test is_fp8_tensor returns False for non-tuple."""
        is_fp8_tensor = _load_fp8_utils()
        t = paddle.randn([2, 3])
        self.assertFalse(is_fp8_tensor(t))

    def test_non_tuple_string_returns_false(self):
        """Test is_fp8_tensor returns False for string."""
        is_fp8_tensor = _load_fp8_utils()
        self.assertFalse(is_fp8_tensor("not_a_tuple"))

    def test_wrong_length_tuple_raises(self):
        """Test is_fp8_tensor raises ValueError for tuple with wrong length."""
        is_fp8_tensor = _load_fp8_utils()
        t = paddle.randn([2, 3])
        # The function unpacks as (tensor, scale), so 3-element tuple raises
        with self.assertRaises(ValueError):
            is_fp8_tensor((t, t, t))

    def test_correct_fp8_tuple_returns_true(self):
        """Test is_fp8_tensor returns True for valid FP8 tuple."""
        is_fp8_tensor = _load_fp8_utils()
        fp8_tensor = _make_fp8_tensor([2, 3], paddle.float8_e4m3fn)
        scale = paddle.randn([1], dtype=paddle.float32)
        self.assertTrue(is_fp8_tensor((fp8_tensor, scale)))

    def test_wrong_tensor_dtype_returns_false(self):
        """Test is_fp8_tensor returns False when tensor dtype is not fp8."""
        is_fp8_tensor = _load_fp8_utils()
        tensor = paddle.randn([2, 3], dtype=paddle.float32)
        scale = paddle.randn([1], dtype=paddle.float32)
        self.assertFalse(is_fp8_tensor((tensor, scale)))

    def test_wrong_scale_dtype_returns_false(self):
        """Test is_fp8_tensor returns False when scale dtype is not float32."""
        is_fp8_tensor = _load_fp8_utils()
        fp8_tensor = _make_fp8_tensor([2, 3], paddle.float8_e4m3fn)
        scale = paddle.randn([1], dtype=paddle.float16)
        self.assertFalse(is_fp8_tensor((fp8_tensor, scale)))

    def test_e5m2_assertion(self):
        """Test is_fp8_tensor raises assertion for float8_e5m2."""
        is_fp8_tensor = _load_fp8_utils()
        e5m2_tensor = _make_fp8_tensor([2, 3], paddle.float8_e5m2)
        scale = paddle.randn([1], dtype=paddle.float32)
        with self.assertRaises(AssertionError):
            is_fp8_tensor((e5m2_tensor, scale))


if __name__ == "__main__":
    unittest.main()
