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

import unittest

import paddle


class TestFlashattnAutoCast(unittest.TestCase):
    """Tests for flashattn_auto_cast function."""

    def test_flashattn_auto_cast_no_cast_needed(self):
        """Test when tensors are already target dtype."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            flashattn_auto_cast,
        )

        q = paddle.randn([2, 4, 8, 16], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8, 16], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8, 16], dtype=paddle.bfloat16)
        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16)
        self.assertEqual(oq.dtype, paddle.bfloat16)
        self.assertEqual(ok.dtype, paddle.bfloat16)
        self.assertEqual(ov.dtype, paddle.bfloat16)

    def test_flashattn_auto_cast_float32_to_bfloat16(self):
        """Test casting float32 to bfloat16."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            flashattn_auto_cast,
        )

        q = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16)
        self.assertEqual(oq.dtype, paddle.bfloat16)
        self.assertEqual(ok.dtype, paddle.bfloat16)
        self.assertEqual(ov.dtype, paddle.bfloat16)

    def test_flashattn_auto_cast_preserves_values(self):
        """Test that casting preserves approximate values."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            flashattn_auto_cast,
        )

        q = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.float32)
        self.assertTrue(paddle.allclose(oq, q))

    def test_flashattn_auto_cast_default_dtype(self):
        """Test default dtype is bfloat16."""
        import inspect

        from paddleformers.fleet.refined_recompute.flash_attn import (
            flashattn_auto_cast,
        )

        sig = inspect.signature(flashattn_auto_cast)
        self.assertEqual(sig.parameters["dtype"].default, paddle.bfloat16)

    def test_flashattn_auto_cast_mixed_dtypes(self):
        """Test with mixed input dtypes."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            flashattn_auto_cast,
        )

        q = paddle.randn([2, 4, 8, 16], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8, 16], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8, 16], dtype=paddle.bfloat16)
        oq, ok, ov = flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16)
        self.assertEqual(oq.dtype, paddle.bfloat16)
        self.assertEqual(ok.dtype, paddle.bfloat16)
        self.assertEqual(ov.dtype, paddle.bfloat16)


class TestGetFaVersion(unittest.TestCase):
    """Tests for _get_fa_version function."""

    def test_get_fa_version_returns_int(self):
        """Test _get_fa_version returns an integer."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            _get_fa_version,
        )

        version = _get_fa_version(64)
        self.assertIsInstance(version, int)

    def test_get_fa_version_valid_values(self):
        """Test _get_fa_version returns 2 or 3."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            _get_fa_version,
        )

        version = _get_fa_version(64)
        self.assertIn(version, [2, 3, 4])

    def test_get_fa_version_different_hdim(self):
        """Test _get_fa_version with different head dims."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            _get_fa_version,
        )

        for hdim in [32, 64, 128]:
            version = _get_fa_version(hdim)
            self.assertIn(version, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
