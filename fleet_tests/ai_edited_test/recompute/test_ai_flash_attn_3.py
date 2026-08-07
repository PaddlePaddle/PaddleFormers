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


# Extra tests for paddlefleet/refined_recompute/flash_attn.py
# Focus on: _get_fa_version, flashattn_auto_cast,
# RefinedRcomputeFlashAttention, FlashAttnFunctor

import unittest

import paddle


class TestGetFaVersion(unittest.TestCase):
    """Tests for _get_fa_version function."""

    def test_returns_int(self):
        """Test that _get_fa_version returns an integer."""
        from paddleformers.fleet.refined_recompute.flash_attn import _get_fa_version

        result = _get_fa_version(64)
        self.assertIsInstance(result, int)

    def test_returns_2_or_3(self):
        """Test that _get_fa_version returns 2 or 3."""
        from paddleformers.fleet.refined_recompute.flash_attn import _get_fa_version

        result = _get_fa_version(64)
        self.assertIn(result, [2, 3, 4])


class TestFlashattnAutoCast(unittest.TestCase):
    """Tests for flashattn_auto_cast function."""

    def test_casts_to_bfloat16(self):
        """Test that tensors are cast to bfloat16."""
        from paddleformers.fleet.refined_recompute.flash_attn import flashattn_auto_cast

        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_no_cast_when_same_dtype(self):
        """Test no cast when tensor already has target dtype."""
        from paddleformers.fleet.refined_recompute.flash_attn import flashattn_auto_cast

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )
        # Should return the same tensors (no copy)
        self.assertTrue(q_out is q)
        self.assertTrue(k_out is k)
        self.assertTrue(v_out is v)

    def test_casts_float16_to_bfloat16(self):
        """Test casting float16 to bfloat16."""
        from paddleformers.fleet.refined_recompute.flash_attn import flashattn_auto_cast

        q = paddle.randn([2, 4, 8], dtype=paddle.float16)
        k = paddle.randn([2, 4, 8], dtype=paddle.float16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)

        q_out, k_out, v_out = flashattn_auto_cast(
            q, k, v, dtype=paddle.bfloat16
        )
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_preserves_shape(self):
        """Test that shape is preserved after casting."""
        from paddleformers.fleet.refined_recompute.flash_attn import flashattn_auto_cast

        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)

        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)
        self.assertEqual(v_out.shape, v.shape)


class TestRefinedRcomputeFlashAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashAttention class."""

    def test_init_creates_queue(self):
        """Test that __init__ creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        rr = RefinedRcomputeFlashAttention()
        self.assertIsNotNone(rr._hold_tensors_queue)

    def test_is_callable(self):
        """Test that instance is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        rr = RefinedRcomputeFlashAttention()
        self.assertTrue(callable(rr))


class TestRefinedRcomputeFlashMaskAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention class."""

    def test_init_creates_queue(self):
        """Test that __init__ creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        rr = RefinedRcomputeFlashMaskAttention()
        self.assertIsNotNone(rr._hold_tensors_queue)

    def test_is_callable(self):
        """Test that instance is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        rr = RefinedRcomputeFlashMaskAttention()
        self.assertTrue(callable(rr))


class TestRefinedRcomputeFlashMaskCpAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskCpAttention class."""

    def test_init_creates_queue(self):
        """Test that __init__ creates a queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        self.assertIsNotNone(rr._hold_tensors_queue)

    def test_is_callable(self):
        """Test that instance is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        self.assertTrue(callable(rr))

    def test_first_fwd_dropout_not_supported(self):
        """Test that _first_fwd raises NotImplementedError for dropout > 0."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        with self.assertRaises(NotImplementedError):
            rr._first_fwd(
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randint(0, 10, [1, 4]),
                dropout=0.1,
            )

    def test_first_fwd_causal_not_supported(self):
        """Test that _first_fwd raises NotImplementedError for causal=True."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        with self.assertRaises(NotImplementedError):
            rr._first_fwd(
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randint(0, 10, [1, 4]),
                causal=True,
            )

    def test_first_fwd_fixed_seed_offset_not_supported(self):
        """Test _first_fwd raises NotImplementedError for fixed_seed_offset."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        rr = RefinedRcomputeFlashMaskCpAttention()
        with self.assertRaises(NotImplementedError):
            rr._first_fwd(
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randn([1, 4, 8]),
                paddle.randint(0, 10, [1, 4]),
                fixed_seed_offset=42,
            )


class TestFlashAttnFunctor(unittest.TestCase):
    """Tests for FlashAttnFunctor PyLayer."""

    def test_class_exists(self):
        """Test that FlashAttnFunctor can be imported."""
        from paddleformers.fleet.refined_recompute.flash_attn import FlashAttnFunctor

        self.assertTrue(callable(FlashAttnFunctor))


class TestFlashMaskAttnFunctor(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor PyLayer."""

    def test_class_exists(self):
        """Test that FlashMaskAttnFunctor can be imported."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnFunctor,
        )

        self.assertTrue(callable(FlashMaskAttnFunctor))


class TestFlashMaskAttnCpFunctor(unittest.TestCase):
    """Tests for FlashMaskAttnCpFunctor PyLayer."""

    def test_class_exists(self):
        """Test that FlashMaskAttnCpFunctor can be imported."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            FlashMaskAttnCpFunctor,
        )

        self.assertTrue(callable(FlashMaskAttnCpFunctor))


if __name__ == "__main__":
    unittest.main()
