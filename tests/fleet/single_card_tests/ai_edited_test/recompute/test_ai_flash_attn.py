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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import queue
import unittest
from unittest.mock import MagicMock, patch

import paddle

_SKIP_FLASH_ATTN = False
try:
    from paddleformers.fleet.recompute.flash_attn import FlashAttnFunctor  # noqa: F401

    _SKIP_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


class TestFlashattnAutoCast(unittest.TestCase):
    """Tests for flashattn_auto_cast function."""


class TestFlashAttnFunctor(unittest.TestCase):
    """Tests for FlashAttnFunctor PyLayer."""

    def test_forward_invalid_version(self):
        """Test that FlashAttnFunctor.forward raises for invalid fa_version."""
        from paddleformers.fleet.refined_recompute.flash_attn import FlashAttnFunctor

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        hold_tensors = {"result_attention": q, "causal": True, "softmax_lse": k}

        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
            return_value=99,
        ):
            with self.assertRaises(ValueError) as ctx:
                FlashAttnFunctor.apply(q, k, v, hold_tensors)
            self.assertIn("Invalid flash attention version", str(ctx.exception))


class TestRefinedRcomputeFlashAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashAttention class."""

    def test_init_creates_queue(self):
        """Test that init creates a hold_tensors_queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        attn = RefinedRcomputeFlashAttention()
        self.assertIsInstance(attn._hold_tensors_queue, queue.Queue)
        self.assertTrue(attn._hold_tensors_queue.empty())

    def test_callable(self):
        """Test that the instance is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashAttention,
        )

        attn = RefinedRcomputeFlashAttention()
        self.assertTrue(callable(attn))


class TestFlashMaskAttnFunctor(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor PyLayer."""


class TestRefinedRcomputeFlashMaskAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention class."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=2,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_puts_in_queue(self, mock_tracer, mock_flashmask, mock_version):
        """Test _first_fwd stores hold_tensors in the queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        out = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        mock_flashmask.return_value = (
            out,
            None,
            paddle.randn([2, 4], dtype=paddle.float32),
            paddle.randn([2], dtype=paddle.float32),
        )

        attn = RefinedRcomputeFlashMaskAttention()
        result = attn.forward(q, k, v, startend, training=True)

        self.assertTrue(result is not None)
        self.assertFalse(attn._hold_tensors_queue.empty())
        hold_tensors = attn._hold_tensors_queue.get()
        self.assertIn("result_attention", hold_tensors)
        self.assertIn("softmax_lse", hold_tensors)
        self.assertIn("seed_offset", hold_tensors)
        self.assertIn("causal", hold_tensors)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=99,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_invalid_version_raises(self, mock_tracer, mock_flashmask, mock_version):
        """Test _first_fwd raises for invalid FA version."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        attn = RefinedRcomputeFlashMaskAttention()
        with self.assertRaises(ValueError) as ctx:
            attn.forward(q, k, v, startend)
        self.assertIn("Invalid flash attention version", str(ctx.exception))

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=3,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention_v2")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.inspect.signature")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_version_3_with_block_mask(self, mock_tracer, mock_sig, mock_flashmask_v2, mock_version):
        """Test _first_fwd v3 with block_mask parameter in signature."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        # flashmask_attention has block_mask in its signature
        mock_sig.return_value.parameters = {"block_mask": MagicMock()}

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        out = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        mock_flashmask_v2.return_value = (
            out,
            paddle.randn([2, 4], dtype=paddle.float32),
        )

        attn = RefinedRcomputeFlashMaskAttention()
        result = attn.forward(q, k, v, startend, causal=False)
        self.assertTrue(result is not None)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=3,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention_v2")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.inspect.signature")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_version_3_without_block_mask(self, mock_tracer, mock_sig, mock_flashmask_v2, mock_version):
        """Test _first_fwd v3 without block_mask parameter in signature."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskAttention,
        )

        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        # flashmask_attention does NOT have block_mask in its signature
        mock_sig.return_value.parameters = {"q": MagicMock()}

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        out = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        mock_flashmask_v2.return_value = (
            out,
            paddle.randn([2, 4], dtype=paddle.float32),
        )

        attn = RefinedRcomputeFlashMaskAttention()
        result = attn.forward(q, k, v, startend, causal=False)
        self.assertTrue(result is not None)


class TestFlashMaskAttnCpFunctor(unittest.TestCase):
    """Tests for FlashMaskAttnCpFunctor PyLayer."""


class TestRefinedRcomputeFlashMaskCpAttention(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskCpAttention class."""

    def test_init_creates_queue(self):
        """Test that init creates a hold_tensors_queue."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        attn = RefinedRcomputeFlashMaskCpAttention()
        self.assertIsInstance(attn._hold_tensors_queue, queue.Queue)
        self.assertTrue(attn._hold_tensors_queue.empty())

    def test_callable(self):
        """Test that the instance is callable."""
        from paddleformers.fleet.refined_recompute.flash_attn import (
            RefinedRcomputeFlashMaskCpAttention,
        )

        attn = RefinedRcomputeFlashMaskCpAttention()
        self.assertTrue(callable(attn))


class TestRefinedRecomputeModuleImport(unittest.TestCase):
    """Tests for refined_recompute module __init__.py."""
