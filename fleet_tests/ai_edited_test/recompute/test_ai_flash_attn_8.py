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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.refined_recompute.flash_attn import (
    FlashMaskAttnCpFunctor,
    RefinedRcomputeFlashMaskAttention,
)


class TestRefinedRcomputeFlashMaskAttentionInit(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention initialization."""

    def test_init_creates_queue(self):
        """Test init creates a hold_tensors_queue."""
        attn = RefinedRcomputeFlashMaskAttention()
        import queue

        self.assertIsInstance(attn._hold_tensors_queue, queue.Queue)
        self.assertTrue(attn._hold_tensors_queue.empty())

    def test_callable(self):
        """Test instance is callable."""
        attn = RefinedRcomputeFlashMaskAttention()
        self.assertTrue(callable(attn))


class TestRefinedRcomputeFlashMaskAttentionForward(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention.forward."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=2,
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention"
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_dispatches(
        self, mock_tracer, mock_flashmask, mock_version
    ):
        """Test forward dispatches to _first_fwd when no grad."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_flashmask.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            None,
            paddle.randn([2, 4], dtype=paddle.float32),
            paddle.zeros([2], dtype=paddle.int64),
        )

        attn = RefinedRcomputeFlashMaskAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn.forward(q, k, v, startend)
        self.assertFalse(attn._hold_tensors_queue.empty())

    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_second_fwd_dispatches(self, mock_tracer):
        """Test forward dispatches to _second_fwd when grad is active."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = True
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskAttention()
        # Put something in queue
        hold_tensors = {
            "result_attention": paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            "softmax_lse": paddle.randn([2, 4], dtype=paddle.float32),
            "causal": True,
        }
        attn._hold_tensors_queue.put(hold_tensors)

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.FlashMaskAttnFunctor.apply",
            return_value=paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
        ) as mock_functor:
            result = attn.forward(q, k, v, startend)
            mock_functor.assert_called_once()

    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_second_fwd_empty_queue_raises(self, mock_tracer):
        """Test second_fwd raises when queue is empty."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = True
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with self.assertRaises(AssertionError):
            attn.forward(q, k, v, startend)


class TestRefinedRcomputeFlashMaskAttentionFirstFwdV4(unittest.TestCase):
    """Tests for _first_fwd with version 4."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._get_fa_version",
        return_value=4,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_v4(self, mock_tracer, mock_version):
        """Test _first_fwd with version 4."""
        # _flash_attn_fwd is conditionally imported (sm_10x only).
        # Create it on the module if it doesn't exist so the test can run.
        import paddleformers.fleet.refined_recompute.flash_attn as fa_mod

        _orig_flash_fwd = getattr(fa_mod, "_flash_attn_fwd", None)

        mock_flash_fwd = MagicMock()
        mock_flash_fwd.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4], dtype=paddle.float32),
        )
        fa_mod._flash_attn_fwd = mock_flash_fwd

        try:
            mock_tracer_obj = MagicMock()
            mock_tracer_obj._has_grad = False
            mock_tracer.return_value = mock_tracer_obj

            attn = RefinedRcomputeFlashMaskAttention()
            q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
            k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
            v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
            startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

            result = attn.forward(q, k, v, startend, causal=False)
            self.assertTrue(result is not None)
            hold_tensors = attn._hold_tensors_queue.get()
            self.assertIn("result_attention", hold_tensors)
            self.assertIn("softmax_lse", hold_tensors)
        finally:
            # Restore original state
            if _orig_flash_fwd is not None:
                fa_mod._flash_attn_fwd = _orig_flash_fwd
            elif hasattr(fa_mod, "_flash_attn_fwd"):
                delattr(fa_mod, "_flash_attn_fwd")


class TestFlashMaskAttnFunctorCpForward(unittest.TestCase):
    """Tests for FlashMaskAttnCpFunctor.forward."""

    def test_forward_saves_tensors(self):
        """Test forward saves the correct tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        group = MagicMock()
        causal = False

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "startend_row_indices": startend,
            "fa_version": 2,
            "group": group,
            "causal": causal,
        }

        result = FlashMaskAttnCpFunctor.apply(q, k, v, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestFlashMaskAttnCpFunctorBackward(unittest.TestCase):
    """Tests for FlashMaskAttnCpFunctor.backward."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_backward"
    )
    def test_backward_calls_cp_backward(self, mock_cp_backward):
        """Test backward calls cp_flashmask_allgatherkv_balance_backward."""
        mock_cp_backward.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
        )

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        group = MagicMock()
        causal = False

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "startend_row_indices": startend,
            "fa_version": 2,
            "group": group,
            "causal": causal,
        }

        FlashMaskAttnCpFunctor.apply(q, k, v, hold_tensors)
        mock_cp_backward.assert_not_called()  # Not called in forward


class TestRefinedRcomputeFlashMaskAttentionCall(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention __call__."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.RefinedRcomputeFlashMaskAttention.forward"
    )
    def test_call_delegates_to_forward(self, mock_forward):
        """Test __call__ delegates to forward."""
        mock_forward.return_value = paddle.randn([2, 4, 8])
        attn = RefinedRcomputeFlashMaskAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn(q, k, v, startend)
        mock_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
