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
    RefinedRcomputeFlashMaskCpAttention,
)


class TestRefinedRcomputeFlashMaskCpAttentionInit(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskCpAttention initialization."""

    def test_init_creates_queue(self):
        """Test init creates a hold_tensors_queue."""
        attn = RefinedRcomputeFlashMaskCpAttention()
        import queue

        self.assertIsInstance(attn._hold_tensors_queue, queue.Queue)
        self.assertTrue(attn._hold_tensors_queue.empty())

    def test_callable(self):
        """Test instance is callable."""
        attn = RefinedRcomputeFlashMaskCpAttention()
        self.assertTrue(callable(attn))


class TestRefinedRcomputeFlashMaskCpAttentionForward(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskCpAttention.forward."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_forward"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.fleet.get_hybrid_communicate_group"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_first_fwd_no_grad(self, mock_tracer, mock_hcg, mock_cp_forward):
        """Test forward dispatches to _first_fwd when no grad."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_hcg.return_value.get_context_parallel_group.return_value = (
            MagicMock()
        )

        mock_cp_forward.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4], dtype=paddle.float32),
            paddle.to_tensor([0, 4, 8], dtype=paddle.int32),
            2,
        )

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn.forward(q, k, v, startend)
        self.assertFalse(attn._hold_tensors_queue.empty())

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_second_fwd_with_grad(self, mock_tracer):
        """Test forward dispatches to _second_fwd when grad is active."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = True
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        hold_tensors = {
            "result_attention": paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            "softmax_lse": paddle.randn([2, 4], dtype=paddle.float32),
            "startend_row_indices": paddle.to_tensor(
                [0, 4, 8], dtype=paddle.int32
            ),
            "group": MagicMock(),
            "causal": False,
            "fa_version": 2,
        }
        attn._hold_tensors_queue.put(hold_tensors)

        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.FlashMaskAttnCpFunctor.apply",
            return_value=paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
        ) as mock_functor:
            result = attn.forward(q, k, v, startend)
            mock_functor.assert_called_once()

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_second_fwd_empty_queue_raises(self, mock_tracer):
        """Test second_fwd raises when queue is empty."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = True
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with self.assertRaises(AssertionError):
            attn.forward(q, k, v, startend)


class TestRefinedRcomputeFlashMaskCpAttentionDropoutRaises(unittest.TestCase):
    """Tests for dropout/causal validation."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_forward"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.fleet.get_hybrid_communicate_group"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_dropout_not_supported(
        self, mock_tracer, mock_hcg, mock_cp_forward
    ):
        """Test dropout > 0 raises NotImplementedError."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with self.assertRaises(NotImplementedError):
            attn.forward(q, k, v, startend, dropout=0.1)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_forward"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.fleet.get_hybrid_communicate_group"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_causal_not_supported(self, mock_tracer, mock_hcg, mock_cp_forward):
        """Test causal=True raises NotImplementedError."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with self.assertRaises(NotImplementedError):
            attn.forward(q, k, v, startend, causal=True)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_forward"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.fleet.get_hybrid_communicate_group"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_fixed_seed_offset_not_supported(
        self, mock_tracer, mock_hcg, mock_cp_forward
    ):
        """Test fixed_seed_offset not None raises NotImplementedError."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        with self.assertRaises(NotImplementedError):
            attn.forward(q, k, v, startend, fixed_seed_offset=MagicMock())


class TestRefinedRcomputeFlashMaskCpAttentionSeqLenAssertion(unittest.TestCase):
    """Tests for query sequence length assertion."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.cp_flashmask_allgatherkv_balance_forward"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.fleet.get_hybrid_communicate_group"
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer"
    )
    def test_query_seq_len_must_be_even(
        self, mock_tracer, mock_hcg, mock_cp_forward
    ):
        """Test assertion that query seq len must be divisible by 2."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 3, 8], dtype=paddle.bfloat16)  # odd seq len
        k = paddle.randn([2, 3, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 3, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 3, 6], dtype=paddle.int32)

        with self.assertRaises(AssertionError):
            attn.forward(q, k, v, startend)


class TestRefinedRcomputeFlashMaskCpAttentionCall(unittest.TestCase):
    """Tests for __call__ delegation."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.RefinedRcomputeFlashMaskCpAttention.forward"
    )
    def test_call_delegates_to_forward(self, mock_forward):
        """Test __call__ delegates to forward."""
        mock_forward.return_value = paddle.randn([2, 4, 8])
        attn = RefinedRcomputeFlashMaskCpAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn(q, k, v, startend)
        mock_forward.assert_called_once()


class TestFlashMaskAttnCpFunctorForwardAndBackward(unittest.TestCase):
    """Tests for FlashMaskAttnCpFunctor."""

    def test_forward_returns_result_attention(self):
        """Test forward returns result_attention from hold_tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": paddle.randn([2, 4], dtype=paddle.float32),
            "startend_row_indices": paddle.to_tensor(
                [0, 4, 8], dtype=paddle.int32
            ),
            "group": MagicMock(),
            "causal": False,
            "fa_version": 2,
        }

        result = FlashMaskAttnCpFunctor.apply(q, k, v, hold_tensors)
        self.assertTrue(result is result_attn)


if __name__ == "__main__":
    unittest.main()
