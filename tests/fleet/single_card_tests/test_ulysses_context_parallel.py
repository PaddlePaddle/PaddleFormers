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

"""
Unit tests for Ulysses context parallel changes:
- _ulysses_generate_layout_params
- _ulysses_single_all_to_all
- UlyssesAlltoAll (forward/backward)
- flashmask_attention_ulysses
- flashmask_attention_cp mode routing (dualchunk_allgather / contiguous_a2a / invalid)
- mode.startswith("contiguous") changes in ContextParallelScatterOp/GatherOp/AllGatherOp
- TransformerConfig cp_balance_mode validation
- DotProductAttention contiguous_a2a logic
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import paddle

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."),
)

from paddleformers.fleet.context_parallel_utils import (
    _ulysses_generate_layout_params,
)

# =============================================================================
# Tests for _ulysses_generate_layout_params
# =============================================================================


class TestUlyssesGenerateLayoutParams(unittest.TestCase):
    """Test all 4 branches of _ulysses_generate_layout_params."""

    def test_batch_dim0_scatter_seq(self):
        """batch_dim_idx=0, scatter_idx < 2: scatter sequence, gather heads."""
        # Input shape: [bs=2, global_seq=8, num_local_head=2, head_dim=4]
        inp = paddle.randn([2, 8, 2, 4])
        seq_world_size = 2
        scatter_idx = 1  # < 2

        pre_perm, pre_shape, post_perm, post_shape = (
            _ulysses_generate_layout_params(
                scatter_idx,
                batch_dim_idx=0,
                seq_world_size=seq_world_size,
                input=inp,
            )
        )

        self.assertEqual(pre_shape, [2, 2, 4, 2, 4])
        self.assertEqual(pre_perm, (1, 0, 2, 3, 4))
        self.assertEqual(post_perm, (1, 2, 0, 3, 4))
        self.assertEqual(post_shape, [2, 4, 4, 4])

    def test_batch_dim0_scatter_heads(self):
        """batch_dim_idx=0, scatter_idx >= 2: scatter heads, gather sequence."""
        # Input shape: [bs=2, local_seq=4, num_total_head=4, head_dim=8]
        inp = paddle.randn([2, 4, 4, 8])
        seq_world_size = 2
        scatter_idx = 2

        pre_perm, pre_shape, post_perm, post_shape = (
            _ulysses_generate_layout_params(
                scatter_idx,
                batch_dim_idx=0,
                seq_world_size=seq_world_size,
                input=inp,
            )
        )

        self.assertEqual(pre_shape, [2, 4, 2, 2, 8])
        self.assertEqual(pre_perm, (2, 0, 1, 3, 4))
        self.assertEqual(post_perm, (1, 0, 2, 3, 4))
        self.assertEqual(post_shape, [2, 8, 2, 8])

    def test_batch_dim0_scatter_heads_not_divisible(self):
        """batch_dim_idx=0, scatter_idx >= 2: heads not divisible by world size."""
        inp = paddle.randn([2, 4, 3, 8])  # 3 heads, world_size=2
        with self.assertRaises(AssertionError):
            _ulysses_generate_layout_params(
                scatter_idx=2, batch_dim_idx=0, seq_world_size=2, input=inp
            )

    def test_batch_dim1_scatter_seq(self):
        """batch_dim_idx=1, scatter_idx < 2: layout [seq, batch, heads, head_dim]."""
        # Input shape: [global_seq=8, bs=2, num_local_head=2, head_dim=4]
        inp = paddle.randn([8, 2, 2, 4])
        seq_world_size = 2
        scatter_idx = 0

        pre_perm, pre_shape, post_perm, post_shape = (
            _ulysses_generate_layout_params(
                scatter_idx,
                batch_dim_idx=1,
                seq_world_size=seq_world_size,
                input=inp,
            )
        )

        self.assertEqual(pre_shape, [2, 4, 2, 2, 4])
        self.assertIsNone(pre_perm)
        self.assertEqual(post_perm, (1, 2, 0, 3, 4))
        self.assertEqual(post_shape, [4, 2, 4, 4])

    def test_batch_dim1_scatter_heads(self):
        """batch_dim_idx=1, scatter_idx >= 2: layout [seq, batch, heads, head_dim]."""
        # Input shape: [local_seq=4, bs=2, num_total_head=4, head_dim=8]
        inp = paddle.randn([4, 2, 4, 8])
        seq_world_size = 2
        scatter_idx = 3

        pre_perm, pre_shape, post_perm, post_shape = (
            _ulysses_generate_layout_params(
                scatter_idx,
                batch_dim_idx=1,
                seq_world_size=seq_world_size,
                input=inp,
            )
        )

        self.assertEqual(pre_shape, [4, 2, 2, 2, 8])
        self.assertEqual(pre_perm, (2, 0, 1, 3, 4))
        self.assertIsNone(post_perm)
        self.assertEqual(post_shape, [8, 2, 2, 8])

    def test_batch_dim1_scatter_heads_not_divisible(self):
        """batch_dim_idx=1, scatter_idx >= 2: heads not divisible."""
        inp = paddle.randn([4, 2, 5, 8])  # 5 heads, world_size=2
        with self.assertRaises(AssertionError):
            _ulysses_generate_layout_params(
                scatter_idx=2, batch_dim_idx=1, seq_world_size=2, input=inp
            )


# PLACEHOLDER_SECTION_2


# =============================================================================
# Tests for _ulysses_single_all_to_all
# =============================================================================


class TestUlyssesSingleAllToAll(unittest.TestCase):
    """Test _ulysses_single_all_to_all with mocked dist operations."""

    @patch("paddleformers.fleet.context_parallel_utils.dist.alltoall")
    @patch(
        "paddleformers.fleet.context_parallel_utils.dist.get_world_size", return_value=2
    )
    def test_basic_forward_batch_dim0_scatter_heads(self, mock_ws, mock_a2a):
        """scatter_idx=2, batch_dim_idx=0: scatter heads, gather sequence."""
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all,
        )

        inp = paddle.randn(
            [2, 4, 4, 8]
        )  # [bs, local_seq, num_total_head, head_dim]
        group = MagicMock()

        def fake_alltoall(out, inp_t, group=None):
            out.copy_(inp_t, False)

        mock_a2a.side_effect = fake_alltoall

        result = _ulysses_single_all_to_all(
            inp, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )
        # After scatter heads (4 heads / 2 ranks = 2), gather seq (4 * 2 = 8)
        self.assertEqual(result.shape, [2, 8, 2, 8])

    @patch("paddleformers.fleet.context_parallel_utils.dist.alltoall")
    @patch(
        "paddleformers.fleet.context_parallel_utils.dist.get_world_size", return_value=2
    )
    def test_basic_forward_batch_dim0_scatter_seq(self, mock_ws, mock_a2a):
        """scatter_idx=1, batch_dim_idx=0: scatter seq, gather heads."""
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all,
        )

        inp = paddle.randn(
            [2, 8, 2, 8]
        )  # [bs, full_seq, local_heads, head_dim]
        group = MagicMock()

        def fake_alltoall(out, inp_t, group=None):
            out.copy_(inp_t, False)

        mock_a2a.side_effect = fake_alltoall

        result = _ulysses_single_all_to_all(
            inp, scatter_idx=1, gather_idx=2, batch_dim_idx=0, group=group
        )
        # After scatter seq (8/2=4), gather heads (2*2=4)
        self.assertEqual(result.shape, [2, 4, 4, 8])

    @patch("paddleformers.fleet.context_parallel_utils.dist.alltoall")
    @patch(
        "paddleformers.fleet.context_parallel_utils.dist.get_world_size", return_value=2
    )
    def test_batch_dim1_scatter_seq_no_pre_permute(self, mock_ws, mock_a2a):
        """batch_dim_idx=1, scatter_idx=0: pre_perm is None."""
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all,
        )

        inp = paddle.randn([8, 2, 2, 4])  # [seq, bs, heads, head_dim]
        group = MagicMock()

        def fake_alltoall(out, inp_t, group=None):
            out.copy_(inp_t, False)

        mock_a2a.side_effect = fake_alltoall

        result = _ulysses_single_all_to_all(
            inp, scatter_idx=0, gather_idx=2, batch_dim_idx=1, group=group
        )
        self.assertEqual(result.shape, [4, 2, 4, 4])

    @patch("paddleformers.fleet.context_parallel_utils.dist.alltoall")
    @patch(
        "paddleformers.fleet.context_parallel_utils.dist.get_world_size", return_value=2
    )
    def test_batch_dim1_scatter_heads_no_post_permute(self, mock_ws, mock_a2a):
        """batch_dim_idx=1, scatter_idx=2: post_perm is None."""
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all,
        )

        inp = paddle.randn([4, 2, 4, 8])  # [local_seq, bs, heads, head_dim]
        group = MagicMock()

        def fake_alltoall(out, inp_t, group=None):
            out.copy_(inp_t, False)

        mock_a2a.side_effect = fake_alltoall

        result = _ulysses_single_all_to_all(
            inp, scatter_idx=2, gather_idx=0, batch_dim_idx=1, group=group
        )
        self.assertEqual(result.shape, [8, 2, 2, 8])


# PLACEHOLDER_SECTION_3


# =============================================================================
# Tests for UlyssesAlltoAll PyLayer
# =============================================================================


class TestUlyssesAlltoAll(unittest.TestCase):
    """Test UlyssesAlltoAll forward and backward."""

    @patch("paddleformers.fleet.context_parallel_utils.dist.alltoall")
    @patch(
        "paddleformers.fleet.context_parallel_utils.dist.get_world_size", return_value=2
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils._ulysses_fused_supported",
        return_value=False,
    )
    def test_forward(self, mock_supported, mock_ws, mock_a2a):
        """UlyssesAlltoAll.forward should call _ulysses_single_all_to_all."""
        from paddleformers.fleet.context_parallel_utils import UlyssesAlltoAll

        def fake_alltoall(out, inp_t, group=None):
            out.copy_(inp_t, False)

        mock_a2a.side_effect = fake_alltoall
        group = MagicMock()

        inp = paddle.randn([2, 4, 4, 8])
        result = UlyssesAlltoAll.apply(
            inp, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )
        self.assertEqual(result.shape, [2, 8, 2, 8])

    @patch("paddleformers.fleet.context_parallel_utils._ulysses_fused_supported")
    @patch(
        "paddleformers.fleet.context_parallel_utils._ulysses_single_all_to_all_fused"
    )
    def test_forward_uses_fused_path(self, mock_a2a, mock_supported):
        """Forward on the production path (batch_dim_idx=0, 4-D) uses the fused
        kernel with scatter_idx=2."""
        from paddleformers.fleet.context_parallel_utils import UlyssesAlltoAll

        mock_supported.return_value = True
        mock_a2a.return_value = paddle.randn([2, 8, 2, 8])
        group = MagicMock()

        inp = paddle.randn([2, 4, 4, 8])
        inp.stop_gradient = False
        UlyssesAlltoAll.apply(
            inp, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group
        )

        # Fused signature: _ulysses_single_all_to_all_fused(input, scatter_idx, group)
        call_args = mock_a2a.call_args_list[0]
        self.assertEqual(call_args[0][1], 2)

    @patch("paddleformers.fleet.context_parallel_utils._ulysses_fused_supported")
    @patch(
        "paddleformers.fleet.context_parallel_utils._ulysses_single_all_to_all_fused"
    )
    def test_backward_uses_fused_path_with_swapped_index(
        self, mock_a2a, mock_supported
    ):
        """Backward on the production path (batch_dim_idx=0, 4-D) calls the fused
        a2a with the swapped (gather) index."""
        from paddleformers.fleet.context_parallel_utils import UlyssesAlltoAll

        mock_supported.return_value = True

        # Call backward directly via the static method
        mock_ctx = MagicMock()
        mock_ctx.scatter_idx = 2
        mock_ctx.gather_idx = 1
        mock_ctx.batch_dim_idx = 0
        mock_ctx.group = MagicMock()

        grad_output = paddle.randn([2, 8, 2, 8])
        mock_a2a.return_value = paddle.randn([2, 4, 4, 8])

        UlyssesAlltoAll.backward(mock_ctx, grad_output)

        # The inverse a2a swaps scatter/gather; the fused path passes the
        # backward scatter index (= ctx.gather_idx) as the fused scatter_idx.
        mock_a2a.assert_called_once()
        call_args = mock_a2a.call_args
        # positional args: (grad_output, gather_idx, group)
        self.assertIs(call_args[0][0], grad_output)
        self.assertEqual(
            call_args[0][1], 1
        )  # swapped scatter index = gather_idx
        self.assertIs(call_args[0][2], mock_ctx.group)

    @patch("paddleformers.fleet.context_parallel_utils._ulysses_single_all_to_all")
    @patch("paddleformers.fleet.context_parallel_utils._ulysses_fused_supported")
    def test_backward_falls_back_swaps_scatter_gather(
        self, mock_supported, mock_a2a
    ):
        """When the fused path is unsupported, backward falls back to the
        reference a2a with swapped (scatter, gather) indices."""
        from paddleformers.fleet.context_parallel_utils import UlyssesAlltoAll

        mock_supported.return_value = False
        mock_a2a.return_value = paddle.randn([2, 4, 4, 8])

        mock_ctx = MagicMock()
        mock_ctx.scatter_idx = 2
        mock_ctx.gather_idx = 1
        mock_ctx.batch_dim_idx = 0
        mock_ctx.group = MagicMock()

        grad_output = paddle.randn([2, 8, 2, 8])
        UlyssesAlltoAll.backward(mock_ctx, grad_output)

        # Reference signature: (grad_output, gather_idx, scatter_idx,
        # batch_dim_idx, group) -- scatter and gather are swapped.
        mock_a2a.assert_called_once()
        call_args = mock_a2a.call_args
        self.assertIs(call_args[0][0], grad_output)
        self.assertEqual(call_args[0][1], 1)  # was gather_idx=1
        self.assertEqual(call_args[0][2], 2)  # was scatter_idx=2
        self.assertEqual(call_args[0][3], 0)  # batch_dim_idx
        self.assertIs(call_args[0][4], mock_ctx.group)


# PLACEHOLDER_SECTION_4


# =============================================================================
# Tests for flashmask_attention_ulysses
# =============================================================================


class TestFlashmaskAttentionUlysses(unittest.TestCase):
    """Test flashmask_attention_ulysses validation and happy path."""

    def _mock_fleet_context(self, cp_size=2, cp_rank=0):
        """Helper to mock fleet hybrid communicate group."""
        mock_hcg = MagicMock()
        mock_cp_group = MagicMock()
        mock_cp_group.nranks = cp_size
        mock_cp_group.rank = cp_rank
        mock_hcg.get_context_parallel_group.return_value = mock_cp_group
        return mock_hcg, mock_cp_group

    def test_learnable_sink_raises(self):
        """Should raise NotImplementedError if learnable_sink is provided."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, _ = self._mock_fleet_context()
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                flashmask_attention_ulysses(
                    query=paddle.randn([2, 4, 4, 8]),
                    key=paddle.randn([2, 4, 4, 8]),
                    value=paddle.randn([2, 4, 4, 8]),
                    startend_row_indices=paddle.zeros(
                        [2, 1, 8, 2], dtype="int32"
                    ),
                    learnable_sink=paddle.randn([1]),
                )
            self.assertIn("learnable_sink", str(ctx.exception))

    def test_softmax_scale_raises(self):
        """Should raise NotImplementedError if softmax_scale is provided."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, _ = self._mock_fleet_context()
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                flashmask_attention_ulysses(
                    query=paddle.randn([2, 4, 4, 8]),
                    key=paddle.randn([2, 4, 4, 8]),
                    value=paddle.randn([2, 4, 4, 8]),
                    startend_row_indices=paddle.zeros(
                        [2, 1, 8, 2], dtype="int32"
                    ),
                    softmax_scale=0.5,
                )
            self.assertIn("softmax_scale", str(ctx.exception))

    def test_mismatched_heads_raises(self):
        """Should assert if q/k/v heads are not equal."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, _ = self._mock_fleet_context()
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                flashmask_attention_ulysses(
                    query=paddle.randn([2, 4, 4, 8]),
                    key=paddle.randn([2, 4, 2, 8]),  # different heads
                    value=paddle.randn([2, 4, 4, 8]),
                    startend_row_indices=paddle.zeros(
                        [2, 1, 8, 2], dtype="int32"
                    ),
                )
            self.assertIn("q_heads == k_heads == v_heads", str(ctx.exception))

    def test_heads_not_divisible_by_cp_raises(self):
        """Should assert if num_heads not divisible by cp_size."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, _ = self._mock_fleet_context(cp_size=3)
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                flashmask_attention_ulysses(
                    query=paddle.randn([2, 4, 4, 8]),
                    key=paddle.randn([2, 4, 4, 8]),
                    value=paddle.randn([2, 4, 4, 8]),
                    startend_row_indices=paddle.zeros(
                        [2, 1, 8, 2], dtype="int32"
                    ),
                )
            self.assertIn("divisible", str(ctx.exception))

    def test_invalid_mask_heads_raises(self):
        """Should assert if mask head dim is neither 1 nor num_kv_heads."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, _ = self._mock_fleet_context(cp_size=2)
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                flashmask_attention_ulysses(
                    query=paddle.randn([2, 4, 4, 8]),
                    key=paddle.randn([2, 4, 4, 8]),
                    value=paddle.randn([2, 4, 4, 8]),
                    startend_row_indices=paddle.zeros(
                        [2, 3, 8, 2], dtype="int32"
                    ),  # 3 != 1 and 3 != 4
                )
            self.assertIn("head dim must be 1", str(ctx.exception))

    @patch("paddlefleet_ops.flash_mask_facade.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.UlyssesAlltoAll.apply")
    def test_happy_path_broadcast_mask(self, mock_a2a_apply, mock_fm_attn):
        """Normal path with num_mask_heads=1 (broadcast), should not slice mask."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, mock_cp_group = self._mock_fleet_context(cp_size=2, cp_rank=0)

        # Mock UlyssesAlltoAll.apply to return correctly shaped tensors
        def a2a_side_effect(inp, scatter_idx, gather_idx, batch_dim_idx, group):
            if scatter_idx == 2:
                # scatter heads, gather seq: [2, 4, 4, 8] -> [2, 8, 2, 8]
                return paddle.randn([2, 8, 2, 8])
            else:
                # scatter seq, gather heads: [2, 8, 2, 8] -> [2, 4, 4, 8]
                return paddle.randn([2, 4, 4, 8])

        mock_a2a_apply.side_effect = a2a_side_effect
        mock_fm_attn.return_value = paddle.randn([2, 8, 2, 8])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = flashmask_attention_ulysses(
                query=paddle.randn([2, 4, 4, 8]),
                key=paddle.randn([2, 4, 4, 8]),
                value=paddle.randn([2, 4, 4, 8]),
                startend_row_indices=paddle.zeros([2, 1, 8, 2], dtype="int32"),
                causal=True,
            )
        self.assertEqual(result.shape, [2, 4, 4, 8])
        mock_fm_attn.assert_called_once()

    @patch("paddlefleet_ops.flash_mask_facade.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.UlyssesAlltoAll.apply")
    def test_happy_path_per_head_mask_slicing(
        self, mock_a2a_apply, mock_fm_attn
    ):
        """When num_mask_heads == num_kv_heads, mask is sliced per rank."""
        from paddleformers.fleet.context_parallel_utils import (
            flashmask_attention_ulysses,
        )

        mock_hcg, mock_cp_group = self._mock_fleet_context(cp_size=2, cp_rank=1)

        def a2a_side_effect(inp, scatter_idx, gather_idx, batch_dim_idx, group):
            if scatter_idx == 2:
                return paddle.randn([2, 8, 2, 8])
            else:
                return paddle.randn([2, 4, 4, 8])

        mock_a2a_apply.side_effect = a2a_side_effect
        mock_fm_attn.return_value = paddle.randn([2, 8, 2, 8])

        startend = paddle.zeros(
            [2, 4, 8, 2], dtype="int32"
        )  # num_mask_heads=4==num_kv_heads

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = flashmask_attention_ulysses(
                query=paddle.randn([2, 4, 4, 8]),
                key=paddle.randn([2, 4, 4, 8]),
                value=paddle.randn([2, 4, 4, 8]),
                startend_row_indices=startend,
                causal=False,
            )

        # Verify mask was sliced: rank=1, heads_per_rank=2, so head_start=2, head_end=4
        fm_call_kwargs = mock_fm_attn.call_args
        used_mask = (
            fm_call_kwargs[1]["startend_row_indices"]
            if fm_call_kwargs[1]
            else fm_call_kwargs[0][3]
        )
        self.assertEqual(used_mask.shape[1], 2)  # sliced from 4 to 2


# PLACEHOLDER_SECTION_5


# =============================================================================
# Tests for flashmask_attention_cp mode routing
# =============================================================================


class TestFlashmaskAttentionCpModeRouting(unittest.TestCase):
    """Test mode routing in flashmask_attention_cp."""

    @patch("paddleformers.fleet.context_parallel_utils.FlashMaskContextParallel.apply")
    def test_dualchunk_allgather_mode(self, mock_apply):
        """mode='dualchunk_allgather' should call FlashMaskContextParallel.apply."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        mock_apply.return_value = paddle.randn([2, 4, 4, 8])
        q = paddle.randn([2, 4, 4, 8])
        k = paddle.randn([2, 4, 4, 8])
        v = paddle.randn([2, 4, 4, 8])
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        result = flashmask_attention_cp(
            q, k, v, mask, mode="dualchunk_allgather"
        )
        mock_apply.assert_called_once()
        self.assertEqual(result.shape, [2, 4, 4, 8])

    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention_ulysses")
    def test_contiguous_a2a_mode(self, mock_ulysses):
        """mode='contiguous_a2a' should call flashmask_attention_ulysses."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        mock_ulysses.return_value = paddle.randn([2, 4, 4, 8])
        q = paddle.randn([2, 4, 4, 8])
        k = paddle.randn([2, 4, 4, 8])
        v = paddle.randn([2, 4, 4, 8])
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        result = flashmask_attention_cp(
            q, k, v, mask, mode="contiguous_a2a", training=True
        )
        mock_ulysses.assert_called_once()

    def test_contiguous_a2a_fixed_seed_offset_raises(self):
        """mode='contiguous_a2a' with fixed_seed_offset should raise."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        with self.assertRaises(NotImplementedError) as ctx:
            flashmask_attention_cp(
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.zeros([2, 1, 4, 2], dtype="int32"),
                fixed_seed_offset=paddle.to_tensor([42]),
                mode="contiguous_a2a",
            )
        self.assertIn("fixed_seed_offset", str(ctx.exception))

    def test_contiguous_a2a_dropout_raises(self):
        """mode='contiguous_a2a' with dropout != 0.0 should raise."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        with self.assertRaises(NotImplementedError) as ctx:
            flashmask_attention_cp(
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.zeros([2, 1, 4, 2], dtype="int32"),
                dropout=0.1,
                mode="contiguous_a2a",
            )
        self.assertIn("dropout", str(ctx.exception))

    def test_contiguous_a2a_training_false_raises(self):
        """mode='contiguous_a2a' with training=False should raise."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        with self.assertRaises(NotImplementedError) as ctx:
            flashmask_attention_cp(
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.zeros([2, 1, 4, 2], dtype="int32"),
                training=False,
                mode="contiguous_a2a",
            )
        self.assertIn("training", str(ctx.exception))

    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        from paddleformers.fleet.context_parallel_utils import flashmask_attention_cp

        with self.assertRaises(ValueError) as ctx:
            flashmask_attention_cp(
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.randn([2, 4, 4, 8]),
                paddle.zeros([2, 1, 4, 2], dtype="int32"),
                mode="nonexistent_mode",
            )
        self.assertIn("invalid cp_balance_mode", str(ctx.exception))


# PLACEHOLDER_SECTION_6


# =============================================================================
# Tests for mode.startswith("contiguous") changes in ScatterOp/GatherOp/AllGatherOp
# =============================================================================


class TestContextParallelOpsContiguousMode(unittest.TestCase):
    """Test that mode.startswith('contiguous') routes correctly."""

    def _mock_hcg(self, cp_world_size=2):
        mock_hcg = MagicMock()
        mock_group = MagicMock()
        mock_group.nranks = cp_world_size
        mock_group.rank = 0
        mock_hcg.get_context_parallel_world_size.return_value = cp_world_size
        mock_hcg.get_context_parallel_group.return_value = mock_group
        return mock_hcg

    @patch("paddleformers.fleet.context_parallel_utils.scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.scatter_balance")
    def test_scatter_op_contiguous_allgather(
        self, mock_balance, mock_contiguous
    ):
        """mode='contiguous_allgather' should call scatter_contiguous."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp

        mock_hcg = self._mock_hcg()
        mock_contiguous.return_value = paddle.randn([4, 4])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            ctx = MagicMock()
            ContextParallelScatterOp.forward(
                ctx, paddle.randn([8, 4]), axis=0, mode="contiguous_allgather"
            )
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.scatter_balance")
    def test_scatter_op_contiguous_a2a(self, mock_balance, mock_contiguous):
        """mode='contiguous_a2a' should also call scatter_contiguous (startswith)."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp

        mock_hcg = self._mock_hcg()
        mock_contiguous.return_value = paddle.randn([4, 4])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            ctx = MagicMock()
            ContextParallelScatterOp.forward(
                ctx, paddle.randn([8, 4]), axis=0, mode="contiguous_a2a"
            )
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.scatter_balance")
    def test_scatter_op_dualchunk_uses_balance(
        self, mock_balance, mock_contiguous
    ):
        """mode='dualchunk_allgather' should call scatter_balance."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp

        mock_hcg = self._mock_hcg()
        mock_balance.return_value = paddle.randn([4, 4])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            ctx = MagicMock()
            ContextParallelScatterOp.forward(
                ctx, paddle.randn([8, 4]), axis=0, mode="dualchunk_allgather"
            )
        mock_balance.assert_called_once()
        mock_contiguous.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.all_gather_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.all_gather_balance")
    def test_gather_op_contiguous_a2a(self, mock_balance, mock_contiguous):
        """ContextParallelGatherOp with mode 'contiguous_a2a'."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelGatherOp

        mock_hcg = self._mock_hcg()
        mock_contiguous.return_value = paddle.randn([8, 4])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            ctx = MagicMock()
            ContextParallelGatherOp.forward(
                ctx, paddle.randn([4, 4]), axis=0, mode="contiguous_a2a"
            )
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.all_gather_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.all_gather_balance")
    def test_allgather_op_contiguous_a2a(self, mock_balance, mock_contiguous):
        """ContextParallelAllGatherOp with mode 'contiguous_a2a'."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_hcg = self._mock_hcg()
        mock_contiguous.return_value = paddle.randn([8, 4])

        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            ctx = MagicMock()
            ContextParallelAllGatherOp.forward(
                ctx, paddle.randn([4, 4]), axis=0, mode="contiguous_a2a"
            )
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()


# PLACEHOLDER_SECTION_7


# =============================================================================
# Tests for TransformerConfig cp_balance_mode validation
# =============================================================================


class TestTransformerConfigCpBalanceMode(unittest.TestCase):
    """Test cp_balance_mode validation added in __post_init__."""

    def test_valid_dualchunk_allgather(self):
        """'dualchunk_allgather' should be accepted."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(cp_balance_mode="dualchunk_allgather")
        self.assertEqual(config.cp_balance_mode, "dualchunk_allgather")

    def test_valid_contiguous_allgather(self):
        """'contiguous_allgather' should be accepted."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(cp_balance_mode="contiguous_allgather")
        self.assertEqual(config.cp_balance_mode, "contiguous_allgather")

    def test_valid_contiguous_a2a(self):
        """'contiguous_a2a' should be accepted."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(cp_balance_mode="contiguous_a2a")
        self.assertEqual(config.cp_balance_mode, "contiguous_a2a")

    def test_invalid_mode_raises(self):
        """Invalid cp_balance_mode should raise ValueError."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError) as ctx:
            TransformerConfig(cp_balance_mode="invalid_mode")
        self.assertIn("invalid", str(ctx.exception))


# PLACEHOLDER_SECTION_8


# =============================================================================
# Tests for DotProductAttention contiguous_a2a logic
# =============================================================================


class TestDotProductAttentionContiguousA2a(unittest.TestCase):
    """Test DotProductAttention changes for contiguous_a2a mode.

    Tests the actual DotProductAttention.forward() method with mocked
    flashmask_attention functions to cover the modified lines 345-350,
    475-494, and 540 in dot_product_attention.py.
    """

    def _make_config(self, cp_balance_mode="contiguous_a2a", **kwargs):
        """Create a real TransformerConfig for testing."""
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig
        from paddleformers.fleet.utils import (
            init_method_normal,
            scaled_init_method_normal,
        )

        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "softmax_scale": None,
            "use_bias": True,
            "recompute_granularity": None,
            "recompute_modules": None,
            "init_method": init_method_normal(0.02),
            "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
            "rms_norm_eps": 1e-5,
            "context_parallel_size": kwargs.get("context_parallel_size", 2),
            "sequence_parallel": False,
            "apply_query_key_layer_scaling": False,
            "sliding_window": None,
            "window_attn_skip_freq": None,
            "fp16": False,
            "bf16": True,
            "masked_softmax_fusion": False,
            "attention_softmax_in_fp32": True,
            "attention_dropout": 0.0,
            "softmax_type": "vanilla",
            "fa_version": None,
            "cp_balance_mode": cp_balance_mode,
            "multi_latent_attention": kwargs.get(
                "multi_latent_attention", True
            ),
        }
        defaults.update(kwargs)
        return TransformerConfig(**defaults)

    def _make_attn_instance(self, config):
        """Create a DotProductAttention with mocked CP world size."""
        from paddleformers.fleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType

        with patch(
            "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
            return_value=config.context_parallel_size,
        ):
            attn = DotProductAttention(
                config=config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
            )
        return attn

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    def test_forward_contiguous_a2a_passes_mode_kwarg(self, mock_cp_attn):
        """DotProductAttention.forward with contiguous_a2a passes mode='contiguous_a2a' to flashmask_attention_cp."""
        config = self._make_config(
            cp_balance_mode="contiguous_a2a", multi_latent_attention=True
        )
        attn = self._make_attn_instance(config)
        attn.eval()

        mock_cp_attn.return_value = paddle.randn(
            [2, 4, 4, 32], dtype="bfloat16"
        )

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with patch(
            "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
            return_value=2,
        ):
            result = attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=False,
            )

        # Verify mode kwarg was passed
        mock_cp_attn.assert_called_once()
        call_kwargs = mock_cp_attn.call_args[1]
        self.assertEqual(call_kwargs.get("mode"), "contiguous_a2a")

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    def test_forward_contiguous_a2a_skips_expand_mask(self, mock_cp_attn):
        """contiguous_a2a mode should NOT call expand_attn_mask_startend_row_indices_for_cp."""
        config = self._make_config(
            cp_balance_mode="contiguous_a2a", multi_latent_attention=True
        )
        attn = self._make_attn_instance(config)
        attn.eval()

        mock_cp_attn.return_value = paddle.randn(
            [2, 4, 4, 32], dtype="bfloat16"
        )

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with (
            patch(
                "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                attn,
                "expand_attn_mask_startend_row_indices_for_cp",
                side_effect=AssertionError("Should not be called"),
            ) as mock_expand,
        ):
            result = attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=False,
            )

        mock_expand.assert_not_called()

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    def test_forward_dualchunk_calls_expand_mask(self, mock_cp_attn):
        """dualchunk_allgather mode SHOULD call expand_attn_mask_startend_row_indices_for_cp."""
        config = self._make_config(cp_balance_mode="dualchunk_allgather")
        attn = self._make_attn_instance(config)
        attn.eval()

        mock_cp_attn.return_value = paddle.randn(
            [2, 4, 4, 32], dtype="bfloat16"
        )
        expanded_mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with (
            patch(
                "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                attn,
                "expand_attn_mask_startend_row_indices_for_cp",
                return_value=expanded_mask,
            ) as mock_expand,
        ):
            result = attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=False,
            )

        mock_expand.assert_called_once()
        # Verify mode is dualchunk_allgather
        call_kwargs = mock_cp_attn.call_args[1]
        self.assertEqual(call_kwargs.get("mode"), "dualchunk_allgather")

    def test_forward_contiguous_a2a_rr_passes_mode_kwarg(self):
        """contiguous_a2a + refined recompute dispatches to the RR CP wrapper."""
        config = self._make_config(
            cp_balance_mode="contiguous_a2a", multi_latent_attention=True
        )
        attn = self._make_attn_instance(config)
        attn.eval()

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with (
            patch(
                "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                attn,
                "rr_flashmask_attention_cp_func",
                return_value=paddle.randn([2, 4, 4, 32], dtype="bfloat16"),
            ) as mock_rr_cp,
        ):
            attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=True,
            )

        mock_rr_cp.assert_called_once()
        call_kwargs = mock_rr_cp.call_args.kwargs
        self.assertEqual(call_kwargs.get("mode"), "contiguous_a2a")

    def test_forward_contiguous_a2a_without_mla_raises(self):
        """contiguous_a2a without multi_latent_attention should raise."""
        config = self._make_config(
            cp_balance_mode="contiguous_a2a", multi_latent_attention=False
        )
        attn = self._make_attn_instance(config)
        attn.eval()

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with (
            patch(
                "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            self.assertRaises(NotImplementedError) as ctx,
        ):
            attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=False,
            )

        self.assertIn("only supports mla", str(ctx.exception))

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    def test_forward_contiguous_a2a_preserves_causal(self, mock_cp_attn):
        """contiguous_a2a should preserve the original is_causal (not force it to False)."""
        from paddleformers.fleet.transformer.enums import AttnMaskType

        config = self._make_config(
            cp_balance_mode="contiguous_a2a", multi_latent_attention=True
        )
        attn = self._make_attn_instance(config)
        attn.eval()

        mock_cp_attn.return_value = paddle.randn(
            [2, 4, 4, 32], dtype="bfloat16"
        )

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with patch(
            "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
            return_value=2,
        ):
            result = attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                attn_mask_type=AttnMaskType.causal,
                use_rr_flash_attention=False,
            )

        # For causal attn_mask_type, is_causal should be True (not forced to False like dualchunk)
        call_kwargs = mock_cp_attn.call_args[1]
        self.assertTrue(call_kwargs.get("causal", False))

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    def test_forward_dualchunk_forces_causal_false(self, mock_cp_attn):
        """dualchunk_allgather should force is_causal=False."""
        config = self._make_config(cp_balance_mode="dualchunk_allgather")
        attn = self._make_attn_instance(config)
        attn.eval()

        mock_cp_attn.return_value = paddle.randn(
            [2, 4, 4, 32], dtype="bfloat16"
        )
        expanded_mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        q = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        k = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        v = paddle.randn([2, 4, 4, 32], dtype="bfloat16")
        mask = paddle.zeros([2, 1, 4, 2], dtype="int32")

        with (
            patch(
                "paddleformers.fleet.transformer.dot_product_attention.get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                attn,
                "expand_attn_mask_startend_row_indices_for_cp",
                return_value=expanded_mask,
            ),
        ):
            result = attn(
                q,
                k,
                v,
                None,
                attn_mask_startend_row_indices=mask,
                use_rr_flash_attention=False,
            )

        call_kwargs = mock_cp_attn.call_args[1]
        self.assertFalse(call_kwargs.get("causal", True))


# =============================================================================
# Tests for backward of ContextParallelScatterOp/GatherOp with contiguous modes
# =============================================================================


class TestContextParallelOpsBackward(unittest.TestCase):
    """Test backward paths for contiguous mode in ScatterOp/GatherOp."""

    @patch("paddleformers.fleet.context_parallel_utils.all_gather_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.all_gather_balance")
    def test_scatter_op_backward_contiguous(
        self, mock_balance, mock_contiguous
    ):
        """ScatterOp backward with contiguous mode calls all_gather_contiguous."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp

        mock_contiguous.return_value = paddle.randn([8, 4])

        ctx = MagicMock()
        ctx.mode = "contiguous_a2a"
        ctx.group = MagicMock()
        ctx.axis = 0

        ContextParallelScatterOp.backward(ctx, paddle.randn([4, 4]))
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.all_gather_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.all_gather_balance")
    def test_scatter_op_backward_balance(self, mock_balance, mock_contiguous):
        """ScatterOp backward with non-contiguous mode calls all_gather_balance."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelScatterOp

        mock_balance.return_value = paddle.randn([8, 4])

        ctx = MagicMock()
        ctx.mode = "dualchunk_allgather"
        ctx.group = MagicMock()
        ctx.axis = 0

        ContextParallelScatterOp.backward(ctx, paddle.randn([4, 4]))
        mock_balance.assert_called_once()
        mock_contiguous.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.scatter_balance")
    def test_gather_op_backward_contiguous(self, mock_balance, mock_contiguous):
        """GatherOp backward with contiguous mode calls scatter_contiguous."""
        from paddleformers.fleet.context_parallel_utils import ContextParallelGatherOp

        mock_contiguous.return_value = paddle.randn([4, 4])

        ctx = MagicMock()
        ctx.mode = "contiguous_allgather"
        ctx.group = MagicMock()
        ctx.axis = 0

        ContextParallelGatherOp.backward(ctx, paddle.randn([8, 4]))
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.reduce_scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.reduce_scatter_any_axis_balance")
    def test_allgather_op_backward_contiguous(
        self, mock_balance, mock_contiguous
    ):
        """AllGatherOp backward with contiguous mode calls reduce_scatter_contiguous."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_contiguous.return_value = paddle.randn([4, 4])

        ctx = MagicMock()
        ctx.mode = "contiguous_a2a"
        ctx.group = MagicMock()
        ctx.axis = 0

        ContextParallelAllGatherOp.backward(ctx, paddle.randn([8, 4]))
        mock_contiguous.assert_called_once()
        mock_balance.assert_not_called()

    @patch("paddleformers.fleet.context_parallel_utils.reduce_scatter_contiguous")
    @patch("paddleformers.fleet.context_parallel_utils.reduce_scatter_any_axis_balance")
    def test_allgather_op_backward_balance(self, mock_balance, mock_contiguous):
        """AllGatherOp backward with non-contiguous mode calls reduce_scatter_balance."""
        from paddleformers.fleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        mock_balance.return_value = paddle.randn([4, 4])

        ctx = MagicMock()
        ctx.mode = "dualchunk_allgather"
        ctx.group = MagicMock()
        ctx.axis = 0

        ContextParallelAllGatherOp.backward(ctx, paddle.randn([8, 4]))
        mock_balance.assert_called_once()
        mock_contiguous.assert_not_called()


# =============================================================================
# Tests that actually execute the fused Triton permute kernels
# =============================================================================


def _fused_kernels_available():
    """Whether the CUDA + Triton fused kernels can actually run here.

    Kept import-time cheap and side-effect free: it must NOT switch the global
    default device (doing so at ``@skipUnless`` evaluation time would leak GPU
    placement into the mock-based dispatch tests above and make them bypass
    their mocks into the real fused path). It also swallows *any* optional-
    backend failure (driver init, Triton runtime, missing GPU, etc.) so an
    unavailable backend skips these tests instead of erroring at collection.
    The GPU device switch itself happens in ``setUpClass`` below.
    """
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        import triton  # noqa: F401

        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (  # noqa: F401
            _head2seq_post_kernel,
            _seq2head_pre_kernel,
        )
    except Exception:
        return False
    return True


def _triton_available():
    """Triton importable (the fused op module can be loaded). Does not
    require a GPU: the shape-validation paths raise before any kernel runs."""
    try:
        import triton  # noqa: F401

        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (  # noqa: F401
            ulysses_single_all_to_all_fused,
        )
    except (ImportError, OSError):
        return False
    return True


@unittest.skipUnless(
    _fused_kernels_available(), "requires CUDA + Triton fused kernels"
)
class TestUlyssesFusedKernels(unittest.TestCase):
    """Run the fused permute kernels and compare their index mapping against
    an explicit paddle reshape/transpose reference (bit-exact).

    These exercise the actual `_seq2head_*` / `_head2seq_*` kernels for P>1,
    which the mock-based dispatch tests above do not, guarding the index math
    and buffer layout against silent regressions.
    """

    P, B, SLOC, HLOC, D = 4, 2, 3, 2, 8

    @classmethod
    def setUpClass(cls):
        # Switch to GPU only for these kernel tests and restore the previous
        # default device afterwards, so the mock-based dispatch tests keep
        # their CPU default (where _ulysses_fused_supported returns False).
        cls._prev_device = paddle.get_device()
        paddle.set_device("gpu")

    @classmethod
    def tearDownClass(cls):
        paddle.set_device(cls._prev_device)

    def _launch(self, kernel, src, dst):
        import triton

        P, b, Sloc, Hloc, D = self.P, self.B, self.SLOC, self.HLOC, self.D
        seg = Hloc * D
        block = min(triton.next_power_of_2(seg), 4096)
        grid = (P * b * Sloc, triton.cdiv(seg, block))
        kernel[grid](src, dst, P, b, Sloc, Hloc, D, seg, BLOCK=block)

    def test_seq2head_pre_matches_reference(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            _seq2head_pre_kernel,
        )

        P, b, Sloc, Hloc, D = self.P, self.B, self.SLOC, self.HLOC, self.D
        x = paddle.randn([b, Sloc, P * Hloc, D])
        out = paddle.empty([P, b, Sloc, Hloc, D], dtype=x.dtype)
        self._launch(_seq2head_pre_kernel, x, out)
        ref = x.reshape([b, Sloc, P, Hloc, D]).transpose([2, 0, 1, 3, 4])
        self.assertTrue(paddle.equal_all(out, ref).item())

    def test_seq2head_post_matches_reference(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            _seq2head_post_kernel,
        )

        P, b, Sloc, Hloc, D = self.P, self.B, self.SLOC, self.HLOC, self.D
        recv = paddle.randn([P, b, Sloc, Hloc, D])
        out = paddle.empty([b, P * Sloc, Hloc, D], dtype=recv.dtype)
        self._launch(_seq2head_post_kernel, recv, out)
        ref = recv.transpose([1, 0, 2, 3, 4]).reshape([b, P * Sloc, Hloc, D])
        self.assertTrue(paddle.equal_all(out, ref).item())

    def test_head2seq_pre_matches_reference(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            _head2seq_pre_kernel,
        )

        P, b, Sloc, Hloc, D = self.P, self.B, self.SLOC, self.HLOC, self.D
        g = paddle.randn([b, P * Sloc, Hloc, D])
        out = paddle.empty([P, b, Sloc, Hloc, D], dtype=g.dtype)
        self._launch(_head2seq_pre_kernel, g, out)
        ref = g.reshape([b, P, Sloc, Hloc, D]).transpose([1, 0, 2, 3, 4])
        self.assertTrue(paddle.equal_all(out, ref).item())

    def test_head2seq_post_matches_reference(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            _head2seq_post_kernel,
        )

        P, b, Sloc, Hloc, D = self.P, self.B, self.SLOC, self.HLOC, self.D
        recv = paddle.randn([P, b, Sloc, Hloc, D])
        out = paddle.empty([b, Sloc, P * Hloc, D], dtype=recv.dtype)
        self._launch(_head2seq_post_kernel, recv, out)
        ref = recv.transpose([1, 2, 0, 3, 4]).reshape([b, Sloc, P * Hloc, D])
        self.assertTrue(paddle.equal_all(out, ref).item())

    def test_fused_full_a2a_matches_reference_single_rank(self):
        """End-to-end fused vs reference all-to-all on a 1-rank group (P=1):
        exercises the public wrapper, pre+post kernels, send-buffer reuse and
        the alltoall_single call together."""
        from unittest.mock import patch

        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all,
        )
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            ulysses_single_all_to_all_fused,
        )

        b, Sloc, H, D = 2, 4, 4, 8

        def fake_a2a_single(recv, send, group=None, use_calc_stream=True):
            recv.copy_(send, False)  # 1-rank all-to-all is a copy

        for scatter_idx, gather_idx in ((2, 1), (1, 2)):
            inp = paddle.randn([b, Sloc, H, D])
            with (
                patch(
                    "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                    ".get_world_size",
                    return_value=1,
                ),
                patch(
                    "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                    ".stream.alltoall_single",
                    side_effect=fake_a2a_single,
                ),
            ):
                fused = ulysses_single_all_to_all_fused(inp, scatter_idx, None)
            with (
                patch(
                    "paddleformers.fleet.context_parallel_utils.dist.get_world_size",
                    return_value=1,
                ),
                patch(
                    "paddleformers.fleet.context_parallel_utils.dist.alltoall",
                    side_effect=lambda out, i, group=None: out.copy_(i, False),
                ),
            ):
                ref = _ulysses_single_all_to_all(
                    inp, scatter_idx, gather_idx, 0, None
                )
            self.assertEqual(fused.shape, ref.shape)
            self.assertTrue(paddle.equal_all(fused, ref).item())

    def test_fused_supported_true_for_gpu_4d(self):
        """The real guard accepts the production layout on GPU: exercises the
        deferred Triton import and the underlying support check (returns True),
        which the mock-based dispatch tests never run."""
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_fused_supported,
        )

        x = paddle.randn([2, 4, 8, 16])  # batch_dim0, 4-D, on GPU
        self.assertTrue(_ulysses_fused_supported(2, 0, x))

    def test_wrapper_dispatches_to_fused_op(self):
        """The `_ulysses_single_all_to_all_fused` thin wrapper imports and
        calls the Triton op (P=1 mock keeps it single-rank)."""
        from unittest.mock import patch

        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_single_all_to_all_fused,
        )

        def fake_a2a_single(recv, send, group=None, use_calc_stream=True):
            recv.copy_(send, False)

        inp = paddle.randn([2, 4, 4, 8])
        with (
            patch(
                "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                ".get_world_size",
                return_value=1,
            ),
            patch(
                "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                ".stream.alltoall_single",
                side_effect=fake_a2a_single,
            ),
        ):
            out = _ulysses_single_all_to_all_fused(inp, 2, None)
        self.assertEqual(out.shape, [2, 4, 4, 8])


class TestUlyssesFusedSupportGuard(unittest.TestCase):
    """Exercise the non-Triton `_ulysses_fused_supported` guard directly
    (the dispatch tests above mock it out). CPU-safe: the reject branches and
    the import-failure fallback do not touch a GPU."""

    def test_false_for_non_batch_dim0(self):
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_fused_supported,
        )

        # batch_dim_idx != 0 short-circuits to the reference path regardless
        # of build/device, so this covers the early `return False` guard.
        x = paddle.randn([2, 4, 8, 16])
        self.assertFalse(_ulysses_fused_supported(2, 1, x))

    def test_false_when_triton_import_fails(self):
        from paddleformers.fleet.context_parallel_utils import (
            _ulysses_fused_supported,
        )

        # Force the first guard to pass (compiled-with-cuda + gpu place + 4-D)
        # then make the deferred Triton import raise, hitting the except path.
        fake_input = MagicMock()
        fake_input.shape = [2, 4, 8, 16]
        fake_input.place.is_gpu_place.return_value = True
        with (
            patch.object(paddle, "is_compiled_with_cuda", return_value=True),
            patch.dict(
                sys.modules,
                {"paddleformers.fleet.triton_ops.ulysses_alltoall_fused": None},
            ),
        ):
            self.assertFalse(_ulysses_fused_supported(2, 0, fake_input))


@unittest.skipUnless(_triton_available(), "requires Triton fused op module")
class TestUlyssesFusedOpValidation(unittest.TestCase):
    """Shape-divisibility validation in the fused op. The ValueError is raised
    before any kernel launch, so these run without a GPU."""

    def test_raises_on_indivisible_heads(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            ulysses_single_all_to_all_fused,
        )

        x = paddle.randn([2, 4, 5, 8])  # H=5 not divisible by world size 2
        with (
            patch(
                "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                ".get_world_size",
                return_value=2,
            ),
            self.assertRaises(ValueError),
        ):
            ulysses_single_all_to_all_fused(x, 2, None)

    def test_raises_on_indivisible_seq(self):
        from paddleformers.fleet.triton_ops.ulysses_alltoall_fused import (
            ulysses_single_all_to_all_fused,
        )

        x = paddle.randn([2, 5, 4, 8])  # Nglob=5 not divisible by world size 2
        with (
            patch(
                "paddleformers.fleet.triton_ops.ulysses_alltoall_fused.dist"
                ".get_world_size",
                return_value=2,
            ),
            self.assertRaises(ValueError),
        ):
            ulysses_single_all_to_all_fused(x, 1, None)


if __name__ == "__main__":
    unittest.main()
