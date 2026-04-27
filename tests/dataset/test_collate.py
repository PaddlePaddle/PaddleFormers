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

import unittest

import numpy as np

from paddleformers.datasets.collate import (
    calc_padding_size,
    gen_attn_mask_startend_row_indices,
    gen_self_attn_mask,
    pad_batch_data,
)

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------


class _TrainingArgs:
    """Minimal stub for training_args used by calc_padding_size."""

    def __init__(self, cp_size=1, tp_size=1, sequence_parallel=False, fp8=False):
        self.context_parallel_size = cp_size
        self.tensor_model_parallel_size = tp_size
        self.sequence_parallel = sequence_parallel
        self.fp8 = fp8


# ---------------------------------------------------------------------------
# calc_padding_size
# ---------------------------------------------------------------------------


class TestCalcPaddingSize(unittest.TestCase):
    def test_no_parallelism_no_padding(self):
        """cp=1, sp=1 → padding_to_size=1, result equals seq_len."""
        args = _TrainingArgs(cp_size=1, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(100, args), 100)
        self.assertEqual(calc_padding_size(1, args), 1)

    def test_cp2_pads_to_multiple_of_4(self):
        """cp=2, sp=1 → padding_to_size=4 (2*2), seq_len rounded up to multiple of 4."""
        args = _TrainingArgs(cp_size=2, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(5, args), 8)
        self.assertEqual(calc_padding_size(8, args), 8)
        self.assertEqual(calc_padding_size(9, args), 12)

    def test_sequence_parallel_uses_tp_size(self):
        """sequence_parallel=True activates tp_size in the computation."""
        args = _TrainingArgs(cp_size=1, tp_size=2, sequence_parallel=True)
        # padding_to_size = 2 * 1 * 2 = 4
        self.assertEqual(calc_padding_size(5, args), 8)

    def test_fp8_rounds_padding_to_size_to_multiple_of_4(self):
        """fp8=True applies an additional round-up of padding_to_size to multiples of 4."""
        # cp=2, sp=1 → base padding_to_size=2; after fp8: ceil(2/4)*4=4; final=4*2=8
        args = _TrainingArgs(cp_size=2, tp_size=1, sequence_parallel=False, fp8=True)
        result = calc_padding_size(1, args)
        # Must be a multiple of 8
        self.assertEqual(result % 8, 0)

    def test_already_aligned_is_unchanged(self):
        """A sequence already aligned to the padding size should not be padded further."""
        args = _TrainingArgs(cp_size=2, tp_size=1, sequence_parallel=False)
        # padding_to_size=4; 16 is already a multiple of 4
        self.assertEqual(calc_padding_size(16, args), 16)


# ---------------------------------------------------------------------------
# pad_batch_data
# ---------------------------------------------------------------------------


class TestPadBatchData(unittest.TestCase):
    def test_pads_to_max_length(self):
        insts = [[1, 2, 3], [4, 5]]
        result = pad_batch_data(insts, pad_idx=0)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result[1], [4, 5, 0])

    def test_pads_to_explicit_max_seq_len(self):
        insts = [[1, 2], [3]]
        result = pad_batch_data(insts, pad_idx=-1, max_seq_len=5)
        self.assertEqual(result.shape, (2, 5))
        np.testing.assert_array_equal(result[0], [1, 2, -1, -1, -1])

    def test_single_sequence_no_padding_needed(self):
        insts = [[10, 20, 30]]
        result = pad_batch_data(insts, pad_idx=0)
        np.testing.assert_array_equal(result[0], [10, 20, 30])


# ---------------------------------------------------------------------------
# gen_self_attn_mask
# ---------------------------------------------------------------------------


class TestGenSelfAttnMask(unittest.TestCase):
    def test_output_shape(self):
        """Result shape must be (1, 1, max_seq_len, max_seq_len)."""
        token_ids = [[1, 2, 3], [4, 5]]
        mask = gen_self_attn_mask(token_ids, max_seq_len=8, use_global_causal_attn=False)
        self.assertEqual(mask.shape, (1, 1, 8, 8))

    def test_causal_within_each_segment(self):
        """Without global causal, each segment is independently causal (lower triangular)."""
        token_ids = [[1, 2, 3]]
        mask = gen_self_attn_mask(token_ids, max_seq_len=3, use_global_causal_attn=False)
        seg = mask[0, 0, :3, :3]
        # Lower triangular
        self.assertEqual(seg[0, 0], 1.0)
        self.assertEqual(seg[0, 1], 0.0)
        self.assertEqual(seg[1, 0], 1.0)
        self.assertEqual(seg[1, 1], 1.0)

    def test_no_cross_segment_attention(self):
        """Tokens in segment B should not attend to tokens in segment A."""
        # Two segments of length 2 each
        token_ids = [[1, 2], [3, 4]]
        mask = gen_self_attn_mask(token_ids, max_seq_len=4, use_global_causal_attn=False)
        # token 2 (offset 2, seg B) attending to token 0 (seg A) should be 0
        self.assertEqual(mask[0, 0, 2, 0], 0.0)
        self.assertEqual(mask[0, 0, 3, 1], 0.0)

    def test_global_causal_attn_is_single_lower_triangular(self):
        """use_global_causal_attn=True treats all segments as one causal sequence."""
        token_ids = [[1, 2], [3, 4]]
        mask = gen_self_attn_mask(token_ids, max_seq_len=4, use_global_causal_attn=True)
        seq = mask[0, 0, :4, :4]
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(seq, expected)

    def test_padding_area_is_zero(self):
        """Positions beyond the actual sequence length should remain 0."""
        token_ids = [[1, 2]]
        mask = gen_self_attn_mask(token_ids, max_seq_len=5, use_global_causal_attn=False)
        # Rows / cols 2-4 are padding
        np.testing.assert_array_equal(mask[0, 0, 2:, :], 0.0)


# ---------------------------------------------------------------------------
# gen_attn_mask_startend_row_indices
# ---------------------------------------------------------------------------


class TestGenAttnMaskStartendRowIndices(unittest.TestCase):
    def test_output_shape(self):
        """Result shape must be (1, 1, max_seq_len, 1)."""
        token_ids = [[1, 2, 3]]
        result = gen_attn_mask_startend_row_indices(token_ids, max_seq_len=3, use_global_causal_attn=False)
        self.assertEqual(result.shape, (1, 1, 3, 1))

    def test_dtype_is_int32(self):
        token_ids = [[1, 2]]
        result = gen_attn_mask_startend_row_indices(token_ids, max_seq_len=2, use_global_causal_attn=False)
        self.assertEqual(result.dtype, np.int32)

    def test_non_global_each_token_points_to_segment_end(self):
        """Each token index should point to the end of its own segment (exclusive)."""
        # Segment A: tokens 0,1,2 → end = 3; segment B: tokens 3,4 → end = 5
        token_ids = [[0, 0, 0], [0, 0]]
        result = gen_attn_mask_startend_row_indices(token_ids, max_seq_len=5, use_global_causal_attn=False)
        indices = result[0, 0, :, 0].tolist()
        self.assertEqual(indices[0], 3)
        self.assertEqual(indices[1], 3)
        self.assertEqual(indices[2], 3)
        self.assertEqual(indices[3], 5)
        self.assertEqual(indices[4], 5)

    def test_global_causal_all_tokens_point_to_total_length(self):
        """With global causal, every token points to the total sequence length."""
        token_ids = [[0, 0], [0]]
        result = gen_attn_mask_startend_row_indices(token_ids, max_seq_len=3, use_global_causal_attn=True)
        indices = result[0, 0, :3, 0].tolist()
        self.assertTrue(all(v == 3 for v in indices))

    def test_padding_area_is_ascending(self):
        """Positions in the padding area (beyond sequence) should be ascending."""
        token_ids = [[0, 0]]
        result = gen_attn_mask_startend_row_indices(token_ids, max_seq_len=5, use_global_causal_attn=False)
        pad_indices = result[0, 0, 2:, 0].tolist()
        # Should be [2, 3, 4] (range(offset, max_seq_len))
        self.assertEqual(pad_indices, [2, 3, 4])
