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
    gen_mtp_attn_mask,
    gen_mtp_attn_mask_startend_row_indices,
    gen_mtp_layer_mask,
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
        args = _TrainingArgs(cp_size=1, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(100, args), 100)

    def test_cp2_pads_to_multiple_of_4(self):
        args = _TrainingArgs(cp_size=2, tp_size=1, sequence_parallel=False)
        self.assertEqual(calc_padding_size(5, args), 8)
        self.assertEqual(calc_padding_size(8, args), 8)

    def test_sequence_parallel_uses_tp_size(self):
        args = _TrainingArgs(cp_size=1, tp_size=2, sequence_parallel=True)
        self.assertEqual(calc_padding_size(5, args), 8)


# ---------------------------------------------------------------------------
# pad_batch_data
# ---------------------------------------------------------------------------


class TestPadBatchData(unittest.TestCase):
    def test_pads_to_max_length(self):
        result = pad_batch_data([[1, 2, 3], [4, 5]], pad_idx=0)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_array_equal(result[1], [4, 5, 0])

    def test_pads_to_explicit_max_seq_len(self):
        result = pad_batch_data([[1, 2], [3]], pad_idx=-1, max_seq_len=5)
        np.testing.assert_array_equal(result[0], [1, 2, -1, -1, -1])


# ---------------------------------------------------------------------------
# gen_self_attn_mask
# ---------------------------------------------------------------------------


class TestGenSelfAttnMask(unittest.TestCase):
    def test_output_shape(self):
        mask = gen_self_attn_mask([[1, 2, 3], [4, 5]], max_seq_len=8, use_global_causal_attn=False)
        self.assertEqual(mask.shape, (1, 1, 8, 8))

    def test_no_cross_segment_attention(self):
        mask = gen_self_attn_mask([[1, 2], [3, 4]], max_seq_len=4, use_global_causal_attn=False)
        self.assertEqual(mask[0, 0, 2, 0], 0.0)

    def test_global_causal_is_lower_triangular(self):
        mask = gen_self_attn_mask([[1, 2], [3, 4]], max_seq_len=4, use_global_causal_attn=True)
        np.testing.assert_array_equal(mask[0, 0, :4, :4], np.tril(np.ones((4, 4))))


# ---------------------------------------------------------------------------
# gen_attn_mask_startend_row_indices
# ---------------------------------------------------------------------------


class TestGenAttnMaskStartendRowIndices(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        result = gen_attn_mask_startend_row_indices([[1, 2, 3]], max_seq_len=3, use_global_causal_attn=False)
        self.assertEqual(result.shape, (1, 1, 3, 1))
        self.assertEqual(result.dtype, np.int32)

    def test_each_token_points_to_segment_end(self):
        # Segment A: tokens 0-2 → end=3; Segment B: tokens 3-4 → end=5
        result = gen_attn_mask_startend_row_indices([[0, 0, 0], [0, 0]], max_seq_len=5, use_global_causal_attn=False)
        indices = result[0, 0, :, 0].tolist()
        self.assertEqual(indices[:3], [3, 3, 3])
        self.assertEqual(indices[3:], [5, 5])

    def test_padding_area_is_ascending(self):
        result = gen_attn_mask_startend_row_indices([[0, 0]], max_seq_len=5, use_global_causal_attn=False)
        self.assertEqual(result[0, 0, 2:, 0].tolist(), [2, 3, 4])


# ---------------------------------------------------------------------------
# Shared fixture
#
# BATCH = [[1, 2, EOS], [4, 5, 6]]
# total_len=6, mtp_depth=2, max_seq_len=8
# internal_boundaries=[3]
#
# Boundary shift rule (layer k): shifted = original - (k+1)
#   Layer 0: 3-1=2 → blocks [0:2], [2:6]
#   Layer 1: 3-2=1 → blocks [0:1], [1:6]
# ---------------------------------------------------------------------------

EOS = 3
BATCH = [[1, 2, EOS], [4, 5, 6]]
MTP_DEPTH = 2
TOTAL_LEN = 6
MAX_SEQ_LEN = TOTAL_LEN + MTP_DEPTH  # 8


# ---------------------------------------------------------------------------
# gen_mtp_attn_mask
# ---------------------------------------------------------------------------


class TestGenMtpAttnMask(unittest.TestCase):
    def _call(self, use_global_causal_attn):
        return gen_mtp_attn_mask(BATCH, MAX_SEQ_LEN, MTP_DEPTH, use_global_causal_attn)

    def test_output_shape(self):
        self.assertEqual(self._call(False).shape, (MTP_DEPTH, 1, MAX_SEQ_LEN, MAX_SEQ_LEN))

    def test_global_causal_is_lower_triangular(self):
        mask = self._call(True)
        expected = np.tril(np.ones((TOTAL_LEN, TOTAL_LEN), dtype=np.float32))
        for k in range(MTP_DEPTH):
            np.testing.assert_array_equal(mask[k, 0, :TOTAL_LEN, :TOTAL_LEN], expected)

    def test_layer0_block_boundaries(self):
        """Layer 0: blocks [0:2] and [2:6], no cross-block attention."""
        m = self._call(False)[0, 0]
        np.testing.assert_array_equal(m[:2, :2], np.tril(np.ones((2, 2))))
        np.testing.assert_array_equal(m[2:6, 2:6], np.tril(np.ones((4, 4))))
        np.testing.assert_array_equal(m[2:6, :2], 0.0)

    def test_layer1_block_boundaries(self):
        """Layer 1: blocks [0:1] and [1:6], no cross-block attention."""
        m = self._call(False)[1, 0]
        self.assertEqual(m[0, 0], 1.0)
        np.testing.assert_array_equal(m[1:6, 1:6], np.tril(np.ones((5, 5))))
        np.testing.assert_array_equal(m[1:6, :1], 0.0)

    def test_padding_area_is_zero(self):
        mask = self._call(False)
        for k in range(MTP_DEPTH):
            np.testing.assert_array_equal(mask[k, 0, TOTAL_LEN:, :], 0.0)


# ---------------------------------------------------------------------------
# gen_mtp_attn_mask_startend_row_indices
# ---------------------------------------------------------------------------


class TestGenMtpAttnMaskStartendRowIndices(unittest.TestCase):
    def _call(self, use_global_causal_attn):
        return gen_mtp_attn_mask_startend_row_indices(BATCH, MAX_SEQ_LEN, MTP_DEPTH, use_global_causal_attn)

    def test_output_shape_and_dtype(self):
        result = self._call(False)
        self.assertEqual(result.shape, (MTP_DEPTH, 1, MAX_SEQ_LEN, 1))
        self.assertEqual(result.dtype, np.int32)

    def test_layer0_end_row_values(self):
        """Layer 0: positions 0-1 → end=2, positions 2-5 → end=6."""
        indices = self._call(False)[0, 0, :TOTAL_LEN, 0].tolist()
        self.assertEqual(indices[:2], [2, 2])
        self.assertEqual(indices[2:], [6, 6, 6, 6])

    def test_layer1_end_row_values(self):
        """Layer 1: position 0 → end=1, positions 1-5 → end=6."""
        indices = self._call(False)[1, 0, :TOTAL_LEN, 0].tolist()
        self.assertEqual(indices[0], 1)
        self.assertEqual(indices[1:], [6, 6, 6, 6, 6])

    def test_padding_area_is_ascending(self):
        result = self._call(False)
        for k in range(MTP_DEPTH):
            self.assertEqual(result[k, 0, TOTAL_LEN:, 0].tolist(), list(range(TOTAL_LEN, MAX_SEQ_LEN)))

    def test_consistency_with_2d_mask(self):
        """startend_row_indices must agree with the 2D matrix version."""
        mask_2d = gen_mtp_attn_mask(BATCH, MAX_SEQ_LEN, MTP_DEPTH, use_global_causal_attn=False)
        result = self._call(False)
        for k in range(MTP_DEPTH):
            for pos in range(TOTAL_LEN):
                end = int(result[k, 0, pos, 0])
                np.testing.assert_array_equal(
                    mask_2d[k, 0, pos, end:TOTAL_LEN],
                    0.0,
                    err_msg=f"layer={k} pos={pos} end={end}",
                )


# ---------------------------------------------------------------------------
# gen_mtp_layer_mask
# ---------------------------------------------------------------------------
#
# all_token_ids=[1,2,EOS,4,5,6], ids_mtp=[5,6], ids_ori=[1,2,3,4]
# Layer 0: mtp_ids=[2,3,4,5] → EOS@1 → mask[1]=0
# Layer 1: mtp_ids=[3,4,5,6] → EOS@0 → mask[0]=0
# ---------------------------------------------------------------------------


class TestGenMtpLayerMask(unittest.TestCase):
    def _call(self, eos_token_id=None):
        return gen_mtp_layer_mask(BATCH, MAX_SEQ_LEN, MTP_DEPTH, eos_token_id)

    def test_output_shape_and_dtype(self):
        result = self._call()
        self.assertEqual(result.shape, (MTP_DEPTH, MAX_SEQ_LEN))
        self.assertEqual(result.dtype, np.int32)

    def test_no_eos_all_ones(self):
        np.testing.assert_array_equal(self._call(None), np.ones((MTP_DEPTH, MAX_SEQ_LEN), dtype=np.int32))

    def test_layer0_eos_zeroed_at_shifted_position(self):
        """Layer 0: EOS at shifted position 1 → mask[0,1]=0, rest=1."""
        result = self._call(EOS)
        self.assertEqual(result[0, 1], 0)
        np.testing.assert_array_equal(np.delete(result[0], 1), 1)

    def test_layer1_eos_zeroed_at_shifted_position(self):
        """Layer 1: EOS at shifted position 0 → mask[1,0]=0, rest=1."""
        result = self._call(EOS)
        self.assertEqual(result[1, 0], 0)
        np.testing.assert_array_equal(result[1, 1:], 1)

    def test_padding_area_is_one(self):
        result = self._call(EOS)
        for k in range(MTP_DEPTH):
            np.testing.assert_array_equal(result[k, TOTAL_LEN:], 1)


if __name__ == "__main__":
    unittest.main()
