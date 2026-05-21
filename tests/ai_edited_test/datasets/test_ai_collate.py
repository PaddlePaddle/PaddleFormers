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

import unittest
from unittest.mock import MagicMock

import numpy as np

from paddleformers.datasets.collate import (
    calc_padding_size,
    dpo_collate_fn,
    mm_dpo_collate_fn,
)
from paddleformers.datasets.DPODataset import Sequence


class TestCalcPaddingSize(unittest.TestCase):
    """Tests for calc_padding_size function."""

    def _make_training_args(self, cp_size=1, sp_size=1, fp8=False, sequence_parallel=True):
        args = MagicMock()
        args.context_parallel_size = cp_size
        args.tensor_model_parallel_size = sp_size
        args.sequence_parallel = sequence_parallel
        args.fp8 = fp8
        return args

    def test_basic_no_parallel(self):
        """Test with no parallelism (cp=1, sp=1)."""
        args = self._make_training_args(cp_size=1, sp_size=1, sequence_parallel=False)
        result = calc_padding_size(100, args)
        self.assertEqual(result, 100)

    def test_with_sequence_parallel(self):
        """Test with sequence parallel (sp=2)."""
        args = self._make_training_args(cp_size=1, sp_size=2, sequence_parallel=True)
        result = calc_padding_size(100, args)
        # padding_to_size = 2 * 1 * 2 = 4, ceil(100/4)*4 = 100
        self.assertEqual(result, 100)

    def test_with_context_parallel(self):
        """Test with context parallel (cp=2)."""
        args = self._make_training_args(cp_size=2, sp_size=1, sequence_parallel=False)
        result = calc_padding_size(5, args)
        # padding_to_size = 2 * 2 * 1 = 4, ceil(5/4)*4 = 8
        self.assertEqual(result, 8)

    def test_with_fp8(self):
        """Test with fp8 enabled (rounds up to multiple of 4)."""
        args = self._make_training_args(cp_size=1, sp_size=2, sequence_parallel=True, fp8=True)
        result = calc_padding_size(5, args)
        # padding_to_size = 2 (cp*sp>1), then (2+3)//4*4 = 4, then 4*1*2=8, ceil(5/8)*8=8
        self.assertEqual(result, 8)

    def test_already_aligned(self):
        """Test when seq_len is already aligned."""
        args = self._make_training_args(cp_size=2, sp_size=2, sequence_parallel=True)
        result = calc_padding_size(8, args)
        # padding_to_size = 2*2*2=8, ceil(8/8)*8 = 8
        self.assertEqual(result, 8)

    def test_with_sp_not_enabled(self):
        """Test when sequence_parallel is False, sp_size is ignored."""
        args = self._make_training_args(cp_size=1, sp_size=4, sequence_parallel=False)
        result = calc_padding_size(10, args)
        # sp_size is 1 (since sequence_parallel=False), padding_to_size=1, ceil(10/1)*1=10
        self.assertEqual(result, 10)


class TestDPOCollateFn(unittest.TestCase):
    """Tests for dpo_collate_fn function."""

    def _make_sequence(
        self,
        token_ids,
        position_ids,
        response_labels,
        response_index,
        attention_mask=None,
        attn_mask_startend_row_indices=None,
        score_delta=0.0,
    ):
        return Sequence(
            token_ids=token_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            response_labels=response_labels,
            response_index=response_index,
            score_delta=score_delta,
            has_mm=[False],
        )

    def _make_training_args(self, cp_size=1, sp_size=1, fp8=False, sequence_parallel=True):
        args = MagicMock()
        args.context_parallel_size = cp_size
        args.tensor_model_parallel_size = sp_size
        args.sequence_parallel = sequence_parallel
        args.fp8 = fp8
        return args

    def test_basic_collate_with_attention_mask(self):
        """Test basic collation with attention_mask."""
        seq1 = self._make_sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            response_labels=[-100, 1, 2],
            response_index=[0, 1, 3],
            attention_mask=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        )
        seq2 = self._make_sequence(
            token_ids=[4, 5],
            position_ids=[0, 1],
            response_labels=[-100, 5],
            response_index=[0, 1, 2],
            attention_mask=[[1, 0], [1, 1]],
        )
        batch = [[seq1], [seq2]]
        tokenizer = MagicMock()
        args = self._make_training_args()

        result = dpo_collate_fn(batch, tokenizer, args, max_seq_len=4, use_filtered_label_loss=False)

        self.assertIn("input_ids", result)
        self.assertIn("position_ids", result)
        self.assertIn("response_labels", result)
        self.assertIn("response_indexs", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(result["input_ids"].shape[0], 2)
        self.assertEqual(result["position_ids"].shape[0], 2)

    def test_collate_with_attn_mask_startend_row_indices(self):
        """Test collation with attn_mask_startend_row_indices."""
        seq1 = self._make_sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            response_labels=[-100, 1, 2],
            response_index=[0, 1, 3],
            attention_mask=None,
            attn_mask_startend_row_indices=[0, 1],
        )
        seq2 = self._make_sequence(
            token_ids=[4, 5],
            position_ids=[0, 1],
            response_labels=[-100, 5],
            response_index=[0, 1, 2],
            attention_mask=None,
            attn_mask_startend_row_indices=[0, 1],
        )
        batch = [[seq1], [seq2]]
        tokenizer = MagicMock()
        args = self._make_training_args()

        result = dpo_collate_fn(batch, tokenizer, args, max_seq_len=4, use_filtered_label_loss=False)

        self.assertIn("attn_mask_startend_row_indices", result)

    def test_collate_with_score_delta(self):
        """Test collation with score_delta enabled."""
        seq1 = self._make_sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            response_labels=[-100, 1, 2],
            response_index=[0, 1, 3],
            attention_mask=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            score_delta=0.5,
        )
        batch = [[seq1]]
        tokenizer = MagicMock()
        args = self._make_training_args()

        result = dpo_collate_fn(
            batch, tokenizer, args, max_seq_len=4, use_filtered_label_loss=True, use_response_score_delta=True
        )

        self.assertIn("score_deltas", result)

    def test_collate_padding_free(self):
        """Test collation in padding_free mode."""
        seq1 = self._make_sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            response_labels=[-100, 1, 2],
            response_index=[0, 1, 3],
            attention_mask=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        )
        seq2 = self._make_sequence(
            token_ids=[4, 5],
            position_ids=[0, 1],
            response_labels=[-100, 5],
            response_index=[0, 1, 2],
            attention_mask=[[1, 0], [1, 1]],
        )
        batch = [[seq1, seq2]]
        tokenizer = MagicMock()
        args = self._make_training_args()

        result = dpo_collate_fn(
            batch, tokenizer, args, max_seq_len=None, padding_free=True, use_filtered_label_loss=False
        )

        self.assertIn("input_ids", result)
        # In padding_free mode, sequences are concatenated into one
        self.assertEqual(result["input_ids"].shape[0], 1)

    def test_collate_no_max_seq_len_uses_max_of_batch(self):
        """Test that max_seq_len is computed from batch if not provided."""
        seq1 = self._make_sequence(
            token_ids=[1, 2, 3],
            position_ids=[0, 1, 2],
            response_labels=[-100, 1, 2],
            response_index=[0, 1, 3],
            attention_mask=[[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        )
        batch = [[seq1]]
        tokenizer = MagicMock()
        args = self._make_training_args()

        result = dpo_collate_fn(batch, tokenizer, args, max_seq_len=None, use_filtered_label_loss=False)
        self.assertIn("input_ids", result)


class TestMMDPOCollateFn(unittest.TestCase):
    """Tests for mm_dpo_collate_fn function."""

    def test_mm_dpo_collate_fn_exists(self):
        """Test that mm_dpo_collate_fn is callable."""
        self.assertTrue(callable(mm_dpo_collate_fn))

    def test_mm_dpo_collate_fn_accepts_model_param(self):
        """Test that mm_dpo_collate_fn has model parameter."""
        import inspect

        sig = inspect.signature(mm_dpo_collate_fn)
        self.assertIn("model", sig.parameters)

    def test_mm_dpo_collate_fn_sequence_has_mm_inputs(self):
        """Test that Sequence has mm_inputs field for mm_dpo_collate_fn."""
        seq = Sequence(
            token_ids=[1, 2],
            position_ids=[0, 1],
            attention_mask=None,
            attn_mask_startend_row_indices=None,
            response_labels=[-100, 1],
            response_index=[0, 1, 2],
            score_delta=0.0,
            has_mm=[True, True],
            mm_inputs={"pixel_values": np.random.randn(1, 3, 224, 224)},
        )
        self.assertIn("pixel_values", seq.mm_inputs)


if __name__ == "__main__":
    unittest.main()
