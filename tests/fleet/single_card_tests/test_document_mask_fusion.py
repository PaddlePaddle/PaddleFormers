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

"""
Unit tests for ``paddleformers.fleet.triton_ops.document_mask_fusion``.

Every triton kernel is validated against the *original* (pre-fusion)
implementation imported from ``csa_attention`` — the fusion replaced those
functions, so they are the ground-truth reference.
"""

import unittest

import numpy as np
import paddle

from paddleformers.fleet.transformer.csa_attention import (
    CSADocMaskMetadata,
    _build_compress_topk_idxs_from_valid_range,
    _build_window_topk_idxs_from_doc_bounds,
    compact_kv_score_cutoff,
)
from paddleformers.fleet.triton_ops.document_mask_fusion import (
    compressed_doc_start_triton,
    compressed_topk_idxs_triton,
    cutoff_compact_triton,
    document_mask_triton,
    window_topk_idxs_triton,
)


class TestDocumentMaskFusion(unittest.TestCase):
    def setUp(self):
        self.ratio = 4
        self.window_size = 4
        self.batch_size = 1
        doc_lens = [5, 14, 3, 8]
        pad = 2

        startend_row_indices = []
        cum = 0
        for length in doc_lens:
            cum += length
            startend_row_indices += [cum] * length
        # padding continues the last doc's end value
        startend_row_indices += [cum] * pad

        self.seqlen = sum(doc_lens) + pad  # 32
        self.startend_row_indices = paddle.to_tensor(
            startend_row_indices, dtype="int32"
        ).reshape([1, 1, self.seqlen, 1])

        # --- reference doc boundaries (original implementation) ---
        self.meta_ref = CSADocMaskMetadata.build(
            self.ratio,
            self.batch_size,
            self.seqlen,
            self.startend_row_indices,
            dense_mode=False,
        )
        positions = paddle.arange(self.seqlen, dtype="int64")
        self.pos_in_doc_ref = positions - self.meta_ref.doc_start_per_pos

    def test_document_mask_kernel(self):
        """document_mask_fwd_kernel -> (doc_start, doc_len, pos_in_doc)."""
        doc_start, doc_len, pos_in_doc = document_mask_triton(
            self.startend_row_indices.flatten()
        )
        np.testing.assert_array_equal(
            doc_start, self.meta_ref.doc_start_per_pos
        )
        np.testing.assert_array_equal(doc_len, self.meta_ref.doc_len_per_pos)
        np.testing.assert_array_equal(pos_in_doc, self.pos_in_doc_ref)

    def test_cutoff_compact_kernel(self):
        """cutoff_compact_kernel -> gather idx / pos / n / is_first / comp_pos."""
        ratio = self.ratio
        seqlen = self.seqlen
        total_cutoff = self.meta_ref.doc_lens_cutoff.sum().item()
        actual_n_compressed = total_cutoff // ratio

        # ---- reference from original functions ----
        identity = paddle.arange(seqlen, dtype="int64").reshape([1, seqlen, 1])
        src_idx, _ = compact_kv_score_cutoff(
            self.meta_ref.doc_starts,
            self.meta_ref.doc_lens_cutoff,
            self.meta_ref.doc_starts_cutoff,
            total_cutoff,
            identity,
            identity,
        )
        gather_idx_ref = src_idx.flatten()  # [total_cutoff]
        cutoff_pos_ref = paddle.gather(self.pos_in_doc_ref, gather_idx_ref)

        # ---- kernel output ----
        (
            gather_idx,
            cutoff_pos,
            n_cutoff,
            is_first,
            compressed_pos,
        ) = cutoff_compact_triton(
            self.pos_in_doc_ref, self.meta_ref.doc_len_per_pos, ratio
        )

        self.assertEqual(n_cutoff.item(), total_cutoff)
        np.testing.assert_array_equal(gather_idx[:n_cutoff], gather_idx_ref)
        np.testing.assert_array_equal(cutoff_pos[:n_cutoff], cutoff_pos_ref)
        np.testing.assert_array_equal(
            is_first[:actual_n_compressed],
            self.meta_ref.get_is_first_compressed_group(),
        )

        # compressed_pos_in_doc has no original reference
        compressed_pos_ref = paddle.concat(
            [
                paddle.arange(0, doc_len, ratio)
                for doc_len in self.meta_ref.doc_lens_cutoff
            ],
        )
        np.testing.assert_array_equal(
            compressed_pos[:actual_n_compressed], compressed_pos_ref
        )

    def test_window_topk_idxs_kernel(self):
        """window_topk_idxs_kernel -> [1, seqlen, window_size]."""
        window_ref = _build_window_topk_idxs_from_doc_bounds(
            self.batch_size,
            self.seqlen,
            self.window_size,
            self.meta_ref.doc_start_per_pos,
            self.meta_ref.is_valid,
        )
        window_out = window_topk_idxs_triton(
            self.meta_ref.doc_start_per_pos,
            self.meta_ref.doc_len_per_pos,
            self.window_size,
        )
        np.testing.assert_array_equal(window_out, window_ref)

    def test_compressed_topk_idxs_kernel(self):
        """compressed_topk_idxs_kernel -> [1, seqlen, seqlen // ratio]."""
        n_compressed = self.seqlen // self.ratio
        valid_range = self.meta_ref.valid_range

        compressed_doc_start = compressed_doc_start_triton(
            self.startend_row_indices.flatten(),
            self.meta_ref.doc_start_per_pos,
            self.ratio,
        )

        for offset in [0, 1000]:
            ref = _build_compress_topk_idxs_from_valid_range(
                self.batch_size, self.seqlen, n_compressed, offset, valid_range
            )
            out = compressed_topk_idxs_triton(
                compressed_doc_start,
                self.pos_in_doc_ref,
                self.meta_ref.doc_len_per_pos,
                self.ratio,
                offset,
            )
            np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    unittest.main()
