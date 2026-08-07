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

"""Unit tests for ``build_hysparse_valid_range`` (flashmask -> valid_range).

This closes the precision-critical gap that the HySparse block-score / gather
op tests leave open: those tests feed ``valid_range`` built DIRECTLY from
``doc_lens`` (``_multidoc_valid_range``), bypassing the production path that
DERIVES ``valid_range`` from the flashmask ``startend_row_indices`` via a
cummax document-boundary scan. A bug in that derivation (wrong ``bos`` per
token) would silently mis-place every block coordinate and hurt precision, yet
go uncaught by the operator tests.

Here we build a synthetic flashmask (LTS ``[B, H, S, 1]`` whose value is each
token's exclusive document end -- the same convention as ``utils.get_doc_lens``)
from arbitrary-length (NOT 64-aligned) packed documents, and assert
``build_hysparse_valid_range`` reproduces the ground-truth ``[bos, eos)`` built
straight from the document layout -- for the pure document range, the windowed
range, the single-document (``None``) fast path, mask-batch broadcast, and the
multi-head (read head-0) case. Pure host math -- no GPU / SM requirement.
"""

import unittest

import paddle

from paddleformers.fleet.transformer.multi_latent_attention import (
    build_hysparse_valid_range,
)


def _doc_bounds(doc_lens):
    """[(start, end)] cumulative document boundaries for the packed sequence."""
    bounds, off = [], 0
    for L in doc_lens:
        bounds.append((off, off + L))
        off += L
    return bounds


def _flashmask_from_doc_lens(doc_lens, h=1, batch=1):
    """Synthetic flashmask LTS [B, H, S, 1] int32: per token its doc end (excl).

    Mirrors the flashmask document-causal convention consumed by
    ``get_doc_lens`` / ``build_hysparse_valid_range``: the value stored at
    position ``t`` is the exclusive end of the document containing ``t``.
    """
    s = sum(doc_lens)
    de = paddle.zeros([s], dtype="int32")
    for ds, dee in _doc_bounds(doc_lens):
        de[ds:dee] = dee
    return de.reshape([1, 1, s, 1]).expand([batch, h, s, 1]).contiguous()


def _reference_valid_range(doc_lens, batch=1, window_size=None):
    """Ground-truth [B, S, 2] int32 built straight from the document layout.

    ``eos = pos + 1`` (causal); ``bos = doc_start`` (document mask), optionally
    clamped up to ``pos - window_size + 1`` for the causal sliding window.
    """
    s = sum(doc_lens)
    bos = paddle.zeros([s], dtype="int64")
    eos = paddle.zeros([s], dtype="int64")
    for ds, dee in _doc_bounds(doc_lens):
        pos = paddle.arange(ds, dee, dtype="int64")
        start = paddle.full([dee - ds], ds, dtype="int64")
        if window_size is not None and window_size > 0:
            start = paddle.maximum(start, pos - window_size + 1)
        bos[ds:dee] = start
        eos[ds:dee] = pos + 1
    vr = paddle.stack([bos, eos], axis=-1).cast("int32")  # [S, 2]
    return vr.reshape([1, s, 2]).expand([batch, s, 2]).contiguous()


class TestBuildHySparseValidRange(unittest.TestCase):
    def _assert_equal(self, got, ref, msg=""):
        got_np = got.astype("int32").numpy()
        ref_np = ref.astype("int32").numpy()
        self.assertEqual(got_np.shape, ref_np.shape, f"shape mismatch {msg}")
        n_diff = int((got_np != ref_np).sum())
        self.assertEqual(
            n_diff, 0, f"{n_diff} valid_range entries differ {msg}"
        )

    def test_single_doc_via_flashmask(self):
        # One document spanning the whole sequence: bos=0, eos=t+1.
        doc_lens = [200]
        mask = _flashmask_from_doc_lens(doc_lens)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1)
        ref = _reference_valid_range(doc_lens)
        self._assert_equal(got, ref, "(single doc)")

    def test_none_mask_is_single_doc(self):
        # attn_mask_startend_row_indices=None => single document, bos=0.
        s = 137
        got = build_hysparse_valid_range(None, s, 1)
        ref = _reference_valid_range([s])
        self._assert_equal(got, ref, "(None mask)")

    def test_multidoc_unaligned(self):
        # Several docs, all unaligned to 64 -- the packed-training regime.
        doc_lens = [40, 88, 133, 27]
        mask = _flashmask_from_doc_lens(doc_lens)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1)
        ref = _reference_valid_range(doc_lens)
        self._assert_equal(got, ref, "(multidoc unaligned)")

    def test_multidoc_windowed(self):
        # Windowed clamp: bos = max(doc_start, pos - window + 1).
        doc_lens = [40, 88, 133, 27]
        window = 32
        mask = _flashmask_from_doc_lens(doc_lens)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1, window_size=window)
        ref = _reference_valid_range(doc_lens, window_size=window)
        self._assert_equal(got, ref, "(multidoc windowed)")

    def test_window_larger_than_doc_is_noop(self):
        # window >= longest doc => clamp never binds, equals pure doc range.
        doc_lens = [40, 88, 27]
        window = 1000
        mask = _flashmask_from_doc_lens(doc_lens)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1, window_size=window)
        ref = _reference_valid_range(doc_lens)  # no window clamp binds
        self._assert_equal(got, ref, "(window >= doc len)")

    def test_mask_batch_broadcast(self):
        # Flashmask batch=1 must broadcast over a data batch > 1.
        doc_lens = [50, 100, 70]
        mask = _flashmask_from_doc_lens(doc_lens, batch=1)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 4)
        ref = _reference_valid_range(doc_lens, batch=4)
        self._assert_equal(got, ref, "(mask batch broadcast)")

    def test_multihead_reads_head0(self):
        # Multi-head flashmask (all heads identical doc layout): head-0 read
        # must recover the same document valid_range.
        doc_lens = [40, 88, 133, 27]
        mask = _flashmask_from_doc_lens(doc_lens, h=8)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1)
        ref = _reference_valid_range(doc_lens)
        self._assert_equal(got, ref, "(multihead head-0)")

    def test_matches_get_doc_lens_convention(self):
        # Cross-check: bos recovered here must be consistent with the repo's
        # get_doc_lens document-length derivation from the same flashmask.
        from paddleformers.fleet.transformer.utils import get_doc_lens

        doc_lens = [40, 88, 133, 27]
        mask = _flashmask_from_doc_lens(doc_lens)
        s = sum(doc_lens)
        got = build_hysparse_valid_range(mask, s, 1).astype("int64").numpy()
        derived_lens = get_doc_lens(mask).numpy().tolist()
        self.assertEqual(
            derived_lens,
            doc_lens,
            f"get_doc_lens {derived_lens} != {doc_lens}",
        )
        # eos - bos at each doc's last token equals that doc's length.
        for ds, dee in _doc_bounds(doc_lens):
            last = dee - 1
            bos_last = int(got[0, last, 0])
            eos_last = int(got[0, last, 1])
            self.assertEqual(
                eos_last - bos_last,
                dee - ds,
                f"doc [{ds},{dee}) last-token span {eos_last - bos_last} "
                f"!= len {dee - ds}",
            )

    def test_bidirectional_bound_mask_rejected(self):
        # A document mask must carry a single exclusive-doc-end bound on the
        # last axis. A 2-bound (start+end / bidirectional) layout would make
        # [:, 0, :, 0] no longer the doc-end, so it must be rejected loudly.
        doc_lens = [40, 88, 27]
        mask = _flashmask_from_doc_lens(doc_lens)  # [1, 1, S, 1]
        s = sum(doc_lens)
        two_bound = paddle.concat([mask, mask], axis=-1)  # [1, 1, S, 2]
        with self.assertRaises(ValueError):
            build_hysparse_valid_range(two_bound, s, 1)


if __name__ == "__main__":
    unittest.main()
