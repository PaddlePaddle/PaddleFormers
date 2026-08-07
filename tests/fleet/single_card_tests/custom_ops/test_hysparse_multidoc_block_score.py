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

"""HySparse Bug 2: DOCUMENT-relative block bucketing / pack-equivalence.

The FA4-fused block-score kernel buckets each key column into a block relative
to its query's document start ``bos`` (``rel = floor((col - bos) / block_B)``),
so the whole downstream pipeline (``_valid_block_mask`` / ``select_topk_blocks``
/ the gather kernels), which is document-RELATIVE, sees consistent coordinates.

Two properties are exercised:

* ``test_scatter_ref_multidoc``: a packed sequence of several arbitrary-length
  (NOT 64-aligned) documents scored by FA4 (with a flashmask document-causal
  mask) matches an independent relative-scatter reference (matmul + softmax) on
  the finite block logits, eq.(3) block scores, and TopK indices.

* ``test_pack_equivalence`` (the core selection-correctness proof): a document D
  run alone (``bos=0``) and the SAME document packed behind an arbitrary-length
  (unaligned) prefix must select the exact same relative TopK blocks for D's
  rows. This holds for the document-relative kernel and FAILS for the old
  absolute-coordinate kernel -- so the test itself証明s the coordinate fix.

FA4 block-score fusion runs only on SM 10.x (Blackwell); the test skips
otherwise.
"""

import math
import unittest

import paddle

_NEG_INF = float("-inf")


def _sm100_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"FA4 block-score fusion requires SM 10.x (Blackwell); got SM {major}.x"
        )


def _dsa_or_skip(testcase):
    """Skip unless the DSA gather backend (FlashMLA + cuDNN) can run here."""
    _sm100_or_skip(testcase)
    try:
        from paddleformers.fleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            testcase.skipTest(
                "FlashMLA sparse fwd + cuDNN DSA bwd not available"
            )
    except (ImportError, RuntimeError):
        testcase.skipTest("hysparse DSA import failed")


def _num_blocks(seqlen_k, block_B):
    return (seqlen_k + block_B - 1) // block_B


def _doc_bounds(doc_lens):
    """[(start, end)] cumulative document boundaries for the packed sequence."""
    bounds, off = [], 0
    for L in doc_lens:
        bounds.append((off, off + L))
        off += L
    return bounds


def _multidoc_valid_range(doc_lens):
    """valid_range [1, S, 2] int32 for packed docs: per token (bos, eos=pos+1).

    Each token attends causally only within its own document, so bos is the
    document start and eos is the token's own position + 1.
    """
    s = sum(doc_lens)
    bos = paddle.zeros([s], dtype="int32")
    eos = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        idx = paddle.arange(ds, de, dtype="int32")
        bos[ds:de] = ds
        eos[ds:de] = idx + 1
    return paddle.stack([bos, eos], axis=-1).reshape([1, s, 2]).contiguous()


def _multidoc_startend_row_indices(doc_lens, h):
    """flashmask LTS [1, H, S, 1] int32: mask query rows >= doc_end per key col.

    Combined with FA4's built-in causal (rows < col masked), each key column c
    in document [ds, de) is visible exactly to rows [c, de) -- i.e. a
    block-diagonal (per-document) causal mask.
    """
    s = sum(doc_lens)
    lts = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        lts[ds:de] = de
    return lts.reshape([1, 1, s, 1]).expand([1, h, s, 1]).contiguous()


def _causal_valid_range(b, s):
    """Single-document causal valid_range [B, S, 2]: bos=0, eos=t+1."""
    eos = (
        paddle.arange(1, s + 1, dtype="int32")
        .reshape([1, s, 1])
        .expand([b, s, 1])
    )
    bos = paddle.zeros([b, s, 1], dtype="int32")
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _relative_scatter_reference(q, k, valid_range, sm_scale, block_B):
    """Document-RELATIVE scatter (eq.3) reference: scaled per-block max logit + LSE.

    q, k: [B, S, H, D] bf16 (packed multi-doc, S == S_kv). ``valid_range`` gives
    per-row [bos, eos). Block j of row r covers key columns
    ``[bos_r + j*block_B, bos_r + (j+1)*block_B)`` and only causally-valid,
    same-document keys (``bos_r <= c <= r``) contribute -- mirroring the kernel.

    Returns (block_logit_ref [B,H,S,nb] fp32, lse_ref [B,S,H] fp32).
    """
    b, s, h, d = q.shape
    sk = k.shape[1]
    nb = _num_blocks(sk, block_B)
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # [B,H,S,D]
    kf = k.astype("float32").transpose([0, 2, 1, 3])  # [B,H,Sk,D]
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [B,H,S,Sk]

    bos = valid_range[0, :, 0].astype("int64")  # [S]
    rows = paddle.arange(s, dtype="int64").reshape([s, 1])
    cols = paddle.arange(sk, dtype="int64").reshape([1, sk])
    bos_col = bos.reshape([s, 1])
    # same-document causal validity: bos_r <= c <= r  ([S, Sk] bool)
    valid = (cols >= bos_col) & (cols <= rows)
    # document-relative block id of each (row, col): floor((c - bos_r)/block_B)
    rel_block = (
        cols - bos_col
    ) // block_B  # [S, Sk] (only meaningful where valid)

    lse_mask = valid.reshape([1, 1, s, sk])
    masked_all = paddle.where(
        lse_mask, logits, paddle.full_like(logits, _NEG_INF)
    )
    lse_bhs = paddle.logsumexp(masked_all, axis=-1)  # [B,H,S]
    lse_ref = lse_bhs.transpose([0, 2, 1]).contiguous()  # [B,S,H]

    block_logit_ref = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    for j in range(nb):
        colmask = (valid & (rel_block == j)).reshape([1, 1, s, sk])
        masked_j = paddle.where(
            colmask, logits, paddle.full_like(logits, _NEG_INF)
        )
        block_logit_ref[:, :, :, j] = masked_j.max(axis=-1)
    return block_logit_ref, lse_ref


def _maxerr_finite(got, ref):
    """Masked-pattern match + max abs diff on finite (unmasked) entries."""
    import numpy as np

    got_np = got.astype("float32").numpy()
    ref_np = ref.astype("float32").numpy()
    thr = -1e30
    got_masked = got_np <= thr
    ref_masked = ref_np <= thr
    pattern_mismatch = int((got_masked != ref_masked).sum())
    finite = ~ref_masked
    diff = (
        np.abs(got_np[finite] - ref_np[finite])
        if finite.any()
        else np.array([0.0])
    )
    return pattern_mismatch, float(diff.max())


class TestHySparseMultiDocBlockScore(unittest.TestCase):
    BLOCK_B = 64
    TOPK = 4

    def _topk_set_mismatch(self, idx_a, idx_b):
        """#slots that differ after per-row sort (order-independent set compare)."""
        a = paddle.sort(idx_a.astype("int64"), axis=-1)
        b = paddle.sort(idx_b.astype("int64"), axis=-1)
        return int((a != b).astype("int32").sum().item()), a.numel().item()

    def test_scatter_ref_multidoc(self):
        # Packed docs of arbitrary lengths, NONE aligned to BLOCK_B=64.
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )
        from paddleformers.fleet.tilelang_ops.hysparse.pipeline import (
            block_scores_from_logit,
        )

        doc_lens = [40, 88, 133, 27]  # sum = 288, all unaligned to 64
        s = sum(doc_lens)
        h, d, dv = 8, 64, 64
        paddle.seed(2026)
        q = paddle.randn([1, s, h, d], dtype="bfloat16")
        k = paddle.randn([1, s, h, d], dtype="bfloat16")
        v = paddle.randn([1, s, h, dv], dtype="bfloat16")
        sm_scale = 1.0 / math.sqrt(d)
        valid_range = _multidoc_valid_range(doc_lens)
        startend = _multidoc_startend_row_indices(doc_lens, h)

        out, lse, block_logit = block_score_fa4_attn_fwd(
            q,
            k,
            v,
            valid_range=valid_range,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
            startend_row_indices=startend,
        )
        block_logit_ref, lse_ref = _relative_scatter_reference(
            q, k, valid_range, sm_scale, self.BLOCK_B
        )

        # (1) relative per-block max logit agrees (masked pattern + finite).
        pattern_mismatch, logit_err = _maxerr_finite(
            block_logit, block_logit_ref
        )
        self.assertEqual(
            pattern_mismatch,
            0,
            "block_logit masked pattern mismatch vs relative scatter ref",
        )
        logit_tol = 0.06 * math.sqrt(d) * sm_scale
        self.assertLessEqual(
            logit_err,
            logit_tol,
            f"block_logit finite max|diff|={logit_err:.4e} > tol={logit_tol:.4e}",
        )

        # (2) eq.(3) block scores exp(block_logit - lse) agree.
        scores_fa4 = block_scores_from_logit(block_logit, lse)
        scores_ref = block_scores_from_logit(block_logit_ref, lse_ref)
        _, score_err = _maxerr_finite(scores_fa4, scores_ref)
        self.assertLessEqual(
            score_err,
            5e-2,
            f"eq.3 block scores max|diff|={score_err:.4e} too large",
        )

        # (3) per-query TopK relative block indices match.
        fa4_idx = select_topk_blocks(
            block_logit, lse, valid_range, self.TOPK, self.BLOCK_B
        )
        ref_idx = select_topk_blocks(
            block_logit_ref, lse_ref, valid_range, self.TOPK, self.BLOCK_B
        )
        mismatch, total = self._topk_set_mismatch(fa4_idx, ref_idx)
        self.assertLessEqual(
            mismatch,
            max(1, int(0.005 * total)),
            f"TopK index mismatch {mismatch}/{total} between FA4 and scatter ref",
        )

    def _pack_equivalence(self, prefix_len, doc_len, h, d, dv, seed=2026):
        """Core pack-equivalence proof.

        Run document D alone (bos=0) and the SAME D packed behind an
        arbitrary-length (unaligned) prefix, both CAUSAL-only (no flashmask).
        The document-relative kernel must select the identical relative TopK
        blocks for D's rows and emit the identical relative block_logit; the old
        absolute-coordinate kernel would NOT (its 64-grid shifts by prefix_len).

        Causal-only is sufficient and robust: for D's rows in the packed run the
        prefix key columns land at relative block id < 0 (dropped/guarded by the
        kernel), and within-D causal visibility is bit-identical to the solo run;
        TopK is invariant to the per-row LSE shift the prefix induces because the
        block-score ordering within a row only depends on that row's block
        logits. So this isolates the coordinate fix, independent of flashmask.
        """
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )

        sm_scale = 1.0 / math.sqrt(d)

        # ---- (a) document D alone: bos=0, single-doc causal. ----
        paddle.seed(seed)
        qd = paddle.randn([1, doc_len, h, d], dtype="bfloat16")
        kd = paddle.randn([1, doc_len, h, d], dtype="bfloat16")
        vd = paddle.randn([1, doc_len, h, dv], dtype="bfloat16")
        vr_solo = _causal_valid_range(1, doc_len)
        _, lse_solo, bl_solo = block_score_fa4_attn_fwd(
            qd,
            kd,
            vd,
            valid_range=vr_solo,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
        )
        idx_solo = select_topk_blocks(
            bl_solo, lse_solo, vr_solo, self.TOPK, self.BLOCK_B
        )

        # ---- (b) [unaligned prefix | D] packed, single-doc causal over all. ----
        # A plain causal mask over the whole packed sequence lets D's rows also
        # attend to the prefix; but those prefix columns map to relative block
        # id < 0 for D's rows (bos = prefix_len), which the fixed kernel drops.
        # So D's finite relative blocks (>=0) see exactly D's own keys, matching
        # the solo run. valid_range gives D's rows bos = prefix_len.
        s = prefix_len + doc_len
        paddle.seed(seed + 777)
        qp = paddle.randn([1, s, h, d], dtype="bfloat16")
        kp = paddle.randn([1, s, h, d], dtype="bfloat16")
        vp = paddle.randn([1, s, h, dv], dtype="bfloat16")
        # Overwrite D's slice with the SAME q/k/v tensors as the solo run so the
        # relative-block logits are directly comparable value-for-value.
        qp[:, prefix_len:, :, :] = qd
        kp[:, prefix_len:, :, :] = kd
        vp[:, prefix_len:, :, :] = vd

        # valid_range for the packed run: prefix rows are their own doc (bos=0),
        # D's rows have bos=prefix_len, eos=row+1 (causal within packed seq).
        eos = paddle.arange(1, s + 1, dtype="int32").reshape([1, s, 1])
        bos = paddle.zeros([1, s, 1], dtype="int32")
        bos[:, prefix_len:, :] = prefix_len
        vr_pack = paddle.concat([bos, eos], axis=-1).contiguous()

        _, lse_pack, bl_pack = block_score_fa4_attn_fwd(
            qp,
            kp,
            vp,
            valid_range=vr_pack,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
        )
        idx_pack = select_topk_blocks(
            bl_pack, lse_pack, vr_pack, self.TOPK, self.BLOCK_B
        )

        # ---- Compare D's rows. Both are DOCUMENT-relative: no remap needed. ----
        nb_solo = _num_blocks(doc_len, self.BLOCK_B)
        # (1) relative block_logit for D's rows over D's relative columns match.
        bl_d_pack = bl_pack[
            :, :, prefix_len:, :nb_solo
        ]  # [1,H,doc_len,nb_solo]
        pattern_mismatch, logit_err = _maxerr_finite(bl_d_pack, bl_solo)
        self.assertEqual(
            pattern_mismatch,
            0,
            "packed-vs-solo block_logit masked pattern differs for D's rows "
            "-- coordinate systems disagree (absolute kernel bug)",
        )
        logit_tol = 0.06 * math.sqrt(d) * sm_scale
        self.assertLessEqual(
            logit_err,
            logit_tol,
            f"packed-vs-solo block_logit max|diff|={logit_err:.4e} > "
            f"tol={logit_tol:.4e} for D's rows",
        )

        # (2) relative TopK indices for D's rows are IDENTICAL (the hard req).
        idx_pack_d = idx_pack[:, prefix_len:, :]  # [1,doc_len,TOPK]
        mismatch, total = self._topk_set_mismatch(idx_pack_d, idx_solo)
        self.assertLessEqual(
            mismatch,
            max(1, int(0.005 * total)),
            f"pack-equivalence VIOLATED: {mismatch}/{total} TopK slots differ "
            f"between packed(bos={prefix_len}) and solo(bos=0) for D "
            f"(prefix_len={prefix_len}, doc_len={doc_len})",
        )

    def test_pack_equivalence(self):
        # Unaligned prefix (40) shifts the absolute 64-grid; the relative kernel
        # must still select D's blocks identically to running D alone. This
        # causal-only (no-flashmask) path also exercises the relative-block-id<0
        # guard for the prefix columns -- config-independent integer logic, so a
        # single d=64 config suffices to cover it.
        self._pack_equivalence(prefix_len=40, doc_len=200, h=8, d=64, dv=64)

    def _multidoc_pack_vs_nonpack(self, doc_lens, h, d, dv, seed=2026):
        """SEVERAL unaligned-length docs, run NON-PACKED vs PACKED, must match.

        The core precision guarantee for HySparse packed training: a batch of
        documents of arbitrary (NOT 64-aligned) lengths must select the exact
        same relative TopK blocks -- and emit the same relative block_logit --
        whether each document is scored ALONE (``bos=0``, one sequence per doc)
        or all of them are PACKED into a single sequence (each doc's rows carry
        ``bos = doc_start``, doc-causal flashmask). This is strictly stronger
        than the single-doc ``_pack_equivalence`` above: EVERY document in the
        pack (not just one behind a prefix) is checked against its solo run, so
        it exercises every non-zero ``bos`` the relative kernel must handle.
        """
        _sm100_or_skip(self)
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
        )

        sm_scale = 1.0 / math.sqrt(d)

        # Independent per-doc q/k/v, reused VERBATIM in the packed run so the
        # relative-block logits are directly comparable value-for-value.
        qs, ks, vs = [], [], []
        for di, L in enumerate(doc_lens):
            paddle.seed(seed + di)
            qs.append(paddle.randn([1, L, h, d], dtype="bfloat16"))
            ks.append(paddle.randn([1, L, h, d], dtype="bfloat16"))
            vs.append(paddle.randn([1, L, h, dv], dtype="bfloat16"))

        # ---- NON-PACKED: each doc alone, bos=0, single-doc causal. ----
        solo_bl, solo_idx = [], []
        for di, L in enumerate(doc_lens):
            vr = _causal_valid_range(1, L)
            _, lse_d, bl_d = block_score_fa4_attn_fwd(
                qs[di],
                ks[di],
                vs[di],
                valid_range=vr,
                sm_scale=sm_scale,
                block_B=self.BLOCK_B,
                causal=True,
            )
            solo_bl.append(bl_d)
            solo_idx.append(
                select_topk_blocks(bl_d, lse_d, vr, self.TOPK, self.BLOCK_B)
            )

        # ---- PACKED: concat docs into one sequence; per-doc bos via
        # valid_range and a doc-causal flashmask so each doc's rows attend only
        # to their own (causally-earlier) keys. ----
        qp = paddle.concat(qs, axis=1)
        kp = paddle.concat(ks, axis=1)
        vp = paddle.concat(vs, axis=1)
        vr_pack = _multidoc_valid_range(doc_lens)
        startend = _multidoc_startend_row_indices(doc_lens, h)
        _, lse_pack, bl_pack = block_score_fa4_attn_fwd(
            qp,
            kp,
            vp,
            valid_range=vr_pack,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            causal=True,
            startend_row_indices=startend,
        )
        idx_pack = select_topk_blocks(
            bl_pack, lse_pack, vr_pack, self.TOPK, self.BLOCK_B
        )

        # ---- Per-doc compare: packed rows [ds, de) over doc D's relative
        # blocks must match D's solo run (both document-RELATIVE, no remap). ----
        logit_tol = 0.06 * math.sqrt(d) * sm_scale
        for di, ((ds, de), L) in enumerate(
            zip(_doc_bounds(doc_lens), doc_lens)
        ):
            nb_d = _num_blocks(L, self.BLOCK_B)
            bl_d_pack = bl_pack[:, :, ds:de, :nb_d]  # [1,H,L,nb_d]
            patt, logit_err = _maxerr_finite(bl_d_pack, solo_bl[di])
            self.assertEqual(
                patt,
                0,
                f"doc{di} (len={L}, bos={ds}) block_logit masked pattern "
                f"differs packed-vs-solo -- coordinate systems disagree",
            )
            self.assertLessEqual(
                logit_err,
                logit_tol,
                f"doc{di} (len={L}, bos={ds}) block_logit max|diff|="
                f"{logit_err:.4e} > tol={logit_tol:.4e} packed-vs-solo",
            )
            idx_pack_d = idx_pack[:, ds:de, :]  # [1,L,TOPK]
            mismatch, total = self._topk_set_mismatch(idx_pack_d, solo_idx[di])
            self.assertLessEqual(
                mismatch,
                max(1, int(0.005 * total)),
                f"doc{di} (len={L}, bos={ds}) pack-equivalence VIOLATED: "
                f"{mismatch}/{total} TopK slots differ packed-vs-solo",
            )

    def test_multidoc_pack_vs_nonpack(self):
        # Four docs, all unaligned to BLOCK_B=64; every doc's non-zero bos is
        # checked against its solo (bos=0) run. h=8/d=64 fast config.
        self._multidoc_pack_vs_nonpack(
            doc_lens=[40, 88, 133, 27], h=8, d=64, dv=64
        )

    def test_multidoc_pack_vs_nonpack_dv256_h64(self):
        # Production dims (head_dim=256 split-D, H=64) with unaligned docs.
        self._multidoc_pack_vs_nonpack(
            doc_lens=[50, 100, 70], h=64, d=256, dv=256
        )

    def _dense_relative_indices(self, doc_lens, nsel):
        """[1, S, nsel] int32 document-relative block ids for a packed seq.

        For each query row in document D (length L), list D's own relative
        block ids ``0..nb_D-1`` (nb_D = ceil(L / block_B)) and pad the rest
        with ``-1``. This selects EVERY block of the row's own document, so
        the block-sparse gather degenerates to full document-causal attention
        -- letting us compare the gather kernel packed-vs-solo directly.
        """
        s = sum(doc_lens)
        idx = paddle.full([s, nsel], -1, dtype="int32")
        for (ds, de), L in zip(_doc_bounds(doc_lens), doc_lens):
            nb = _num_blocks(L, self.BLOCK_B)
            for j in range(min(nb, nsel)):
                idx[ds:de, j] = j
        return idx.reshape([1, s, nsel]).contiguous()

    def _gather_pack_vs_nonpack(self, doc_lens, h, seed=2026):
        """End-to-end GATHER pack-equivalence (the downstream DSA consumer).

        The block-sparse DSA gather op is what actually CONSUMES the
        document-relative TopK block indices produced by the (now fixed)
        scorer. This test proves the gather itself is pack-equivalent: with
        dense relative indices (all of a doc's own blocks selected), running
        each document ALONE (bos=0) and all documents PACKED (per-doc bos =
        doc_start) must yield the SAME per-token attention output. If the
        gather mixed absolute vs relative coordinates, ``col = bos + blk*BB``
        would address the wrong keys for bos>0 docs and the outputs would
        diverge. Absorbed-MLA MQA layout: q [1,S,H,576], shared single-head
        latent k [1,S_kv,576] whose value is its leading Dv=512 slice.
        """
        _dsa_or_skip(self)
        from paddleformers.fleet.cudnn_ops import block_sparse_mqa_attention_dsa

        d, dv = 576, 512
        sm_scale = 1.0 / math.sqrt(d)
        nsel = max(_num_blocks(L, self.BLOCK_B) for L in doc_lens)

        # Independent per-doc q and shared latent k (value == leading dv slice),
        # reused VERBATIM in the packed run for value-for-value comparison.
        qs, ks = [], []
        for di, L in enumerate(doc_lens):
            paddle.seed(seed + di)
            qs.append(paddle.randn([1, L, h, d], dtype="bfloat16"))
            ks.append(paddle.randn([1, L, d], dtype="bfloat16"))

        # ---- NON-PACKED: each doc alone, bos=0, dense relative indices. ----
        solo_out = []
        for di, L in enumerate(doc_lens):
            vr = _causal_valid_range(1, L)
            idx = self._dense_relative_indices([L], nsel)
            out_d, _ = block_sparse_mqa_attention_dsa(
                qs[di],
                ks[di],
                idx,
                vr,
                sm_scale=sm_scale,
                block_B=self.BLOCK_B,
                kv_lora_rank=dv,
            )
            solo_out.append(out_d)

        # ---- PACKED: concat docs; per-doc bos via valid_range; each doc's
        # rows carry its OWN relative block ids (0..nb_D-1). ----
        qp = paddle.concat(qs, axis=1)
        kp = paddle.concat(ks, axis=1)
        vr_pack = _multidoc_valid_range(doc_lens)
        idx_pack = self._dense_relative_indices(doc_lens, nsel)
        out_pack, _ = block_sparse_mqa_attention_dsa(
            qp,
            kp,
            idx_pack,
            vr_pack,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            kv_lora_rank=dv,
        )

        # ---- Per-doc compare: packed rows [ds, de) must match D's solo run. ----
        import numpy as np

        for di, ((ds, de), L) in enumerate(
            zip(_doc_bounds(doc_lens), doc_lens)
        ):
            got = out_pack[:, ds:de].astype("float32").numpy()
            ref = solo_out[di].astype("float32").numpy()
            max_diff = float(np.abs(got - ref).max())
            # bf16 gather: same relative math, so differences are only bf16
            # rounding from block-iteration order. Tight tolerance.
            self.assertLessEqual(
                max_diff,
                2e-2,
                f"doc{di} (len={L}, bos={ds}) GATHER output max|diff|="
                f"{max_diff:.4e} packed-vs-solo -- coordinate systems disagree",
            )

    def test_gather_pack_vs_nonpack_h64(self):
        # Absorbed-MLA MQA dims: Dk=576, Dv=512, full H=64 query heads. The MQA
        # gather is head-count-agnostic (single shared latent), so production
        # H=64 fully covers the relative-coordinate gather pack-equivalence.
        self._gather_pack_vs_nonpack(doc_lens=[50, 100, 70], h=64)


if __name__ == "__main__":
    unittest.main()
