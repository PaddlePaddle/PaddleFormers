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

"""End-to-end HySparse precision test over the WHOLE forward flow's three
attention structures, each checked against a pure scattered-op (eager fp32)
reference, plus their combination:

1. **FULL-layer block scoring** -- FA4-fused ``block_score_fa4_attn_fwd`` +
   ``select_topk_blocks`` -> per-query document-relative TopK block indices.
2. **SWA main path** -- ``sliding_window_mqa_attention`` (windowed MQA over the
   Dk=576 / Dv=512 absorbed-MLA shared latent).
3. **Block-sparse gather branch** -- ``block_sparse_mqa_attention_dsa``
   (FlashMLA sparse fwd + cuDNN DSA bwd), gathering exactly the blocks selected
   in (1).

The consumer (``MLASelfAttention``) sums the SWA main and sparse-gather branch
outputs (after a shared value-absorb + gating that are identical linear ops on
both paths and so do not affect attention-precision alignment). This test
therefore also checks ``SWA_out + sparse_out`` against the summed eager
references.

To isolate kernel precision from TopK tie-flips, the KERNEL-selected block
indices are fed into BOTH the kernel gather and the eager gather reference.

Single-document: tight alignment (cosine > 0.99). Multi-document packed:
acceptable error (cosine > 0.98) -- bf16 rounding + block-iteration order only.

Runs only where the DSA backend is available (SM100 + FlashMLA + cuDNN
frontend); skips otherwise.
"""

import math
import unittest

import paddle

_NEG_INF = float("-inf")
_DK = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)
_DV = 512  # value == leading kv_lora_rank slice of the shared latent


def _dsa_or_skip(testcase):
    """Skip unless the DSA gather backend (FlashMLA + cuDNN, SM100) can run."""
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        testcase.skipTest(
            f"HySparse whole-flow requires SM 10.x (Blackwell); got SM {major}.x"
        )
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
    bounds, off = [], 0
    for L in doc_lens:
        bounds.append((off, off + L))
        off += L
    return bounds


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


def _doc_valid_range(doc_lens):
    """Document-anchored causal [1, S, 2]: bos=doc_start, eos=pos+1."""
    s = sum(doc_lens)
    bos = paddle.zeros([s], dtype="int32")
    eos = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        idx = paddle.arange(ds, de, dtype="int32")
        bos[ds:de] = ds
        eos[ds:de] = idx + 1
    return paddle.stack([bos, eos], axis=-1).reshape([1, s, 2]).contiguous()


def _window_valid_range(doc_lens, window):
    """Windowed causal [1, S, 2]: bos=max(doc_start, pos-W+1), eos=pos+1."""
    s = sum(doc_lens)
    bos = paddle.zeros([s], dtype="int32")
    eos = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        for pos in range(ds, de):
            bos[pos] = max(ds, pos - window + 1)
            eos[pos] = pos + 1
    return paddle.stack([bos, eos], axis=-1).reshape([1, s, 2]).contiguous()


def _multidoc_startend_row_indices(doc_lens, h):
    """flashmask LTS [1, H, S, 1]: mask query rows >= doc_end per key col."""
    s = sum(doc_lens)
    lts = paddle.zeros([s], dtype="int32")
    for ds, de in _doc_bounds(doc_lens):
        lts[ds:de] = de
    return lts.reshape([1, 1, s, 1]).expand([1, h, s, 1]).contiguous()


def _range_allow_mask(valid_range, s_kv):
    """Bool [1, S, S_kv]: col allowed iff bos <= col < eos (windowed/doc range)."""
    bos = valid_range[..., 0:1].astype("int64")  # [1,S,1]
    eos = valid_range[..., 1:2].astype("int64")  # [1,S,1]
    cols = paddle.arange(s_kv, dtype="int64").reshape([1, 1, s_kv])
    return (cols >= bos) & (cols < eos)  # [1,S,S_kv]


def _block_allow_mask(indices, valid_range, s_kv, block_B):
    """Bool [1, S, S_kv]: col allowed iff in a selected block AND in [bos, eos).

    Mirrors the DSA op's block->token expansion + doc/causal masking exactly.
    """
    import numpy as np

    idx = indices.numpy()
    vr = valid_range.numpy()
    b, s, _ = idx.shape
    allow = np.zeros([b, s, s_kv], dtype=bool)
    for bi in range(b):
        for i in range(s):
            bos, eos = int(vr[bi, i, 0]), int(vr[bi, i, 1])
            for blk in idx[bi, i]:
                if blk < 0:
                    continue
                c0 = bos + int(blk) * block_B
                for col in range(c0, min(c0 + block_B, eos)):
                    if 0 <= col < s_kv:
                        allow[bi, i, col] = True
    return paddle.to_tensor(allow)


def _eager_masked_mqa(q, k, v, allow, sm_scale):
    """Differentiable-free dense masked MQA reference (fp32).

    q [1,S,H,Dk], k [1,S_kv,Dk], v [1,S_kv,Dv], allow [1,S,S_kv] bool.
    Returns out [1,S,H,Dv] fp32; rows with no allowed key emit 0.
    """
    qf = q.astype("float32")
    kf = k.astype("float32")
    vf = v.astype("float32")
    logits = paddle.einsum("bshd,bcd->bshc", qf, kf) * sm_scale  # [1,S,H,Skv]
    mask = allow.unsqueeze(2)  # [1,S,1,Skv]
    neg = paddle.full_like(logits, _NEG_INF)
    logits = paddle.where(mask, logits, neg)
    row_has = allow.any(axis=-1)  # [1,S]
    m = logits.max(axis=-1, keepdim=True)
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    p = paddle.exp(logits - m)
    denom = p.sum(axis=-1, keepdim=True)
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = p / denom
    out = paddle.einsum("bshc,bcv->bshv", p, vf)  # [1,S,H,Dv]
    out = out * row_has.astype("float32").unsqueeze(-1).unsqueeze(-1)
    return out


def _eager_block_logit_lse(q, k, valid_range, sm_scale, block_B):
    """Document-RELATIVE scatter (eq.3) reference for the FA4 MHA block scorer.

    q, k [1,S,H,d] decompressed MHA (independent heads, head_dim <= 256 as FA4
    requires). Block j of row r covers relative key columns
    ``[bos_r + j*block_B, bos_r + (j+1)*block_B)`` restricted to causally-valid
    same-document keys (``bos_r <= c < eos_r``). Returns
    (block_logit [1,H,S,nb] fp32 scaled per-block max logit, lse [1,S,H] fp32).
    """
    b, s, h, d = q.shape
    sk = k.shape[1]
    nb = _num_blocks(sk, block_B)
    qf = q.astype("float32").transpose([0, 2, 1, 3])  # [1,H,S,d]
    kf = k.astype("float32").transpose([0, 2, 1, 3])  # [1,H,Sk,d]
    logits = paddle.matmul(qf, kf, transpose_y=True) * sm_scale  # [1,H,S,Sk]

    bos = valid_range[0, :, 0].astype("int64").reshape([s, 1])  # [S,1]
    eos = valid_range[0, :, 1].astype("int64").reshape([s, 1])  # [S,1]
    cols = paddle.arange(sk, dtype="int64").reshape([1, sk])
    valid = (cols >= bos) & (cols < eos)  # [S, Sk]
    rel_block = (cols - bos) // block_B  # [S, Sk] (meaningful where valid)

    vmask = valid.reshape([1, 1, s, sk])
    masked_all = paddle.where(vmask, logits, paddle.full_like(logits, _NEG_INF))
    lse_bhs = paddle.logsumexp(masked_all, axis=-1)  # [1,H,S]
    lse = lse_bhs.transpose([0, 2, 1]).contiguous()  # [1,S,H]

    block_logit = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    for j in range(nb):
        cm = (valid & (rel_block == j)).reshape([1, 1, s, sk])
        mj = paddle.where(cm, logits, paddle.full_like(logits, _NEG_INF))
        block_logit[:, :, :, j] = mj.max(axis=-1)
    return block_logit, lse


class TestHySparseWholeFlowPrecision(unittest.TestCase):
    BLOCK_B = 64
    TOPK = 4
    WINDOW = 128

    def _topk_set_mismatch(self, idx_a, idx_b):
        a = paddle.sort(idx_a.astype("int64"), axis=-1)
        b = paddle.sort(idx_b.astype("int64"), axis=-1)
        return int((a != b).astype("int32").sum().item()), a.numel().item()

    def _run_whole_flow(self, doc_lens, h, tol_cos, seed=2026):
        """Build the 3-structure flow and check every structure + their sum
        against a pure eager fp32 reference.

        The structures live in DIFFERENT layers and use DIFFERENT tensor dims,
        sharing only the document-relative block coordinates:

        * Structure 1 (full-layer scoring) runs on DECOMPRESSED MHA with head_dim
          ``d_score`` <= 256 (FA4 flash_mask caps head_dim at 256): independent
          query/key/value heads. It emits the document-relative TopK block
          indices consumed downstream.
        * Structures 2/3 (SWA main + block-sparse gather) run on the ABSORBED-MLA
          MQA latent: query [1,S,H,576], a single shared K/V head k [1,S,576]
          whose value is its leading Dv=512 slice. They consume the SAME block
          indices produced by structure 1 (block coordinates are layer-agnostic
          document-relative ids).
        """
        _dsa_or_skip(self)
        from paddleformers.fleet.cudnn_ops import block_sparse_mqa_attention_dsa
        from paddleformers.fleet.tilelang_ops.hysparse import (
            block_score_fa4_attn_fwd,
            select_topk_blocks,
            sliding_window_mqa_attention,
        )

        s = sum(doc_lens)
        multi = len(doc_lens) > 1
        paddle.seed(seed)

        doc_vr = _doc_valid_range(doc_lens)
        window_vr = _window_valid_range(doc_lens, self.WINDOW)
        startend = (
            _multidoc_startend_row_indices(doc_lens, h) if multi else None
        )

        # ---- Structure 1: FA4-fused block scoring + TopK selection. ----
        # Full layers score DECOMPRESSED MHA (independent heads, head_dim<=256).
        d_score, dv_score = 192, 128  # qk_nope+qk_rope / v_head_dim, <= 256.
        sm_score = 1.0 / math.sqrt(d_score)
        q_score = paddle.randn(
            [1, s, h, d_score], dtype="bfloat16"
        ).contiguous()
        k_score = paddle.randn(
            [1, s, h, d_score], dtype="bfloat16"
        ).contiguous()
        v_score = paddle.randn(
            [1, s, h, dv_score], dtype="bfloat16"
        ).contiguous()
        _, lse1, block_logit = block_score_fa4_attn_fwd(
            q_score,
            k_score,
            v_score,
            valid_range=doc_vr,
            sm_scale=sm_score,
            block_B=self.BLOCK_B,
            causal=True,
            startend_row_indices=startend,
        )
        block_indices = select_topk_blocks(
            block_logit, lse1, doc_vr, self.TOPK, self.BLOCK_B
        )
        # eager block-score selection over the SAME relative coordinates.
        bl_ref, lse_ref = _eager_block_logit_lse(
            q_score, k_score, doc_vr, sm_score, self.BLOCK_B
        )
        idx_ref = select_topk_blocks(
            bl_ref, lse_ref, doc_vr, self.TOPK, self.BLOCK_B
        )
        mism, total = self._topk_set_mismatch(block_indices, idx_ref)
        self.assertLessEqual(
            mism,
            max(1, int(0.01 * total)),
            f"[scoring] TopK indices differ from eager ref: {mism}/{total}",
        )

        # ---- Structures 2/3 run on the ABSORBED-MLA MQA latent. ----
        sm_scale = 1.0 / math.sqrt(_DK)
        query = paddle.randn([1, s, h, _DK], dtype="bfloat16").contiguous()
        shared_k = paddle.randn([1, s, _DK], dtype="bfloat16").contiguous()
        shared_v = shared_k[:, :, :_DV].contiguous()

        # ---- Structure 2: SWA main path (windowed MQA). ----
        swa_out, _ = sliding_window_mqa_attention(
            query,
            shared_k,
            shared_v,
            window_vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
        )  # [1,S,H,Dv]
        swa_allow = _range_allow_mask(window_vr, s)
        swa_ref = _eager_masked_mqa(
            query, shared_k, shared_v, swa_allow, sm_scale
        )
        cos_swa = _cos(swa_out, swa_ref)
        self.assertGreater(
            cos_swa, tol_cos, f"[SWA] cos={cos_swa:.6f} <= {tol_cos}"
        )

        # ---- Structure 3: block-sparse gather (DSA), KERNEL indices. ----
        sparse_out, _ = block_sparse_mqa_attention_dsa(
            query,
            shared_k,
            block_indices,
            doc_vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            kv_lora_rank=_DV,
        )
        sparse_out = sparse_out.reshape([1, s, h, _DV])
        # eager gather over the exact same kernel-selected blocks.
        sp_allow = _block_allow_mask(block_indices, doc_vr, s, self.BLOCK_B)
        sparse_ref = _eager_masked_mqa(
            query, shared_k, shared_v, sp_allow, sm_scale
        )
        cos_sp = _cos(sparse_out, sparse_ref)
        self.assertGreater(
            cos_sp, tol_cos, f"[sparse] cos={cos_sp:.6f} <= {tol_cos}"
        )

        # ---- Combined: SWA_out + sparse_out (the consumer's branch sum). ----
        total_kernel = swa_out + sparse_out
        total_ref = swa_ref + sparse_ref
        cos_tot = _cos(total_kernel, total_ref)
        self.assertGreater(
            cos_tot, tol_cos, f"[combined] cos={cos_tot:.6f} <= {tol_cos}"
        )
        print(
            f"\n[whole-flow docs={doc_lens} h={h}] "
            f"scoring TopK mism={mism}/{total} | "
            f"cos SWA={cos_swa:.6f} sparse={cos_sp:.6f} combined={cos_tot:.6f} "
            f"(tol={tol_cos})"
        )
        return cos_swa, cos_sp, cos_tot

    def test_whole_flow_single_doc(self):
        # One document (bos=0 everywhere): tight kernel-vs-eager alignment.
        self._run_whole_flow(doc_lens=[320], h=64, tol_cos=0.99)

    def test_whole_flow_multidoc(self):
        # Packed docs of arbitrary (NOT 64-aligned) lengths: acceptable error.
        # Every doc's non-zero bos exercises the document-relative coordinates
        # across all three structures at once.
        self._run_whole_flow(doc_lens=[40, 88, 133, 27], h=64, tol_cos=0.98)


if __name__ == "__main__":
    unittest.main()
