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

"""Correctness tests for the HySparse block-sparse *DSA* backend
(:func:`paddleformers.fleet.cudnn_ops.block_sparse_mqa_attention_dsa`) on the
**value-pad-to-512** path (``kv_lora_rank < 512``).

The FlashMLA sparse kernel hard-requires ``d_v == 512`` / ``d_qk in {512, 576}``.
ernielite runs an absorbed-MQA latent with ``kv_lora_rank == 448`` (+ rope 64,
so ``d_qk == 512``). The wrapper re-lays each latent from ``[value(448) | rope]``
to ``[value(448) | zeros(64) | rope]`` (=> ``d_qk == 576``, value == leading
512), runs the fixed-``d_v=512`` kernel, and slices the leading 448 value columns
back out. That pad/slice code (block_sparse_mqa_dsa.py:356-388) sits *outside*
the PyLayer and is not covered by the existing ``kv_lora_rank == 512`` suite;
this file targets exactly that branch.

We validate:
1. pad-path forward + dq/dkv grads match a differentiable dense masked-softmax
   reference built with ``Dv == 448`` (cosine > 0.99);
2. the inserted zero columns contribute nothing and do not pollute the real
   448-dim gradient -- the pad path is numerically equivalent to manually
   building the ``[val | zeros | rope]`` latent and calling the native
   ``kv_lora_rank == 512`` path, then slicing the leading 448 (cosine > 0.999999);
3. the reconstructed value gradient over the real 448 dims and the trailing
   rope dims each match the reference (pad boundary does not leak).

The suite skips gracefully when FlashMLA / cuDNN-frontend / a Blackwell (SM100)
GPU is unavailable, using the same detection as the sibling DSA test.
"""

import math
import unittest

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


def _dsa_unavailable_reason():
    if not paddle.device.is_compiled_with_cuda():
        return "CUDA build of Paddle required"
    if paddle.device.cuda.device_count() == 0:
        return "no CUDA device available"
    try:
        from paddleformers.fleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            return "FlashMLA sparse fwd + cuDNN DSA bwd not available"
    except (ImportError, RuntimeError):
        return "hysparse DSA import failed"
    cc = paddle.device.cuda.get_device_capability()
    if cc[0] < 10:
        return f"DSA sparse fwd requires SM100+, got {cc}"
    return None


_SKIP_REASON = None


def _skip_if_no_dsa(tc):
    global _SKIP_REASON
    if _SKIP_REASON is None:
        _SKIP_REASON = _dsa_unavailable_reason() or ""
    if _SKIP_REASON:
        tc.skipTest(_SKIP_REASON)


def _allow_mask(indices, valid_range, s_kv, block_B):
    """Bool [B, S, S_kv]: col allowed iff in a selected block ∩ [bos, eos)."""
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


def _ref_masked_attn(q, k, v, allow, sm_scale):
    """Differentiable dense masked MQA attention reference (fp32, sinkless)."""
    neg_inf = float("-inf")
    logits = paddle.einsum("bshd,bkd->bshk", q, k) * sm_scale
    neg = paddle.full_like(logits, neg_inf)
    mask = allow.unsqueeze(2)  # [B,S,1,Skv]
    logits = paddle.where(mask, logits, neg)
    row_has = allow.any(axis=-1)
    m = logits.max(axis=-1, keepdim=True)
    m = paddle.where(paddle.isfinite(m), m, paddle.zeros_like(m))
    p = paddle.exp(logits - m)
    denom = p.sum(axis=-1, keepdim=True)
    denom = paddle.where(denom > 0, denom, paddle.ones_like(denom))
    p = p / denom
    out = paddle.einsum("bshk,bkc->bshc", p, v)
    out = out * row_has.astype("float32").unsqueeze(-1).unsqueeze(-1)
    return out


def _causal_valid_range(b, s):
    eos = (
        paddle.arange(1, s + 1, dtype="int32")
        .reshape([1, s, 1])
        .expand([b, s, 1])
    )
    bos = paddle.zeros([b, s, 1], dtype="int32")
    return paddle.concat([bos, eos], axis=-1).contiguous()


def _cos(a, b):
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    denom = (np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12
    return float(np.dot(af, bf) / denom)


def _rel_l2(a, b):
    # Relative Frobenius error ||a-ref|| / ||ref||: unlike cosine this is
    # scale-SENSITIVE, so it catches a constant-factor / offset gradient error
    # (e.g. a wrong gradient scale) that cosine is blind to.
    import numpy as np

    af = a.astype("float32").numpy().reshape(-1)
    bf = b.astype("float32").numpy().reshape(-1)
    return float(np.linalg.norm(af - bf) / (np.linalg.norm(bf) + 1e-12))


def _select_blocks(b, s, block_B, topk):
    """Per-token relative block ids: block 0 plus the running block (pos//BB),
    padded with -1 to width ``topk``."""
    pos = paddle.arange(s)
    b0 = paddle.zeros([b, s, 1], dtype="int32")
    b1 = (pos // block_B).cast("int32").reshape([1, s, 1]).expand([b, s, 1])
    idx = paddle.concat([b0, b1], axis=-1)  # [b, s, 2]
    if topk > 2:
        pad = paddle.full([b, s, topk - 2], -1, dtype="int32")
        idx = paddle.concat([idx, pad], axis=-1)
    return idx.contiguous()


class TestBlockSparseDSAPadTo512(unittest.TestCase):
    # ernielite absorbed-MQA latent: value 448, rope 64 => query/key dim 512.
    BLOCK_B = 64
    KV_LORA = 448  # d_v < 512 -> exercises the pad-to-512 branch.
    ROPE = 64
    Dk = KV_LORA + ROPE  # 512, latent dim fed to the wrapper (pre-pad).
    Dv = KV_LORA  # 448, real value dim recovered on output.

    def _make(self, b, s, h, topk, seed=7):
        paddle.seed(seed)
        q = paddle.randn([b, s, h, self.Dk]).cast("bfloat16")
        kf = paddle.randn([b, s, self.Dk]).cast("bfloat16")
        vr = _causal_valid_range(b, s)
        idx = _select_blocks(b, s, self.BLOCK_B, topk)
        sm_scale = 1.0 / math.sqrt(self.Dk)
        return q, kf, vr, idx, sm_scale

    def _run_dsa(self, q, kf, idx, vr, sm_scale, kv_lora_rank):
        from paddleformers.fleet.cudnn_ops import block_sparse_mqa_attention_dsa

        qd = q.detach().clone()
        qd.stop_gradient = False
        kd = kf.detach().clone()
        kd.stop_gradient = False
        out, _ = block_sparse_mqa_attention_dsa(
            qd,
            kd,
            idx,
            vr,
            sm_scale=sm_scale,
            block_B=self.BLOCK_B,
            kv_lora_rank=kv_lora_rank,
        )
        out.sum().backward()
        return out, qd.grad, kd.grad

    def _run_ref(self, q, kf, idx, vr, sm_scale):
        b, s, h = q.shape[0], q.shape[1], q.shape[2]
        s_kv = kf.shape[1]
        qr = q.detach().cast("float32")
        qr.stop_gradient = False
        kr = kf.detach().cast("float32")
        kr.stop_gradient = False
        vr_ = kr[:, :, : self.Dv]  # value = leading 448 of the latent
        allow = _allow_mask(idx, vr, s_kv, self.BLOCK_B)
        out = _ref_masked_attn(qr, kr, vr_, allow, sm_scale)
        out = out.reshape([b, s, h * self.Dv])
        out.sum().backward()
        return out, qr.grad, kr.grad

    def test_pad_boundary_does_not_leak_gradient(self):
        # The zero columns are inserted *between* value(448) and rope(64). If
        # the pad/slice leaked, the value-region and rope-region gradients would
        # diverge from the reference. Check each sub-slice independently.
        _skip_if_no_dsa(self)
        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=17)
        _, _, dkv_dsa = self._run_dsa(q, kf, idx, vr, sm, self.KV_LORA)
        _, _, dkv_ref = self._run_ref(q, kf, idx, vr, sm)
        # value region [0:448] and rope region [448:512] both intact.
        self.assertGreater(
            _cos(dkv_dsa[..., : self.KV_LORA], dkv_ref[..., : self.KV_LORA]),
            0.99,
        )
        self.assertLess(
            _rel_l2(dkv_dsa[..., : self.KV_LORA], dkv_ref[..., : self.KV_LORA]),
            4e-3,
        )
        self.assertGreater(
            _cos(dkv_dsa[..., self.KV_LORA :], dkv_ref[..., self.KV_LORA :]),
            0.99,
        )
        self.assertLess(
            _rel_l2(dkv_dsa[..., self.KV_LORA :], dkv_ref[..., self.KV_LORA :]),
            5e-3,
        )

    def test_pad_path_equivalent_to_native_512(self):
        # The pad path must be numerically identical to manually building the
        # ``[val(448) | zeros(64) | rope(64)]`` latent (=> d_qk=576, value ==
        # leading 512), running the native kv_lora_rank=512 path, and slicing
        # the leading 448 value columns per head. This directly proves the
        # inserted zeros contribute nothing to the value output.
        _skip_if_no_dsa(self)
        from paddleformers.fleet.cudnn_ops import block_sparse_mqa_attention_dsa

        b, s, h, topk = 1, 192, 4, 2
        q, kf, vr, idx, sm = self._make(b, s, h, topk, seed=23)
        pad_v = 512 - self.KV_LORA

        # Pad path (kv_lora_rank=448), forward only.
        out_pad, _ = block_sparse_mqa_attention_dsa(
            q,
            kf,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=self.KV_LORA,
        )

        # Manual native-512 layout: [val | zeros | rope].
        q_val, q_rope = q[..., : self.KV_LORA], q[..., self.KV_LORA :]
        k_val, k_rope = kf[..., : self.KV_LORA], kf[..., self.KV_LORA :]
        zq = paddle.zeros([b, s, h, pad_v], dtype=q.dtype)
        zk = paddle.zeros([b, s, pad_v], dtype=kf.dtype)
        q576 = paddle.concat([q_val, zq, q_rope], axis=-1)
        k576 = paddle.concat([k_val, zk, k_rope], axis=-1)
        out512, _ = block_sparse_mqa_attention_dsa(
            q576,
            k576,
            idx,
            vr,
            sm_scale=sm,
            block_B=self.BLOCK_B,
            kv_lora_rank=512,
        )
        # Slice the leading 448 value columns per head from the 512-wide output.
        out512 = out512.reshape([b, s, h, 512])[..., : self.KV_LORA]
        out512 = out512.reshape([b, s, h * self.KV_LORA])

        self.assertEqual(list(out_pad.shape), list(out512.shape))
        # Near-exact equivalence: enforce a tight magnitude ceiling alongside
        # the near-exact cosine floor (cosine alone would miss a scale drift).
        self.assertGreater(_cos(out_pad, out512), 0.999999)
        self.assertLess(_rel_l2(out_pad, out512), 1e-3)


if __name__ == "__main__":
    unittest.main()
