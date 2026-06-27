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

"""Tests for cuDNN-frontend CSA indexer backward and related integration points.

Covers:
- csa_indexer_bwd_cudnn.py (wrapper + _to_bf16 helper)
- cudnn_ops/__init__.py (lazy __getattr__)
- csa_attention.py (TileLangCSAIndexerLossAutoScaler cudnn backward branch)
- transformer_config.py (csa_indexer_backend field + validation)
"""

import unittest

import paddle
import paddle.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_close(actual, expected, rtol, atol, name):
    a = actual.cast("float32")
    e = expected.cast("float32")
    if not paddle.allclose(a, e, rtol=rtol, atol=atol).item():
        diff = (a - e).abs()
        denom = e.abs().clip(min=1e-12)
        raise AssertionError(
            f"{name} mismatch: max abs={diff.max().item():.4e} "
            f"max rel={(diff / denom).max().item():.4e}"
        )


def _make_inputs(b, sq, sk, h, d, topk, seed=2026):
    paddle.seed(seed)
    index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
    index_k = paddle.randn([b, sk, d]).astype("bfloat16")
    weights = paddle.randn([b, sq, h]).astype("bfloat16")
    target = paddle.nn.functional.softmax(
        paddle.randn([b, sq, topk]).astype("float32"), axis=-1
    )
    topk_probs = paddle.nn.functional.softmax(
        paddle.randn([b, sq, topk]).astype("float32"), axis=-1
    )
    topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
    mask = paddle.rand([b, sq, topk]) < 0.05
    topk_indices = paddle.where(
        mask, paddle.full_like(topk_indices, -1), topk_indices
    )
    return index_q, weights, index_k, target, topk_probs, topk_indices


def _ref_csa_indexer_backward(
    index_q,
    weights,
    index_k_comp,
    topk_indices,
    target,
    topk_probs,
    loss_coeff,
    grad_loss,
):
    """Pure-paddle small-op reference for the cuDNN indexer backward.

    Reproduces the forward score computation from _ref_csa_indexer_topk
    (test_tilelang_csa_indexer.py:93), then uses paddle.autograd to get
    exact gradients. This is the mathematical definition of the backward.
    """
    b, sq, h, d = index_q.shape
    sk = index_k_comp.shape[1]
    topk = topk_indices.shape[-1]

    q = index_q.cast("float32").detach()
    q.stop_gradient = False
    k = index_k_comp.cast("float32").detach()
    k.stop_gradient = False
    w = weights.cast("float32").detach()
    w.stop_gradient = False

    # Forward: scores [B, S, S_comp]
    # cuDNN kernel order: einsum → scale → ReLU → weighted sum
    scores = paddle.einsum("bshd,btd->bsht", q, k)
    scores = scores * (d**-0.5)
    scores = F.relu(scores)
    scores = (scores * w.unsqueeze(-1)).sum(axis=2)  # [B, S, S_comp]

    # Gather scores at topk positions, mask invalid (-1) indices
    idx = topk_indices.cast("int64")
    valid_mask = topk_indices >= 0  # [B, S, topk]
    safe_idx = paddle.where(valid_mask, idx, paddle.zeros_like(idx))
    topk_scores = paddle.take_along_axis(
        scores, safe_idx, axis=-1
    )  # [B, S, topk]
    topk_scores = paddle.where(
        valid_mask, topk_scores, paddle.zeros_like(topk_scores)
    )

    # cuDNN internal loss: sum((topk_probs - target) * topk_scores) * scale * grad_loss
    scale = loss_coeff / float(b * sq)
    grad_scores_weight = (topk_probs - target) * scale
    if grad_loss is not None:
        grad_scores_weight = grad_scores_weight * grad_loss
    loss = (grad_scores_weight.detach() * topk_scores).sum()

    loss.backward()
    return q.grad, w.grad, k.grad


# ---------------------------------------------------------------------------
# Tests: csa_indexer_bwd_cudnn.py (_to_bf16 + csa_indexer_bwd wrapper)
# ---------------------------------------------------------------------------


class TestCudnnHelpers(unittest.TestCase):
    """Cover _to_bf16 helper in csa_indexer_bwd_cudnn.py."""

    def setUp(self):
        from paddleformers.fleet.cudnn_ops.indexer.csa_indexer_bwd_cudnn import _to_bf16

        self._to_bf16 = _to_bf16

    def test_to_bf16_noop(self):
        t = paddle.zeros([2, 3], dtype="bfloat16")
        out = self._to_bf16(t)
        self.assertIs(out, t)

    def test_to_bf16_cast(self):
        t = paddle.zeros([2, 3], dtype="float32")
        out = self._to_bf16(t)
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertIsNot(out, t)

    def test_to_bf16_fp16(self):
        t = paddle.zeros([2, 3], dtype="float16")
        out = self._to_bf16(t)
        self.assertEqual(out.dtype, paddle.bfloat16)


@unittest.skipIf(
    not paddle.device.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] != 10,
    "CudnnCsaIndexerBwd requires Blackwell GPU (SM100)",
)
class TestCudnnCsaIndexerBwd(unittest.TestCase):
    """Cover csa_indexer_bwd wrapper: dtype casts, grad_loss handling, clone semantics."""

    def setUp(self):
        from paddleformers.fleet.cudnn_ops import csa_indexer_bwd

        self.cudnn_fn = csa_indexer_bwd

    def test_basic_output_shapes(self):
        """Basic call returns correct shapes and dtypes."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))
        self.assertEqual(grad_w.shape, list(weights.shape))
        self.assertEqual(grad_k.shape, list(index_k.shape))
        self.assertEqual(grad_q.dtype, paddle.bfloat16)
        self.assertEqual(grad_w.dtype, paddle.bfloat16)
        self.assertEqual(grad_k.dtype, paddle.bfloat16)

    def test_grad_loss_none_equals_one(self):
        """grad_loss=None should be treated as scalar 1.0."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=11)
        )
        grad_q_none, grad_w_none, grad_k_none = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=None,
        )
        grad_q_one, grad_w_one, grad_k_one = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(grad_q_none, grad_q_one, 1e-3, 1e-3, "d_index_q")
        _assert_close(grad_w_none, grad_w_one, 1e-3, 1e-3, "d_weights")
        _assert_close(grad_k_none, grad_k_one, 1e-3, 1e-3, "d_index_k")

    def test_grad_loss_bf16_cast(self):
        """grad_loss in non-fp32 dtype should be cast to fp32 internally."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=12)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="bfloat16"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_fp32_inputs_cast_and_grads_restored(self):
        """fp32 inputs should be cast to bf16 internally; grads restored to fp32."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=13)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q.cast("float32"),
            weights.cast("float32"),
            index_k.cast("float32"),
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.dtype, paddle.float32)
        self.assertEqual(grad_w.dtype, paddle.float32)
        self.assertEqual(grad_k.dtype, paddle.float32)

    def test_topk_indices_int64_cast(self):
        """int64 topk_indices should be cast to int32 internally."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=14)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices.cast("int64"),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_target_predict_bf16_cast(self):
        """bf16 target/topk_probs should be cast to fp32 (new buffer)."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=15)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target.cast("bfloat16"),
            topk_probs.cast("bfloat16"),
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_custom_block_I(self):
        """block_I=64 covers the non-default tile size path."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=16)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
            block_I=64,
        )
        self.assertEqual(grad_q.shape, list(index_q.shape))

    def test_saved_tensors_not_mutated(self):
        """Wrapper must clone target/topk_probs; cuDNN overwrites in place."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=17)
        )
        target_before = target.clone()
        topk_probs_before = topk_probs.clone()
        self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        self.assertTrue(paddle.allclose(target, target_before).item())
        self.assertTrue(paddle.allclose(topk_probs, topk_probs_before).item())

    def test_grad_loss_scales_output(self):
        """grad_loss=2.0 should produce ~2x the gradients of grad_loss=1.0."""
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=18)
        )
        grad_q_1, _, _ = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_2, _, _ = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(2.0, dtype="float32"),
        )
        _assert_close(
            grad_q_2, grad_q_1 * 2, rtol=1e-2, atol=1e-3, name="2x scale"
        )

    def test_parity_against_reference(self):
        """Numerical parity: cuDNN backward vs pure-paddle autograd reference.

        The reference uses the same forward math as _ref_csa_indexer_topk
        (einsum → relu → weighted sum → scale) and paddle.autograd for the
        backward. This validates that the cuDNN kernel computes the correct
        gradients, not just that it runs without error.

        Tolerance: bf16 accumulation + different reduction order → rtol/atol=5e-2.
        """
        b, sq, sk, h, d, topk = 1, 1024, 256, 64, 128, 128
        loss_coeff = 0.01
        grad_loss_val = 1.0

        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=42)
        )

        # cuDNN result
        grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(grad_loss_val, dtype="float32"),
        )

        # Pure-paddle reference
        grad_q_ref, grad_w_ref, grad_k_ref = _ref_csa_indexer_backward(
            index_q,
            weights,
            index_k,
            topk_indices,
            target,
            topk_probs,
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(grad_loss_val, dtype="float32"),
        )

        _assert_close(
            grad_q_cudnn, grad_q_ref, rtol=5e-2, atol=5e-2, name="d_index_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_ref, rtol=5e-2, atol=5e-2, name="d_weights"
        )
        _assert_close(
            grad_k_cudnn,
            grad_k_ref,
            rtol=5e-2,
            atol=5e-2,
            name="d_index_k_comp",
        )

    def test_parity_small_sq(self):
        """Parity on a small sq for fast debugging."""
        b, sq, sk, h, d, topk = 1, 16, 128, 64, 128, 128
        loss_coeff = 0.1
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=100)
        )
        grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_ref, grad_w_ref, grad_k_ref = _ref_csa_indexer_backward(
            index_q,
            weights,
            index_k,
            topk_indices,
            target,
            topk_probs,
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(
            grad_q_cudnn, grad_q_ref, rtol=5e-2, atol=5e-2, name="small d_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_ref, rtol=5e-2, atol=5e-2, name="small d_w"
        )
        _assert_close(
            grad_k_cudnn, grad_k_ref, rtol=5e-2, atol=5e-2, name="small d_k"
        )

    def test_parity_topk_equals_sk(self):
        """Parity when topk == sk (full coverage, no -1 indices)."""
        b, sq, sk, h, d, topk = 1, 64, 128, 64, 128, 128
        loss_coeff = 0.05
        paddle.seed(200)
        index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([b, sk, d]).astype("bfloat16")
        weights = paddle.randn([b, sq, h]).astype("bfloat16")
        target = F.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )
        topk_probs = F.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )
        # All indices valid (no -1), cover full sk range
        topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")

        grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_ref, grad_w_ref, grad_k_ref = _ref_csa_indexer_backward(
            index_q,
            weights,
            index_k,
            topk_indices,
            target,
            topk_probs,
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(
            grad_q_cudnn, grad_q_ref, rtol=5e-2, atol=5e-2, name="full d_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_ref, rtol=5e-2, atol=5e-2, name="full d_w"
        )
        _assert_close(
            grad_k_cudnn, grad_k_ref, rtol=5e-2, atol=5e-2, name="full d_k"
        )

    def test_parity_large_loss_coeff(self):
        """Parity with a larger loss_coeff to exercise scaling path."""
        b, sq, sk, h, d, topk = 1, 128, 256, 64, 128, 128
        loss_coeff = 1.0
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=300)
        )
        grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_ref, grad_w_ref, grad_k_ref = _ref_csa_indexer_backward(
            index_q,
            weights,
            index_k,
            topk_indices,
            target,
            topk_probs,
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(
            grad_q_cudnn, grad_q_ref, rtol=5e-2, atol=5e-2, name="lcoeff d_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_ref, rtol=5e-2, atol=5e-2, name="lcoeff d_w"
        )
        _assert_close(
            grad_k_cudnn, grad_k_ref, rtol=5e-2, atol=5e-2, name="lcoeff d_k"
        )

    def test_invalid_indices_zero_contribution(self):
        """Positions with topk_indices == -1 should not contribute to gradients."""
        b, sq, sk, h, d, topk = 1, 64, 256, 64, 128, 128
        paddle.seed(400)
        index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
        index_k = paddle.randn([b, sk, d]).astype("bfloat16")
        weights = paddle.randn([b, sq, h]).astype("bfloat16")
        target = F.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )
        topk_probs = F.softmax(
            paddle.randn([b, sq, topk]).astype("float32"), axis=-1
        )

        # All valid indices
        topk_indices_valid = paddle.randint(0, sk, [b, sq, topk]).astype(
            "int32"
        )
        # Set half to -1
        topk_indices_half = topk_indices_valid.clone()
        topk_indices_half[:, :, topk // 2 :] = -1

        grad_q_full, grad_w_full, grad_k_full = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices_valid.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_half, grad_w_half, grad_k_half = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices_half.clone(),
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        # Gradient magnitudes should differ (masked positions don't contribute)
        # At minimum, half-masked should have smaller or equal gradient norm
        norm_full = grad_q_full.cast("float32").abs().sum().item()
        norm_half = grad_q_half.cast("float32").abs().sum().item()
        self.assertGreater(norm_full, 0.0)
        # With half indices masked, gradient should be different
        self.assertFalse(
            paddle.allclose(
                grad_q_full.cast("float32"),
                grad_q_half.cast("float32"),
                rtol=1e-6,
                atol=1e-6,
            ).item(),
            "Masking half indices should change gradients",
        )

    def test_no_nan_inf_in_grads(self):
        """All gradient outputs must be finite (no NaN/Inf)."""
        b, sq, sk, h, d, topk = 1, 128, 256, 64, 128, 128
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=500)
        )
        grad_q, grad_w, grad_k = self.cudnn_fn(
            index_q,
            weights,
            index_k,
            target,
            topk_probs,
            topk_indices,
            loss_coeff=0.01,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        for name, g in (("d_q", grad_q), ("d_w", grad_w), ("d_k", grad_k)):
            self.assertTrue(
                paddle.isfinite(g.cast("float32")).all().item(),
                f"{name} contains NaN/Inf",
            )
            self.assertGreater(
                g.cast("float32").abs().max().item(),
                0.0,
                f"{name} is identically zero",
            )

    def test_parity_multi_batch(self):
        """Parity with b>1 to verify batch dimension correctness."""
        b, sq, sk, h, d, topk = 2, 64, 128, 64, 128, 128
        loss_coeff = 0.01
        index_q, weights, index_k, target, topk_probs, topk_indices = (
            _make_inputs(b, sq, sk, h, d, topk, seed=600)
        )
        grad_q_cudnn, grad_w_cudnn, grad_k_cudnn = self.cudnn_fn(
            index_q.clone(),
            weights.clone(),
            index_k.clone(),
            target.clone(),
            topk_probs.clone(),
            topk_indices.clone(),
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        grad_q_ref, grad_w_ref, grad_k_ref = _ref_csa_indexer_backward(
            index_q,
            weights,
            index_k,
            topk_indices,
            target,
            topk_probs,
            loss_coeff=loss_coeff,
            grad_loss=paddle.to_tensor(1.0, dtype="float32"),
        )
        _assert_close(
            grad_q_cudnn, grad_q_ref, rtol=5e-2, atol=5e-2, name="batch d_q"
        )
        _assert_close(
            grad_w_cudnn, grad_w_ref, rtol=5e-2, atol=5e-2, name="batch d_w"
        )
        _assert_close(
            grad_k_cudnn, grad_k_ref, rtol=5e-2, atol=5e-2, name="batch d_k"
        )


# ---------------------------------------------------------------------------
# Tests: cudnn_ops/__init__.py imports
# ---------------------------------------------------------------------------


class TestCudnnOpsInit(unittest.TestCase):
    """Cover paddleformers.fleet.cudnn_ops.__init__.py direct imports."""

    def test_export_csa_indexer_bwd(self):
        import paddleformers.fleet.cudnn_ops as cudnn_ops_mod

        fn = cudnn_ops_mod.csa_indexer_bwd
        self.assertTrue(callable(fn))

    def test_unknown_attribute_raises(self):
        import paddleformers.fleet.cudnn_ops as cudnn_ops_mod

        with self.assertRaises(AttributeError):
            _ = cudnn_ops_mod.does_not_exist_xyz

    def test_all_list(self):
        import paddleformers.fleet.cudnn_ops as cudnn_ops_mod

        self.assertIn("csa_indexer_bwd", cudnn_ops_mod.__all__)


# ---------------------------------------------------------------------------
# Tests: csa_attention.py PyLayer cudnn branches
# ---------------------------------------------------------------------------


class TestTileLangCSAIndexerLossAutoScalerCudnn(unittest.TestCase):
    """Cover TileLangCSAIndexerLossAutoScaler.backward cudnn branch."""

    def setUp(self):
        from paddleformers.fleet.transformer.csa_attention import (
            DSAIndexerLossAutoScaler,
            TileLangCSAIndexerLossAutoScaler,
        )

        self.AutoScaler = TileLangCSAIndexerLossAutoScaler
        self.DSAScaler = DSAIndexerLossAutoScaler
        self._orig_scale = DSAIndexerLossAutoScaler._main_loss_backward_scale

    def tearDown(self):
        if hasattr(self, "DSAScaler"):
            self.DSAScaler._main_loss_backward_scale = self._orig_scale

    def _run_with_scale(self, scale):
        import paddleformers.fleet.cudnn_ops as cudnn_ops_mod

        captured = {}

        def fake_bwd(
            index_q,
            weights,
            index_k_comp,
            target,
            topk_probs,
            topk_indices,
            loss_coeff,
            grad_loss=None,
            block_I=128,
        ):
            captured["grad_loss"] = grad_loss
            captured["loss_coeff"] = loss_coeff
            return (
                paddle.zeros_like(index_q),
                paddle.zeros_like(weights),
                paddle.zeros_like(index_k_comp),
            )

        orig = cudnn_ops_mod.__dict__.get("csa_indexer_bwd")
        cudnn_ops_mod.csa_indexer_bwd = fake_bwd
        try:
            self.DSAScaler._main_loss_backward_scale = scale

            b, sq, sk, h, d, topk = 1, 4, 4, 1, 8, 4
            index_q = paddle.randn([b, sq, h, d]).astype("bfloat16")
            weights = paddle.randn([b, sq, h]).astype("bfloat16")
            index_k = paddle.randn([b, sk, d]).astype("bfloat16")
            topk_indices = paddle.randint(0, sk, [b, sq, topk]).astype("int32")
            topk_probs = paddle.nn.functional.softmax(
                paddle.randn([b, sq, topk]).astype("float32"), axis=-1
            )
            target = paddle.nn.functional.softmax(
                paddle.randn([b, sq, topk]).astype("float32"), axis=-1
            )

            class FakeCtx:
                pass

            ctx = FakeCtx()
            ctx.saved_tensor = lambda: (
                index_q,
                weights,
                index_k,
                topk_indices,
                topk_probs,
                target,
            )
            ctx.loss_coeff = 0.01
            ctx.indexer_backend = "cudnn"

            grad_output = paddle.ones_like(weights)
            self.AutoScaler.backward(ctx, grad_output)
        finally:
            if orig is not None:
                cudnn_ops_mod.csa_indexer_bwd = orig
            else:
                cudnn_ops_mod.__dict__.pop("csa_indexer_bwd", None)
        return captured

    def test_scale_none(self):
        captured = self._run_with_scale(None)
        self.assertIsNone(captured["grad_loss"])

    def test_scale_paddle_tensor(self):
        scale_t = paddle.to_tensor(2.5, dtype="float32")
        captured = self._run_with_scale(scale_t)
        self.assertIs(captured["grad_loss"], scale_t)

    def test_scale_python_float(self):
        captured = self._run_with_scale(0.7)
        self.assertIsInstance(captured["grad_loss"], paddle.Tensor)
        self.assertEqual(captured["grad_loss"].dtype, paddle.float32)
        self.assertAlmostEqual(
            float(captured["grad_loss"].item()), 0.7, places=5
        )

    def test_unknown_backend_raises(self):
        """Unknown backend in AutoScaler should raise NotImplementedError."""
        from paddleformers.fleet.transformer.csa_attention import (
            DSAIndexerLossAutoScaler,
        )

        DSAIndexerLossAutoScaler._main_loss_backward_scale = None
        b, sq, sk, h, d, topk = 1, 4, 4, 1, 8, 4

        class FakeCtx:
            pass

        ctx = FakeCtx()
        ctx.saved_tensor = lambda: (
            paddle.randn([b, sq, h, d]).astype("bfloat16"),
            paddle.randn([b, sq, h]).astype("bfloat16"),
            paddle.randn([b, sk, d]).astype("bfloat16"),
            paddle.randint(0, sk, [b, sq, topk]).astype("int32"),
            paddle.randn([b, sq, topk]).astype("float32"),
            paddle.randn([b, sq, topk]).astype("float32"),
        )
        ctx.loss_coeff = 0.01
        ctx.indexer_backend = "bad_backend"

        with self.assertRaises(NotImplementedError):
            self.AutoScaler.backward(ctx, paddle.ones([b, sq, h]))


# ---------------------------------------------------------------------------
# Tests: transformer_config.py (csa_indexer_backend field + validation)
# ---------------------------------------------------------------------------


class TestTransformerConfigCsaIndexerBackend(unittest.TestCase):
    """Cover csa_indexer_backend field default and __post_init__ validation."""

    def test_default_value(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        cfg = TransformerConfig()
        self.assertEqual(cfg.csa_indexer_backend, "tilelang")

    def test_valid_cudnn_value(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        cfg = TransformerConfig(
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[4],
            csa_indexer_backend="cudnn",
        )
        self.assertEqual(cfg.csa_indexer_backend, "cudnn")

    def test_valid_tilelang_value(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        cfg = TransformerConfig(
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[4],
            csa_indexer_backend="tilelang",
        )
        self.assertEqual(cfg.csa_indexer_backend, "tilelang")

    def test_invalid_value_raises(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        with self.assertRaises(ValueError) as cm:
            TransformerConfig(
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[4],
                csa_indexer_backend="invalid_backend",
            )
        self.assertIn("invalid_backend", str(cm.exception))

    def test_attribute_map_registered(self):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        self.assertIn("csa_indexer_backend", TransformerConfig.transform_rules)


# ---------------------------------------------------------------------------
# Tests: indexer/__init__.py re-export
# ---------------------------------------------------------------------------


class TestIndexerSubpackageInit(unittest.TestCase):
    """Cover paddleformers.fleet.cudnn_ops.indexer.__init__.py re-export."""

    def test_import_from_indexer(self):
        from paddleformers.fleet.cudnn_ops.indexer import csa_indexer_bwd

        self.assertTrue(callable(csa_indexer_bwd))

    def test_indexer_all(self):
        import paddleformers.fleet.cudnn_ops.indexer as indexer_mod

        self.assertIn("csa_indexer_bwd", indexer_mod.__all__)


if __name__ == "__main__":
    unittest.main()
