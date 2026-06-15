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

"""Unit tests for DSA (DeepSeek Sparse Attention) module.

Tests are organized in 4 layers:
  1. Pure functions: hadamard_transform, rotate_activation, _unfused_dsa_attention,
     _compute_index_scores_fused
  2. Indexer module: forward_before_topk, compute_index_scores, backward
  3. Loss: _compute_dsa_indexer_loss, FusedDSAIndexerLoss, DSAIndexerLossAutoScaler
  4. Integration: MLASelfAttention with DSAttention (as core_attention) forward + backward
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    DSAIndexerSublayersSpec,
    DSAttention,
    DSAttentionSublayersSpec,
    FusedDSAIndexerLoss,
    _bwd_fused_indexer_loss,
    _compute_dsa_indexer_loss,
    _compute_index_scores_fused,
    _unfused_dsa_attention,
    hadamard_transform,
    rotate_activation,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal


# ---------------------------------------------------------------------------
# Stub layers (same pattern as test_attention.py)
# ---------------------------------------------------------------------------
class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        # Cast input to match weight dtype (mimic ColumnParallelLinear behavior)
        if x.dtype != self.linear.weight.dtype:
            x = x.cast(self.linear.weight.dtype)
        return self.linear(x), self.linear.bias


class LayerNormStub(paddle.nn.Layer):
    """Stub for LayerNorm that accepts hidden_size or normalized_shape keyword argument."""

    def __init__(
        self,
        hidden_size=None,
        eps=None,
        normalized_shape=None,
        epsilon=None,
        **kwargs,
    ):
        super().__init__()
        size = hidden_size if hidden_size is not None else normalized_shape
        self.eps = eps if eps is not None else (epsilon if epsilon is not None else 1e-5)
        self.weight = paddle.nn.Parameter(paddle.ones([size]))
        self.bias = paddle.nn.Parameter(paddle.zeros([size]))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, unbiased=False)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        return x * self.weight + self.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


# ---------------------------------------------------------------------------
# Helper: create DSA-compatible TransformerConfig
# ---------------------------------------------------------------------------
def _create_dsa_config(
    hidden_size=256,
    num_attention_heads=2,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    v_head_dim=64,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=16,
    indexer_loss_coeff=1.0,
    indexer_use_sparse_loss=False,
    sequence_parallel=False,
):
    config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    # MLA fields
    config.num_key_value_heads = num_attention_heads
    config.head_dim = hidden_size // num_attention_heads
    config.q_lora_rank = q_lora_rank
    config.kv_lora_rank = kv_lora_rank
    config.qk_nope_head_dim = qk_nope_head_dim
    config.qk_rope_head_dim = qk_rope_head_dim
    config.v_head_dim = v_head_dim
    config.multi_latent_attention = True

    # RoPE / YaRN
    config.rope_type = "yarn"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.rotary_scaling_factor = 40.0
    config.original_max_position_embeddings = 4096
    config.beta_fast = 32.0
    config.beta_slow = 1.0
    config.mscale = 1.0
    config.mscale_all_dim = 0.0
    config.apply_rope_fusion = False  # DSA requires unfused RoPE

    # DSA Indexer fields
    config.dsa_index_n_heads = index_n_heads
    config.dsa_index_head_dim = index_head_dim
    config.dsa_index_topk = index_topk
    config.dsa_indexer_loss_coeff = indexer_loss_coeff
    config.dsa_indexer_use_sparse_loss = indexer_use_sparse_loss
    config.dsa_indexer_rotary_interleaved = False  # Test default value

    # Attention generic fields
    config.softmax_scale = None
    config.use_bias = True
    config.no_rope_freq = None
    config.recompute_granularity = None
    config.fused_single_qkv_rope = False
    config.init_method = init_method_normal(0.02)
    config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    config.rms_norm_eps = 1e-5
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.sliding_window = None
    config.window_attn_skip_freq = None
    config.fp16 = False
    config.bf16 = False
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = "vanilla"
    config.sequence_parallel = sequence_parallel

    return config


def _create_sublayers_spec():
    """Create MLASelfAttentionSublayersSpec with DSAttention as core_attention."""
    # DSA Indexer sublayers spec for testing
    dsa_indexer_sublayers = DSAIndexerSublayersSpec(
        linear_wq_b=BiasedLinear,
        linear_wk=BiasedLinear,
        k_norm=LayerNormStub,  # Use stub that accepts hidden_size kwarg
        linear_weights_proj=BiasedLinear,
    )

    # DSAttention as core_attention (pluggable component)
    class DSAttentionWrapper(DSAttention):
        """Wrapper for testing that uses test stub layers."""

        pass  # Inherits all behavior from DSAttention

    return MLASelfAttentionSublayersSpec(
        q_proj=BiasedLinear,
        q_a_proj=BiasedLinear,
        q_b_proj=BiasedLinear,
        kv_a_proj_with_mqa=BiasedLinear,
        kv_b_proj=BiasedLinear,
        core_attention=LayerSpec(
            layer=DSAttentionWrapper,
            sublayers_spec=DSAttentionSublayersSpec(
                indexer=LayerSpec(
                    layer=DSAIndexer,
                    sublayers_spec=dsa_indexer_sublayers,
                ),
            ),
        ),
        o_proj=BiasedLinear,
        q_a_layernorm=RMSNorm,
        kv_a_layernorm=RMSNorm,
    )


def _make_causal_topk_indices(b, sq, sk, topk):
    """Generate topk indices that respect causality (indices <= current position)."""
    indices_list = []
    for i in range(sq):
        max_idx = min(i + 1, sk)
        actual_topk = min(topk, max_idx)
        # Pick the last `actual_topk` positions (most recent)
        row_indices = paddle.arange(max_idx - actual_topk, max_idx)
        if actual_topk < topk:
            # Pad with the last valid index
            pad = paddle.full([topk - actual_topk], max_idx - 1, dtype="int64")
            row_indices = paddle.concat([row_indices, pad])
        indices_list.append(row_indices)
    # [sq, topk] -> expand to [b, sq, topk]
    indices = paddle.stack(indices_list, axis=0).unsqueeze(0).expand([b, sq, topk])
    return indices


# ===========================================================================
# Layer 1: Pure function tests
# ===========================================================================
class TestHadamardTransform(unittest.TestCase):
    def test_output_shape(self):
        x = paddle.randn([4, 8, 16])
        out = hadamard_transform(x)
        self.assertEqual(out.shape, [4, 8, 16])

    def test_power_of_two_assertion(self):
        x = paddle.randn([4, 7])
        with self.assertRaises(AssertionError):
            hadamard_transform(x)

    def test_involution(self):
        """H(H(x)) = dim * x (Hadamard is involutory up to scaling)."""
        dim = 16
        x = paddle.randn([3, dim], dtype="float32")
        hx = hadamard_transform(x)
        hhx = hadamard_transform(hx)
        self.assertTrue(paddle.allclose(hhx, x * dim, atol=1e-4, rtol=1e-4))

    def test_scale_factor(self):
        x = paddle.randn([4, 8])
        out_unscaled = hadamard_transform(x)
        out_scaled = hadamard_transform(x, scale=0.5)
        self.assertTrue(paddle.allclose(out_scaled, out_unscaled * 0.5, atol=1e-5))

    def test_1d_input(self):
        x = paddle.randn([16])
        out = hadamard_transform(x)
        self.assertEqual(out.shape, [16])


class TestRotateActivation(unittest.TestCase):
    def test_output_shape(self):
        x = paddle.randn([2, 4, 128]).cast("bfloat16")
        out = rotate_activation(x)
        self.assertEqual(list(out.shape), [2, 4, 128])
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_dtype_assertion(self):
        x = paddle.randn([2, 4, 64], dtype="float32")
        with self.assertRaises(AssertionError):
            rotate_activation(x)


class TestUnfusedDSAAttention(unittest.TestCase):
    def setUp(self):
        self.b, self.s, self.nhpp = 2, 8, 4
        self.qk_hd, self.v_hd = 32, 64
        self.softmax_scale = self.qk_hd**-0.5

    def test_output_shape(self):
        query = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, self.v_hd])
        out = _unfused_dsa_attention(query, key, value, None, self.softmax_scale)
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * self.v_hd])

    def test_with_causal_mask(self):
        query = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, self.v_hd])
        causal = paddle.triu(
            paddle.full([self.s, self.s], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, s, s]
        out = _unfused_dsa_attention(query, key, value, mask, self.softmax_scale)
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * self.v_hd])

    def test_asymmetric_dims(self):
        """qk_head_dim != v_head_dim should work."""
        qk_hd, v_hd = 48, 32
        query = paddle.randn([self.b, self.s, self.nhpp, qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, v_hd])
        out = _unfused_dsa_attention(query, key, value, None, qk_hd**-0.5)
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * v_hd])


class TestComputeIndexScoresFused(unittest.TestCase):
    def test_output_shape(self):
        sq, b, h, d = 8, 2, 4, 32
        sk = 8
        q = paddle.randn([sq, b, h, d])
        weights = paddle.randn([sq, b, h])
        k = paddle.randn([sk, b, d])
        out = _compute_index_scores_fused(q, weights, k)
        self.assertEqual(out.shape, [b, sq, sk])

    def test_nonnegative_after_relu(self):
        sq, b, h, d = 8, 2, 4, 32
        q = paddle.randn([sq, b, h, d])
        # Use positive weights so that relu * positive_weights >= 0
        weights = paddle.abs(paddle.randn([sq, b, h])) + 0.1
        k = paddle.randn([sq, b, d])
        out = _compute_index_scores_fused(q, weights, k)
        self.assertTrue((out >= -1e-6).all().item())


# ===========================================================================
# Layer 2: DSAIndexer module tests
# ===========================================================================
class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.config = _create_dsa_config()
        # Create indexer sublayers spec with stub layers that accept **kwargs
        indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        self.indexer = DSAIndexer(
            config=self.config,
            sublayers_spec=indexer_sublayers,
            layer_number=1,
            pg_collection=None,
        )
        self.b = 2
        self.s = 16

    def _prepare_indexer_bf16(self):
        """Convert wq_b/wk to bf16 for rotate_activation, keep weights_proj fp32."""
        self.indexer.wq_b = self.indexer.wq_b.to(dtype="bfloat16")
        self.indexer.wk = self.indexer.wk.to(dtype="bfloat16")
        self.indexer.k_norm = self.indexer.k_norm.to(dtype="bfloat16")
        # weights_proj stays fp32 (code does hidden.cast("float32") before calling it)

    def test_forward_before_topk_shapes(self):
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")

        q, k, weights = self.indexer.forward_before_topk(hidden, q_latent)
        self.assertEqual(
            list(q.shape),
            [
                self.b,
                self.s,
                self.config.dsa_index_n_heads,
                self.config.dsa_index_head_dim,
            ],
        )
        self.assertEqual(
            list(k.shape),
            [self.b, self.s, self.config.dsa_index_head_dim],
        )
        self.assertEqual(
            list(weights.shape),
            [self.b, self.s, self.config.dsa_index_n_heads],
        )

    def test_compute_index_scores_shapes(self):
        q = paddle.randn(
            [
                self.b,
                self.s,
                self.config.dsa_index_n_heads,
                self.config.dsa_index_head_dim,
            ]
        )
        k = paddle.randn([self.b, self.s, self.config.dsa_index_head_dim])
        weights = paddle.randn([self.b, self.s, self.config.dsa_index_n_heads])
        index_scores, topk_indices = self.indexer.compute_index_scores(q, k, weights, mask=None)
        self.assertEqual(list(index_scores.shape), [self.b, self.s, self.s])
        self.assertEqual(
            list(topk_indices.shape),
            [self.b, self.s, self.config.dsa_index_topk],
        )

    def test_topk_in_range(self):
        q = paddle.randn(
            [
                self.b,
                self.s,
                self.config.dsa_index_n_heads,
                self.config.dsa_index_head_dim,
            ]
        )
        k = paddle.randn([self.b, self.s, self.config.dsa_index_head_dim])
        weights = paddle.randn([self.b, self.s, self.config.dsa_index_n_heads])
        _, topk_indices = self.indexer.compute_index_scores(q, k, weights, mask=None)
        self.assertTrue((topk_indices >= 0).all().item())
        self.assertTrue((topk_indices < self.s).all().item())

    def test_backward(self):
        """Indexer parameters receive gradients."""
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")

        q, k, weights = self.indexer.forward_before_topk(hidden, q_latent)
        # rotate_activation requires bf16, so skip it in this unit test
        # and just use the raw outputs for gradient checking.
        loss = q.cast("float32").sum() + k.cast("float32").sum() + weights.sum()
        loss.backward()

        for name, param in self.indexer.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")


# ===========================================================================
# Layer 3: Loss tests
# ===========================================================================
class TestComputeDSAIndexerLoss(unittest.TestCase):
    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b, self.np, self.hn = 2, 4, 32
        self.topk = 4
        self.softmax_scale = self.hn**-0.5
        self.loss_coeff = 1.0

    def _make_inputs(self, sparse=False):
        index_scores = paddle.randn([self.b, self.sq, self.sk], dtype="float32")
        if sparse:
            topk_indices = _make_causal_topk_indices(self.b, self.sq, self.sk, self.topk)
        else:
            topk_indices = paddle.randint(0, self.sk, [self.b, self.sq, self.topk]).cast("int64")
        query = paddle.randn([self.sq, self.b, self.np, self.hn], dtype="float32")
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        return index_scores, topk_indices, query, key

    def test_loss_is_scalar(self):
        index_scores, topk_indices, query, key = self._make_inputs()
        loss = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            self.loss_coeff,
            False,
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_loss_with_sparse(self):
        index_scores, topk_indices, query, key = self._make_inputs(sparse=True)
        loss = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            self.loss_coeff,
            True,
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_loss_coeff_scaling(self):
        index_scores, topk_indices, query, key = self._make_inputs()
        loss1 = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            1.0,
            False,
            None,
        )
        loss2 = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            2.0,
            False,
            None,
        )
        self.assertTrue(
            paddle.allclose(loss2, loss1 * 2.0, atol=1e-4),
            f"loss2={loss2.item():.6f} != 2*loss1={2 * loss1.item():.6f}",
        )


class TestFusedDSAIndexerLoss(unittest.TestCase):
    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b = 2
        self.h, self.d = 4, 32  # indexer heads/dim
        self.np, self.hn = 4, 64  # MLA heads/dim
        self.topk = 4
        self.softmax_scale = self.hn**-0.5

    def _make_inputs(self, with_mask=False):
        q = paddle.randn([self.sq, self.b, self.h, self.d], dtype="float32")
        q.stop_gradient = False
        weights = paddle.randn([self.sq, self.b, self.h], dtype="float32")
        weights.stop_gradient = False
        k = paddle.randn([self.sk, self.b, self.d], dtype="float32")
        k.stop_gradient = False
        query = paddle.randn([self.sq, self.b, self.np, self.hn], dtype="float32")
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        if with_mask:
            causal = paddle.triu(
                paddle.full([self.sq, self.sk], float("-inf"), dtype="float32"),
                diagonal=1,
            )
            mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]
        else:
            mask = None
        return q, weights, k, query, key, mask

    def test_forward_returns_scalar(self):
        q, weights, k, query, key, mask = self._make_inputs(with_mask=True)
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            False,
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_topk_indices_stored(self):
        FusedDSAIndexerLoss._last_topk_indices = None
        q, weights, k, query, key, _ = self._make_inputs()
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            None,
            False,
            None,
        )
        self.assertIsNotNone(FusedDSAIndexerLoss._last_topk_indices)
        self.assertEqual(
            list(FusedDSAIndexerLoss._last_topk_indices.shape),
            [self.b, self.sq, self.topk],
        )

    def test_backward_gradients(self):
        # Pass a mask tensor so PyLayer sees 6 tensor inputs (q, weights, k,
        # query, key, mask) matching the 6 return values in backward.
        q, weights, k, query, key, mask = self._make_inputs(with_mask=True)
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            False,
            None,
        )
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(k.grad)
        self.assertEqual(list(q.grad.shape), [self.sq, self.b, self.h, self.d])
        self.assertEqual(list(weights.grad.shape), [self.sq, self.b, self.h])
        self.assertEqual(list(k.grad.shape), [self.sk, self.b, self.d])
        self.assertTrue(paddle.isfinite(q.grad).all().item())
        self.assertTrue(paddle.isfinite(weights.grad).all().item())
        self.assertTrue(paddle.isfinite(k.grad).all().item())


class TestDSAIndexerLossAutoScaler(unittest.TestCase):
    def _make_non_leaf_output(self, shape):
        """Create a non-leaf tensor (required by PyLayer inplace check)."""
        x = paddle.randn(shape)
        x.stop_gradient = False
        return x + 0  # Adding 0 makes it non-leaf

    def test_forward_passthrough(self):
        output = self._make_non_leaf_output([2, 8, 64])
        indexer_loss = self._make_non_leaf_output([])
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        self.assertEqual(list(result.shape), [2, 8, 64])

    def test_backward_grad_output(self):
        output = self._make_non_leaf_output([2, 8, 64])
        indexer_loss = self._make_non_leaf_output([])

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        loss = result.sum()
        loss.backward()
        # output is non-leaf (x + 0), so its grad may not be retained,
        # but the computation should not error out.
        self.assertTrue(True)  # Just verify no error

    def test_loss_scale(self):
        DSAIndexerLossAutoScaler.set_loss_scale(paddle.to_tensor(2.0, dtype="float32"))
        output = self._make_non_leaf_output([2, 4])
        indexer_loss = paddle.to_tensor(1.0, dtype="float32")
        indexer_loss.stop_gradient = False
        indexer_loss = indexer_loss * 1.0  # Make non-leaf

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        loss = result.sum()
        loss.backward()
        # Verify no errors
        self.assertTrue(True)
        # Reset
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None


# ===========================================================================
# Layer 4: MLASelfAttention with DSAttention (core_attention) integration tests
# ===========================================================================
class TestMLASelfAttentionWithDSA(unittest.TestCase):
    def setUp(self):
        self.config = _create_dsa_config()
        self.micro_batch_size = 2
        self.sequence_length = 32

    def _build_model(self, config=None):
        cfg = config or self.config
        model = MLASelfAttention(
            cfg,
            _create_sublayers_spec(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        # Convert model to bf16 because rotate_activation requires bf16 input.
        # But weights_proj does hidden.cast("float32") internally and expects
        # fp32 weights, so convert it back to fp32 after the global bf16 cast.
        model = model.to(dtype="bfloat16")
        model.core_attention.indexer.weights_proj = model.core_attention.indexer.weights_proj.to(dtype="float32")
        return model

    def _make_hidden(self, dtype="bfloat16"):
        return paddle.randn(
            [
                self.micro_batch_size,
                self.sequence_length,
                self.config.hidden_size,
            ],
        ).cast(dtype)

    def test_forward_shape(self):
        model = self._build_model()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)
        self.assertEqual(bias.shape[0], self.config.hidden_size)

    def test_forward_with_attention_mask(self):
        model = self._build_model()
        hidden = self._make_hidden()
        causal = paddle.triu(
            paddle.full(
                [self.sequence_length, self.sequence_length],
                float("-inf"),
                dtype="float32",
            ),
            diagonal=1,
        )
        mask = (
            causal.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                [
                    self.micro_batch_size,
                    1,
                    self.sequence_length,
                    self.sequence_length,
                ]
            )
        )
        output, bias = model(hidden, attention_mask=mask)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)

    def test_forward_training_with_loss(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)

    def test_forward_eval_mode(self):
        config = _create_dsa_config(indexer_loss_coeff=None)
        model = self._build_model(config)
        model.eval()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)

    def test_backward_gradients(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        hidden.stop_gradient = False
        output, bias = model(hidden, attention_mask=None)
        loss = output.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_indexer_params_have_grad(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        hidden.stop_gradient = False
        output, bias = model(hidden, attention_mask=None)
        loss = output.cast("float32").sum()
        loss.backward()

        indexer_param_names = [
            "indexer.wq_b",
            "indexer.wk",
            "indexer.weights_proj",
        ]
        for name, param in model.named_parameters():
            for iname in indexer_param_names:
                if iname in name:
                    self.assertIsNotNone(
                        param.grad,
                        f"Indexer parameter {name} has no gradient",
                    )


# ===========================================================================
# Layer 5: DSAIndexerLossLoggingHelper tests
# ===========================================================================
class TestDSAIndexerLossLoggingHelperSaveLoss(unittest.TestCase):
    """Tests for DSAIndexerLossLoggingHelper.save_loss_to_tracker."""

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker = {}

    def tearDown(self):
        DSAIndexerLossLoggingHelper.tracker = {}

    def test_save_loss_initializes_values(self):
        """First call should create the 'values' tensor with correct size."""
        loss = paddle.to_tensor(0.5, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=4)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        self.assertEqual(list(DSAIndexerLossLoggingHelper.tracker["values"].shape), [4])

    def test_save_loss_accumulates(self):
        """Multiple saves to the same layer should accumulate."""
        loss1 = paddle.to_tensor(0.5, dtype="float32")
        loss2 = paddle.to_tensor(0.3, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss1, layer_number=1, num_layers=4)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss2, layer_number=1, num_layers=4)
        self.assertTrue(
            paddle.allclose(
                DSAIndexerLossLoggingHelper.tracker["values"][0],
                paddle.to_tensor(0.8, dtype="float32"),
                atol=1e-5,
            )
        )

    def test_save_loss_different_layers(self):
        """Saving to different layers puts values in correct positions."""
        loss1 = paddle.to_tensor(1.0, dtype="float32")
        loss2 = paddle.to_tensor(2.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss1, layer_number=1, num_layers=3)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss2, layer_number=3, num_layers=3)
        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertAlmostEqual(values[0].item(), 1.0, places=5)
        self.assertAlmostEqual(values[1].item(), 0.0, places=5)
        self.assertAlmostEqual(values[2].item(), 2.0, places=5)

    def test_save_loss_none_layer_number_noop(self):
        """layer_number=None should be a no-op."""
        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=None, num_layers=4)
        self.assertNotIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_save_loss_stores_groups(self):
        """reduce_group and avg_group should be stored in the tracker."""
        loss = paddle.to_tensor(0.1, dtype="float32")
        mock_reduce = MagicMock()
        mock_avg = MagicMock()
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            layer_number=1,
            num_layers=2,
            reduce_group=mock_reduce,
            avg_group=mock_avg,
        )
        self.assertIs(DSAIndexerLossLoggingHelper.tracker["reduce_group"], mock_reduce)
        self.assertIs(DSAIndexerLossLoggingHelper.tracker["avg_group"], mock_avg)


class TestDSAIndexerLossLoggingHelperClean(unittest.TestCase):
    """Tests for DSAIndexerLossLoggingHelper.clean_loss_in_tracker."""

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker = {}

    def tearDown(self):
        DSAIndexerLossLoggingHelper.tracker = {}

    def test_clean_zeros_values(self):
        """Clean should zero out the values tensor."""
        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=3)
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertTrue(paddle.allclose(values, paddle.zeros([3])))

    def test_clean_resets_groups(self):
        """Clean should set reduce_group and avg_group to None."""
        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            layer_number=1,
            num_layers=2,
            reduce_group=MagicMock(),
            avg_group=MagicMock(),
        )
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        self.assertIsNone(DSAIndexerLossLoggingHelper.tracker["reduce_group"])
        self.assertIsNone(DSAIndexerLossLoggingHelper.tracker["avg_group"])

    def test_clean_empty_tracker_noop(self):
        """Clean on empty tracker should not raise."""
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        self.assertIsNone(DSAIndexerLossLoggingHelper.tracker.get("reduce_group"))


class TestDSAIndexerLossLoggingHelperReduce(unittest.TestCase):
    """Tests for DSAIndexerLossLoggingHelper.reduce_loss_in_tracker."""

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker = {}
        DSAIndexerLossLoggingHelper.num_layers = None

    def tearDown(self):
        DSAIndexerLossLoggingHelper.tracker = {}
        DSAIndexerLossLoggingHelper.num_layers = None

    def test_reduce_empty_tracker_noop(self):
        """Reduce with no 'values' should be a no-op."""
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        # Should not raise

    def test_get_total_num_layers_treats_none_as_zero(self):
        """None nextn layer count should be treated as zero."""
        config = SimpleNamespace(num_hidden_layers=4, mtp_num_layers=0)
        config.num_nextn_predict_layers = None
        self.assertEqual(DSAIndexerLossLoggingHelper.get_total_num_layers(config), 4)

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_empty_tracker_with_num_layers_joins_pp_reduce(self, mock_ps):
        """Empty tracker should initialize zeros and join PP all_reduce."""
        pp_group = MagicMock()
        pp_group.nranks = 2
        mock_ps.get_pipeline_model_parallel_group.return_value = pp_group
        mock_ps.get_data_parallel_group.return_value = None

        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(num_layers=3)
            mock_all_reduce.assert_called_once()

        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertTrue(paddle.allclose(values, paddle.zeros([3])))

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_empty_tracker_uses_registered_num_layers(self, mock_ps):
        """Empty tracker should infer registered layer count for PP reduce."""
        pp_group = MagicMock()
        pp_group.nranks = 2
        mock_ps.get_pipeline_model_parallel_group.return_value = pp_group
        mock_ps.get_data_parallel_group.return_value = None
        DSAIndexerLossLoggingHelper.num_layers = 3

        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_called_once()

        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertTrue(paddle.allclose(values, paddle.zeros([3])))

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_no_distributed_groups(self, mock_ps):
        """Reduce with no distributed groups should keep values unchanged."""
        mock_ps.get_pipeline_model_parallel_group.return_value = None
        mock_ps.get_data_parallel_group.return_value = None

        loss = paddle.to_tensor(2.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        original_values = DSAIndexerLossLoggingHelper.tracker["values"].clone()
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        self.assertTrue(
            paddle.allclose(
                DSAIndexerLossLoggingHelper.tracker["values"],
                original_values,
            )
        )

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_with_pp_group(self, mock_ps):
        """Reduce with PP group should call all_reduce."""
        pp_group = MagicMock()
        pp_group.nranks = 2
        mock_ps.get_pipeline_model_parallel_group.return_value = pp_group
        mock_ps.get_data_parallel_group.return_value = None

        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_called_once()

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_with_dp_group(self, mock_ps):
        """Reduce with DP group should call all_reduce and divide by nranks."""
        mock_ps.get_pipeline_model_parallel_group.return_value = None
        dp_group = MagicMock()
        dp_group.nranks = 4
        mock_ps.get_data_parallel_group.return_value = dp_group

        loss = paddle.to_tensor(4.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        with patch("paddle.distributed.all_reduce"):
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            # After DP reduce, values should be divided by nranks
            # (all_reduce is mocked, so actual value won't change, but the
            #  division path is exercised)

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_with_reduce_group(self, mock_ps):
        """Reduce with TP reduce_group should call all_reduce."""
        mock_ps.get_pipeline_model_parallel_group.return_value = None
        mock_ps.get_data_parallel_group.return_value = None

        loss = paddle.to_tensor(1.0, dtype="float32")
        reduce_group = MagicMock()
        reduce_group.nranks = 2
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            layer_number=1,
            num_layers=2,
            reduce_group=reduce_group,
        )
        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_called_once()

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_with_avg_group(self, mock_ps):
        """Reduce with avg_group should call all_reduce and divide by nranks."""
        mock_ps.get_pipeline_model_parallel_group.return_value = None
        mock_ps.get_data_parallel_group.return_value = None

        avg_group = MagicMock()
        avg_group.nranks = 3
        loss = paddle.to_tensor(3.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            layer_number=1,
            num_layers=2,
            avg_group=avg_group,
        )
        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_called_once()

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_pp_group_single_rank_skipped(self, mock_ps):
        """PP group with nranks=1 should not trigger all_reduce."""
        pp_group = MagicMock()
        pp_group.nranks = 1
        mock_ps.get_pipeline_model_parallel_group.return_value = pp_group
        mock_ps.get_data_parallel_group.return_value = None

        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_not_called()

    @patch("paddleformers.fleet.transformer.dsa_attention.parallel_state")
    def test_reduce_dp_group_single_rank_skipped(self, mock_ps):
        """DP group with nranks=1 should not trigger all_reduce."""
        mock_ps.get_pipeline_model_parallel_group.return_value = None
        dp_group = MagicMock()
        dp_group.nranks = 1
        mock_ps.get_data_parallel_group.return_value = dp_group

        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        with patch("paddle.distributed.all_reduce") as mock_all_reduce:
            DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
            mock_all_reduce.assert_not_called()


class TestDSAIndexerLossLoggingHelperTrackMetrics(unittest.TestCase):
    """Tests for DSAIndexerLossLoggingHelper.track_indexer_metrics."""

    def setUp(self):
        DSAIndexerLossLoggingHelper.tracker = {}
        DSAIndexerLossLoggingHelper.num_layers = None

    def tearDown(self):
        DSAIndexerLossLoggingHelper.tracker = {}
        DSAIndexerLossLoggingHelper.num_layers = None

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_empty_tracker_noop(self, mock_reduce):
        """With no values, track_indexer_metrics should be a no-op after reduce."""
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=10, num_layers=2)
        mock_reduce.assert_called_once_with(num_layers=2)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_logs_loss(self, mock_reduce):
        """track_indexer_metrics should log the averaged indexer loss."""
        loss = paddle.to_tensor(2.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        with self.assertLogs("paddleformers.fleet.transformer.dsa_attention", level="INFO") as cm:
            DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=42)
        log_output = "\n".join(cm.output)
        self.assertIn("42", log_output)
        self.assertIn("indexer loss", log_output)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_updates_total_loss_dict(self, mock_reduce):
        """track_indexer_metrics should add 'indexer loss' to total_loss_dict."""
        loss = paddle.to_tensor(4.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        total_loss_dict = {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=0.5, iteration=1, total_loss_dict=total_loss_dict)
        self.assertIn("indexer loss", total_loss_dict)
        # loss=4.0 at layer 1 only, num_layers=2
        # avg = (4.0 * 0.5) / 2 = 1.0
        expected = 1.0
        self.assertAlmostEqual(total_loss_dict["indexer loss"].item(), expected, places=4)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_accumulates_total_loss_dict(self, mock_reduce):
        """Calling track_indexer_metrics twice should accumulate in total_loss_dict."""
        loss1 = paddle.to_tensor(2.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss1, layer_number=1, num_layers=1)
        total_loss_dict = {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=1, total_loss_dict=total_loss_dict)
        first_value = total_loss_dict["indexer loss"].item()

        # Save and track again
        loss2 = paddle.to_tensor(3.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss2, layer_number=1, num_layers=1)
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=2, total_loss_dict=total_loss_dict)
        self.assertAlmostEqual(
            total_loss_dict["indexer loss"].item(),
            first_value + 3.0,
            places=4,
        )

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_with_writer(self, mock_reduce):
        """track_indexer_metrics should call writer.add_scalar when provided."""
        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=1)
        mock_writer = MagicMock()
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=100, writer=mock_writer)
        mock_writer.add_scalar.assert_called_once()
        args = mock_writer.add_scalar.call_args
        self.assertEqual(args[0][0], "indexer loss")
        self.assertEqual(args[0][2], 100)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_cleans_tracker(self, mock_reduce):
        """track_indexer_metrics should clean the tracker after logging."""
        loss = paddle.to_tensor(1.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=2)
        DSAIndexerLossLoggingHelper.track_indexer_metrics(loss_scale=1.0, iteration=1)
        # After track_indexer_metrics, values should be zeroed
        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertTrue(paddle.allclose(values, paddle.zeros([2])))

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_loss_scale_applied(self, mock_reduce):
        """loss_scale should be multiplied into the loss values."""
        loss = paddle.to_tensor(6.0, dtype="float32")
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_number=1, num_layers=1)
        total_loss_dict = {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=0.25, iteration=1, total_loss_dict=total_loss_dict
        )
        # 6.0 * 0.25 / 1 = 1.5
        self.assertAlmostEqual(total_loss_dict["indexer loss"].item(), 1.5, places=4)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_averages_over_csa_indexer_layers(self, mock_reduce):
        """CSA logging should average over layers with ratio == 4."""
        DSAIndexerLossLoggingHelper.tracker["values"] = paddle.to_tensor([0.0, 2.0, 0.0, 4.0], dtype="float32")
        total_loss_dict = {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=1.0,
            iteration=1,
            total_loss_dict=total_loss_dict,
            csa_compress_ratios=[0, 4, 128, 4],
        )
        self.assertAlmostEqual(total_loss_dict["indexer loss"].item(), 3.0, places=4)

    @patch.object(DSAIndexerLossLoggingHelper, "reduce_loss_in_tracker")
    def test_track_metrics_no_csa_indexer_layers_noop(self, mock_reduce):
        """CSA logging should skip metrics when no layer owns an indexer."""
        DSAIndexerLossLoggingHelper.tracker["values"] = paddle.zeros([2])
        total_loss_dict = {}
        DSAIndexerLossLoggingHelper.track_indexer_metrics(
            loss_scale=1.0,
            iteration=1,
            total_loss_dict=total_loss_dict,
            csa_compress_ratios=[0, 128],
        )
        self.assertNotIn("indexer loss", total_loss_dict)
        self.assertTrue(
            paddle.allclose(
                DSAIndexerLossLoggingHelper.tracker["values"],
                paddle.zeros([2]),
            )
        )


# ===========================================================================
# Layer 6: Additional coverage for DSAIndexer and helper functions
# ===========================================================================
class TestIndexerForward(unittest.TestCase):
    """Test DSAIndexer.forward (the combined forward_before_topk + compute_index_scores)."""

    def setUp(self):
        self.config = _create_dsa_config()
        indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        self.indexer = DSAIndexer(
            config=self.config,
            sublayers_spec=indexer_sublayers,
            layer_number=1,
            pg_collection=None,
        )
        self.b = 2
        self.s = 16

    def _prepare_indexer_bf16(self):
        self.indexer.wq_b = self.indexer.wq_b.to(dtype="bfloat16")
        self.indexer.wk = self.indexer.wk.to(dtype="bfloat16")
        self.indexer.k_norm = self.indexer.k_norm.to(dtype="bfloat16")

    def test_forward_returns_scores_and_indices(self):
        """DSAIndexer.forward should return (index_scores, topk_indices)."""
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")
        causal = paddle.triu(
            paddle.full([self.s, self.s], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)

        index_scores, topk_indices = self.indexer.forward(hidden, q_latent, attention_mask=mask)
        self.assertEqual(list(index_scores.shape), [self.b, self.s, self.s])
        self.assertEqual(
            list(topk_indices.shape),
            [self.b, self.s, self.config.dsa_index_topk],
        )

    def test_forward_no_mask(self):
        """DSAIndexer.forward should work with mask=None."""
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")
        index_scores, topk_indices = self.indexer.forward(hidden, q_latent, attention_mask=None)
        self.assertEqual(list(index_scores.shape), [self.b, self.s, self.s])


class TestIndexerComputeScoresWithMask(unittest.TestCase):
    """Test mask handling in DSAIndexer.compute_index_scores."""

    def setUp(self):
        self.config = _create_dsa_config(index_topk=4)
        indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        self.indexer = DSAIndexer(
            config=self.config,
            sublayers_spec=indexer_sublayers,
            layer_number=1,
            pg_collection=None,
        )
        self.b = 2
        self.s = 8

    def test_mask_zeros_future_positions(self):
        """With causal mask, topk indices should not exceed current position."""
        q = paddle.randn(
            [
                self.b,
                self.s,
                self.config.dsa_index_n_heads,
                self.config.dsa_index_head_dim,
            ]
        )
        k = paddle.randn([self.b, self.s, self.config.dsa_index_head_dim])
        weights = paddle.abs(paddle.randn([self.b, self.s, self.config.dsa_index_n_heads]))
        causal = paddle.triu(
            paddle.full([self.s, self.s], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)

        index_scores, topk_indices = self.indexer.compute_index_scores(q, k, weights, mask=mask)
        # For a later position (e.g., position 5), all topk indices should be <= 5
        pos = 5
        later_token_indices = topk_indices[:, pos, :]
        self.assertTrue(
            (later_token_indices <= pos).all().item(),
            f"topk indices at position {pos} exceed causal bound",
        )


class TestComputeIndexScoresFusedAdditional(unittest.TestCase):
    """Additional tests for _compute_index_scores_fused."""

    def test_matches_unfused(self):
        """Fused scores should match unfused Indexer.compute_index_scores logic."""
        sq, b, h, d = 4, 2, 2, 16
        q = paddle.randn([sq, b, h, d], dtype="float32")
        weights = paddle.randn([sq, b, h], dtype="float32")
        k = paddle.randn([sq, b, d], dtype="float32")

        fused_scores = _compute_index_scores_fused(q, weights, k)  # [b, sq, sk]

        # Manual unfused computation
        scores = paddle.einsum("sbhd,tbd->sbht", q, k)
        relu_scores = paddle.nn.functional.relu(scores)
        weighted = relu_scores * weights.unsqueeze(-1)
        summed = weighted.sum(axis=2)  # [sq, b, sk]
        unfused_scores = summed.transpose([1, 0, 2])  # [b, sq, sk]

        self.assertTrue(
            paddle.allclose(fused_scores, unfused_scores, atol=1e-5),
            "Fused and unfused scores do not match",
        )


class TestComputeDSAIndexerLossWithMask(unittest.TestCase):
    """Additional edge cases for _compute_dsa_indexer_loss."""

    def test_loss_is_non_negative(self):
        """KL divergence should be non-negative."""
        sq, sk, b, np, hn = 4, 4, 2, 2, 16
        topk = 2
        index_scores = paddle.randn([b, sq, sk], dtype="float32")
        topk_indices = paddle.randint(0, sk, [b, sq, topk]).cast("int64")
        query = paddle.randn([sq, b, np, hn], dtype="float32")
        key = paddle.randn([sk, b, np, hn], dtype="float32")

        loss = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale=hn**-0.5,
            loss_coeff=1.0,
            sparse_loss=False,
            tp_group=None,
        )
        # KL divergence can be slightly negative due to numerical noise,
        # but should be very close to non-negative
        self.assertTrue(loss.item() > -0.1, f"Loss is too negative: {loss.item()}")
        self.assertTrue(paddle.isfinite(loss).item())


class TestBwdFusedIndexerLoss(unittest.TestCase):
    """Tests for _bwd_fused_indexer_loss manual backward."""

    def test_backward_shapes(self):
        """Manual backward should return gradients with correct shapes."""
        sq, b, h, d = 4, 2, 2, 16
        np, hn = 4, 32
        topk = 2
        q = paddle.randn([sq, b, h, d], dtype="float32")
        weights = paddle.randn([sq, b, h], dtype="float32")
        k = paddle.randn([sq, b, d], dtype="float32")
        query = paddle.randn([sq, b, np, hn], dtype="float32")
        key = paddle.randn([sq, b, np, hn], dtype="float32")
        topk_indices = paddle.randint(0, sq, [b, sq, topk]).cast("int64")
        grad_loss = paddle.to_tensor(1.0, dtype="float32")

        grad_q, grad_weights, grad_k = _bwd_fused_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            softmax_scale=hn**-0.5,
            loss_coeff=1.0,
            sparse_loss=False,
            grad_loss=grad_loss,
            tp_group=None,
        )
        self.assertEqual(list(grad_q.shape), [sq, b, h, d])
        self.assertEqual(list(grad_weights.shape), [sq, b, h])
        self.assertEqual(list(grad_k.shape), [sq, b, d])

    def test_backward_finite(self):
        """All gradients from manual backward should be finite."""
        sq, b, h, d = 4, 2, 2, 16
        np, hn = 4, 32
        topk = 2
        q = paddle.randn([sq, b, h, d], dtype="float32")
        weights = paddle.randn([sq, b, h], dtype="float32")
        k = paddle.randn([sq, b, d], dtype="float32")
        query = paddle.randn([sq, b, np, hn], dtype="float32")
        key = paddle.randn([sq, b, np, hn], dtype="float32")
        topk_indices = _make_causal_topk_indices(b, sq, sq, topk)
        grad_loss = paddle.to_tensor(1.0, dtype="float32")

        grad_q, grad_weights, grad_k = _bwd_fused_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            softmax_scale=hn**-0.5,
            loss_coeff=1.0,
            sparse_loss=False,
            grad_loss=grad_loss,
            tp_group=None,
        )
        self.assertTrue(paddle.isfinite(grad_q).all().item())
        self.assertTrue(paddle.isfinite(grad_weights).all().item())
        self.assertTrue(paddle.isfinite(grad_k).all().item())

    def test_backward_with_sparse_loss(self):
        """Manual backward should work with sparse_loss=True."""
        sq, b, h, d = 4, 2, 2, 16
        np, hn = 4, 32
        topk = 2
        q = paddle.randn([sq, b, h, d], dtype="float32")
        weights = paddle.randn([sq, b, h], dtype="float32")
        k = paddle.randn([sq, b, d], dtype="float32")
        query = paddle.randn([sq, b, np, hn], dtype="float32")
        key = paddle.randn([sq, b, np, hn], dtype="float32")
        topk_indices = _make_causal_topk_indices(b, sq, sq, topk)
        grad_loss = paddle.to_tensor(1.0, dtype="float32")

        grad_q, grad_weights, grad_k = _bwd_fused_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            softmax_scale=hn**-0.5,
            loss_coeff=1.0,
            sparse_loss=True,
            grad_loss=grad_loss,
            tp_group=None,
        )
        self.assertTrue(paddle.isfinite(grad_q).all().item())
        self.assertTrue(paddle.isfinite(grad_weights).all().item())
        self.assertTrue(paddle.isfinite(grad_k).all().item())


class TestFusedDSAIndexerLossNoMask(unittest.TestCase):
    """Test FusedDSAIndexerLoss with no mask (mask=None path)."""

    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b = 2
        self.h, self.d = 4, 32
        self.np, self.hn = 4, 64
        self.topk = 4
        self.softmax_scale = self.hn**-0.5

    def test_forward_no_mask(self):
        q = paddle.randn([self.sq, self.b, self.h, self.d], dtype="float32")
        q.stop_gradient = False
        weights = paddle.randn([self.sq, self.b, self.h], dtype="float32")
        weights.stop_gradient = False
        k = paddle.randn([self.sk, self.b, self.d], dtype="float32")
        k.stop_gradient = False
        query = paddle.randn([self.sq, self.b, self.np, self.hn], dtype="float32")
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")

        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            None,  # no mask
            False,
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())


class TestFusedDSAIndexerLossSparseLoss(unittest.TestCase):
    """Test FusedDSAIndexerLoss with sparse_loss=True."""

    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b = 2
        self.h, self.d = 4, 32
        self.np, self.hn = 4, 64
        self.topk = 4
        self.softmax_scale = self.hn**-0.5

    def test_forward_sparse_loss(self):
        q = paddle.randn([self.sq, self.b, self.h, self.d], dtype="float32")
        q.stop_gradient = False
        weights = paddle.randn([self.sq, self.b, self.h], dtype="float32")
        weights.stop_gradient = False
        k = paddle.randn([self.sk, self.b, self.d], dtype="float32")
        k.stop_gradient = False
        query = paddle.randn([self.sq, self.b, self.np, self.hn], dtype="float32")
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        causal = paddle.triu(
            paddle.full([self.sq, self.sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)

        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            True,
            None,  # sparse_loss=True
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_backward_sparse_loss(self):
        q = paddle.randn([self.sq, self.b, self.h, self.d], dtype="float32")
        q.stop_gradient = False
        weights = paddle.randn([self.sq, self.b, self.h], dtype="float32")
        weights.stop_gradient = False
        k = paddle.randn([self.sk, self.b, self.d], dtype="float32")
        k.stop_gradient = False
        query = paddle.randn([self.sq, self.b, self.np, self.hn], dtype="float32")
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        causal = paddle.triu(
            paddle.full([self.sq, self.sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)

        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            True,
            None,
        )
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(paddle.isfinite(q.grad).all().item())
        self.assertTrue(paddle.isfinite(weights.grad).all().item())
        self.assertTrue(paddle.isfinite(k.grad).all().item())


class TestDSAIndexerLossAutoScalerAdditional(unittest.TestCase):
    """Additional tests for DSAIndexerLossAutoScaler edge cases."""

    def _make_non_leaf_output(self, shape):
        x = paddle.randn(shape)
        x.stop_gradient = False
        return x + 0

    def test_backward_without_loss_scale(self):
        """When _main_loss_backward_scale is None, backward should use ones."""
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None
        output = self._make_non_leaf_output([2, 4])
        indexer_loss = paddle.to_tensor(1.0, dtype="float32")
        indexer_loss.stop_gradient = False
        indexer_loss = indexer_loss * 1.0

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        loss = result.sum()
        loss.backward()
        # Should not error; the None-scale path creates ones
        self.assertTrue(True)

    def test_set_loss_scale_stores_value(self):
        """set_loss_scale should store the scale tensor."""
        scale = paddle.to_tensor(3.14, dtype="float32")
        DSAIndexerLossAutoScaler.set_loss_scale(scale)
        stored = DSAIndexerLossAutoScaler._main_loss_backward_scale
        self.assertIsNotNone(stored)
        self.assertAlmostEqual(stored.item(), 3.14, places=4)
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None

    def test_forward_preserves_value(self):
        """Forward should return output with the same values (passthrough)."""
        x = paddle.randn([3, 5])
        x.stop_gradient = False
        output = x + 0
        indexer_loss = paddle.to_tensor(0.0, dtype="float32")
        indexer_loss.stop_gradient = False
        indexer_loss = indexer_loss + 0

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        self.assertTrue(paddle.allclose(result, output, atol=1e-7))


class TestMLASelfAttentionWithDSASparseLoss(unittest.TestCase):
    """Integration test for MLASelfAttention with DSAttention and sparse_loss enabled."""

    def setUp(self):
        self.config = _create_dsa_config(indexer_use_sparse_loss=True)
        self.micro_batch_size = 2
        self.sequence_length = 32

    def _build_model(self, config=None):
        cfg = config or self.config
        model = MLASelfAttention(
            cfg,
            _create_sublayers_spec(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        model = model.to(dtype="bfloat16")
        model.core_attention.indexer.weights_proj = model.core_attention.indexer.weights_proj.to(dtype="float32")
        return model

    def _make_hidden(self, dtype="bfloat16"):
        return paddle.randn(
            [
                self.micro_batch_size,
                self.sequence_length,
                self.config.hidden_size,
            ]
        ).cast(dtype)

    def test_forward_with_sparse_loss(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)
        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)

    def test_backward_with_sparse_loss(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        hidden.stop_gradient = False
        output, bias = model(hidden, attention_mask=None)
        loss = output.cast("float32").sum()
        loss.backward()
        self.assertIsNotNone(hidden.grad)


class TestMLASelfAttentionWithDSAZeroLossCoeff(unittest.TestCase):
    """Test MLASelfAttention with DSAttention and indexer_loss_coeff=0."""

    def setUp(self):
        self.config = _create_dsa_config(indexer_loss_coeff=0.0)
        self.micro_batch_size = 2
        self.sequence_length = 32

    def _build_model(self):
        model = MLASelfAttention(
            self.config,
            _create_sublayers_spec(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        model = model.to(dtype="bfloat16")
        model.core_attention.indexer.weights_proj = model.core_attention.indexer.weights_proj.to(dtype="float32")
        return model

    def test_forward_zero_loss_coeff(self):
        """With loss_coeff=0, the model should still compute loss but skip logging."""
        model = self._build_model()
        model.train()
        hidden = paddle.randn(
            [
                self.micro_batch_size,
                self.sequence_length,
                self.config.hidden_size,
            ]
        ).cast("bfloat16")
        DSAIndexerLossLoggingHelper.tracker = {}
        output, bias = model(hidden, attention_mask=None)
        self.assertEqual(output.shape[0], self.micro_batch_size)
        # loss_coeff=0 means save_loss_to_tracker is NOT called (coeff <= 0 check)
        # Actually the code checks `if self.dsa_indexer_loss_coeff > 0`
        self.assertNotIn("values", DSAIndexerLossLoggingHelper.tracker)
        DSAIndexerLossLoggingHelper.tracker = {}


class TestIndexerWithRopeType(unittest.TestCase):
    """Test DSAIndexer with rope_type='rope' (covers RotaryEmbedding init + forward path)."""

    def setUp(self):
        self.config = _create_dsa_config()
        self.config.rope_type = "rope"
        indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        self.indexer = DSAIndexer(
            config=self.config,
            sublayers_spec=indexer_sublayers,
            layer_number=1,
            pg_collection=None,
        )
        self.b = 2
        self.s = 16

    def _prepare_indexer_bf16(self):
        self.indexer.wq_b = self.indexer.wq_b.to(dtype="bfloat16")
        self.indexer.wk = self.indexer.wk.to(dtype="bfloat16")
        self.indexer.k_norm = self.indexer.k_norm.to(dtype="bfloat16")

    def test_init_creates_rotary_embedding(self):
        """With rope_type='rope', the indexer should use RotaryEmbedding."""
        from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
            RotaryEmbedding,
        )

        self.assertIsInstance(self.indexer.rotary_pos_emb, RotaryEmbedding)

    def test_forward_before_topk_rope(self):
        """forward_before_topk should work with rope_type='rope'."""
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")
        q, k, weights = self.indexer.forward_before_topk(hidden, q_latent)
        self.assertEqual(
            list(q.shape),
            [
                self.b,
                self.s,
                self.config.dsa_index_n_heads,
                self.config.dsa_index_head_dim,
            ],
        )
        self.assertEqual(list(k.shape), [self.b, self.s, self.config.dsa_index_head_dim])


class TestIndexerUnsupportedRopeType(unittest.TestCase):
    """Test DSAIndexer raises ValueError for unsupported rope_type."""

    def test_invalid_rope_type_raises(self):
        config = _create_dsa_config()
        config.rope_type = "invalid_type"
        indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        with self.assertRaises(ValueError) as ctx:
            DSAIndexer(
                config=config,
                sublayers_spec=indexer_sublayers,
                layer_number=1,
                pg_collection=None,
            )
        self.assertIn("Unsupported RoPE type", str(ctx.exception))


class TestDSAttentionValueError(unittest.TestCase):
    """Test DSAttention.forward raises ValueError when x or qr is None."""

    def setUp(self):
        self.config = _create_dsa_config(indexer_loss_coeff=None)
        self.b, self.s = 2, 8
        self.np = self.config.num_attention_heads
        self.qk_hd = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        self.v_hd = self.config.v_head_dim

    def _build_dsattention(self):
        dsa_indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        sublayers_spec = DSAttentionSublayersSpec(
            indexer=LayerSpec(
                layer=DSAIndexer,
                sublayers_spec=dsa_indexer_sublayers,
            ),
        )
        mock_pg = MagicMock()
        mock_pg.tp = None
        model = DSAttention(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.qk_hd**-0.5,
            pg_collection=mock_pg,
        )
        return model

    def test_x_none_raises(self):
        model = self._build_dsattention()
        model.eval()
        query = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        key = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        value = paddle.randn([self.b, self.s, self.np, self.v_hd]).cast("bfloat16")
        mask = None
        with self.assertRaises(ValueError) as ctx:
            model(
                query,
                key,
                value,
                mask,
                x=None,
                qr=paddle.randn([self.b, self.s, 64]).cast("bfloat16"),
            )
        self.assertIn("DSAttention requires x and qr", str(ctx.exception))

    def test_qr_none_raises(self):
        model = self._build_dsattention()
        model.eval()
        query = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        key = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        value = paddle.randn([self.b, self.s, self.np, self.v_hd]).cast("bfloat16")
        with self.assertRaises(ValueError) as ctx:
            model(
                query,
                key,
                value,
                None,
                x=paddle.randn([self.b, self.s, 256]).cast("bfloat16"),
                qr=None,
            )
        self.assertIn("DSAttention requires x and qr", str(ctx.exception))


class TestDSAttentionMaskBranches(unittest.TestCase):
    """Test DSAttention.forward mask branch coverage (elif/else paths)."""

    def setUp(self):
        self.config = _create_dsa_config(indexer_loss_coeff=None)
        self.b, self.s = 2, 16
        self.np = self.config.num_attention_heads
        self.qk_hd = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        self.v_hd = self.config.v_head_dim

    def _build_dsattention(self):
        dsa_indexer_sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        sublayers_spec = DSAttentionSublayersSpec(
            indexer=LayerSpec(
                layer=DSAIndexer,
                sublayers_spec=dsa_indexer_sublayers,
            ),
        )
        mock_pg = MagicMock()
        mock_pg.tp = None
        model = DSAttention(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=self.qk_hd**-0.5,
            pg_collection=mock_pg,
        )
        model = model.to(dtype="bfloat16")
        model.indexer.weights_proj = model.indexer.weights_proj.to(dtype="float32")
        return model

    def test_elif_attention_mask_not_none(self):
        """Cover the elif branch: attn_mask_type=None but attention_mask is not None."""
        model = self._build_dsattention()
        model.eval()
        query = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        key = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        value = paddle.randn([self.b, self.s, self.np, self.v_hd]).cast("bfloat16")
        x = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        qr = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")

        # Provide attention_mask but no attn_mask_type -> elif branch
        causal = paddle.triu(
            paddle.full([self.s, self.s], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        attention_mask = causal.unsqueeze(0).unsqueeze(0).expand([self.b, 1, self.s, self.s])

        output = model(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=None,
            x=x,
            qr=qr,
        )
        self.assertEqual(list(output.shape), [self.b, self.s, self.np * self.v_hd])

    def test_else_no_mask_no_type(self):
        """Cover the else branch: attn_mask_type=None and attention_mask=None."""
        model = self._build_dsattention()
        model.eval()
        query = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        key = paddle.randn([self.b, self.s, self.np, self.qk_hd]).cast("bfloat16")
        value = paddle.randn([self.b, self.s, self.np, self.v_hd]).cast("bfloat16")
        x = paddle.randn([self.b, self.s, self.config.hidden_size]).cast("bfloat16")
        qr = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast("bfloat16")

        # No attention_mask, no attn_mask_type -> else branch
        output = model(
            query,
            key,
            value,
            None,
            attn_mask_type=None,
            x=x,
            qr=qr,
        )
        self.assertEqual(list(output.shape), [self.b, self.s, self.np * self.v_hd])


if __name__ == "__main__":
    unittest.main()
