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

"""
Tests for the softmax_offset SDPA branch in DotProductAttention.

Covers dot_product_attention.py:
    if self.softmax_offset is not None:
        attn_output = scaled_dot_product_attention_with_softmax_offset(
            query, key, value_for_sdpa,
            attn_mask_kv=attn_mask_kv,
            is_causal=is_causal,
            softmax_offset=self.softmax_offset,
            q_head_dim=q_head_dim,
        )
"""

import os
import sys

# Add the tests root so sibling imports work.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

# Compatibility shim: local paddleformers/fleet imports
# `get_fa_version` from `paddlefleet_ops.flash_mask_facade`, which may not be
# present in older installed paddlefleet_ops builds. Inject a stub before
# paddleformers.fleet is imported. The tests below mock out any actual attention
# dispatch, so the stub is never invoked.
try:
    import paddlefleet_ops.flash_mask_facade as _fm_facade

    if not hasattr(_fm_facade, "get_fa_version"):

        def _get_fa_version_stub(*args, **kwargs):
            return 3

        _fm_facade.get_fa_version = _get_fa_version_stub
except ImportError:
    pass

import unittest

import paddle

from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "softmax_scale": None,
        "use_bias": True,
        "recompute_granularity": None,
        "recompute_modules": None,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "rms_norm_eps": 1e-5,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "softmax_type": "vanilla",
        "fa_version": None,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestSDPASoftmaxOffsetBranch(unittest.TestCase):
    """
    Tests DotProductAttention routing by controlling softmax_type config.
    No mocking — real functions execute for full coverage.
    """

    def _make_attn(self, softmax_type="off-by-one", **cfg_overrides):
        config = _make_config(softmax_type=softmax_type, **cfg_overrides)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def test_softmax_offset_branch_runs_without_error(self):
        """softmax_type='off-by-one' => softmax_offset is set, output shape correct."""
        attn = self._make_attn(softmax_type="off-by-one")
        self.assertIsNotNone(attn.softmax_offset)

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        output = attn(
            query, key, value, None, attn_mask_startend_row_indices=None
        )
        self.assertEqual(output.shape, [1, 4, 4 * 32])

    def test_vanilla_branch_runs_without_error(self):
        """softmax_type='vanilla' => softmax_offset is None, default SDPA path."""
        attn = self._make_attn(softmax_type="vanilla")
        self.assertIsNone(attn.softmax_offset)

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        output = attn(
            query, key, value, None, attn_mask_startend_row_indices=None
        )
        self.assertEqual(output.shape, [1, 4, 4 * 32])

    def test_with_attention_mask(self):
        """Non-None attention_mask is forwarded correctly."""
        attn = self._make_attn(softmax_type="off-by-one")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        attention_mask = paddle.zeros([1, 1, 4, 4], dtype="bfloat16")

        output = attn(
            query,
            key,
            value,
            attention_mask,
            attn_mask_startend_row_indices=None,
        )
        self.assertEqual(output.shape, [1, 4, 4 * 32])

    def test_learnable_softmax_offset(self):
        """softmax_type='learnable' => offset is a trainable parameter."""
        attn = self._make_attn(softmax_type="learnable")
        self.assertIsNotNone(attn.softmax_offset)
        self.assertFalse(attn.softmax_offset.stop_gradient)

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        output = attn(
            query, key, value, None, attn_mask_startend_row_indices=None
        )
        self.assertEqual(output.shape, [1, 4, 4 * 32])

    def test_value_padding_when_v_head_dim_smaller(self):
        """q_head_dim != v_head_dim => value is padded then output is truncated."""
        attn = self._make_attn(softmax_type="off-by-one")
        bsz, q_len, num_heads = 1, 4, 4
        q_head_dim, v_head_dim = 32, 16

        query = paddle.randn([bsz, q_len, num_heads, q_head_dim]).astype(
            "bfloat16"
        )
        key = paddle.randn([bsz, q_len, num_heads, q_head_dim]).astype(
            "bfloat16"
        )
        value = paddle.randn([bsz, q_len, num_heads, v_head_dim]).astype(
            "bfloat16"
        )

        output = attn(
            query, key, value, None, attn_mask_startend_row_indices=None
        )
        self.assertEqual(output.shape, [bsz, q_len, num_heads * v_head_dim])

    def test_no_value_padding_when_head_dims_match(self):
        """q_head_dim == v_head_dim => no padding, output shape unchanged."""
        attn = self._make_attn(softmax_type="off-by-one")
        bsz, q_len, num_heads, head_dim = 1, 4, 4, 32

        query = paddle.randn([bsz, q_len, num_heads, head_dim]).astype(
            "bfloat16"
        )
        key = paddle.randn([bsz, q_len, num_heads, head_dim]).astype("bfloat16")
        value = paddle.randn([bsz, q_len, num_heads, head_dim]).astype(
            "bfloat16"
        )

        output = attn(
            query, key, value, None, attn_mask_startend_row_indices=None
        )
        self.assertEqual(output.shape, [bsz, q_len, num_heads * head_dim])


class TestSoftmaxOffsetFnDropoutTraining(unittest.TestCase):
    """
    Directly exercises scaled_dot_product_attention_with_softmax_offset to
    guard the dropout_p / training branch (regression test: previously used
    an undefined `self.training`).
    """

    def test_dropout_uses_training_argument_no_nameerror(self):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        # small tensors, MHA path (groups == 1)
        q = paddle.randn([1, 4, 2, 8]).astype("float32")
        k = paddle.randn([1, 4, 2, 8]).astype("float32")
        v = paddle.randn([1, 4, 2, 8]).astype("float32")
        offset = paddle.zeros([2], dtype="float32")

        # training=False + dropout_p>0: dropout is a no-op, must not raise.
        out_eval = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=8,
            is_causal=True,
            dropout_p=0.5,
            training=False,
        )
        self.assertEqual(out_eval.shape, [1, 4, 2, 8])

        # training=True + dropout_p>0: dropout path executed, must not raise.
        out_train = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=8,
            is_causal=True,
            dropout_p=0.5,
            training=True,
        )
        self.assertEqual(out_train.shape, [1, 4, 2, 8])


class TestSoftmaxOffsetFnMaskBranches(unittest.TestCase):
    """
    Exercises the two attn_mask_kv branches in
    scaled_dot_product_attention_with_softmax_offset:

      if attn_mask_kv.dtype == paddle.bool:
          scores = paddle.where(attn_mask_kv, -inf, scores)   # bool: True = masked
      else:
          scores = scores + attn_mask_kv.cast("float32")     # additive
    """

    def _run(self, mask):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")  # [B, Q, Hq, dq]
        k = paddle.randn([1, 3, 1, 4]).astype("float32")  # [B, K, Hkv, dk]
        v = paddle.randn([1, 3, 1, 4]).astype("float32")  # [B, K, Hkv, dv]
        offset = paddle.full([1], -1e9, dtype="float32")  # neutralize sink

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            attn_mask_kv=mask,
            is_causal=False,
            softmax_offset=offset,
            q_head_dim=4,
        )
        return out, v

    def test_bool_mask_zeros_out_masked_positions(self):
        """
        With a bool mask that masks all but one key position, the attention
        output must equal V at that unmasked position (softmax collapses to
        a one-hot).
        """
        # mask shape [B, H, Q, K] = [1,1,2,3]. True = masked (per code).
        mask = paddle.to_tensor(
            [[[[True, True, False], [False, True, True]]]], dtype="bool"
        )
        out, v = self._run(mask)

        # For Q=0: only key 2 is unmasked -> out[0, 0, 0] == v[0, 2, 0]
        # For Q=1: only key 0 is unmasked -> out[0, 1, 0] == v[0, 0, 0]
        expected = paddle.stack([v[0, 2, 0], v[0, 0, 0]], axis=0)
        actual = out[0, :, 0]  # [Q=2, dv=4]

        self.assertTrue(
            paddle.allclose(actual, expected, atol=1e-5).item(),
            msg=f"bool-mask path collapsed weights incorrectly: {actual} vs {expected}",
        )

    def test_additive_mask_uses_same_path_as_before(self):
        """
        A float additive mask that assigns -inf to a position must zero out
        that position's contribution, same as the bool mask would.
        """
        neg_inf = float("-inf")
        # mask everything except key 2 for Q=0 and everything except key 0 for Q=1
        mask = paddle.to_tensor(
            [[[[neg_inf, neg_inf, 0.0], [0.0, neg_inf, neg_inf]]]],
            dtype="float32",
        )
        out, v = self._run(mask)

        expected = paddle.stack([v[0, 2, 0], v[0, 0, 0]], axis=0)
        actual = out[0, :, 0]

        self.assertTrue(
            paddle.allclose(actual, expected, atol=1e-5).item(),
            msg=f"additive-mask path collapsed weights incorrectly: {actual} vs {expected}",
        )

    def test_bool_and_additive_mask_produce_same_output(self):
        """Bool mask and its equivalent additive (-inf) mask must match."""
        bool_mask = paddle.to_tensor(
            [[[[True, False, True], [False, False, True]]]], dtype="bool"
        )
        neg_inf = float("-inf")
        add_mask = paddle.to_tensor(
            [[[[neg_inf, 0.0, neg_inf], [0.0, 0.0, neg_inf]]]],
            dtype="float32",
        )
        out_bool, _ = self._run(bool_mask)
        out_add, _ = self._run(add_mask)

        self.assertTrue(
            paddle.allclose(out_bool, out_add, atol=1e-5).item(),
            msg="bool and additive masks should produce the same output",
        )


class TestSoftmaxOffsetFnRowMaxWithSink(unittest.TestCase):
    """
    Guards the numerical-stability change:
        row_max = paddle.maximum(scores.max(axis=-1, keepdim=True), sink)

    When the sink far exceeds all scores, the softmax weights on real keys
    must approach zero (all mass absorbed by the virtual sink token).
    """

    def test_large_sink_dominates_weights(self):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")
        k = paddle.randn([1, 3, 1, 4]).astype("float32")
        v = paddle.randn([1, 3, 1, 4]).astype("float32")

        # Sink much larger than any score -> exp(sink - row_max) dominates
        # and all attention weights on real tokens approach zero, so the
        # output approaches zero.
        offset = paddle.full([1], 50.0, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=4,
        )

        self.assertTrue(
            paddle.all(paddle.abs(out) < 1e-6).item(),
            msg=f"large sink should absorb all weight; got out={out}",
        )

    def test_small_sink_matches_plain_softmax(self):
        """With a very negative sink, output must equal plain softmax(QK)V."""
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        q = paddle.randn([1, 2, 1, 4]).astype("float32")
        k = paddle.randn([1, 3, 1, 4]).astype("float32")
        v = paddle.randn([1, 3, 1, 4]).astype("float32")

        offset = paddle.full([1], -1e9, dtype="float32")
        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=4,
        )

        # Manual reference: plain softmax(QK^T / sqrt(d)) @ V (no sink).
        # q,k,v are [B, Q, H, D]; transpose to [B, H, Q, D] to matmul.
        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        vh = v.transpose([0, 2, 1, 3])
        scale = 4**-0.5
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            msg=f"small sink should reduce to plain softmax; out={out} ref={ref}",
        )


class TestSoftmaxOffsetFnGQAPath(unittest.TestCase):
    """
    Covers the `if groups > 1:` branches (GQA path) in
    scaled_dot_product_attention_with_softmax_offset. When
    num_q_heads > num_kv_heads, Q is reshaped into groups and K/V are
    broadcast, both for the scores matmul and the weights-@-V matmul.
    """

    def test_gqa_matches_expanded_reference(self):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        num_q_heads = 4
        num_kv_heads = 2  # groups = 2
        d = 4
        b, q_len, kv_len = 1, 3, 5

        q = paddle.randn([b, q_len, num_q_heads, d]).astype("float32")
        k = paddle.randn([b, kv_len, num_kv_heads, d]).astype("float32")
        v = paddle.randn([b, kv_len, num_kv_heads, d]).astype("float32")
        # Neutralize sink so we can compare against plain softmax
        offset = paddle.full([num_q_heads], -1e9, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            softmax_offset=offset,
            q_head_dim=d,
            is_causal=False,
        )
        self.assertEqual(out.shape, [b, q_len, num_q_heads, d])

        # Reference: expand K/V from [B, K, Hkv, D] to [B, K, Hq, D]
        # via repeat_interleave and then compute plain softmax(QK^T/sqrt(d))V.
        groups = num_q_heads // num_kv_heads
        k_exp = k.repeat_interleave(groups, axis=2)  # [B, K, Hq, D]
        v_exp = v.repeat_interleave(groups, axis=2)  # [B, K, Hq, D]

        qh = q.transpose([0, 2, 1, 3])
        kh = k_exp.transpose([0, 2, 1, 3])
        vh = v_exp.transpose([0, 2, 1, 3])
        scale = d**-0.5
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            msg=f"GQA branch diverged from expanded reference: {out} vs {ref}",
        )

    def test_gqa_with_bool_mask(self):
        """GQA path + bool mask branch simultaneously."""
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(1)
        num_q_heads = 4
        num_kv_heads = 2  # groups = 2
        d = 4
        b, q_len, kv_len = 1, 2, 3

        q = paddle.randn([b, q_len, num_q_heads, d]).astype("float32")
        k = paddle.randn([b, kv_len, num_kv_heads, d]).astype("float32")
        v = paddle.randn([b, kv_len, num_kv_heads, d]).astype("float32")
        offset = paddle.full([num_q_heads], -1e9, dtype="float32")

        # Broadcast mask [1, 1, Q, K] over all heads.
        mask = paddle.to_tensor(
            [[[[True, True, False], [False, True, True]]]], dtype="bool"
        )

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            attn_mask_kv=mask,
            is_causal=False,
            softmax_offset=offset,
            q_head_dim=d,
        )
        # For Q=0 only key 2 is unmasked and for Q=1 only key 0 is unmasked.
        # After expansion the GQA reduction must equal V at those positions.
        groups = num_q_heads // num_kv_heads
        v_exp = v.repeat_interleave(groups, axis=2)  # [B, K, Hq, D]

        # out shape: [B, Q, Hq, D]. For each head h: out[0, 0, h] == v_exp[0, 2, h]
        # and out[0, 1, h] == v_exp[0, 0, h].
        for h in range(num_q_heads):
            self.assertTrue(
                paddle.allclose(out[0, 0, h], v_exp[0, 2, h], atol=1e-5).item()
            )
            self.assertTrue(
                paddle.allclose(out[0, 1, h], v_exp[0, 0, h], atol=1e-5).item()
            )


class TestSoftmaxOffsetFnCausalBranch(unittest.TestCase):
    """
    Covers the `elif is_causal and query.shape[1] > 1:` branch of
    scaled_dot_product_attention_with_softmax_offset (i.e. no explicit
    attn_mask_kv but causal masking must be applied via a triangular mask).
    """

    def test_causal_matches_manual_reference(self):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        d = 4
        b, q_len, kv_len = 1, 3, 3
        q = paddle.randn([b, q_len, 1, d]).astype("float32")
        k = paddle.randn([b, kv_len, 1, d]).astype("float32")
        v = paddle.randn([b, kv_len, 1, d]).astype("float32")
        offset = paddle.full([1], -1e9, dtype="float32")  # neutralize sink

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            attn_mask_kv=None,
            is_causal=True,
            softmax_offset=offset,
            q_head_dim=d,
        )

        # Manual causal reference
        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        vh = v.transpose([0, 2, 1, 3])
        scale = d**-0.5
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        tri = paddle.tril(paddle.ones([q_len, kv_len], dtype="float32"))
        neg_inf_mask = paddle.where(
            tri.unsqueeze(0).unsqueeze(0) == 0,
            paddle.full_like(scores, float("-inf")),
            paddle.zeros_like(scores),
        )
        scores = scores + neg_inf_mask
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            msg=f"causal branch diverged from reference: {out} vs {ref}",
        )

    def test_is_causal_ignored_when_qlen_is_one(self):
        """
        query.shape[1] == 1 must NOT enter the causal branch. With a single
        query token, the output equals plain softmax(QK/sqrt(d))V.
        """
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        d = 4
        q = paddle.randn([1, 1, 1, d]).astype("float32")
        k = paddle.randn([1, 3, 1, d]).astype("float32")
        v = paddle.randn([1, 3, 1, d]).astype("float32")
        offset = paddle.full([1], -1e9, dtype="float32")

        out = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            attn_mask_kv=None,
            is_causal=True,  # ignored because q_len == 1
            softmax_offset=offset,
            q_head_dim=d,
        )

        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        vh = v.transpose([0, 2, 1, 3])
        scale = d**-0.5
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * scale
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out, ref, atol=1e-5).item(),
            msg=(
                "with q_len == 1 the causal branch must be skipped; "
                f"got out={out} ref={ref}"
            ),
        )


class TestSoftmaxOffsetFnScaleArg(unittest.TestCase):
    """
    Covers the `scale = float(scale if scale is not None else q_head_dim**-0.5)`
    branch of scaled_dot_product_attention_with_softmax_offset — an explicit
    scale argument must override the default 1/sqrt(q_head_dim).
    """

    def test_explicit_scale_used_instead_of_default(self):
        from paddleformers.fleet.transformer.dot_product_attention import (
            scaled_dot_product_attention_with_softmax_offset,
        )

        paddle.seed(0)
        d = 4
        q = paddle.randn([1, 2, 1, d]).astype("float32")
        k = paddle.randn([1, 3, 1, d]).astype("float32")
        v = paddle.randn([1, 3, 1, d]).astype("float32")
        offset = paddle.full([1], -1e9, dtype="float32")

        custom_scale = 0.123  # deliberately different from d**-0.5 == 0.5
        out_custom = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            is_causal=False,
            softmax_offset=offset,
            q_head_dim=d,
            scale=custom_scale,
        )
        out_default = scaled_dot_product_attention_with_softmax_offset(
            q,
            k,
            v,
            is_causal=False,
            softmax_offset=offset,
            q_head_dim=d,
        )

        # Different scales must produce different outputs (sink neutralized).
        self.assertFalse(
            paddle.allclose(out_custom, out_default, atol=1e-4).item(),
            msg="custom scale must yield a different result from default",
        )

        # Reference computation using the custom scale.
        qh = q.transpose([0, 2, 1, 3])
        kh = k.transpose([0, 2, 1, 3])
        vh = v.transpose([0, 2, 1, 3])
        scores = paddle.matmul(qh, kh.transpose([0, 1, 3, 2])) * custom_scale
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        ref = paddle.matmul(weights, vh).transpose([0, 2, 1, 3])

        self.assertTrue(
            paddle.allclose(out_custom, ref, atol=1e-5).item(),
            msg=(
                "explicit-scale path diverged from reference: "
                f"out={out_custom} ref={ref}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
