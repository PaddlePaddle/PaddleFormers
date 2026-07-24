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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import numpy as np
import paddle

from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
    _EagerQKScoresFn,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 2,
        "context_parallel_size": 1,
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "attention_dropout": 0.0,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "window_attn_skip_freq": None,
        "softmax_type": "vanilla",
        "flashmask_use_varlen": False,
        "params_dtype": "float32",
        "perform_initialization": True,
        "init_method": paddle.nn.initializer.Normal(0.02),
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestEagerQKScoresFn(unittest.TestCase):
    """Tests for the _EagerQKScoresFn PyLayer (forward + custom backward)."""

    def test_forward_matches_scaled_matmul(self):
        """Forward must equal scale * (query @ key_t)."""
        paddle.seed(0)
        b, sq, sk, hn = 2, 3, 4, 5
        query = paddle.randn([b, sq, hn])
        key_t = paddle.randn([b, hn, sk])
        scale = 0.7

        scores = _EagerQKScoresFn.apply(query, key_t, scale)
        ref = paddle.matmul(query, key_t) * scale

        self.assertEqual(scores.shape, [b, sq, sk])
        np.testing.assert_allclose(
            scores.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_backward_matches_autograd_reference(self):
        """The explicit matmul backward must match autograd through a plain
        scaled matmul."""
        paddle.seed(1)
        b, sq, sk, hn = 2, 3, 4, 5
        scale = 0.7

        query = paddle.randn([b, sq, hn])
        query.stop_gradient = False
        key_t = paddle.randn([b, hn, sk])
        key_t.stop_gradient = False

        scores = _EagerQKScoresFn.apply(query, key_t, scale)
        scores.sum().backward()
        grad_q = query.grad.numpy().copy()
        grad_k = key_t.grad.numpy().copy()

        query_ref = paddle.to_tensor(query.numpy())
        query_ref.stop_gradient = False
        key_ref = paddle.to_tensor(key_t.numpy())
        key_ref.stop_gradient = False
        (paddle.matmul(query_ref, key_ref) * scale).sum().backward()

        np.testing.assert_allclose(
            grad_q, query_ref.grad.numpy(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            grad_k, key_ref.grad.numpy(), rtol=1e-5, atol=1e-5
        )


class TestDotProductAttentionUseAccuracyCompatible(unittest.TestCase):
    """Tests for the use_accuracy_compatible code paths in
    DotProductAttention.forward (float32 inputs go through the eager QK
    scores branch)."""

    def _build(self, use_accuracy_compatible):
        config = _make_config(use_accuracy_compatible=use_accuracy_compatible)
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def _inputs(self, seed=0):
        paddle.seed(seed)
        bsz, seq_len, num_heads, head_dim = 1, 4, 4, 16
        query = paddle.randn([bsz, seq_len, num_heads, head_dim])
        key = paddle.randn([bsz, seq_len, num_heads, head_dim])
        value = paddle.randn([bsz, seq_len, num_heads, head_dim])
        return query, key, value, (bsz, seq_len, num_heads, head_dim)

    def test_forward_runs_eager_qk_branch(self):
        """use_accuracy_compatible=True must produce a valid output shape
        even when _attn_implementation is the default (non-eager)."""
        attn = self._build(True)
        query, key, value, (bsz, seq_len, nh, hd) = self._inputs()
        # bool mask: True = masked-out (upper triangle, strict)
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )

        out = attn(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )
        self.assertEqual(out.shape, [bsz, seq_len, nh * hd])

    def test_forces_softmax_in_fp32(self):
        """The accuracy-compatible path must force softmax_in_fp32 on the
        softmax module."""
        attn = self._build(True)
        attn.scale_mask_softmax.softmax_in_fp32 = False
        query, key, value, (bsz, seq_len, _, _) = self._inputs()
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )

        attn(query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal)
        self.assertTrue(attn.scale_mask_softmax.softmax_in_fp32)

    def test_float32_mask_is_converted(self):
        """A PaddleFormers-style float32 mask (1.0=attend, 0.0=mask) must be
        converted to bool semantics and yield the same result as the
        equivalent bool mask (True=masked-out)."""
        query, key, value, (bsz, seq_len, nh, hd) = self._inputs(seed=5)

        # float mask: lower-triangle ones (attend), strict upper-triangle zeros (mask)
        float_mask = paddle.tril(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="float32")
        )
        # equivalent bool mask: True where masked-out == strict upper triangle
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )

        attn_float = self._build(True)
        out_float = attn_float(
            query, key, value, float_mask, attn_mask_type=AttnMaskType.causal
        )

        attn_bool = self._build(True)
        out_bool = attn_bool(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )

        np.testing.assert_allclose(
            out_float.numpy(), out_bool.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_compatible_matches_default_baddbmm(self):
        """With float32 inputs and the same bool mask, the eager QK branch
        (use_accuracy_compatible=True) must match the default baddbmm branch
        (use_accuracy_compatible=False)."""
        query, key, value, (bsz, seq_len, nh, hd) = self._inputs(seed=9)
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )

        out_compat = self._build(True)(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )
        out_default = self._build(False)(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )

        np.testing.assert_allclose(
            out_compat.numpy(), out_default.numpy(), rtol=1e-4, atol=1e-4
        )

    def test_forward_backward_grads(self):
        """Gradients must flow through the eager QK scores PyLayer."""
        attn = self._build(True)
        query, key, value, (bsz, seq_len, _, _) = self._inputs(seed=13)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )

        out = attn(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )
        out.sum().backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)


class TestDotProductAttentionEagerSoftmaxDtype(unittest.TestCase):
    """Cover the dtype-specific softmax configuration in the eager /
    accuracy-compatible path (input_in_bf16 / input_in_fp16 branches)."""

    def _build_eager(self, params_dtype, bf16=False, fp16=False):
        config = _make_config(
            params_dtype=params_dtype,
            bf16=bf16,
            fp16=fp16,
        )
        config._attn_implementation = "eager"
        return DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

    def _run(self, attn, dtype):
        paddle.seed(21)
        bsz, seq_len, num_heads, head_dim = 1, 4, 4, 16
        query = paddle.randn([bsz, seq_len, num_heads, head_dim]).astype(dtype)
        key = paddle.randn([bsz, seq_len, num_heads, head_dim]).astype(dtype)
        value = paddle.randn([bsz, seq_len, num_heads, head_dim]).astype(dtype)
        bool_mask = paddle.triu(
            paddle.ones([bsz, 1, seq_len, seq_len], dtype="bool"),
            diagonal=1,
        )
        return attn(
            query, key, value, bool_mask, attn_mask_type=AttnMaskType.causal
        )

    def test_bf16_sets_input_in_bf16(self):
        """bfloat16 scores set input_in_bf16=True, input_in_fp16=False."""
        attn = self._build_eager("bfloat16", bf16=True)
        if not hasattr(attn.scale_mask_softmax, "input_in_bf16"):
            self.skipTest("softmax module has no input_in_bf16 attribute")
        out = self._run(attn, "bfloat16")
        self.assertEqual(out.dtype, paddle.bfloat16)
        self.assertTrue(attn.scale_mask_softmax.input_in_bf16)
        self.assertFalse(attn.scale_mask_softmax.input_in_fp16)

    def test_fp16_sets_input_in_fp16(self):
        """float16 scores set input_in_fp16=True, input_in_bf16=False."""
        attn = self._build_eager("float16", fp16=True)
        if not hasattr(attn.scale_mask_softmax, "input_in_bf16"):
            self.skipTest("softmax module has no input_in_bf16 attribute")
        out = self._run(attn, "float16")
        self.assertEqual(out.dtype, paddle.float16)
        self.assertTrue(attn.scale_mask_softmax.input_in_fp16)
        self.assertFalse(attn.scale_mask_softmax.input_in_bf16)

    def test_fp32_clears_both_flags(self):
        """float32 scores clear both input_in_bf16 and input_in_fp16."""
        attn = self._build_eager("float32")
        if not hasattr(attn.scale_mask_softmax, "input_in_bf16"):
            self.skipTest("softmax module has no input_in_bf16 attribute")
        self._run(attn, "float32")
        self.assertFalse(attn.scale_mask_softmax.input_in_bf16)
        self.assertFalse(attn.scale_mask_softmax.input_in_fp16)


if __name__ == "__main__":
    unittest.main()
