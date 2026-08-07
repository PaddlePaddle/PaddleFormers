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
Tests for softmax_scale branches in DotProductAttention.
Covers:
- _has_custom_softmax_scale flag behavior
- RR flash attention assert (lines 439, 541, 547)
- SDPA path with custom scale (line 493-495)
- flashmask path with fm_kwargs (line 450-454)
- flashmask CP path with extra_kwargs (line 589-596)
- context_parallel_utils.py softmax_scale NotImplementedError
"""

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

import math
import unittest
from unittest.mock import MagicMock, patch

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


class TestHasCustomSoftmaxScaleFlag(unittest.TestCase):
    """Tests for _has_custom_softmax_scale flag initialization."""

    def test_default_scale_flag_false(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertFalse(attn._has_custom_softmax_scale)

    def test_custom_scale_flag_true(self):
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.5,
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertEqual(attn.softmax_scale, 0.5)

    def test_apply_query_key_layer_scaling_sets_flag(self):
        config = _make_config(apply_query_key_layer_scaling=True)
        attn = DotProductAttention(
            config=config,
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        expected = (1.0 / math.sqrt(32)) / 2
        self.assertAlmostEqual(attn.softmax_scale, expected, places=5)

    def test_custom_scale_with_layer_scaling(self):
        config = _make_config(apply_query_key_layer_scaling=True)
        attn = DotProductAttention(
            config=config,
            layer_number=3,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.9,
        )
        self.assertTrue(attn._has_custom_softmax_scale)
        self.assertAlmostEqual(attn.softmax_scale, 0.9 / 3, places=5)


class TestRRFlashAttentionSoftmaxScaleAssert(unittest.TestCase):
    """
    Tests that RefinedRecompute flash attention paths raise AssertionError
    when _has_custom_softmax_scale is True.
    Covers dot_product_attention.py lines 439, 541, 547.
    """

    def _make_attn(
        self,
        softmax_scale=None,
        context_parallel_size=1,
        apply_query_key_layer_scaling=False,
    ):
        config = _make_config(
            context_parallel_size=context_parallel_size,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        )
        attn = DotProductAttention(
            config=config,
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=softmax_scale,
        )
        return attn

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_rr_packed_seq_path_raises_with_custom_scale(self, mock_fm):
        """Line 439: packed_seq path + use_rr + custom scale -> AssertionError."""
        attn = self._make_attn(softmax_scale=0.5)
        attn.rr_flashmask_attention_func = MagicMock()

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")
        packed_seq = MagicMock()
        packed_seq.cu_seqlens_kv = paddle.to_tensor([0, 2, 4])

        with self.assertRaises(NotImplementedError) as ctx:
            attn(
                query,
                key,
                value,
                None,
                attn_mask_startend_row_indices=startend,
                packed_seq_params=packed_seq,
                use_rr_flash_attention=True,
            )
        self.assertIn("RefinedRcomputeFlashMaskAttention", str(ctx.exception))

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_rr_packed_seq_path_ok_without_custom_scale(self, mock_fm):
        """Line 439: packed_seq path + use_rr + default scale -> no error."""
        attn = self._make_attn(softmax_scale=None)
        mock_rr = MagicMock(
            return_value=paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        )
        attn.rr_flashmask_attention_func = mock_rr

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")
        packed_seq = MagicMock()
        packed_seq.cu_seqlens_kv = paddle.to_tensor([0, 2, 4])

        result = attn(
            query,
            key,
            value,
            None,
            attn_mask_startend_row_indices=startend,
            packed_seq_params=packed_seq,
            use_rr_flash_attention=True,
        )
        mock_rr.assert_called_once()


class TestCPRRFlashAttentionSoftmaxScaleAssert(unittest.TestCase):
    """
    Tests that CP + RR flash attention raises AssertionError with custom scale.
    Covers dot_product_attention.py line 541 (CP + RR path).
    """

    def _make_cp_attn(self, softmax_scale=None):
        config = _make_config(context_parallel_size=2)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=softmax_scale,
        )
        return attn

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.flashmask_attention_cp"
    )
    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_cp_rr_path_raises_with_custom_scale(self, mock_fm, mock_fm_cp):
        """Line 541: CP>1 + use_rr + custom scale -> AssertionError."""
        attn = self._make_cp_attn(softmax_scale=0.5)
        attn.rr_flashmask_attention_cp_func = MagicMock()
        attn.expand_attn_mask_startend_row_indices_for_cp = MagicMock(
            return_value=paddle.zeros([1, 1, 4, 2], dtype="int32")
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")

        with self.assertRaises(NotImplementedError) as ctx:
            attn(
                query,
                key,
                value,
                None,
                attn_mask_startend_row_indices=startend,
                use_rr_flash_attention=True,
            )
        self.assertIn("RefinedRcomputeFlashMaskAttention", str(ctx.exception))


class TestNonCPRRFlashAttentionSoftmaxScaleAssert(unittest.TestCase):
    """
    Tests that non-CP RR flash attention raises AssertionError with custom scale.
    Covers dot_product_attention.py line 547 (non-CP, RR path).
    """

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_non_cp_rr_path_raises_with_custom_scale(self, mock_fm):
        """Line 547: CP=1 + use_rr + custom scale -> AssertionError."""
        config = _make_config(context_parallel_size=1)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.5,
        )
        attn.rr_flashmask_attention_func = MagicMock()

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")

        with self.assertRaises(NotImplementedError) as ctx:
            attn(
                query,
                key,
                value,
                None,
                attn_mask_startend_row_indices=startend,
                use_rr_flash_attention=True,
            )
        self.assertIn("RefinedRcomputeFlashMaskAttention", str(ctx.exception))

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_non_cp_rr_path_ok_without_custom_scale(self, mock_fm):
        """Line 547: CP=1 + use_rr + default scale -> no error."""
        config = _make_config(context_parallel_size=1)
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_rr = MagicMock(
            return_value=paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        )
        attn.rr_flashmask_attention_func = mock_rr

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")

        result = attn(
            query,
            key,
            value,
            None,
            attn_mask_startend_row_indices=startend,
            use_rr_flash_attention=True,
        )
        mock_rr.assert_called_once()


class TestSDPASoftmaxScale(unittest.TestCase):
    """
    Tests that SDPA path passes scale when _has_custom_softmax_scale is True.
    Covers dot_product_attention.py lines 493-495.
    """

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.paddle.nn.functional.scaled_dot_product_attention"
    )
    def test_sdpa_passes_custom_scale(self, mock_sdpa):
        """SDPA path includes scale= when custom softmax_scale is set."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.25,
        )
        mock_sdpa.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertIn("scale", call_kwargs)
        self.assertEqual(call_kwargs["scale"], 0.25)

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.paddle.nn.functional.scaled_dot_product_attention"
    )
    def test_sdpa_no_scale_when_default(self, mock_sdpa):
        """SDPA path does NOT include scale= when using default softmax_scale."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_sdpa.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertNotIn("scale", call_kwargs)

    @patch(
        "paddleformers.fleet.transformer.dot_product_attention.paddle.nn.functional.scaled_dot_product_attention"
    )
    def test_sdpa_passes_scale_with_layer_scaling(self, mock_sdpa):
        """SDPA passes scale when apply_query_key_layer_scaling is True."""
        config = _make_config(apply_query_key_layer_scaling=True)
        attn = DotProductAttention(
            config=config,
            layer_number=4,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_sdpa.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        attn(query, key, value, None, attn_mask_startend_row_indices=None)
        call_kwargs = mock_sdpa.call_args[1]
        self.assertIn("scale", call_kwargs)
        expected = (1.0 / math.sqrt(32)) / 4
        self.assertAlmostEqual(call_kwargs["scale"], expected, places=5)


class TestFlashmaskFmKwargsSoftmaxScale(unittest.TestCase):
    """
    Tests that flashmask non-CP path passes softmax_scale via fm_kwargs.
    Covers dot_product_attention.py lines 450-454.
    """

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_fm_kwargs_with_custom_scale(self, mock_fm):
        """fm_kwargs includes softmax_scale when _has_custom_softmax_scale is True."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.3,
        )
        mock_fm.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")
        packed_seq = MagicMock()
        packed_seq.cu_seqlens_kv = paddle.to_tensor([0, 2, 4])

        attn(
            query,
            key,
            value,
            None,
            attn_mask_startend_row_indices=startend,
            packed_seq_params=packed_seq,
        )
        call_kwargs = mock_fm.call_args[1]
        self.assertIn("softmax_scale", call_kwargs)
        self.assertEqual(call_kwargs["softmax_scale"], 0.3)

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_fm_kwargs_empty_when_default_scale(self, mock_fm):
        """fm_kwargs is empty when _has_custom_softmax_scale is False."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_fm.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")
        packed_seq = MagicMock()
        packed_seq.cu_seqlens_kv = paddle.to_tensor([0, 2, 4])

        attn(
            query,
            key,
            value,
            None,
            attn_mask_startend_row_indices=startend,
            packed_seq_params=packed_seq,
        )
        call_kwargs = mock_fm.call_args[1]
        self.assertNotIn("softmax_scale", call_kwargs)


class TestFlashmaskExtraKwargsSoftmaxScale(unittest.TestCase):
    """
    Tests that flashmask main path (non-packed, non-CP) passes softmax_scale
    via extra_kwargs. Covers dot_product_attention.py lines 589-596.
    """

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_extra_kwargs_with_custom_scale(self, mock_fm):
        """extra_kwargs includes softmax_scale when _has_custom_softmax_scale is True."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=0.7,
        )
        mock_fm.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")

        attn(query, key, value, None, attn_mask_startend_row_indices=startend)
        call_kwargs = mock_fm.call_args[1]
        self.assertIn("softmax_scale", call_kwargs)
        self.assertEqual(call_kwargs["softmax_scale"], 0.7)

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_extra_kwargs_empty_when_default_scale(self, mock_fm):
        """extra_kwargs has no softmax_scale when _has_custom_softmax_scale is False."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_fm.return_value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")

        attn(query, key, value, None, attn_mask_startend_row_indices=startend)
        call_kwargs = mock_fm.call_args[1]
        self.assertNotIn("softmax_scale", call_kwargs)

    @patch("paddleformers.fleet.transformer.dot_product_attention.flashmask_attention")
    def test_extra_kwargs_empty_when_rr(self, mock_fm):
        """extra_kwargs is empty dict when use_rr_flash_attention=True (default scale)."""
        config = _make_config()
        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        mock_rr = MagicMock(
            return_value=paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        )
        attn.rr_flashmask_attention_func = mock_rr

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 4], dtype="int32")

        attn(
            query,
            key,
            value,
            None,
            attn_mask_startend_row_indices=startend,
            use_rr_flash_attention=True,
        )
        call_kwargs = mock_rr.call_args[1]
        self.assertNotIn("softmax_scale", call_kwargs)


class TestCpFlashmaskSoftmaxScaleNotImplemented(unittest.TestCase):
    """
    Tests that context_parallel_utils.py correctly handles softmax_scale
    for different fa_version values in the forward path:
    - The else branch (fa_version 2/3) passes softmax_scale unconditionally
      to flashmask_attention (the kernel itself may ignore unsupported values).
    - The backward path for fa_version==2 raises NotImplementedError
      when softmax_scale is not None.
    - fa_version==4: passes softmax_scale to _flash_attn_fwd (covered elsewhere)
    """

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 3},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_fa3_passes_softmax_scale_unconditionally(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """fa_version==3 (else branch): softmax_scale is always passed to flashmask_attention."""
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0
        group.world_size = 2

        cp_flashmask_allgatherkv_balance_forward(
            query,
            key,
            value,
            startend,
            learnable_sink=None,
            group=group,
            causal=True,
            is_training=True,
            softmax_scale=0.5,
        )

        call_kwargs = mock_fm.call_args[1]
        self.assertEqual(call_kwargs["softmax_scale"], 0.5)

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 3},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_no_error_with_softmax_scale_none(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """cp_flashmask forward does NOT raise when softmax_scale is None on fa<4."""
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0

        try:
            cp_flashmask_allgatherkv_balance_forward(
                query,
                key,
                value,
                startend,
                learnable_sink=None,
                group=group,
                causal=True,
                is_training=True,
                softmax_scale=None,
            )
        except NotImplementedError:
            self.fail(
                "Should not raise NotImplementedError when softmax_scale is None"
            )

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 3},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_fa3_passes_softmax_scale_when_supported(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """fa_version==3: softmax_scale passed to flashmask_attention unconditionally."""
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0
        group.world_size = 2

        cp_flashmask_allgatherkv_balance_forward(
            query,
            key,
            value,
            startend,
            learnable_sink=None,
            group=group,
            causal=True,
            is_training=True,
            softmax_scale=0.125,
        )

        # Verify softmax_scale was passed through
        call_kwargs = mock_fm.call_args[1]
        self.assertEqual(call_kwargs["softmax_scale"], 0.125)

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 2},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_fa2_passes_softmax_scale(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """fa_version==2 (else branch): softmax_scale is passed through to flashmask_attention.

        Note: fa_version==2 does not actually support softmax_scale at the kernel level,
        but the forward function passes it unconditionally. The backward path will raise
        NotImplementedError for fa2 + softmax_scale != None.
        """
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0

        cp_flashmask_allgatherkv_balance_forward(
            query,
            key,
            value,
            startend,
            learnable_sink=None,
            group=group,
            causal=True,
            is_training=True,
            softmax_scale=0.5,
        )

        # forward passes softmax_scale unconditionally
        call_kwargs = mock_fm.call_args[1]
        self.assertEqual(call_kwargs["softmax_scale"], 0.5)

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 2},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_fa2_ok_without_softmax_scale(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """fa_version==2: no error when softmax_scale is None."""
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0

        try:
            cp_flashmask_allgatherkv_balance_forward(
                query,
                key,
                value,
                startend,
                learnable_sink=None,
                group=group,
                causal=True,
                is_training=True,
                softmax_scale=None,
            )
        except NotImplementedError:
            self.fail(
                "Should not raise NotImplementedError when softmax_scale is None on fa2"
            )

    @patch(
        "paddleformers.fleet.context_parallel_utils.preprocess_index_dual_chunks",
        return_value=paddle.zeros([1, 1, 4, 2], dtype="int32"),
    )
    @patch("paddleformers.fleet.context_parallel_utils.flashmask_attention")
    @patch("paddleformers.fleet.context_parallel_utils.paddle.distributed.all_gather")
    @patch("paddleformers.fleet.context_parallel_utils._flash_mask_available", False)
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": False},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.framework.get_flags",
        return_value={"FLAGS_flash_attn_version": 3},
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_push"
    )
    @patch(
        "paddleformers.fleet.context_parallel_utils.paddle.base.core.nvprof_nvtx_pop"
    )
    def test_forward_fa3_none_scale_passes_none(
        self,
        mock_pop,
        mock_push,
        mock_get_flags,
        mock_get_flags2,
        mock_all_gather,
        mock_fm,
        mock_preprocess,
    ):
        """fa_version==3: when softmax_scale is None, softmax_scale=None is passed through."""
        from paddleformers.fleet.context_parallel_utils import (
            cp_flashmask_allgatherkv_balance_forward,
        )

        def fake_all_gather(out_list, tensor, group):
            for i in range(len(out_list)):
                out_list[i] = tensor

        mock_all_gather.side_effect = fake_all_gather
        mock_fm.return_value = (
            paddle.randn([1, 4, 4, 32]).astype("bfloat16"),
            paddle.randn([1, 4, 1, 32]).astype("float32"),
        )

        query = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        key = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        value = paddle.randn([1, 4, 4, 32]).astype("bfloat16")
        startend = paddle.zeros([1, 1, 4, 2], dtype="int32")
        group = MagicMock()
        group.nranks = 2
        group.rank = 0
        group.world_size = 2

        cp_flashmask_allgatherkv_balance_forward(
            query,
            key,
            value,
            startend,
            learnable_sink=None,
            group=group,
            causal=True,
            is_training=True,
            softmax_scale=None,
        )

        # softmax_scale is passed unconditionally, value should be None
        call_kwargs = mock_fm.call_args[1]
        self.assertIsNone(call_kwargs["softmax_scale"])


if __name__ == "__main__":
    unittest.main()
