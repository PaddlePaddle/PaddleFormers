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
from unittest.mock import MagicMock, patch

import paddle
from paddlefleet_ops.flash_mask_facade import get_fa_version

from paddleformers.fleet.refined_recompute.flash_attn import (
    FlashAttnFunctor,
    FlashMaskAttnFunctor,
    RefinedRcomputeFlashAttention,
    RefinedRcomputeFlashMaskAttention,
    flashattn_auto_cast,
)


class TestGetFAVersion(unittest.TestCase):
    """Tests for get_fa_version function (flash_mask_facade)."""

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="xpu:0",
    )
    def test_xpu_returns_version_2(self, mock_device):
        """Test that XPU device returns version 2."""
        result = get_fa_version(64)
        self.assertEqual(result, 2)

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    def test_gpu_returns_flag_value(self, mock_get_flags, mock_device):
        """Test that GPU returns the FLAGS_flash_attn_version value."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 3}
        with patch(
            "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
            return_value={"FLAGS_cudnn_deterministic": False},
        ):
            result = get_fa_version(64)
            self.assertEqual(result, 3)

    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddlefleet_ops.flash_mask_facade.paddle.base.framework.get_flags")
    @patch(
        "paddlefleet_ops.flash_mask_facade.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_fa3_large_hdim_returns_2(
        self, mock_get_flags, mock_base_flags, mock_device
    ):
        """Test FA3 + deterministic + hdim>128 falls back to version 2."""
        mock_base_flags.return_value = {"FLAGS_flash_attn_version": 3}
        result = get_fa_version(192)
        self.assertEqual(result, 2)


class TestFlashattnAutoCast(unittest.TestCase):
    """Tests for flashattn_auto_cast function."""

    def test_no_cast_needed(self):
        """Test no cast when tensors are already bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_cast_from_float32(self):
        """Test casting from float32 to bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_cast_to_custom_dtype(self):
        """Test casting to custom dtype."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertEqual(q_out.dtype, paddle.float16)
        self.assertEqual(k_out.dtype, paddle.float16)
        self.assertEqual(v_out.dtype, paddle.float16)

    def test_mixed_dtypes(self):
        """Test casting with mixed input dtypes."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)


class TestFlashAttnFunctorForwardVersion3(unittest.TestCase):
    """Tests for FlashAttnFunctor.forward with FA version 3."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=3,
    )
    def test_forward_version_3_saves_correct_tensors(self, mock_version):
        """Test forward with version 3 saves correct tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashAttnFunctor.apply(q, k, v, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestFlashAttnFunctorForwardVersion4(unittest.TestCase):
    """Tests for FlashAttnFunctor.forward with FA version 4."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=4,
    )
    def test_forward_version_4_saves_correct_tensors(self, mock_version):
        """Test forward with version 4 saves correct tensors."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashAttnFunctor.apply(q, k, v, hold_tensors)
        self.assertEqual(result.shape, result_attn.shape)


class TestRefinedRcomputeFlashAttentionFirstFwd(unittest.TestCase):
    """Tests for RefinedRcomputeFlashAttention._first_fwd."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=3,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flash_attn_v3")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_version_3(
        self, mock_tracer, mock_flash_v3, mock_version
    ):
        """Test _first_fwd with version 3 puts tensors in queue."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_flash_v3.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4], dtype=paddle.float32),
        )

        attn = RefinedRcomputeFlashAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        result = attn.forward(q, k, v, training=True)
        self.assertFalse(attn._hold_tensors_queue.empty())

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=2,
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn._C_ops.flash_attn")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_version_2(self, mock_tracer, mock_flash, mock_version):
        """Test _first_fwd with version 2 stores additional fields."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj

        mock_flash.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            None,
            paddle.randn([2, 4], dtype=paddle.float32),
            paddle.zeros([2], dtype=paddle.int64),
        )

        attn = RefinedRcomputeFlashAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)

        result, weights = attn.forward(
            q, k, v, dropout=0.1, return_softmax=True, training=True
        )
        hold_tensors = attn._hold_tensors_queue.get()
        self.assertIn("seed_offset", hold_tensors)
        self.assertIn("dropout", hold_tensors)


class TestFlashMaskAttnFunctorVersion3(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor with FA version 3."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=3,
    )
    def test_forward_version_3(self, mock_version):
        """Test forward with version 3."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashMaskAttnFunctor.apply(
            q, k, v, startend, None, hold_tensors
        )
        self.assertEqual(result.shape, result_attn.shape)


class TestFlashMaskAttnFunctorVersion4(unittest.TestCase):
    """Tests for FlashMaskAttnFunctor with FA version 4."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=4,
    )
    def test_forward_version_4(self, mock_version):
        """Test forward with version 4."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)
        result_attn = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        softmax_lse = paddle.randn([2, 4], dtype=paddle.float32)

        hold_tensors = {
            "result_attention": result_attn,
            "softmax_lse": softmax_lse,
            "causal": True,
        }

        result = FlashMaskAttnFunctor.apply(
            q, k, v, startend, None, hold_tensors
        )
        self.assertEqual(result.shape, result_attn.shape)


class TestRefinedRcomputeFlashMaskAttentionFirstFwdV3(unittest.TestCase):
    """Tests for RefinedRcomputeFlashMaskAttention._first_fwd with v3."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.get_fa_version",
        return_value=3,
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn._C_ops.flashmask_attention_v2"
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.inspect.signature")
    @patch("paddleformers.fleet.refined_recompute.flash_attn.framework._dygraph_tracer")
    def test_first_fwd_v3_with_block_mask(
        self, mock_tracer, mock_sig, mock_flash_v2, mock_version
    ):
        """Test _first_fwd with v3 and block_mask parameter."""
        mock_tracer_obj = MagicMock()
        mock_tracer_obj._has_grad = False
        mock_tracer.return_value = mock_tracer_obj
        mock_sig.return_value.parameters = {"block_mask": MagicMock()}

        mock_flash_v2.return_value = (
            paddle.randn([2, 4, 8], dtype=paddle.bfloat16),
            paddle.randn([2, 4], dtype=paddle.float32),
        )

        attn = RefinedRcomputeFlashMaskAttention()
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        startend = paddle.to_tensor([0, 4, 8], dtype=paddle.int32)

        result = attn.forward(q, k, v, startend, causal=False)
        self.assertTrue(result is not None)


if __name__ == "__main__":
    unittest.main()
