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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.refined_recompute.flash_attn import (
    _get_fa_version,
    flashattn_auto_cast,
)


class TestGetFAVersionXPU(unittest.TestCase):
    """Tests for _get_fa_version with XPU device."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="xpu:0",
    )
    def test_xpu_returns_version_2(self, mock_device):
        """Test XPU device always returns version 2."""
        result = _get_fa_version(64)
        self.assertEqual(result, 2)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="xpu:1",
    )
    def test_xpu_any_id(self, mock_device):
        """Test XPU device with any device ID returns version 2."""
        result = _get_fa_version(128)
        self.assertEqual(result, 2)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="xpu:0",
    )
    def test_xpu_different_hdim(self, mock_device):
        """Test XPU returns version 2 regardless of hdim."""
        for hdim in [32, 64, 128, 256]:
            result = _get_fa_version(hdim)
            self.assertEqual(result, 2)


class TestGetFAVersionGPU(unittest.TestCase):
    """Tests for _get_fa_version with GPU device."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.paddle.base.framework.get_flags")
    def test_gpu_returns_flag_value(self, mock_get_flags, mock_device):
        """Test GPU returns FLAGS_flash_attn_version."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 3}
        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
                return_value=MagicMock(parameters={}),
            ),
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
                return_value={"FLAGS_cudnn_deterministic": False},
            ),
        ):
            result = _get_fa_version(64)
            self.assertEqual(result, 3)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.paddle.base.framework.get_flags")
    def test_gpu_flag_version_2(self, mock_get_flags, mock_device):
        """Test GPU returns 2 when flag is set to 2."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 2}
        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
                return_value=MagicMock(parameters={}),
            ),
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
                return_value={"FLAGS_cudnn_deterministic": False},
            ),
        ):
            result = _get_fa_version(64)
            self.assertEqual(result, 2)


class TestGetFAVersionDeterministic(unittest.TestCase):
    """Tests for _get_fa_version with deterministic mode."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_no_block_mask_returns_2(self, mock_get_flags, mock_device):
        """Test deterministic mode returns 2 when no block_mask param."""
        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
            return_value=MagicMock(parameters={}),
        ):
            result = _get_fa_version(64)
            self.assertEqual(result, 2)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_with_block_mask_small_hdim(self, mock_get_flags, mock_device):
        """Test deterministic with block_mask and small hdim returns 2."""
        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
            return_value=MagicMock(parameters={"block_mask": MagicMock()}),
        ):
            result = _get_fa_version(64)
            self.assertEqual(result, 2)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_with_block_mask_large_hdim(self, mock_get_flags, mock_device):
        """Test deterministic with block_mask and large hdim returns 2."""
        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
            return_value=MagicMock(parameters={"block_mask": MagicMock()}),
        ):
            result = _get_fa_version(256)
            self.assertEqual(result, 2)

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
        return_value={"FLAGS_cudnn_deterministic": True},
    )
    def test_deterministic_with_block_mask_hdim_128(self, mock_get_flags, mock_device):
        """Test deterministic with block_mask and hdim=128 returns 2."""
        with patch(
            "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
            return_value=MagicMock(parameters={"block_mask": MagicMock()}),
        ):
            result = _get_fa_version(128)
            self.assertEqual(result, 2)


class TestGetFAVersionNonDeterministic(unittest.TestCase):
    """Tests for _get_fa_version with non-deterministic mode."""

    @patch(
        "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_device",
        return_value="gpu:0",
    )
    @patch("paddleformers.fleet.refined_recompute.flash_attn.paddle.base.framework.get_flags")
    def test_non_deterministic_returns_flag(self, mock_get_flags, mock_device):
        """Test non-deterministic returns flag value."""
        mock_get_flags.return_value = {"FLAGS_flash_attn_version": 4}
        with (
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.inspect.signature",
                return_value=MagicMock(parameters={}),
            ),
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn.paddle.get_flags",
                return_value={"FLAGS_cudnn_deterministic": False},
            ),
            patch(
                "paddleformers.fleet.refined_recompute.flash_attn._flash_mask_available",
                True,
            ),
        ):
            result = _get_fa_version(64)
            self.assertEqual(result, 4)


class TestFlashattnAutoCastBasic(unittest.TestCase):
    """Tests for flashattn_auto_cast basic behavior."""

    def test_all_same_dtype_bfloat16(self):
        """Test no-op when all inputs are already bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_all_same_dtype_float16(self):
        """Test no-op when all inputs are already float16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float16)
        k = paddle.randn([2, 4, 8], dtype=paddle.float16)
        v = paddle.randn([2, 4, 8], dtype=paddle.float16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_float32_to_bfloat16(self):
        """Test casting float32 tensors to bfloat16."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        self.assertEqual(k_out.dtype, paddle.bfloat16)
        self.assertEqual(v_out.dtype, paddle.bfloat16)

    def test_partial_cast(self):
        """Test only casting tensors that need it."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        v = paddle.randn([2, 4, 8], dtype=paddle.bfloat16)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v)
        self.assertEqual(q_out.dtype, paddle.bfloat16)
        # k and v already bfloat16, should not be cast
        self.assertIs(k_out, k)
        self.assertIs(v_out, v)

    def test_cast_to_float16(self):
        """Test casting to float16 target dtype."""
        q = paddle.randn([2, 4, 8], dtype=paddle.float32)
        k = paddle.randn([2, 4, 8], dtype=paddle.float32)
        v = paddle.randn([2, 4, 8], dtype=paddle.float32)
        q_out, k_out, v_out = flashattn_auto_cast(q, k, v, dtype=paddle.float16)
        self.assertEqual(q_out.dtype, paddle.float16)
        self.assertEqual(k_out.dtype, paddle.float16)
        self.assertEqual(v_out.dtype, paddle.float16)


if __name__ == "__main__":
    unittest.main()
