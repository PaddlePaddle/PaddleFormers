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


# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import unittest

import paddle

from paddleformers.fleet.models.multimodal.context_parallel import (
    get_packed_seq_params,
    get_padding,
)


class TestGetPadding(unittest.TestCase):
    """Test get_padding function."""

    def test_no_padding_needed(self):
        """Test when seq_len is already a multiple of padding_factor."""
        padding = get_padding(
            seq_len=128,
            cp_size=1,
            tp_size=1,
            has_sp=False,
        )
        self.assertEqual(padding, 0)

    def test_sp_only(self):
        """Test padding with sequence parallelism only."""
        padding = get_padding(
            seq_len=10,
            cp_size=1,
            tp_size=4,
            has_sp=True,
        )
        # Should pad to multiple of 4
        expected = 4 - 10 % 4
        if expected == 4:
            expected = 0
        self.assertEqual(padding, expected)

    def test_cp_only(self):
        """Test padding with context parallelism only."""
        padding = get_padding(
            seq_len=10,
            cp_size=4,
            tp_size=1,
            has_sp=False,
        )
        # Should pad to multiple of cp_size * 2 = 8
        expected = 8 - 10 % 8
        if expected == 8:
            expected = 0
        self.assertEqual(padding, expected)

    def test_sp_and_cp(self):
        """Test padding with both SP and CP."""
        padding = get_padding(
            seq_len=10,
            cp_size=2,
            tp_size=2,
            has_sp=True,
        )
        # Should pad to multiple of tp_size * cp_size * 2 = 8
        expected = 8 - 10 % 8
        if expected == 8:
            expected = 0
        self.assertEqual(padding, expected)

    def test_fp8_mxfp8(self):
        """Test padding with FP8 enabled and mxfp8 recipe."""
        padding = get_padding(
            seq_len=10,
            cp_size=1,
            tp_size=1,
            has_sp=False,
            fp8_enabled=True,
            fp8_recipe="mxfp8",
        )
        # mxfp8 uses padding_factor = 32
        expected = 32 - 10 % 32
        if expected == 32:
            expected = 0
        self.assertEqual(padding, expected)

    def test_fp8_non_mxfp8(self):
        """Test padding with FP8 enabled and non-mxfp8 recipe."""
        padding = get_padding(
            seq_len=10,
            cp_size=1,
            tp_size=1,
            has_sp=False,
            fp8_enabled=True,
            fp8_recipe="other",
        )
        # Non-mxfp8 uses padding_factor = 16
        expected = 16 - 10 % 16
        if expected == 16:
            expected = 0
        self.assertEqual(padding, expected)

    def test_decoder_tp_comm_overlap(self):
        """Test padding with decoder TP comm overlap."""
        padding = get_padding(
            seq_len=128,
            cp_size=1,
            tp_size=1,
            has_sp=True,
            decoder_tp_comm_overlap=True,
            decoder_seq_len=256,
        )
        self.assertEqual(padding, 256 - 128)

    def test_decoder_tp_comm_overlap_assert_no_seq_len(self):
        """Test that AssertionError is raised when decoder_seq_len is None."""
        with self.assertRaises(AssertionError):
            get_padding(
                seq_len=128,
                cp_size=1,
                tp_size=1,
                has_sp=True,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=None,
            )

    def test_fp8_disabled(self):
        """Test that no padding when FP8 is disabled."""
        padding = get_padding(
            seq_len=64,
            cp_size=1,
            tp_size=1,
            has_sp=False,
            fp8_enabled=False,
        )
        self.assertEqual(padding, 0)

    def test_cp_sp_priority(self):
        """Test that SP+CP takes priority over FP8."""
        padding = get_padding(
            seq_len=10,
            cp_size=2,
            tp_size=2,
            has_sp=True,
            fp8_enabled=True,
            fp8_recipe="mxfp8",
        )
        # SP+CP has padding_factor = tp_size * cp_size * 2 = 8
        expected = 8 - 10 % 8
        if expected == 8:
            expected = 0
        self.assertEqual(padding, expected)


class TestGetPackedSeqParams(unittest.TestCase):
    """Test get_packed_seq_params function."""

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_basic(self):
        """Test basic packed seq params creation."""
        tokens = paddle.randint(0, 100, [2, 10]).cuda()
        packed_params = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=1,
            use_packed_sequence=False,
        )
        self.assertIsNotNone(packed_params)
        self.assertEqual(packed_params.qkv_format, "sbhd")

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_with_cp_padding(self):
        """Test with CP > 1 and padding."""
        tokens = paddle.randint(0, 100, [2, 10]).cuda()
        packed_params = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=64,
            cp_size=2,
            use_packed_sequence=False,
        )
        self.assertIsNotNone(packed_params)
        # Should use THD format with CP and padding
        self.assertEqual(packed_params.qkv_format, "thd")
        self.assertIsNotNone(packed_params.cu_seqlens_q_padded)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_with_packed_sequence(self):
        """Test with use_packed_sequence=True."""
        tokens = paddle.randint(0, 100, [2, 10]).cuda()
        packed_params = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
            use_packed_sequence=True,
        )
        self.assertIsNotNone(packed_params)
        # With use_packed_sequence and CP, should use THD format
        self.assertEqual(packed_params.qkv_format, "thd")
        self.assertIsNotNone(packed_params.cu_seqlens_q_padded)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(), "Requires CUDA")
    def test_cu_seqlens_shape(self):
        """Test cu_seqlens shape."""
        batch_size = 3
        tokens = paddle.randint(0, 100, [batch_size, 20]).cuda()
        packed_params = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=100,
            padding_needed=0,
            cp_size=1,
        )
        combined_valid_seqlen = 20 + 100
        expected_len = batch_size + 1
        self.assertEqual(packed_params.cu_seqlens_q.shape[0], expected_len)


if __name__ == "__main__":
    unittest.main()
