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


class TestGetPaddingBasic(unittest.TestCase):
    """Test get_padding basic scenarios."""

    def test_no_padding_needed(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        # seq_len=32, no sp, no cp, no fp8
        result = get_padding(seq_len=32, cp_size=1, tp_size=1, has_sp=False)
        self.assertEqual(result, 0)

    def test_fp8_padding(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(
            seq_len=33,
            cp_size=1,
            tp_size=1,
            has_sp=False,
            fp8_enabled=True,
            fp8_recipe="fp8",
        )
        self.assertGreater(result, 0)

    def test_mxfp8_padding(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(
            seq_len=33,
            cp_size=1,
            tp_size=1,
            has_sp=False,
            fp8_enabled=True,
            fp8_recipe="mxfp8",
        )
        self.assertGreater(result, 0)
        # mxfp8 uses padding_factor=32
        # padded = ceil(33/32)*32 = 64
        # result = 64 - 33 = 31
        self.assertEqual(result, 31)


class TestGetPaddingWithSequenceParallel(unittest.TestCase):
    """Test get_padding with sequence parallelism."""

    def test_sp_with_tp(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(seq_len=33, cp_size=1, tp_size=4, has_sp=True)
        # padding_factor = tp_size = 4
        expected = int((33 + 3) // 4 * 4) - 33
        self.assertEqual(result, expected)

    def test_sp_aligned(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(seq_len=32, cp_size=1, tp_size=4, has_sp=True)
        self.assertEqual(result, 0)


class TestGetPaddingWithContextParallel(unittest.TestCase):
    """Test get_padding with context parallelism."""

    def test_cp_only(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(seq_len=33, cp_size=2, tp_size=1, has_sp=False)
        # padding_factor = cp_size * 2 = 4
        expected = int((33 + 3) // 4 * 4) - 33
        self.assertEqual(result, expected)

    def test_cp_with_sp(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(seq_len=33, cp_size=2, tp_size=4, has_sp=True)
        # padding_factor = tp_size * cp_size * 2 = 16
        expected = int((33 + 15) // 16 * 16) - 33
        self.assertEqual(result, expected)

    def test_cp_aligned(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(seq_len=32, cp_size=2, tp_size=4, has_sp=True)
        # 32 is a multiple of 16
        self.assertEqual(result, 0)


class TestGetPaddingWithTPCommOverlap(unittest.TestCase):
    """Test get_padding with TP comm overlap."""

    def test_tp_comm_overlap(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        result = get_padding(
            seq_len=576,
            cp_size=2,
            tp_size=4,
            has_sp=True,
            decoder_tp_comm_overlap=True,
            decoder_seq_len=640,
        )
        self.assertEqual(result, 64)

    def test_tp_comm_overlap_missing_decoder_seq_len_raises(self):
        from paddleformers.fleet.models.multimodal.context_parallel import get_padding

        with self.assertRaises(AssertionError):
            get_padding(
                seq_len=576,
                cp_size=2,
                tp_size=4,
                has_sp=True,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=None,
            )


class TestGetPackedSeqParamsBasic(unittest.TestCase):
    """Test get_packed_seq_params basic scenarios."""

    def test_basic_params(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([2, 10]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=1,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.max_seqlen_q, 586)
        self.assertEqual(result.qkv_format, "sbhd")

    def test_params_with_cp_and_padding(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([2, 10]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=16,
            cp_size=2,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.qkv_format, "thd")
        self.assertIsNotNone(result.cu_seqlens_q_padded)


class TestGetPackedSeqParamsCP(unittest.TestCase):
    """Test get_packed_seq_params with context parallel."""

    def test_cp_without_padding(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([2, 10]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
        )
        self.assertIsNotNone(result)
        # No padding, so padded seqlens should be None
        self.assertIsNone(result.cu_seqlens_q_padded)

    def test_cp_with_use_packed_sequence(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([2, 10]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
            use_packed_sequence=True,
        )
        self.assertIsNotNone(result)
        # use_packed_sequence forces thd format even without padding
        self.assertEqual(result.qkv_format, "thd")


class TestGetPackedSeqParamsValues(unittest.TestCase):
    """Test get_packed_seq_params computed values."""

    def test_total_seqlen(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([3, 20]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=100,
            padding_needed=0,
            cp_size=1,
        )
        # combined_valid_seqlen = 20 + 100 = 120
        # total_seqlen = 3 * 120 = 360
        self.assertEqual(result.total_seqlen_q, 360)

    def test_max_seqlen(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([2, 10]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=576,
            padding_needed=0,
            cp_size=1,
        )
        # max_seqlen = combined_padded_seqlen = 10 + 576 = 586
        self.assertEqual(result.max_seqlen_q, 586)

    def test_cu_seqlens(self):
        import paddle

        from paddleformers.fleet.models.multimodal.context_parallel import (
            get_packed_seq_params,
        )

        tokens = paddle.randn([3, 20]).astype("int32")
        result = get_packed_seq_params(
            tokens=tokens,
            img_seq_len=100,
            padding_needed=0,
            cp_size=1,
        )
        # cu_seqlens = [0, 120, 240, 360]
        self.assertEqual(result.cu_seqlens_q.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
