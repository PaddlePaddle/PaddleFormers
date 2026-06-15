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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


# Tests for src/paddleformers.fleet/packed_seq_params.py

import unittest


class TestPackedSeqParams(unittest.TestCase):
    """Tests for PackedSeqParams dataclass."""

    def test_default_values(self):
        """Test default PackedSeqParams values."""
        from paddleformers.fleet.packed_seq_params import PackedSeqParams

        params = PackedSeqParams()
        self.assertIsNone(params.qkv_format)
        self.assertIsNone(params.cu_seqlens_q)
        self.assertIsNone(params.cu_seqlens_kv)
        self.assertIsNone(params.cu_seqlens_q_padded)
        self.assertIsNone(params.cu_seqlens_kv_padded)
        self.assertIsNone(params.max_seqlen_q)
        self.assertIsNone(params.max_seqlen_kv)
        self.assertIsNone(params.total_seqlen_q)
        self.assertIsNone(params.total_seqlen_kv)

    def test_custom_values(self):
        """Test PackedSeqParams with custom values."""
        import paddle

        from paddleformers.fleet.packed_seq_params import PackedSeqParams

        q_cu = paddle.to_tensor([0, 5, 10, 15], dtype="int32")
        kv_cu = paddle.to_tensor([0, 5, 10, 15], dtype="int32")
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=q_cu,
            cu_seqlens_kv=kv_cu,
            max_seqlen_q=5,
            max_seqlen_kv=5,
            total_seqlen_q=15,
            total_seqlen_kv=15,
        )
        self.assertEqual(params.qkv_format, "thd")
        self.assertEqual(params.max_seqlen_q, 5)
        self.assertEqual(params.max_seqlen_kv, 5)
        self.assertEqual(params.total_seqlen_q, 15)
        self.assertEqual(params.total_seqlen_kv, 15)

    def test_frozen_dataclass(self):
        """Test that PackedSeqParams is a dataclass."""
        import dataclasses

        from paddleformers.fleet.packed_seq_params import PackedSeqParams

        self.assertTrue(dataclasses.is_dataclass(PackedSeqParams))

    def test_partial_values(self):
        """Test PackedSeqParams with partial values."""
        from paddleformers.fleet.packed_seq_params import PackedSeqParams

        params = PackedSeqParams(
            qkv_format="hd",
            max_seqlen_q=128,
        )
        self.assertEqual(params.qkv_format, "hd")
        self.assertEqual(params.max_seqlen_q, 128)
        self.assertIsNone(params.cu_seqlens_q)
        self.assertIsNone(params.max_seqlen_kv)


if __name__ == "__main__":
    unittest.main()
