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

import math
import unittest
from unittest import mock

import numpy as np
import paddle

from paddleformers.fleet.transformer.transformer_config import TransformerConfig

try:
    import paddlefleet_ops
    from paddleformers.fleet.tilelang_ops.attn import sparse_mqa

    _HAS_FLASH_MLA = (
        paddlefleet_ops.is_flash_mla_available()
        and sparse_mqa._flash_mla_sparse_fwd is not None
    )
except (ImportError, RuntimeError, AttributeError):
    _HAS_FLASH_MLA = False


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and _HAS_FLASH_MLA,
    "FlashMLA sparse attention requires CUDA and flash_mla",
)
class TestSparseMQAFlashMLAForward(unittest.TestCase):
    def setUp(self):
        paddle.seed(2026)
        self.batch_size = 2
        self.seq_len = 1024
        self.num_heads = 64
        self.head_dim = 512
        self.topk = 128
        self.softmax_scale = self.head_dim**-0.5

    def _make_inputs(self):
        q = paddle.randn(
            [
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
            ],
            dtype=paddle.bfloat16,
        )
        kv = paddle.randn(
            [self.batch_size, self.seq_len, self.head_dim],
            dtype=paddle.bfloat16,
        )
        attn_sink = paddle.randn([self.num_heads], dtype=paddle.float32)
        topk_idxs = (
            paddle.arange(self.topk, dtype="int32")
            .reshape([1, 1, self.topk])
            .expand([self.batch_size, self.seq_len, self.topk])
        )
        return q, kv, attn_sink, topk_idxs

    def test_flash_mla_forward_matches_tilelang(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()

        tile_out, tile_lse = sparse_mqa.sparse_attn(
            q, kv, attn_sink, topk_idxs, sm_scale=self.softmax_scale
        )

        flash_out, flash_lse = sparse_mqa.sparse_attn(
            q,
            kv,
            attn_sink,
            topk_idxs,
            sm_scale=self.softmax_scale,
            backend="cudnn",
        )

        # flash_mla uses a different lse computation from tilelang
        flash_lse = paddle.logaddexp(flash_lse, attn_sink) / math.log(2.0)

        np.testing.assert_allclose(
            flash_out.float(),
            tile_out.float(),
            rtol=5e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            flash_lse,
            tile_lse,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_flash_mla_forward_with_indexer_lse(self):
        q, kv, attn_sink, topk_idxs = self._make_inputs()
        topk_idxs = topk_idxs[:, :, :96]

        out, lse, lse_indexer = sparse_mqa.flash_mla_sparse_attn(
            q,
            kv,
            attn_sink,
            topk_idxs,
            sm_scale=self.softmax_scale,
            indexer_topk=512,
        )

        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )
        self.assertEqual(
            lse.shape,
            [self.batch_size, self.seq_len, self.num_heads],
        )
        self.assertEqual(
            lse_indexer.shape,
            [self.batch_size, self.seq_len, self.num_heads],
        )

    def test_fallback_path(self):
        old_flash_mla_sparse_fwd = getattr(
            sparse_mqa, "_flash_mla_sparse_fwd", None
        )
        try:
            sparse_mqa._flash_mla_sparse_fwd = None
            with self.assertRaisesRegex(
                RuntimeError, "flash_mla is not available"
            ):
                sparse_mqa.flash_mla_sparse_attn(None, None, None, None)
        finally:
            sparse_mqa._flash_mla_sparse_fwd = old_flash_mla_sparse_fwd

        with mock.patch.object(
            sparse_mqa.paddle.cuda,
            "get_device_capability",
            return_value=(10, 0),
        ):
            self.assertEqual(sparse_mqa._get_topk_alignment(), 64)

        with self.assertRaisesRegex(
            ValueError, "csa_sparse_attn_backend='paddle' is invalid"
        ):
            config = TransformerConfig(
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[0],
                csa_sparse_attn_backend="paddle",
            )


if __name__ == "__main__":
    unittest.main()
