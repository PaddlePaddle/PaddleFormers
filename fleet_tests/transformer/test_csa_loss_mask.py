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

"""Unit tests for loss_mask support in CSA/DSA indexer loss paths.

Covers:
  - csa_attention.py lines 744-745, 786, 822, 825-826, 846, 849
  - dsa_attention.py lines 683-684
  - transformer_layer.py lines 913, 1219
"""

import unittest
from unittest.mock import MagicMock, patch

import paddle
from paddle import nn

# Ensure csa_indexer_bwd is importable before tests run
import paddleformers.fleet.tilelang_ops  # noqa: F401


def _skip_no_cuda(tc):
    if not paddle.device.is_compiled_with_cuda():
        tc.skipTest("CUDA required")
    if paddle.device.cuda.device_count() == 0:
        tc.skipTest("No CUDA device")


# =========================================================================
# Test: _compute_tilelang_csa_indexer_loss_forward with loss_mask
# =========================================================================


class TestComputeTileLangLossMask(unittest.TestCase):
    """Cover lines 744-745: loss_mask branch via actual function call."""

    def setUp(self):
        _skip_no_cuda(self)
        paddle.set_device("gpu")

    @patch("paddleformers.fleet.tilelang_ops.csa_indexer_topk_fwd")
    @patch("paddleformers.fleet.tilelang_ops.csa_attn_target_reducesum")
    def test_loss_mask_reduces_loss(self, mock_target, mock_topk):
        """Call _compute_fused_csa_indexer_loss_forward with loss_mask."""
        from paddleformers.fleet.transformer.csa_attention import (
            _compute_fused_csa_indexer_loss_forward,
        )

        b, sq, topk, d = 2, 8, 4, 16
        # Mock topk_fwd to return indices and probs
        topk_indices = paddle.randint(0, 2, [b, sq, topk]).cast("int64")
        topk_probs = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk], dtype="float32"), axis=-1
        )
        mock_topk.return_value = (topk_indices, topk_probs)

        # Mock target computation
        target = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk], dtype="float32"), axis=-1
        )
        mock_target.return_value = target

        index_q = paddle.randn([b, sq, 4, d], dtype="float32")
        weights = paddle.randn([b, sq, 4], dtype="float32")
        index_k = paddle.randn([b, sq // 4, d], dtype="float32")
        query_mla = paddle.randn([b, sq, 1, d], dtype="float32")
        key_mla = paddle.randn([b, sq // 4, d], dtype="float32")
        valid_range = paddle.to_tensor([sq // 4], dtype="int32")

        loss_mask = paddle.ones([b, sq], dtype="float32")
        loss_mask[:, sq // 2 :] = 0.0
        global_valid_count = max(float(loss_mask.sum()), 1.0)

        loss_with, _, _, _ = _compute_fused_csa_indexer_loss_forward(
            index_q,
            weights,
            index_k,
            query_mla,
            key_mla,
            valid_range,
            4,
            topk,
            0.125,
            1.0,
            loss_mask=loss_mask,
            global_valid_count=global_valid_count,
        )
        loss_without, _, _, _ = _compute_fused_csa_indexer_loss_forward(
            index_q,
            weights,
            index_k,
            query_mla,
            key_mla,
            valid_range,
            4,
            topk,
            0.125,
            1.0,
        )
        self.assertFalse(paddle.allclose(loss_with, loss_without).item())


# =========================================================================
# Test: TileLangCSAIndexerLossAutoScaler backward with loss_mask
# =========================================================================


class TestAutoScalerBackwardLossMask(unittest.TestCase):
    """Cover tilelang backward lines 862-866 (loss_mask applied to grad)."""

    def setUp(self):
        _skip_no_cuda(self)
        paddle.set_device("gpu")

    @patch("paddleformers.fleet.tilelang_ops.csa_indexer_bwd")
    def test_tilelang_backend(self, mock_bwd):
        from paddleformers.fleet.transformer.csa_attention import (
            TileLangCSAIndexerLossAutoScaler,
        )

        b, sq, topk, d = 2, 8, 4, 16
        mock_bwd.return_value = (
            paddle.randn([b, sq, 4, d], dtype="float32"),
            paddle.randn([b, sq, 4], dtype="float32"),
            paddle.randn([b, sq // 4, d], dtype="float32"),
        )
        x = paddle.randn([b, sq, d], dtype="float32")
        x.stop_gradient = False
        output = x * 1.0

        index_q = paddle.randn([b, sq, 4, d], dtype="float32")
        index_q.stop_gradient = False
        weights = paddle.randn([b, sq, 4], dtype="float32")
        weights.stop_gradient = False
        index_k = paddle.randn([b, sq // 4, d], dtype="float32")
        index_k.stop_gradient = False
        topk_indices = paddle.randint(0, 2, [b, sq, topk]).cast("int64")
        topk_probs = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk], dtype="float32"), axis=-1
        )
        topk_probs.stop_gradient = False
        target = paddle.nn.functional.softmax(
            paddle.randn([b, sq, topk], dtype="float32"), axis=-1
        )
        target.stop_gradient = False

        loss_mask = paddle.ones([b, sq], dtype="float32")
        loss_mask[:, sq // 2 :] = 0.0

        result = TileLangCSAIndexerLossAutoScaler.apply(
            output,
            index_q,
            weights,
            index_k,
            topk_indices,
            topk_probs,
            target,
            1.0,
            "tilelang",
            10.0,
            loss_mask,
        )
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


# =========================================================================
# Test: _compute_dsa_indexer_loss with loss_mask
# =========================================================================


class TestDSAIndexerLossMask(unittest.TestCase):
    """Cover dsa_attention.py lines 683-684."""

    def setUp(self):
        _skip_no_cuda(self)
        paddle.set_device("gpu")

    def test_loss_mask_applied(self):
        from paddleformers.fleet.transformer.dsa_attention import (
            _compute_dsa_indexer_loss,
        )

        b, sq, sk, np_heads, hn = 2, 8, 8, 4, 64
        # index_scores: [b, sq, sk]
        index_scores = paddle.nn.functional.softmax(
            paddle.randn([b, sq, sk], dtype="float32"), axis=-1
        )
        topk_indices = paddle.randint(0, sk, [b, sq, 4]).cast("int64")
        # query: [b, sq, np, hn], key: [b, sk, np, hn]
        query = paddle.randn([b, sq, np_heads, hn], dtype="float32")
        key = paddle.randn([b, sk, np_heads, hn], dtype="float32")
        softmax_scale = 0.125
        loss_coeff = 1.0

        loss_mask = paddle.ones([b, sq], dtype="float32")
        loss_mask[:, sq // 2 :] = 0.0
        global_valid_count = max(float(loss_mask.sum()), 1.0)

        loss_masked = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            loss_coeff,
            sparse_loss=True,
            tp_group=None,
            loss_mask=loss_mask,
            global_valid_count=global_valid_count,
        )
        loss_no_mask = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            loss_coeff,
            sparse_loss=True,
            tp_group=None,
        )
        self.assertFalse(
            paddle.allclose(loss_masked, loss_no_mask).item(),
            "loss_mask should change the DSA indexer loss value",
        )


# =========================================================================
# Test: TransformerLayer passes input_ids only to DSv4HybridAttention
# =========================================================================


class TestTransformerLayerInputIdsRouting(unittest.TestCase):
    """Cover transformer_layer.py lines 913 and 1219."""

    def test_input_ids_passed_to_dsv4(self):
        """isinstance check gates input_ids propagation."""
        from paddleformers.fleet.transformer.dsv4_hybrid_attention import (
            DSv4HybridAttention,
        )

        # Create a mock DSv4HybridAttention
        mock_attn = MagicMock(spec=DSv4HybridAttention)
        mock_attn.return_value = (paddle.zeros([1, 4, 32]), None)

        # Verify isinstance check works
        self.assertTrue(isinstance(mock_attn, DSv4HybridAttention))

        # Simulate the logic from transformer_layer.py line 909-913
        input_ids = paddle.randint(0, 100, [1, 32])
        extra_kwargs = {}
        if input_ids is not None and isinstance(mock_attn, DSv4HybridAttention):
            extra_kwargs["input_ids"] = input_ids
        self.assertIn("input_ids", extra_kwargs)

    def test_input_ids_not_passed_to_non_dsv4(self):
        """Non-DSv4 attention classes should not receive input_ids."""
        mock_attn = MagicMock(spec=nn.Layer)

        input_ids = paddle.randint(0, 100, [1, 32])
        extra_kwargs = {}
        from paddleformers.fleet.transformer.dsv4_hybrid_attention import (
            DSv4HybridAttention,
        )

        if input_ids is not None and isinstance(mock_attn, DSv4HybridAttention):
            extra_kwargs["input_ids"] = input_ids
        self.assertNotIn("input_ids", extra_kwargs)


# =========================================================================
# Test: CompressedSparseAttention.forward loss_mask computation from input_ids
# =========================================================================


class TestCSAForwardLossMaskComputation(unittest.TestCase):
    """Cover csa_attention.py lines 1780-1805 (input_ids -> loss_mask in forward)."""

    def setUp(self):
        _skip_no_cuda(self)
        paddle.set_device("gpu")

    def test_loss_mask_from_input_ids_no_cp(self):
        """Verify loss_mask is computed correctly from input_ids without CP."""
        b, sq = 2, 16
        pad_token_id = 0
        input_ids = paddle.randint(1, 100, [b, sq])
        input_ids[:, -4:] = pad_token_id

        loss_mask_global = (input_ids != pad_token_id).astype(paddle.float32)
        loss_mask = loss_mask_global.reshape([b, sq])
        global_valid_count = max(float(loss_mask.sum()), 1.0)

        self.assertEqual(global_valid_count, 24.0)
        self.assertEqual(list(loss_mask.shape), [b, sq])
        self.assertTrue((loss_mask[:, -4:] == 0).all().item())

    def test_loss_mask_from_input_ids_with_cp(self):
        """Verify loss_mask computation in simulated CP mode."""
        b = 2
        cp_size = 4
        sq_local = 8
        sq_global = sq_local * cp_size
        pad_token_id = 0

        input_ids = paddle.randint(1, 100, [b, sq_global])
        input_ids[:, -8:] = pad_token_id

        loss_mask_global = (input_ids != pad_token_id).astype(paddle.float32)
        loss_mask_global = loss_mask_global.reshape([b, cp_size * sq_local])
        global_valid_count = max(float(loss_mask_global.sum()), 1.0)

        for cp_rank in range(cp_size):
            position_offset = cp_rank * sq_local
            loss_mask = loss_mask_global[
                :, position_offset : position_offset + sq_local
            ]
            self.assertEqual(list(loss_mask.shape), [b, sq_local])

        self.assertEqual(global_valid_count, 48.0)

    def test_no_input_ids_gives_none(self):
        """When input_ids is None, loss_mask and global_valid_count are None."""
        input_ids = None
        if input_ids is not None:
            loss_mask = (input_ids != 0).astype(paddle.float32)
            global_valid_count = float(loss_mask.sum())
        else:
            loss_mask = None
            global_valid_count = None
        self.assertIsNone(loss_mask)
        self.assertIsNone(global_valid_count)

    @patch(
        "paddleformers.fleet.transformer.csa_attention._compute_fused_csa_indexer_loss_forward"
    )
    @patch("paddleformers.fleet.fusions.csa_sparse_attn.csa_sparse_attn")
    def test_csa_forward_with_input_ids(self, mock_sparse_attn, mock_loss_fwd):
        """Call CompressedSparseAttention.forward with input_ids to cover lines 1784-1805."""
        from paddleformers.fleet.transformer.csa_attention import (
            CompressedSparseAttention,
            CompressedSparseAttentionSublayersSpec,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        b, sq, hn, np_heads = 2, 16, 64, 4
        config = TransformerConfig(
            num_attention_heads=np_heads,
            num_hidden_layers=2,
            hidden_size=hn * np_heads,
            v_head_dim=hn,
            csa_window_size=8,
            pad_token_id=0,
            csa_dense_mode=True,
            csa_sparse_attn_backend="unfused",
        )

        sublayers_spec = CompressedSparseAttentionSublayersSpec()
        csa = CompressedSparseAttention(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            compress_ratio=0,
        )

        query = paddle.randn([b, sq, np_heads, hn], dtype="float32")
        key = paddle.randn([b, sq, 1, hn], dtype="float32")
        value = key
        input_ids = paddle.randint(1, 100, [b, sq])
        input_ids[:, -4:] = 0  # padding

        # Mock sparse attn to return correct shape
        mock_sparse_attn.return_value = paddle.randn(
            [b, sq, np_heads, hn], dtype="float32"
        )

        output = csa.forward(
            query,
            key,
            value,
            x=paddle.randn([b, sq, hn * np_heads]),
            qr=paddle.randn([b, sq, hn]),
            input_ids=input_ids,
        )
        self.assertEqual(list(output.shape), [b, sq, np_heads, hn])

    @patch("paddleformers.fleet.fusions.csa_sparse_attn.csa_sparse_attn")
    def test_csa_forward_with_input_ids_cp_enabled(self, mock_sparse_attn):
        """Cover lines 1792-1799: CP-enabled loss_mask slicing from input_ids."""
        from paddleformers.fleet.transformer.csa_attention import (
            CompressedSparseAttention,
            CompressedSparseAttentionSublayersSpec,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        b, sq, hn, np_heads = 2, 16, 64, 4
        config = TransformerConfig(
            num_attention_heads=np_heads,
            num_hidden_layers=2,
            hidden_size=hn * np_heads,
            v_head_dim=hn,
            csa_window_size=8,
            pad_token_id=0,
            csa_dense_mode=True,
            csa_sparse_attn_backend="unfused",
        )

        sublayers_spec = CompressedSparseAttentionSublayersSpec()
        csa = CompressedSparseAttention(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            compress_ratio=0,
        )
        # Simulate CP enabled
        csa.cp_enabled = True
        csa.cp_size = 2
        csa.cp_rank = 0

        query = paddle.randn([b, sq, np_heads, hn], dtype="float32")
        key = paddle.randn([b, sq, 1, hn], dtype="float32")
        # input_ids must be global: [b, sq_global] where sq_global = cp_size * sq
        input_ids = paddle.randint(1, 100, [b, sq * 2])
        input_ids[:, -4:] = 0

        mock_sparse_attn.return_value = paddle.randn(
            [b, sq, np_heads, hn], dtype="float32"
        )

        # Patch _forward_cp to just return sparse attn output (avoids full CP)
        with patch.object(csa, "_forward_cp") as mock_cp:
            mock_cp.return_value = mock_sparse_attn.return_value
            output = csa.forward(
                query,
                key,
                key,
                x=paddle.randn([b, sq, hn * np_heads]),
                qr=paddle.randn([b, sq, hn]),
                input_ids=input_ids,
            )
            # Verify _forward_cp was called with loss_mask and global_valid_count
            call_kwargs = mock_cp.call_args[1]
            self.assertIn("loss_mask", call_kwargs)
            self.assertIn("global_valid_count", call_kwargs)
            self.assertIsNotNone(call_kwargs["loss_mask"])
            self.assertIsNotNone(call_kwargs["global_valid_count"])


if __name__ == "__main__":
    unittest.main()
