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

import sys
import unittest
from unittest.mock import patch

import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddleformers.fleet.fusions.csa_sparse_attn import (
    csa_sparse_attn,
    unfused_compressed_sparse_attn,
)
from paddleformers.fleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddleformers.fleet.models.gpt.gpt_layer_specs import (
    get_attention_spec,
    get_gpt_decoder_layers_spec,
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    CSADocMaskMetadata,
    _apply_rope,
    _build_compressed_causal_mask,
    _resolve_csa_indexer_attn_topk_effective,
    _resolve_csa_indexer_loss_topk_effective,
    get_compress_topk_idxs,
    get_valid_range,
    get_window_topk_idxs,
)
from paddleformers.fleet.transformer.dsa_attention import (
    fused_qk_topk_naive,
)
from paddleformers.fleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
    build_document_rope_freqs,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

_SEED = 42


class _FakeGroup:
    def __init__(self, nranks=1):
        self.nranks = nranks
        self.ranks = list(range(nranks))
        self.rank = 0


class _FakePGCollection:
    def __init__(self, tp_nranks=1, cp_nranks=1):
        self.tp = _FakeGroup(tp_nranks)
        self.cp = _FakeGroup(cp_nranks)


def _make_startend_row_indices(doc_lens, seqlen):
    values = []
    doc_end = 0
    for doc_len in doc_lens:
        doc_end += doc_len
        values.extend([doc_end] * doc_len)
    if len(values) < seqlen:
        values.extend([doc_end] * (seqlen - len(values)))
    return paddle.to_tensor(values, dtype="int32").reshape([1, 1, seqlen, 1])


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=8,
    v_head_dim=32,
    qk_pos_emb_head_dim=16,
    q_lora_rank=64,
    o_groups=4,
    o_lora_rank=32,
    csa_compress_ratios=None,
    csa_window_size=16,
    dsa_index_n_heads=4,
    dsa_index_head_dim=32,
    dsa_index_topk=8,
    dsa_indexer_loss_coeff=1.0,
    rope_type="rope",
    apply_rope_fusion=False,
    multi_latent_attention=True,
    num_nextn_predict_layers=0,
    csa_indexer_backend="unfused",
    csa_sparse_attn_backend="unfused",
    tensor_model_parallel_size=1,
    context_parallel_size=1,
):
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]

    return TransformerConfig(
        num_hidden_layers=num_layers,
        num_nextn_predict_layers=num_nextn_predict_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=multi_latent_attention,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_nope_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_rope_head_dim=qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type=rope_type,
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_index_n_heads=dsa_index_n_heads,
        dsa_index_head_dim=dsa_index_head_dim,
        dsa_index_topk=dsa_index_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=apply_rope_fusion,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_indexer_backend=csa_indexer_backend,
        csa_sparse_attn_backend=csa_sparse_attn_backend,
        tensor_model_parallel_size=tensor_model_parallel_size,
        context_parallel_size=context_parallel_size,
    )


def _build_attention(config, layer_number):
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


class TestDSv4HybridConfigAndSpec(unittest.TestCase):
    def test_gpt_layer_local_spec_routes_to_dsv4_hybrid_attention(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config=config,
            multi_latent_attention=False,
            normalization=config.normalization,
        )

        self_attn_spec = spec.sublayers_spec.self_attn
        self.assertIs(self_attn_spec.layer, DSv4HybridSelfAttention)

    def test_config_validation_errors(self):
        with self.assertRaisesRegex(
            ValueError, "csa_compress_ratios to be set"
        ):
            TransformerConfig(
                num_hidden_layers=1,
                hidden_size=256,
                num_attention_heads=8,
                params_dtype=paddle.bfloat16,
                bf16=True,
                multi_latent_attention=True,
                experimental_attention_variant="dsv4_hybrid",
            )

        with self.assertRaisesRegex(ValueError, "must equal num_hidden_layers"):
            _make_config(num_layers=2, csa_compress_ratios=[0])

        # ratio 1 is ambiguous (no compression yet not window) and rejected.
        with self.assertRaisesRegex(ValueError, "is invalid"):
            _make_config(num_layers=1, csa_compress_ratios=[1])

        # ratio 129 is above HCA (128) and rejected.
        with self.assertRaisesRegex(ValueError, "is invalid"):
            _make_config(num_layers=1, csa_compress_ratios=[129])

    def test_csa_compress_ratios_accepts_general_set(self):
        # window (0), CSA over the full [2, 127] range (including non-power-of-2
        # 3 and the boundary 127), and HCA (128) must all be accepted and
        # round-trip through the config.
        ratios = [0, 2, 3, 4, 8, 16, 32, 64, 127, 128]
        cfg = _make_config(num_layers=len(ratios), csa_compress_ratios=ratios)
        self.assertEqual(cfg.csa_compress_ratios, ratios)

    def test_csa_indexer_backend_validation(self):
        for backend in ("unfused", "tilelang", "cudnn"):
            cfg = _make_config(csa_indexer_backend=backend)
            self.assertEqual(cfg.csa_indexer_backend, backend)

        with self.assertRaisesRegex(
            ValueError, "csa_indexer_backend='paddle' is invalid"
        ):
            _make_config(csa_indexer_backend="paddle")

    def test_csa_sparse_attn_backend_validation(self):
        for backend in ("unfused", "tilelang", "cudnn"):
            cfg = _make_config(csa_sparse_attn_backend=backend)
            self.assertEqual(cfg.csa_sparse_attn_backend, backend)

        with self.assertRaisesRegex(
            ValueError, "csa_sparse_attn_backend='paddle' is invalid"
        ):
            _make_config(csa_sparse_attn_backend="paddle")

    def test_csa_cudnn_indexer_allows_config_with_cp(self):
        cfg = _make_config(csa_indexer_backend="cudnn", context_parallel_size=2)
        self.assertEqual(cfg.csa_indexer_backend, "cudnn")
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_csa_rejects_tensor_parallel_gt_one(self):
        cfg = _make_config(
            num_layers=1,
            csa_compress_ratios=[4],
            num_attention_heads=2,
            dsa_index_n_heads=32,
            dsa_index_head_dim=128,
            tensor_model_parallel_size=2,
        )
        with self.assertRaisesRegex(
            NotImplementedError, "does not support tensor parallelism > 1"
        ):
            _build_attention(cfg, layer_number=0)

    def test_removed_tilelang_switches_raise(self):
        removed_switches = (
            (
                "csa_tilelang_enable_sparse_attn",
                "csa_tilelang_enable_sparse_attn has been removed",
            ),
            (
                "csa_tilelang_enable_indexer",
                "csa_tilelang_enable_indexer has been removed",
            ),
            (
                "csa_tilelang_backend",
                "csa_tilelang_backend has been removed",
            ),
        )
        for attr, message in removed_switches:
            with (
                self.subTest(attr=attr),
                self.assertRaisesRegex(ValueError, message),
            ):
                cfg = _make_config()
                setattr(cfg, attr, True)
                cfg.__post_init__()

    def test_csa_rejects_tensor_parallelism(self):
        config = _make_config(num_layers=1, csa_compress_ratios=[4])
        with self.assertRaisesRegex(
            NotImplementedError,
            "got tp=2",
        ):
            CompressedSparseAttention(
                config=config,
                sublayers_spec=CompressedSparseAttentionSublayersSpec(),
                layer_number=0,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                pg_collection=_FakePGCollection(tp_nranks=2),
                compress_ratio=4,
            )

        config = _make_config(num_layers=1, csa_compress_ratios=[4])
        config.tensor_model_parallel_size = 2
        with self.assertRaisesRegex(
            NotImplementedError,
            "got tp=2",
        ):
            CompressedSparseAttention(
                config=config,
                sublayers_spec=CompressedSparseAttentionSublayersSpec(),
                layer_number=0,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                compress_ratio=4,
            )

    def test_phase2_loss_topk_does_not_expand_attention_topk(self):
        config = _make_config(
            dsa_index_topk=2,
        )
        n_compressed = 8

        self.assertEqual(
            _resolve_csa_indexer_loss_topk_effective(
                config, config.dsa_index_topk, n_compressed
            ),
            n_compressed,
        )
        self.assertEqual(
            _resolve_csa_indexer_attn_topk_effective(
                config.dsa_index_topk, n_compressed
            ),
            config.dsa_index_topk,
        )

        config.dsa_indexer_use_sparse_loss = True
        self.assertEqual(
            _resolve_csa_indexer_loss_topk_effective(
                config, config.dsa_index_topk, n_compressed
            ),
            config.dsa_index_topk,
        )


class TestCSAIndexHelpers(unittest.TestCase):
    def test_window_and_compress_indices(self):
        window = get_window_topk_idxs(
            window_size=3,
            batch_size=2,
            seqlen=4,
        )
        self.assertEqual(list(window.shape), [2, 4, 3])
        self.assertEqual(
            window.numpy().tolist()[0],
            [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3]],
        )

        compressed = get_compress_topk_idxs(
            ratio=4,
            batch_size=2,
            seqlen=8,
            offset=8,
        )
        self.assertEqual(list(compressed.shape), [2, 8, 2])
        self.assertEqual(
            compressed.numpy().tolist()[0],
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [8, -1],
                [8, -1],
                [8, -1],
                [8, -1],
                [8, 9],
            ],
        )

    def test_fused_qk_topk_naive_with_mask(self):
        q = paddle.ones([1, 2, 1, 2], dtype="bfloat16")
        k = paddle.to_tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype="bfloat16")
        weights = paddle.ones([1, 2, 1], dtype="float32")
        mask = paddle.to_tensor(
            [[[0.0, float("-inf")], [0.0, 0.0]]], dtype="float32"
        )

        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, 2, mask)

        self.assertEqual(list(index_scores.shape), [1, 2, 2])
        self.assertEqual(list(topk_indices.shape), [1, 2, 2])
        self.assertEqual(topk_indices.numpy().tolist()[0][0][0], 0)


class TestCSADocMaskMetadata(unittest.TestCase):
    def _make_docmask(self):
        return paddle.to_tensor(
            [5, 5, 5, 5, 5, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
            dtype="int32",
        ).reshape([1, 1, 16, 1])

    def test_metadata_matches_expected_docmask_outputs(self):
        ratio = 4
        batch_size = 1
        seqlen = 16
        startend_row_indices = self._make_docmask()
        meta = CSADocMaskMetadata.build(
            ratio, batch_size, seqlen, startend_row_indices
        )

        self.assertIsNotNone(meta)
        self.assertEqual(meta.actual_n_compressed, 2)
        self.assertEqual(meta.doc_lens.numpy().tolist(), [5, 7])
        self.assertEqual(meta.doc_lens_list, [5, 7])
        self.assertIs(meta.doc_lens_list, meta.doc_lens_list)
        self.assertEqual(meta.doc_starts.numpy().tolist(), [0, 5])
        self.assertEqual(meta.doc_lens_cutoff.numpy().tolist(), [4, 4])
        self.assertEqual(meta.doc_starts_cutoff.numpy().tolist(), [0, 4])
        self.assertEqual(
            meta.valid_range.numpy().tolist(),
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ]
            ],
        )
        self.assertEqual(
            meta.get_window_topk_idxs(3).numpy().tolist(),
            [
                [
                    [0, -1, -1],
                    [0, 1, -1],
                    [0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [5, -1, -1],
                    [5, 6, -1],
                    [5, 6, 7],
                    [6, 7, 8],
                    [7, 8, 9],
                    [8, 9, 10],
                    [9, 10, 11],
                    [-1, -1, -1],
                    [-1, -1, -1],
                    [-1, -1, -1],
                    [-1, -1, -1],
                ]
            ],
        )
        self.assertEqual(
            meta.get_compress_topk_idxs(offset=16).numpy().tolist(),
            [
                [
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [16, -1, -1, -1],
                    [16, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, 17, -1, -1],
                    [-1, 17, -1, -1],
                    [-1, 17, -1, -1],
                    [-1, 17, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ]
            ],
        )
        causal_mask = meta.get_compressed_causal_mask()
        self.assertTrue(paddle.isinf(causal_mask[:, :3, :]).all().item())
        self.assertEqual(
            causal_mask[0, 3, :].numpy().tolist(),
            [0.0, -float("inf"), -float("inf"), -float("inf")],
        )
        self.assertEqual(
            causal_mask[0, 8, :].numpy().tolist(),
            [-float("inf"), 0.0, -float("inf"), -float("inf")],
        )
        self.assertTrue(paddle.isinf(causal_mask[:, 12:, :]).all().item())
        self.assertEqual(
            meta.get_is_first_compressed_group().numpy().tolist(),
            [True, True],
        )

    def test_metadata_handles_three_docs_ratio_128(self):
        ratio = 128
        seqlen = 384
        startend_row_indices = _make_startend_row_indices([129, 128, 1], seqlen)
        meta = CSADocMaskMetadata.build(ratio, 1, seqlen, startend_row_indices)

        self.assertEqual(meta.doc_lens.numpy().tolist(), [129, 128, 1])
        self.assertEqual(meta.doc_starts.numpy().tolist(), [0, 129, 257])
        self.assertEqual(meta.doc_lens_cutoff.numpy().tolist(), [128, 128, 0])
        self.assertEqual(meta.doc_starts_cutoff.numpy().tolist(), [0, 128, 256])
        self.assertEqual(meta.actual_n_compressed, 2)
        self.assertEqual(
            meta.get_is_first_compressed_group().numpy().tolist(),
            [True, True],
        )

        valid_range = meta.valid_range.numpy().tolist()[0]
        self.assertEqual(valid_range[126], [0, 0])
        self.assertEqual(valid_range[127], [0, 1])
        self.assertEqual(valid_range[128], [0, 1])
        self.assertEqual(valid_range[129], [0, 0])
        self.assertEqual(valid_range[256], [1, 2])
        self.assertEqual(valid_range[257], [0, 0])
        self.assertEqual(valid_range[-1], [0, 0])

        compressed = meta.get_compress_topk_idxs(offset=seqlen)
        self.assertEqual(compressed[0, 127, :].numpy().tolist(), [384, -1, -1])
        self.assertEqual(compressed[0, 256, :].numpy().tolist(), [-1, 385, -1])
        self.assertEqual(compressed[0, 257, :].numpy().tolist(), [-1, -1, -1])

    def test_metadata_lazy_cache_keys_recompute_when_inputs_change(self):
        meta = CSADocMaskMetadata.build(4, 1, 16, self._make_docmask())

        window_3 = meta.get_window_topk_idxs(3)
        self.assertIs(window_3, meta.get_window_topk_idxs(3))
        window_5 = meta.get_window_topk_idxs(5)
        self.assertIs(window_5, meta.get_window_topk_idxs(5))
        self.assertIsNot(window_3, window_5)
        self.assertTrue(
            paddle.equal_all(
                window_5,
                get_window_topk_idxs(5, 1, 16, self._make_docmask()),
            ).item()
        )

        compressed_16 = meta.get_compress_topk_idxs(offset=16)
        self.assertIs(compressed_16, meta.get_compress_topk_idxs(offset=16))
        compressed_32 = meta.get_compress_topk_idxs(offset=32)
        self.assertIs(compressed_32, meta.get_compress_topk_idxs(offset=32))
        self.assertIsNot(compressed_16, compressed_32)
        self.assertTrue(
            paddle.equal_all(
                compressed_32,
                get_compress_topk_idxs(4, 1, 16, 32, self._make_docmask()),
            ).item()
        )

    def test_metadata_none_when_no_docmask(self):
        self.assertIsNone(CSADocMaskMetadata.build(4, 1, 16, None))

    def test_helpers_reuse_supplied_metadata(self):
        startend_row_indices = self._make_docmask()
        meta = CSADocMaskMetadata.build(4, 1, 16, startend_row_indices)

        window = get_window_topk_idxs(
            3, 1, 16, startend_row_indices, docmask_meta=meta
        )
        compressed = get_compress_topk_idxs(
            4, 1, 16, 16, startend_row_indices, docmask_meta=meta
        )
        valid_range = get_valid_range(
            4, 1, 16, startend_row_indices, docmask_meta=meta
        )
        causal_mask = _build_compressed_causal_mask(
            4, 1, 16, 4, startend_row_indices, docmask_meta=meta
        )

        self.assertIs(window, meta.get_window_topk_idxs(3))
        self.assertIs(compressed, meta.get_compress_topk_idxs(16))
        self.assertIs(valid_range, meta.valid_range)
        self.assertIs(causal_mask, meta.get_compressed_causal_mask())

    def test_metadata_rejects_inconsistent_shape(self):
        with self.assertRaisesRegex(ValueError, "startend_row_indices"):
            CSADocMaskMetadata.build(4, 1, 8, self._make_docmask())


class TestDSv4HybridDocumentRoPE(unittest.TestCase):
    def test_document_rope_freqs_reuses_supplied_doc_lens(self):
        config = _make_config(rope_type="yarn")
        rotary_pos_emb = YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=config.csa_compress_rotary_base,
            scaling_factor=getattr(config, "rotary_scaling_factor", 40),
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 4096
            ),
            beta_fast=getattr(config, "beta_fast", 32),
            beta_slow=getattr(config, "beta_slow", 1),
            mscale=getattr(config, "mscale", 1.0),
            mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
        )
        startend_row_indices = paddle.to_tensor(
            [4, 4, 4, 4, 8, 8, 8, 8], dtype="int32"
        ).reshape([1, 1, 8, 1])
        doc_lens = paddle.to_tensor([4, 4], dtype="int32")

        freqs_from_meta, mscale_from_meta = build_document_rope_freqs(
            rotary_pos_emb,
            8,
            startend_row_indices,
            doc_lens=doc_lens,
        )
        freqs_from_mask, mscale_from_mask = build_document_rope_freqs(
            rotary_pos_emb,
            8,
            startend_row_indices,
        )

        self.assertEqual(mscale_from_meta, mscale_from_mask)
        self.assertTrue(
            paddle.equal_all(
                freqs_from_meta.cast("float32"),
                freqs_from_mask.cast("float32"),
            ).item()
        )

    def test_document_rope_freqs_with_position_offset_pads_to_local_slice(self):
        config = _make_config(rope_type="yarn")
        rotary_pos_emb = YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=config.csa_compress_rotary_base,
            scaling_factor=getattr(config, "rotary_scaling_factor", 40),
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 4096
            ),
            beta_fast=getattr(config, "beta_fast", 32),
            beta_slow=getattr(config, "beta_slow", 1),
            mscale=getattr(config, "mscale", 1.0),
            mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
        )
        sq_local = 4
        position_offset = 4
        needed_len = position_offset + sq_local
        startend_row_indices = paddle.to_tensor(
            [2, 2, 2, 2, 2, 2, 2, 2], dtype="int32"
        ).reshape([1, 1, 8, 1])

        freqs, _ = build_document_rope_freqs(
            rotary_pos_emb,
            sq_local,
            startend_row_indices,
            position_offset=position_offset,
        )
        local_freqs = freqs[
            :, position_offset : position_offset + sq_local, :, :
        ]

        self.assertEqual(
            list(local_freqs.shape),
            [1, sq_local, 1, config.qk_pos_emb_head_dim],
        )
        self.assertTrue(
            paddle.equal_all(
                local_freqs[:, -2:, :, :],
                paddle.zeros_like(local_freqs[:, -2:, :, :]),
            ).item()
        )

    def test_compressed_document_rope_matches_separate_documents(self):
        paddle.seed(_SEED)
        config = _make_config(rope_type="yarn")
        rotary_pos_emb = YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=config.csa_compress_rotary_base,
            scaling_factor=getattr(config, "rotary_scaling_factor", 40),
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 4096
            ),
            beta_fast=getattr(config, "beta_fast", 32),
            beta_slow=getattr(config, "beta_slow", 1),
            mscale=getattr(config, "mscale", 1.0),
            mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
        )
        nope_dim = config.qk_nope_head_dim
        pos_dim = config.qk_rope_head_dim
        ratio = 4

        doc1 = paddle.randn(
            [1, 23 // ratio, 1, config.v_head_dim], dtype="bfloat16"
        )
        doc2 = paddle.randn(
            [1, 9 // ratio, 1, config.v_head_dim], dtype="bfloat16"
        )
        padding = paddle.randn([1, 1, 1, config.v_head_dim], dtype="bfloat16")
        packed = paddle.concat([doc1, doc2, padding], axis=1)

        packed_out = _apply_rope(
            packed,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=32 // ratio,
            ratio=ratio,
            doc_lens_cutoff=paddle.to_tensor([20, 8], dtype="int32"),
        )
        doc1_out = _apply_rope(
            doc1,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=20 // ratio,
            ratio=ratio,
        )
        doc2_out = _apply_rope(
            doc2,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=8 // ratio,
            ratio=ratio,
        )

        self.assertTrue(
            paddle.equal_all(
                packed_out[:, : doc1.shape[1], :, :].cast("float32"),
                doc1_out.cast("float32"),
            ).item()
        )
        self.assertTrue(
            paddle.equal_all(
                packed_out[
                    :, doc1.shape[1] : doc1.shape[1] + doc2.shape[1], :, :
                ].cast("float32"),
                doc2_out.cast("float32"),
            ).item()
        )

    def test_compressed_document_rope_with_padding_matches_separate_documents(
        self,
    ):
        paddle.seed(_SEED)
        config = _make_config(rope_type="yarn")
        rotary_pos_emb = YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=config.csa_compress_rotary_base,
            scaling_factor=getattr(config, "rotary_scaling_factor", 40),
            original_max_position_embeddings=getattr(
                config, "original_max_position_embeddings", 4096
            ),
            beta_fast=getattr(config, "beta_fast", 32),
            beta_slow=getattr(config, "beta_slow", 1),
            mscale=getattr(config, "mscale", 1.0),
            mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
        )
        nope_dim = config.qk_nope_head_dim
        pos_dim = config.qk_rope_head_dim
        ratio = 4

        doc1 = paddle.randn(
            [1, 23 // ratio, 1, config.v_head_dim], dtype="bfloat16"
        )
        doc2 = paddle.randn(
            [1, 7 // ratio, 1, config.v_head_dim], dtype="bfloat16"
        )
        padding = paddle.randn([1, 2, 1, config.v_head_dim], dtype="bfloat16")
        packed = paddle.concat([doc1, doc2, padding], axis=1)

        packed_out = _apply_rope(
            packed,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=32 // ratio,
            ratio=ratio,
            doc_lens_cutoff=paddle.to_tensor([20, 4], dtype="int32"),
        )
        doc1_out = _apply_rope(
            doc1,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=20 // ratio,
            ratio=ratio,
        )
        doc2_out = _apply_rope(
            doc2,
            nope_dim,
            pos_dim,
            rotary_pos_emb,
            config,
            rotary_seq_len=4 // ratio,
            ratio=ratio,
        )

        self.assertTrue(
            paddle.equal_all(
                packed_out[:, : doc1.shape[1], :, :].cast("float32"),
                doc1_out.cast("float32"),
            ).item()
        )
        self.assertTrue(
            paddle.equal_all(
                packed_out[
                    :, doc1.shape[1] : doc1.shape[1] + doc2.shape[1], :, :
                ].cast("float32"),
                doc2_out.cast("float32"),
            ).item()
        )

    def test_attention_module_fused_sparse_matches_dynamic_forward_backward(
        self,
    ):
        old_flag = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        paddle.set_flags({"FLAGS_cudnn_deterministic": 0})
        try:
            paddle.seed(_SEED)
            seq_len = 128
            for ratio in [4]:
                dynamic_config = _make_config(
                    hidden_size=256,
                    num_attention_heads=2,
                    v_head_dim=128,
                    q_lora_rank=64,
                    o_groups=2,
                    o_lora_rank=32,
                    csa_window_size=32,
                    dsa_indexer_loss_coeff=1.0,
                    dsa_index_n_heads=16,
                    csa_compress_ratios=[ratio],
                    num_layers=1,
                    csa_indexer_backend="unfused",
                    csa_sparse_attn_backend="unfused",
                )
                fused_config = _make_config(
                    hidden_size=256,
                    num_attention_heads=2,
                    v_head_dim=128,
                    q_lora_rank=64,
                    o_groups=2,
                    o_lora_rank=32,
                    csa_window_size=32,
                    dsa_indexer_loss_coeff=1.0,
                    dsa_index_n_heads=16,
                    csa_compress_ratios=[ratio],
                    num_layers=1,
                    csa_indexer_backend="tilelang",
                    csa_sparse_attn_backend="tilelang",
                )
                doc_len_cases = [
                    ##################### 1. pad + //
                    (96, 24),  # x
                    (24, 96),
                    (92, 28),  # x
                    (28, 92),  # x
                    (88, 32),
                    ##################### 2. pad + ! //
                    (89, 37),  # x
                    (87, 39),
                    #################### 3. no pad + //
                    (92, 36),  # x
                    (88, 40),
                    (84, 44),
                    (80, 48),
                    ##################### 4. no pad + ! //
                    (91, 37),  # x
                    (37, 91),
                    (90, 38),  # x
                    (89, 39),  # x
                    (87, 41),
                    (86, 42),
                    (85, 43),
                    (83, 45),
                    (82, 46),
                ]

                def assert_close_with_diff(name, actual, expected):
                    actual = actual.cast("float32")
                    expected = expected.cast("float32")
                    diff = (actual - expected).abs()
                    close_mask = paddle.isclose(
                        actual, expected, rtol=5e-1, atol=5e-1
                    )
                    fail_mask = ~close_mask
                    fail_count = int(fail_mask.cast("int64").sum().item())
                    total_count = fail_mask.numel()
                    max_idx = int(diff.flatten().argmax().item())
                    actual_flat = actual.flatten()
                    expected_flat = expected.flatten()
                    diff_flat = diff.flatten()
                    diff_info = (
                        f"{name}: "
                        f"shape={actual.shape}, "
                        f"max={float(diff.max().item())}, "
                        f"mean={float(diff.mean().item())}, "
                        f"fail={fail_count}/{total_count}, "
                        f"max_idx={max_idx}, "
                        f"actual={float(actual_flat[max_idx].item())}, "
                        f"expected={float(expected_flat[max_idx].item())}, "
                        f"abs_diff={float(diff_flat[max_idx].item())}"
                    )
                    print(f"[diff] {diff_info}")
                    fail_details = []
                    if fail_count > 0:
                        fail_indices = paddle.nonzero(
                            fail_mask.flatten()
                        ).flatten()[:8]
                        for i, fail_idx in enumerate(fail_indices):
                            idx = int(fail_idx.item())
                            detail = (
                                f"{name} fail[{i}]: "
                                f"idx={idx}, "
                                f"actual={float(actual_flat[idx].item())}, "
                                f"expected={float(expected_flat[idx].item())}, "
                                f"abs_diff={float(diff_flat[idx].item())}"
                            )
                            print(f"[diff] {detail}")
                            fail_details.append(detail)
                    error_msg = (
                        f"{diff_info}\n" + "\n".join(fail_details)
                        if fail_details
                        else diff_info
                    )
                    self.assertTrue(close_mask.all().item(), error_msg)

                for doc1_len, doc2_len in doc_len_cases:
                    with self.subTest(
                        ratio=ratio,
                        doc1_len=doc1_len,
                        doc2_len=doc2_len,
                    ):
                        model_parallel_cuda_manual_seed(_SEED)
                        dynamic_attn = _build_attention(
                            dynamic_config, layer_number=0
                        )
                        model_parallel_cuda_manual_seed(_SEED)
                        fused_attn = _build_attention(
                            fused_config, layer_number=0
                        )
                        fused_attn.set_state_dict(dynamic_attn.state_dict())
                        dynamic_attn.train()
                        fused_attn.train()

                        padding_len = seq_len - doc1_len - doc2_len
                        print(f"[ghz] {doc1_len=} {doc2_len=} {padding_len=}")
                        hidden = paddle.randn(
                            [1, seq_len, dynamic_config.hidden_size],
                            dtype="bfloat16",
                        )
                        startend_row_indices = paddle.to_tensor(
                            [doc1_len] * doc1_len
                            + [doc1_len + doc2_len] * (doc2_len + padding_len),
                            dtype="int32",
                        ).reshape([1, 1, seq_len, 1])

                        dynamic_hidden = hidden.clone()
                        fused_hidden = hidden.clone()
                        dynamic_hidden.stop_gradient = False
                        fused_hidden.stop_gradient = False

                        valid_len = doc1_len + doc2_len
                        dynamic_out, _ = dynamic_attn(
                            hidden_states=dynamic_hidden,
                            attention_mask=None,
                            attn_mask_startend_row_indices=startend_row_indices,
                        )
                        grad = paddle.randn(
                            dynamic_out.shape, dynamic_out.dtype
                        )
                        if padding_len > 0:
                            grad[:, valid_len:, :] = 0
                        dynamic_out.backward(grad)
                        dynamic_hidden_grad = dynamic_hidden.grad.clone()
                        dynamic_param_grads = {
                            name: param.grad.clone()
                            for name, param in dynamic_attn.named_parameters()
                            if param.grad is not None
                        }

                        fused_out, _ = fused_attn(
                            hidden_states=fused_hidden,
                            attention_mask=None,
                            attn_mask_startend_row_indices=startend_row_indices,
                        )
                        fused_out.backward(grad)

                        assert_close_with_diff(
                            "output",
                            fused_out,
                            dynamic_out,
                        )
                        fused_params = dict(fused_attn.named_parameters())
                        for name, dynamic_grad in dynamic_param_grads.items():
                            fused_grad = fused_params[name].grad
                            self.assertIsNotNone(fused_grad, name)
                            assert_close_with_diff(
                                name, fused_grad, dynamic_grad
                            )
                        assert_close_with_diff(
                            "hidden_grad",
                            fused_hidden.grad,
                            dynamic_hidden_grad,
                        )
        finally:
            paddle.set_flags({"FLAGS_cudnn_deterministic": old_flag})

    def test_attention_module_document_mask_matches_separate_documents(self):
        paddle.seed(_SEED)
        for ratio in [0, 4, 128]:
            config = _make_config(
                hidden_size=64,
                num_attention_heads=2,
                v_head_dim=32,
                q_lora_rank=32,
                o_groups=2,
                o_lora_rank=16,
                csa_window_size=32,
                dsa_indexer_loss_coeff=0.0,
                csa_compress_ratios=[ratio],
                num_layers=1,
            )
            model_parallel_cuda_manual_seed(_SEED)
            attn = _build_attention(config, layer_number=0)
            attn.eval()
            seq_len = 32
            for doc2_len in [9, 7]:
                doc1 = paddle.randn(
                    [1, 23, config.hidden_size], dtype="bfloat16"
                )
                doc2 = paddle.randn(
                    [1, doc2_len, config.hidden_size], dtype="bfloat16"
                )
                padding_len = seq_len - 23 - doc2_len
                if padding_len > 0:
                    padding = paddle.randn(
                        [1, padding_len, config.hidden_size], dtype="bfloat16"
                    )
                    packed = paddle.concat([doc1, doc2, padding], axis=1)
                else:
                    packed = paddle.concat([doc1, doc2], axis=1)

                startend_row_indices = paddle.to_tensor(
                    [23] * 23 + [23 + doc2_len] * (doc2_len + padding_len),
                    dtype="int32",
                ).reshape([1, 1, 32, 1])
                doc1_startend_row_indices = paddle.to_tensor(
                    [23] * 23, dtype="int32"
                ).reshape([1, 1, 23, 1])
                doc2_startend_row_indices = paddle.to_tensor(
                    [doc2_len] * doc2_len, dtype="int32"
                ).reshape([1, 1, doc2_len, 1])

                with paddle.no_grad():
                    packed_out, _ = attn(
                        hidden_states=packed,
                        attention_mask=None,
                        attn_mask_startend_row_indices=startend_row_indices,
                    )
                    doc1_out, _ = attn(
                        hidden_states=doc1,
                        attention_mask=None,
                        attn_mask_startend_row_indices=doc1_startend_row_indices,
                    )
                    doc2_out, _ = attn(
                        hidden_states=doc2,
                        attention_mask=None,
                        attn_mask_startend_row_indices=doc2_startend_row_indices,
                    )

                self.assertTrue(
                    paddle.equal_all(
                        packed_out[:, :23, :].cast("float32"),
                        doc1_out.cast("float32"),
                    ).item()
                )
                self.assertTrue(
                    paddle.equal_all(
                        packed_out[:, 23 : 23 + doc2_len, :].cast("float32"),
                        doc2_out.cast("float32"),
                    ).item()
                )

    def test_attention_top_level_reuses_docmask_metadata_once(self):
        paddle.seed(_SEED)
        config = _make_config(
            hidden_size=64,
            num_attention_heads=2,
            v_head_dim=32,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            csa_window_size=32,
            dsa_indexer_loss_coeff=0.0,
            csa_compress_ratios=[4],
            num_layers=1,
            csa_indexer_backend="unfused",
            csa_sparse_attn_backend="unfused",
        )
        model_parallel_cuda_manual_seed(_SEED)
        attn = _build_attention(config, layer_number=0)
        attn.eval()
        hidden = paddle.randn([1, 64, config.hidden_size], dtype="bfloat16")
        startend_row_indices = _make_startend_row_indices([17, 23, 11], 64)

        with (
            patch(
                "paddleformers.fleet.transformer.dsv4_hybrid_attention.CSADocMaskMetadata.build",
                wraps=CSADocMaskMetadata.build,
            ) as build_meta,
            paddle.no_grad(),
        ):
            out_first, _ = attn(
                hidden_states=hidden,
                attention_mask=None,
                attn_mask_startend_row_indices=startend_row_indices,
            )
            out_second, _ = attn(
                hidden_states=hidden.clone(),
                attention_mask=None,
                attn_mask_startend_row_indices=startend_row_indices,
            )

        self.assertEqual(build_meta.call_count, 2)
        self.assertTrue(
            paddle.equal_all(
                out_first.cast("float32"),
                out_second.cast("float32"),
            ).item()
        )

    def test_top_level_builds_ratio_one_metadata_for_window_only_docmask(self):
        paddle.seed(_SEED)
        config = _make_config(
            hidden_size=64,
            num_attention_heads=2,
            v_head_dim=32,
            q_lora_rank=32,
            o_groups=2,
            o_lora_rank=16,
            csa_compress_ratios=[0],
            num_layers=1,
            csa_indexer_backend="unfused",
            csa_sparse_attn_backend="unfused",
        )
        model_parallel_cuda_manual_seed(_SEED)
        attn = _build_attention(config, layer_number=0)
        attn.eval()
        hidden = paddle.randn([1, 32, config.hidden_size], dtype="bfloat16")
        startend_row_indices = _make_startend_row_indices([17, 11], 32)

        with (
            patch(
                "paddleformers.fleet.transformer.dsv4_hybrid_attention.CSADocMaskMetadata.build",
                wraps=CSADocMaskMetadata.build,
            ) as mocked,
            paddle.no_grad(),
        ):
            attn(
                hidden_states=hidden,
                attention_mask=None,
                attn_mask_startend_row_indices=startend_row_indices,
            )

        self.assertGreaterEqual(mocked.call_count, 1)
        self.assertEqual(mocked.call_args_list[0].args[0], 1)

    @unittest.skipIf(
        sys.version_info < (3, 12), "cuDNN indexer requires Python >= 3.12"
    )
    def test_cudnn_indexer_document_mask_matches_separate_documents(self):
        """Main-path integration: csa_indexer_backend='cudnn' packed-vs-separate.

        Realistic shapes: packed seq_len 4096, sliding window 128, indexer
        top-k 512. Exercises the production wiring in
        CompressedSparseAttention._compute_indexer_compressed_topk_idxs where
        cudnn_indexer_topk_fwd(valid_range=...) overrides the topk producer
        (csa_attention.py). In eval mode the cuDNN indexer selects the
        compressed top-k under document-mask; the pure-Paddle sparse attention
        gathers it. doc0 / doc1 outputs sliced from the packed run must equal
        each document run alone — any cross-document leakage in the cuDNN
        docmask topk (wrong valid_range window or bad local->global remap)
        would break the equality.

        cuDNN indexer constraints force dsa_index_n_heads in {32,64} and
        dsa_index_head_dim=128. Document lengths are kept >= 8 (n_compressed
        >= 2 at ratio 4); the cuDNN indexer forward kernel crashes at
        n_compressed == 1, a pre-existing limitation unrelated to docmask.
        """
        paddle.seed(_SEED)
        ratio = 4  # indexer only exists for ratio == 4
        config = _make_config(
            hidden_size=256,
            num_attention_heads=2,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            q_lora_rank=128,
            o_groups=2,
            o_lora_rank=64,
            csa_window_size=128,
            dsa_index_n_heads=32,  # cuDNN requires {32, 64}
            dsa_index_head_dim=128,  # cuDNN requires 128
            dsa_index_topk=512,
            dsa_indexer_loss_coeff=0.0,
            csa_compress_ratios=[ratio],
            num_layers=1,
            csa_indexer_backend="cudnn",
        )
        model_parallel_cuda_manual_seed(_SEED)
        attn = _build_attention(config, layer_number=0)
        attn.eval()

        seq_len = 4096
        # Two documents + trailing padding (the realistic packed layout).
        # doc1 2000 -> cutoff2000 -> 500 compressed cols
        # doc2 1500 -> cutoff1500 -> 375 compressed cols
        # padding 596. Each doc's n_compressed >= 2 (avoids the n_comp==1 crash).
        doc1_len, doc2_len = 2000, 1500
        padding_len = seq_len - doc1_len - doc2_len

        doc1 = paddle.randn([1, doc1_len, config.hidden_size], dtype="bfloat16")
        doc2 = paddle.randn([1, doc2_len, config.hidden_size], dtype="bfloat16")
        padding = paddle.randn(
            [1, padding_len, config.hidden_size], dtype="bfloat16"
        )
        packed = paddle.concat([doc1, doc2, padding], axis=1)

        startend_row_indices = paddle.to_tensor(
            [doc1_len] * doc1_len
            + [doc1_len + doc2_len] * (doc2_len + padding_len),
            dtype="int32",
        ).reshape([1, 1, seq_len, 1])
        doc1_startend = paddle.to_tensor(
            [doc1_len] * doc1_len, dtype="int32"
        ).reshape([1, 1, doc1_len, 1])
        doc2_startend = paddle.to_tensor(
            [doc2_len] * doc2_len, dtype="int32"
        ).reshape([1, 1, doc2_len, 1])

        with paddle.no_grad():
            packed_out, _ = attn(
                hidden_states=packed,
                attention_mask=None,
                attn_mask_startend_row_indices=startend_row_indices,
            )
            doc1_out, _ = attn(
                hidden_states=doc1,
                attention_mask=None,
                attn_mask_startend_row_indices=doc1_startend,
            )
            doc2_out, _ = attn(
                hidden_states=doc2,
                attention_mask=None,
                attn_mask_startend_row_indices=doc2_startend,
            )

        # Sliced packed doc outputs must match standalone doc runs. allclose
        # (not equal_all): cuDNN radix topk tie order + bf16 reductions admit
        # tiny deviations even on identical per-doc inputs.
        self.assertTrue(
            paddle.allclose(
                packed_out[:, :doc1_len, :].cast("float32"),
                doc1_out.cast("float32"),
                rtol=1e-2,
                atol=1e-2,
            ).item(),
            "cuDNN-indexer docmask: packed doc0 != doc0 alone",
        )
        self.assertTrue(
            paddle.allclose(
                packed_out[:, doc1_len : doc1_len + doc2_len, :].cast(
                    "float32"
                ),
                doc2_out.cast("float32"),
                rtol=1e-2,
                atol=1e-2,
            ).item(),
            "cuDNN-indexer docmask: packed doc1 != doc1 alone",
        )


class TestDSv4HybridAttentionConstructor(unittest.TestCase):
    def test_basic_construction(self):
        paddle.seed(_SEED)
        config = _make_config()
        attn = _build_attention(config, layer_number=1)

        self.assertIsInstance(attn, DSv4HybridSelfAttention)
        self.assertTrue(hasattr(attn, "linear_q_down_proj"))
        self.assertTrue(hasattr(attn, "linear_q_up_proj"))
        self.assertTrue(hasattr(attn, "linear_kv_proj"))
        self.assertTrue(hasattr(attn, "o_proj"))
        self.assertTrue(hasattr(attn, "linear_o_group_proj"))
        self.assertTrue(hasattr(attn, "core_attention"))
        self.assertTrue(hasattr(attn, "q_layernorm"))
        self.assertTrue(hasattr(attn, "kv_layernorm"))

    def test_csa_ratio_builds_and_forward(self):
        # CSA layers accept any integer compress ratio in [2, 127]. Cover small
        # and large powers of two as well as a non-power-of-2 ratio (3).
        ratios = [2, 4, 8, 16, 64, 3]
        for ratio in ratios:
            with self.subTest(ratio=ratio):
                paddle.seed(_SEED)
                config = _make_config(num_layers=1, csa_compress_ratios=[ratio])
                attn = _build_attention(config, layer_number=0)
                attn.eval()

                # Every CSA layer must build a compressor with overlap (coff=2)
                # and a Lightning Indexer.
                self.assertIsNotNone(attn.core_attention.compressor)
                self.assertTrue(attn.core_attention.compressor.overlap)
                self.assertEqual(attn.core_attention.compressor.coff, 2)
                self.assertIsNotNone(attn.core_attention.indexer)

                batch_size = 1
                seq_len = 64  # divisible by every ratio above
                hidden = paddle.randn(
                    [batch_size, seq_len, config.hidden_size],
                    dtype="bfloat16",
                )
                with paddle.no_grad():
                    output, _ = attn(hidden_states=hidden, attention_mask=None)

                self.assertEqual(
                    list(output.shape),
                    [batch_size, seq_len, config.hidden_size],
                )
                self.assertTrue(
                    paddle.isfinite(output.cast("float32")).all().item()
                )

    def test_csa_indexer_count_general(self):
        # A [0, 4, 8, 16, 128] config must produce exactly 3 indexer layers
        # (CSA-4, CSA-8, CSA-16); window (0) and HCA (128) have indexer=None.
        # Matches dsa_attention.py track_indexer_metrics.
        paddle.seed(_SEED)
        ratios = [0, 4, 8, 16, 128]
        config = _make_config(
            num_layers=len(ratios), csa_compress_ratios=ratios
        )

        num_indexer = 0
        for layer_number, ratio in enumerate(ratios):
            attn = _build_attention(config, layer_number=layer_number)
            core = attn.core_attention
            self.assertEqual(core.compress_ratio, ratio)
            if 1 < ratio < 128:
                self.assertIsNotNone(core.indexer)
                num_indexer += 1
            else:
                self.assertIsNone(core.indexer)

        self.assertEqual(num_indexer, 3)

    def test_csa_ratio_boundaries(self):
        # ratio 127 (the upper CSA boundary) builds a CSA layer with overlap
        # and a Lightning Indexer.
        paddle.seed(_SEED)
        config = _make_config(num_layers=1, csa_compress_ratios=[127])
        attn = _build_attention(config, layer_number=0)
        attn.eval()
        self.assertIsNotNone(attn.core_attention.compressor)
        self.assertTrue(attn.core_attention.compressor.overlap)
        self.assertIsNotNone(attn.core_attention.indexer)

        # ratio 1 is ambiguous (no compression yet not window) -> rejected.
        with self.assertRaisesRegex(ValueError, "is invalid"):
            _make_config(num_layers=1, csa_compress_ratios=[1])

        # ratio 129 is above HCA (128) -> rejected.
        with self.assertRaisesRegex(ValueError, "is invalid"):
            _make_config(num_layers=1, csa_compress_ratios=[129])

    def test_q_head_dim_equals_v_head_dim(self):
        paddle.seed(_SEED)
        config = _make_config()
        attn = _build_attention(config, layer_number=1)

        self.assertEqual(attn.q_head_dim, config.v_head_dim)

    def test_rope_base_varies_with_compress_ratio(self):
        paddle.seed(_SEED)
        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)

        for layer_number, ratio in enumerate(ratios):
            attn = _build_attention(config, layer_number=layer_number)
            self.assertIsInstance(
                attn.core_attention, CompressedSparseAttention
            )
            self.assertEqual(attn.core_attention.compress_ratio, ratio)

            expected_base = (
                config.csa_compress_rotary_base
                if ratio > 1
                else config.rotary_base
            )
            dim = config.qk_pos_emb_head_dim
            expected_inv_freq = 1.0 / (
                expected_base
                ** (paddle.arange(0, dim, 2, dtype="float32") / dim)
            )
            self.assertTrue(
                paddle.allclose(
                    attn.rotary_pos_emb.inv_freq.cast("float32"),
                    expected_inv_freq,
                    rtol=1e-5,
                    atol=1e-5,
                ).item()
            )

    def test_mtp_layer_uses_nextn_compress_ratio(self):
        ratios = [0, 4, 128, 4, 128]
        config = _make_config(
            num_layers=4,
            num_nextn_predict_layers=1,
            csa_compress_ratios=ratios,
        )
        spec = get_attention_spec(
            config=config,
            attention_layer_type="dsv4_hybrid_attention",
            attn_mask_type=AttnMaskType.causal,
            is_mtp_layer=True,
        )
        attn = build_spec_layer(spec, config=config, layer_number=0)

        self.assertEqual(
            attn.core_attention.compress_ratio, ratios[config.num_hidden_layers]
        )
        self.assertEqual(
            attn.core_attention.layer_number, config.num_hidden_layers + 1
        )

    def test_non_dense_mtp_spec_uses_mtp_attention_ratio(self):
        ratios = [0, 4, 128, 4, 128]
        config = _make_config(
            num_layers=4,
            num_nextn_predict_layers=1,
            csa_compress_ratios=ratios,
        )
        decoder_specs = get_gpt_decoder_layers_spec(
            config=config,
            normalization=config.normalization,
        )
        mtp_specs = get_gpt_mtp_layers_spec(config=config, spec=decoder_specs)
        mtp_self_attn_spec = mtp_specs[
            0
        ].sublayers_spec.transformer_layer.sublayers_spec.self_attn
        attn = build_spec_layer(
            mtp_self_attn_spec,
            config=config,
            layer_number=0,
        )

        self.assertEqual(
            attn.core_attention.compress_ratio, ratios[config.num_hidden_layers]
        )

    def test_yarn_rope_construction(self):
        config = _make_config(rope_type="yarn")
        attn = _build_attention(config, layer_number=1)
        freqs, mscale = attn.rotary_pos_emb(8, packed_seq=False)

        self.assertEqual(
            list(freqs.shape), [1, 8, 1, config.qk_pos_emb_head_dim]
        )
        self.assertIsInstance(mscale, float)

    def test_o_group_proj_shape(self):
        paddle.seed(_SEED)
        o_groups = 4
        o_lora_rank = 32
        config = _make_config(o_groups=o_groups, o_lora_rank=o_lora_rank)
        attn = _build_attention(config, layer_number=1)

        expected_out = o_groups * o_lora_rank
        expected_in = (
            config.v_head_dim * config.num_attention_heads
        ) // o_groups
        self.assertEqual(
            list(attn.linear_o_group_proj.shape), [expected_out, expected_in]
        )
        self.assertFalse(attn.linear_o_group_proj.stop_gradient)


class TestDSv4HybridFusedSparseAttention(unittest.TestCase):
    def test_fused_matches_unfused_forward_backward(self):
        old_flag = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        paddle.set_flags({"FLAGS_cudnn_deterministic": 0})
        try:
            paddle.seed(_SEED)
            batch_size = 1
            seq_len = 128
            num_heads = 16
            head_dim = 128
            topk = 64
            softmax_scale = head_dim**-0.5

            query = paddle.randn(
                [batch_size, seq_len, num_heads, head_dim],
                dtype=paddle.bfloat16,
            )
            kv_full = paddle.randn(
                [batch_size, seq_len, head_dim], dtype=paddle.bfloat16
            )
            attn_sink = paddle.randn([num_heads], dtype=paddle.float32)
            topk_idxs = (
                paddle.arange(topk, dtype="int32")
                .reshape([1, 1, topk])
                .expand([batch_size, seq_len, topk])
            )

            query.stop_gradient = False
            kv_full.stop_gradient = False
            attn_sink.stop_gradient = False
            fused_out = csa_sparse_attn(
                query,
                kv_full,
                attn_sink,
                topk_idxs,
                softmax_scale,
                backend="tilelang",
            )
            fused_loss = fused_out.cast("float32").sum()
            fused_loss.backward()
            fused_query_grad = query.grad.clone()
            fused_kv_grad = kv_full.grad.clone()
            fused_attn_sink_grad = attn_sink.grad.clone()

            query.clear_gradient()
            kv_full.clear_gradient()
            attn_sink.clear_gradient()
            unfused_out = unfused_compressed_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, softmax_scale
            )
            unfused_loss = unfused_out.cast("float32").sum()
            unfused_loss.backward()

            self.assertTrue(
                paddle.allclose(
                    fused_out.cast("float32"),
                    unfused_out.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_query_grad.cast("float32"),
                    query.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_kv_grad.cast("float32"),
                    kv_full.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_attn_sink_grad.cast("float32"),
                    attn_sink.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
        finally:
            paddle.set_flags({"FLAGS_cudnn_deterministic": old_flag})


class TestDSv4HybridAttentionForwardBackward(unittest.TestCase):
    def setUp(self):
        paddle.seed(_SEED)
        self.config = _make_config(dsa_indexer_loss_coeff=1.0)

    def test_backward_gradient_flow(self):
        batch_size = 2
        seq_len = 64

        for layer_number in [0, 1]:
            attn = _build_attention(self.config, layer_number=layer_number)
            attn.train()
            hidden = paddle.randn(
                [batch_size, seq_len, self.config.hidden_size],
                dtype=paddle.bfloat16,
            )
            hidden.stop_gradient = False

            output, _ = attn(hidden_states=hidden, attention_mask=None)
            loss = output.cast("float32").sum()
            loss.backward()

            self.assertIsNotNone(hidden.grad)
            self.assertTrue(
                paddle.isfinite(hidden.grad.cast("float32")).all().item()
            )
            used_params = [
                name
                for name, param in attn.named_parameters()
                if not param.stop_gradient and param.grad is not None
            ]
            self.assertGreater(len(used_params), 0)
            for name, param in attn.named_parameters():
                if not param.stop_gradient and param.grad is not None:
                    self.assertTrue(
                        paddle.isfinite(param.grad.cast("float32"))
                        .all()
                        .item(),
                        f"Non-finite gradient for parameter {name}",
                    )

    def test_eval_mode(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        attn.eval()
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        with paddle.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        self.assertEqual(
            list(output.shape), [batch_size, seq_len, self.config.hidden_size]
        )
        self.assertTrue(paddle.isfinite(output.cast("float32")).all().item())
        self.assertIsNone(bias)

    def test_different_seq_lengths(self):
        batch_size = 2
        attn = _build_attention(self.config, layer_number=2)

        for seq_len in [32, 64, 128]:
            hidden = paddle.randn(
                [batch_size, seq_len, self.config.hidden_size],
                dtype=paddle.bfloat16,
            )
            output, _ = attn(hidden_states=hidden, attention_mask=None)
            self.assertEqual(
                list(output.shape),
                [batch_size, seq_len, self.config.hidden_size],
            )
            self.assertTrue(
                paddle.isfinite(output.cast("float32")).all().item()
            )

    def test_rope_fusion(self):
        batch_size = 2
        seq_len = 128
        self.config.apply_rope_fusion = True
        attn = _build_attention(self.config, layer_number=2)
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        output, _ = attn(hidden_states=hidden, attention_mask=None)

        self.assertEqual(
            list(output.shape),
            [batch_size, seq_len, self.config.hidden_size],
        )
        self.assertTrue(paddle.isfinite(output.float()).all().item())

    def test_gated_attention(self):
        batch_size = 2
        seq_len = 64
        model_parallel_cuda_manual_seed(_SEED)

        for use_q_lora in [False, True]:
            config = _make_config(dsa_indexer_loss_coeff=1.0)
            config.gated_attention = True
            config.gated_attn_use_q_lora = use_q_lora
            attn = _build_attention(config, layer_number=1)
            attn.recompute_gated_attn = not use_q_lora
            attn.config.sigmoid_gate_fusion = use_q_lora

            self.assertTrue(attn.gated_attention)
            self.assertEqual(attn.gated_attn_use_q_lora, use_q_lora)
            self.assertIsNotNone(attn.gate_proj)

            hidden = paddle.randn(
                [batch_size, seq_len, config.hidden_size],
                dtype=paddle.bfloat16,
            )
            output, bias = attn(hidden_states=hidden, attention_mask=None)

            self.assertEqual(
                list(output.shape),
                [batch_size, seq_len, config.hidden_size],
            )
            self.assertTrue(paddle.isfinite(output.float()).all().item())


class TestDSv4HybridQKV(unittest.TestCase):
    def setUp(self):
        paddle.seed(_SEED)
        self.config = _make_config(dsa_indexer_loss_coeff=0.0)

    def test_qkv_shapes(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(
            hidden
        )

        self.assertEqual(
            list(q.shape),
            [
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.v_head_dim,
            ],
        )
        self.assertEqual(
            list(k.shape), [batch_size, seq_len, 1, self.config.v_head_dim]
        )
        self.assertEqual(
            list(v.shape), [batch_size, seq_len, 1, self.config.v_head_dim]
        )
        self.assertEqual(
            list(q_compressed.shape),
            [batch_size, seq_len, self.config.q_lora_rank],
        )
        self.assertEqual(list(kv_compressed.shape), list(hidden.shape))

    def test_key_equals_value(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        _, key, value, _, _ = attn.get_query_key_value_tensors(hidden)
        self.assertTrue(
            paddle.equal_all(key.cast("float32"), value.cast("float32")).item()
        )


if __name__ == "__main__":
    unittest.main()
