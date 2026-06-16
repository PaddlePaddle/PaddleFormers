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

"""Multi-card unit tests for CSA (CompressedSparseAttention) Context Parallelism.

Tests verify that CSA with CP enabled (local Q slice, all-gather KV) produces
identical forward output and backward gradients compared to the non-CP reference
(full sequence on a single rank).

Covers:
  - CompressedSparseAttention._forward_cp: fwd output matches non-CP forward
  - Backward: input grads (dQ, dK, dX) match the local slice of non-CP grads
  - Backward: parameter grads (compressor, indexer) match after all-reduce across CP
  - DSv4HybridSelfAttention: full-layer fwd+bwd including RoPE offset

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/fleet/multi_card_tests/transformer/test_csa_attention_cp.py
"""

import os
import sys
import types
import unittest

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from paddleformers.fleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddleformers.fleet.transformer.csa_attention import (
    CompressedSparseAttention,
    CompressedSparseAttentionSublayersSpec,
    Compressor,
    CompressorSublayersSpec,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)
from paddleformers.fleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
    DSv4HybridSelfAttentionSublayersSpec,
)

# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------
CP_SIZE = None
CP_RANK = None
CP_GROUP = None

DTYPE = "bfloat16"
FWD_ATOL = 5e-1
BWD_ATOL = 5e-1
COS_SIM_THRESHOLD = 0.95


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    CP_SIZE = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": CP_SIZE,
        "sep_degree": 1,
        "cp_degree": CP_SIZE,
        "ep_degree": CP_SIZE,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    CP_GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks


# ---------------------------------------------------------------------------
# Stub layers (lightweight, no TP distribution)
# ---------------------------------------------------------------------------
class _TestLinear(nn.Layer):
    def __init__(self, input_size, output_size, dtype=None, **kwargs):
        super().__init__()
        if dtype is None:
            dtype = DTYPE
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype=dtype,
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


class _TestRMSNorm(nn.Layer):
    def __init__(self, hidden_size=None, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        normed = x * paddle.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return normed * self.weight.cast(x.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_csa_config(
    compress_ratio=4,
    hidden_size=256,
    head_dim=64,
    q_lora_rank=64,
    csa_window_size=64,
    dsa_index_topk=16,
    dsa_indexer_loss_coeff=0.0,
):
    return types.SimpleNamespace(
        num_attention_heads=8,
        v_head_dim=head_dim,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        qk_pos_emb_head_dim=32,
        csa_window_size=csa_window_size,
        csa_compress_ratios=[compress_ratio],
        csa_dense_mode=False,
        dsa_index_n_heads=16,
        dsa_index_head_dim=32,
        dsa_index_topk=dsa_index_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=False,
        csa_tilelang_enable_indexer=False,
        csa_tilelang_enable_sparse_attn=False,
        init_method=None,
        init_method_std=0.02,
        layernorm_epsilon=1e-5,
        num_hidden_layers=1,
    )


def _build_csa(config, compress_ratio=4, head_dim=64):
    rope = RotaryEmbedding(32, rotary_percent=1.0, rotary_base=160000)
    comp_spec = CompressorSublayersSpec(
        linear_wkv=_TestLinear,
        linear_wgate=_TestLinear,
        norm=_TestRMSNorm,
    )
    indexer_comp_spec = CompressorSublayersSpec(
        linear_wkv=_TestLinear,
        linear_wgate=_TestLinear,
        norm=_TestRMSNorm,
    )
    indexer_sublayers = CSAIndexerSublayersSpec(
        linear_wq_b=_TestLinear,
        linear_weights_proj=_TestLinear,
        compressor=LayerSpec(
            layer=Compressor, sublayers_spec=indexer_comp_spec
        ),
    )
    attn_spec = CompressedSparseAttentionSublayersSpec(
        compressor=LayerSpec(layer=Compressor, sublayers_spec=comp_spec),
        indexer=LayerSpec(layer=CSAIndexer, sublayers_spec=indexer_sublayers),
    )
    return CompressedSparseAttention(
        config=config,
        sublayers_spec=attn_spec,
        layer_number=1,
        attn_mask_type=None,
        attention_type="self",
        k_channels=head_dim,
        v_channels=head_dim,
        compress_ratio=compress_ratio,
        rotary_pos_emb=rope,
    )


def _max_diff(actual, expected):
    a = actual.cast("float32").flatten()
    b = expected.cast("float32").flatten()
    diff = (a - b).abs()
    return diff.max().item()


def _cosine_sim(actual, expected):
    a = actual.cast("float32").flatten()
    b = expected.cast("float32").flatten()
    dot = (a * b).sum()
    return (dot / (a.norm() * b.norm() + 1e-30)).item()


def _run_csa_cp_vs_ref(
    b,
    sq_global,
    head_dim,
    hidden_size,
    q_lora_rank,
    csa_window_size,
    dsa_index_topk,
    dsa_indexer_loss_coeff,
):
    """Run one config: build ref (non-CP) and CP instances, compare fwd+bwd."""
    compress_ratio = 4
    sq_local = sq_global // CP_SIZE

    config = _build_csa_config(
        compress_ratio=compress_ratio,
        hidden_size=hidden_size,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        csa_window_size=csa_window_size,
        dsa_index_topk=dsa_index_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
    )

    # Reference: full sequence, CP disabled
    paddle.seed(2026)
    csa_ref = _build_csa(config, compress_ratio, head_dim)
    csa_ref.cp_group = None
    csa_ref.cp_size = 1
    csa_ref.cp_rank = 0
    csa_ref.cp_enabled = False

    # CP: local slice, CP enabled
    paddle.seed(2026)
    csa_cp = _build_csa(config, compress_ratio, head_dim)
    csa_cp.cp_group = CP_GROUP
    csa_cp.cp_size = CP_SIZE
    csa_cp.cp_rank = CP_RANK
    csa_cp.cp_enabled = True

    if dsa_indexer_loss_coeff > 0:
        csa_ref.train()
        csa_cp.train()

    np_heads = config.num_attention_heads

    # Generate inputs (same seed across ranks)
    paddle.seed(1000)
    query_full = paddle.randn([b, sq_global, np_heads, head_dim], dtype=DTYPE)
    key_full = paddle.randn([b, sq_global, 1, head_dim], dtype=DTYPE)
    x_full = paddle.randn([b, sq_global, hidden_size], dtype=DTYPE)
    qr_full = paddle.randn([b, sq_global, q_lora_rank], dtype=DTYPE)

    # Side A: full-sequence reference
    q_a = query_full.clone()
    q_a.stop_gradient = False
    k_a = key_full.clone()
    k_a.stop_gradient = False
    x_a = x_full.clone()
    x_a.stop_gradient = False
    qr_a = qr_full.clone()
    qr_a.stop_gradient = False
    out_a = csa_ref.forward(q_a, k_a, k_a, None, x=x_a, qr=qr_a)
    out_a.sum().backward()

    # Side B: CP local slice
    s, e = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
    q_b = query_full[:, s:e].clone()
    q_b.stop_gradient = False
    k_b = key_full[:, s:e].clone()
    k_b.stop_gradient = False
    x_b = x_full[:, s:e].clone()
    x_b.stop_gradient = False
    qr_b = qr_full[:, s:e].clone()
    qr_b.stop_gradient = False
    out_b = csa_cp.forward(q_b, k_b, k_b, None, x=x_b, qr=qr_b)
    out_b.sum().backward()

    # All-reduce param grads (simulates ZeRO sharding MEAN × cp_size = SUM)
    for p in csa_cp.parameters():
        if p.grad is not None:
            g = p.grad.contiguous()
            dist.all_reduce(g, group=CP_GROUP)
            paddle.assign(g, p.grad)

    # Assertions
    results = {}
    results["fwd_cos"] = _cosine_sim(out_b, out_a[:, s:e])
    results["fwd_diff"] = _max_diff(out_b, out_a[:, s:e])
    results["dq_cos"] = _cosine_sim(q_b.grad, q_a.grad[:, s:e])
    results["dx_cos"] = _cosine_sim(x_b.grad, x_a.grad[:, s:e])

    # Compressor param grad
    results["comp_wkv_cos"] = _cosine_sim(
        csa_cp.compressor.linear_wkv.weight.grad,
        csa_ref.compressor.linear_wkv.weight.grad,
    )

    if dsa_indexer_loss_coeff > 0:
        results["idx_wq_cos"] = _cosine_sim(
            csa_cp.indexer.linear_wq_b.weight.grad,
            csa_ref.indexer.linear_wq_b.weight.grad,
        )

    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestCSAContextParallel(unittest.TestCase):
    """CompressedSparseAttention CP vs non-CP equivalence."""

    def _assert_results(self, results, msg=""):
        self.assertGreater(
            results["fwd_cos"],
            COS_SIM_THRESHOLD,
            f"{msg} forward cosine sim too low: {results['fwd_cos']:.6f}",
        )
        self.assertGreater(
            results["dq_cos"],
            COS_SIM_THRESHOLD,
            f"{msg} dQ cosine sim too low: {results['dq_cos']:.6f}",
        )
        self.assertGreater(
            results["dx_cos"],
            COS_SIM_THRESHOLD,
            f"{msg} dX cosine sim too low: {results['dx_cos']:.6f}",
        )
        self.assertGreater(
            results["comp_wkv_cos"],
            COS_SIM_THRESHOLD,
            f"{msg} compressor.wkv grad cosine sim too low: {results['comp_wkv_cos']:.6f}",
        )
        if "idx_wq_cos" in results:
            self.assertGreater(
                results["idx_wq_cos"],
                COS_SIM_THRESHOLD,
                f"{msg} indexer.wq grad cosine sim too low: {results['idx_wq_cos']:.6f}",
            )

    def test_basic_fwd_bwd(self):
        """Basic CSA CP: sq=256, window=64, topk=16, no indexer loss."""
        results = _run_csa_cp_vs_ref(
            b=2,
            sq_global=256,
            head_dim=64,
            hidden_size=256,
            q_lora_rank=64,
            csa_window_size=64,
            dsa_index_topk=16,
            dsa_indexer_loss_coeff=0.0,
        )
        self._assert_results(results, "basic")

    def test_large_window(self):
        """CSA CP with larger window covering more of the sequence."""
        results = _run_csa_cp_vs_ref(
            b=2,
            sq_global=256,
            head_dim=64,
            hidden_size=256,
            q_lora_rank=64,
            csa_window_size=128,
            dsa_index_topk=16,
            dsa_indexer_loss_coeff=0.0,
        )
        self._assert_results(results, "large_window")

    def test_with_indexer_loss(self):
        """CSA CP with indexer loss enabled (Paddle reference loss path)."""
        results = _run_csa_cp_vs_ref(
            b=2,
            sq_global=256,
            head_dim=64,
            hidden_size=256,
            q_lora_rank=64,
            csa_window_size=64,
            dsa_index_topk=16,
            dsa_indexer_loss_coeff=1.0,
        )
        self._assert_results(results, "indexer_loss")

    def test_longer_sequence(self):
        """CSA CP with sq=1024 to exercise more CP boundary cases."""
        results = _run_csa_cp_vs_ref(
            b=2,
            sq_global=1024,
            head_dim=64,
            hidden_size=256,
            q_lora_rank=128,
            csa_window_size=64,
            dsa_index_topk=64,
            dsa_indexer_loss_coeff=0.0,
        )
        self._assert_results(results, "long_seq")

    def test_larger_topk(self):
        """CSA CP with large topk relative to compressed sequence."""
        results = _run_csa_cp_vs_ref(
            b=2,
            sq_global=256,
            head_dim=64,
            hidden_size=256,
            q_lora_rank=64,
            csa_window_size=64,
            dsa_index_topk=64,
            dsa_indexer_loss_coeff=0.0,
        )
        self._assert_results(results, "large_topk")


class TestDSv4HybridAttentionCP(unittest.TestCase):
    """DSv4HybridSelfAttention full-layer CP vs non-CP (includes RoPE)."""

    def test_full_layer_fwd_bwd(self):
        """Full DSv4HybridSelfAttention layer: Q/KV proj, RoPE, CSA, inverse RoPE."""
        from paddleformers.fleet.transformer.enums import AttnMaskType
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        head_dim = 64
        hidden_size = 256
        num_heads = 4
        q_lora_rank = 64
        pos_dim = 32
        sq_global = 256
        sq_local = sq_global // CP_SIZE
        b = 2

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )
        config.num_key_value_heads = 1
        config.head_dim = head_dim
        config.q_lora_rank = q_lora_rank
        config.kv_lora_rank = q_lora_rank
        config.qk_nope_head_dim = head_dim - pos_dim
        config.qk_rope_head_dim = pos_dim
        config.qk_pos_emb_head_dim = pos_dim
        config.v_head_dim = head_dim
        config.multi_latent_attention = True
        config.rope_type = "rope"
        config.rotary_base = 10000.0
        config.rotary_interleaved = False
        config.rotary_percent = 1.0
        config.apply_rope_fusion = False
        config.csa_compress_ratios = [4]
        config.csa_compress_rotary_base = 160000.0
        config.csa_window_size = 64
        config.csa_dense_mode = False
        config.dsa_index_n_heads = 8
        config.dsa_index_head_dim = 32
        config.dsa_index_topk = 16
        config.dsa_indexer_loss_coeff = 0.0
        config.dsa_indexer_use_sparse_loss = False
        config.csa_tilelang_enable_indexer = False
        config.csa_tilelang_enable_sparse_attn = False
        config.init_method = None
        config.init_method_std = 0.02
        config.output_layer_init_method = None
        config.layernorm_epsilon = 1e-5
        config.rms_norm_eps = 1e-5
        config.o_groups = 4
        config.o_lora_rank = 64
        config.sequence_parallel = False
        config.tensor_model_parallel_size = 1
        config.cp_balance_mode = "contiguous_allgather"
        config.csa_indexer_backend = "tilelang"

        sublayers = DSv4HybridSelfAttentionSublayersSpec(
            linear_q_down_proj=_TestLinear,
            linear_q_up_proj=_TestLinear,
            linear_kv_proj=_TestLinear,
            core_attention=LayerSpec(
                layer=CompressedSparseAttention,
                sublayers_spec=CompressedSparseAttentionSublayersSpec(
                    compressor=LayerSpec(
                        layer=Compressor,
                        sublayers_spec=CompressorSublayersSpec(
                            linear_wkv=_TestLinear,
                            linear_wgate=_TestLinear,
                            norm=_TestRMSNorm,
                        ),
                    ),
                    indexer=LayerSpec(
                        layer=CSAIndexer,
                        sublayers_spec=CSAIndexerSublayersSpec(
                            linear_wq_b=_TestLinear,
                            linear_weights_proj=_TestLinear,
                            compressor=LayerSpec(
                                layer=Compressor,
                                sublayers_spec=CompressorSublayersSpec(
                                    linear_wkv=_TestLinear,
                                    linear_wgate=_TestLinear,
                                    norm=_TestRMSNorm,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            o_proj=_TestLinear,
            q_layernorm=_TestRMSNorm,
            kv_layernorm=_TestRMSNorm,
        )

        # Build ref (no CP) and CP instances with same weights
        pg_ref = types.SimpleNamespace(tp=None, cp=None)
        pg_cp = types.SimpleNamespace(tp=None, cp=CP_GROUP)

        paddle.seed(2026)
        layer_ref = DSv4HybridSelfAttention(
            config=config,
            sublayers_spec=sublayers,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_ref,
        )

        paddle.seed(2026)
        layer_cp = DSv4HybridSelfAttention(
            config=config,
            sublayers_spec=sublayers,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_cp,
        )

        # Generate inputs
        paddle.seed(1000)
        hidden_full = paddle.randn([b, sq_global, hidden_size], dtype=DTYPE)

        # Side A: full sequence
        h_a = hidden_full.clone()
        h_a.stop_gradient = False
        out_a, _ = layer_ref.forward(h_a)
        out_a.sum().backward()

        # Side B: CP local slice
        s, e = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
        h_b = hidden_full[:, s:e].clone()
        h_b.stop_gradient = False
        out_b, _ = layer_cp.forward(h_b)
        out_b.sum().backward()

        # All-reduce param grads
        for p in layer_cp.parameters():
            if p.grad is not None:
                g = p.grad.contiguous()
                dist.all_reduce(g, group=CP_GROUP)
                paddle.assign(g, p.grad)

        # Verify forward
        fwd_cos = _cosine_sim(out_b, out_a[:, s:e])
        self.assertGreater(
            fwd_cos,
            COS_SIM_THRESHOLD,
            f"Full-layer forward cosine sim too low: {fwd_cos:.6f}",
        )

        # Verify input grad
        dh_cos = _cosine_sim(h_b.grad, h_a.grad[:, s:e])
        self.assertGreater(
            dh_cos,
            COS_SIM_THRESHOLD,
            f"Full-layer dH cosine sim too low: {dh_cos:.6f}",
        )

    def test_full_layer_fwd_bwd_fp8_qat(self):
        """DSv4HybridSelfAttention with use_fp8_qat=True: CP vs non-CP."""
        from paddleformers.fleet.transformer.enums import AttnMaskType
        from paddleformers.fleet.transformer.transformer_config import (
            TransformerConfig,
        )

        head_dim = 128
        hidden_size = 256
        num_heads = 4
        q_lora_rank = 64
        pos_dim = 64
        sq_global = 256
        sq_local = sq_global // CP_SIZE
        b = 2

        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            use_fp8_qat=True,
        )
        config.num_key_value_heads = 1
        config.head_dim = head_dim
        config.q_lora_rank = q_lora_rank
        config.kv_lora_rank = q_lora_rank
        config.qk_nope_head_dim = head_dim - pos_dim
        config.qk_rope_head_dim = pos_dim
        config.qk_pos_emb_head_dim = pos_dim
        config.v_head_dim = head_dim
        config.multi_latent_attention = True
        config.rope_type = "rope"
        config.rotary_base = 10000.0
        config.rotary_interleaved = False
        config.rotary_percent = 1.0
        config.apply_rope_fusion = False
        config.csa_compress_ratios = [4]
        config.csa_compress_rotary_base = 160000.0
        config.csa_window_size = 64
        config.csa_dense_mode = False
        config.dsa_index_n_heads = 8
        config.dsa_index_head_dim = 128
        config.dsa_index_topk = 16
        config.dsa_indexer_loss_coeff = 0.0
        config.dsa_indexer_use_sparse_loss = False
        config.csa_tilelang_enable_indexer = False
        config.csa_tilelang_enable_sparse_attn = False
        config.init_method = None
        config.init_method_std = 0.02
        config.output_layer_init_method = None
        config.layernorm_epsilon = 1e-5
        config.rms_norm_eps = 1e-5
        config.o_groups = 4
        config.o_lora_rank = 64
        config.sequence_parallel = False
        config.tensor_model_parallel_size = 1
        config.cp_balance_mode = "contiguous_allgather"
        config.csa_indexer_backend = "tilelang"

        sublayers = DSv4HybridSelfAttentionSublayersSpec(
            linear_q_down_proj=_TestLinear,
            linear_q_up_proj=_TestLinear,
            linear_kv_proj=_TestLinear,
            core_attention=LayerSpec(
                layer=CompressedSparseAttention,
                sublayers_spec=CompressedSparseAttentionSublayersSpec(
                    compressor=LayerSpec(
                        layer=Compressor,
                        sublayers_spec=CompressorSublayersSpec(
                            linear_wkv=_TestLinear,
                            linear_wgate=_TestLinear,
                            norm=_TestRMSNorm,
                        ),
                    ),
                    indexer=LayerSpec(
                        layer=CSAIndexer,
                        sublayers_spec=CSAIndexerSublayersSpec(
                            linear_wq_b=_TestLinear,
                            linear_weights_proj=_TestLinear,
                            compressor=LayerSpec(
                                layer=Compressor,
                                sublayers_spec=CompressorSublayersSpec(
                                    linear_wkv=_TestLinear,
                                    linear_wgate=_TestLinear,
                                    norm=_TestRMSNorm,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            o_proj=_TestLinear,
            q_layernorm=_TestRMSNorm,
            kv_layernorm=_TestRMSNorm,
        )

        pg_ref = types.SimpleNamespace(tp=None, cp=None)
        pg_cp = types.SimpleNamespace(tp=None, cp=CP_GROUP)

        paddle.seed(2026)
        layer_ref = DSv4HybridSelfAttention(
            config=config,
            sublayers_spec=sublayers,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_ref,
        )

        paddle.seed(2026)
        layer_cp = DSv4HybridSelfAttention(
            config=config,
            sublayers_spec=sublayers,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_cp,
        )

        paddle.seed(1000)
        hidden_full = paddle.randn([b, sq_global, hidden_size], dtype=DTYPE)

        h_a = hidden_full.clone()
        h_a.stop_gradient = False
        out_a, _ = layer_ref.forward(h_a)
        out_a.sum().backward()

        s, e = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
        h_b = hidden_full[:, s:e].clone()
        h_b.stop_gradient = False
        out_b, _ = layer_cp.forward(h_b)
        out_b.sum().backward()

        for p in layer_cp.parameters():
            if p.grad is not None:
                g = p.grad.contiguous()
                dist.all_reduce(g, group=CP_GROUP)
                paddle.assign(g, p.grad)

        fwd_cos = _cosine_sim(out_b, out_a[:, s:e])
        self.assertGreater(
            fwd_cos,
            COS_SIM_THRESHOLD,
            f"FP8 QAT full-layer forward cosine sim too low: {fwd_cos:.6f}",
        )

        dh_cos = _cosine_sim(h_b.grad, h_a.grad[:, s:e])
        self.assertGreater(
            dh_cos,
            COS_SIM_THRESHOLD,
            f"FP8 QAT full-layer dH cosine sim too low: {dh_cos:.6f}",
        )


# ---------------------------------------------------------------------------
# TileLang CP: kernel-level isolation tests
# ---------------------------------------------------------------------------
class TestTileLangIndexerKernelCP(unittest.TestCase):
    """TileLang indexer kernel with real CP group: fwd bit-exact, bwd sum-exact."""

    def setUp(self):
        paddle.enable_compat(scope={"tilelang"}, silent=True)

    def _run_fwd(self, sq_global, topk_effective, h_i=16, d_i=32, ratio=4):
        """Each rank runs kernel with seq_offset, verify == full-seq slice."""
        from paddleformers.fleet.tilelang_ops import csa_indexer_topk_fwd

        b = 2
        sq_local = sq_global // CP_SIZE
        sk = sq_global // ratio

        paddle.seed(8888)
        q = paddle.randn([b, sq_global, h_i, d_i], dtype=DTYPE)
        k = paddle.randn([b, sk, d_i], dtype=DTYPE)
        w = paddle.randn([b, sq_global, h_i], dtype="float32")

        # Full reference
        idx_full, scores_full = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            seq_offset=0,
        )

        # This rank's slice
        s = CP_RANK * sq_local
        idx_local, scores_local = csa_indexer_topk_fwd(
            q[:, s : s + sq_local, :, :],
            k,
            w[:, s : s + sq_local, :],
            ratio=ratio,
            topk_effective=topk_effective,
            seq_offset=s,
        )

        self.assertTrue(
            paddle.equal_all(
                idx_local, idx_full[:, s : s + sq_local, :]
            ).item(),
            f"Fwd indices mismatch: sq={sq_global} topk={topk_effective} "
            f"cp_rank={CP_RANK}",
        )
        score_diff = (
            (scores_local - scores_full[:, s : s + sq_local, :])
            .abs()
            .max()
            .item()
        )
        self.assertLess(
            score_diff,
            1e-6,
            f"Fwd scores diff={score_diff:.2e}: sq={sq_global} "
            f"topk={topk_effective} cp_rank={CP_RANK}",
        )

    def _run_bwd(self, sq_global, topk_effective, h_i=16, d_i=32, ratio=4):
        """Each rank runs bwd; all_reduce dK; verify matches full-seq."""
        from paddleformers.fleet.tilelang_ops import (
            csa_indexer_bwd,
            csa_indexer_topk_fwd,
        )

        b = 2
        sq_local = sq_global // CP_SIZE
        sk = sq_global // ratio

        paddle.seed(9999)
        q = paddle.randn([b, sq_global, h_i, d_i], dtype=DTYPE)
        k = paddle.randn([b, sk, d_i], dtype=DTYPE)
        w = paddle.randn([b, sq_global, h_i], dtype="float32")

        # Full reference
        idx_full, _ = csa_indexer_topk_fwd(
            q,
            k,
            w,
            ratio=ratio,
            topk_effective=topk_effective,
            seq_offset=0,
        )
        paddle.seed(7777)
        grad_full = (
            paddle.randn([b, sq_global, topk_effective], dtype="float32") * 0.01
        )
        dq_full, dw_full, dk_full = csa_indexer_bwd(
            q,
            w,
            k,
            idx_full,
            grad_full,
        )

        # This rank
        s = CP_RANK * sq_local
        idx_local, _ = csa_indexer_topk_fwd(
            q[:, s : s + sq_local, :, :],
            k,
            w[:, s : s + sq_local, :],
            ratio=ratio,
            topk_effective=topk_effective,
            seq_offset=s,
        )
        dq_local, dw_local, dk_local = csa_indexer_bwd(
            q[:, s : s + sq_local, :, :],
            w[:, s : s + sq_local, :],
            k,
            idx_local,
            grad_full[:, s : s + sq_local, :],
        )

        # dQ/dW slice-exact
        dq_diff = (
            (
                dq_local.cast("float32")
                - dq_full[:, s : s + sq_local, :, :].cast("float32")
            )
            .abs()
            .max()
            .item()
        )
        self.assertLess(
            dq_diff, 1e-4, f"dQ diff={dq_diff:.2e} cp_rank={CP_RANK}"
        )

        dw_diff = (
            (
                dw_local.cast("float32")
                - dw_full[:, s : s + sq_local, :].cast("float32")
            )
            .abs()
            .max()
            .item()
        )
        self.assertLess(
            dw_diff, 1e-4, f"dW diff={dw_diff:.2e} cp_rank={CP_RANK}"
        )

        # All-reduce dK and compare to full
        dk_sum = dk_local.cast("float32").contiguous()
        dist.all_reduce(dk_sum, group=CP_GROUP)
        dk_diff = (dk_sum - dk_full.cast("float32")).abs().max().item()
        self.assertLess(dk_diff, 1e-3, f"dK diff={dk_diff:.2e}")

    def test_fwd_sq64_topk8(self):
        self._run_fwd(sq_global=64, topk_effective=8)

    def test_fwd_sq128_topk16(self):
        self._run_fwd(sq_global=128, topk_effective=16)

    def test_fwd_sq256_topk16(self):
        self._run_fwd(sq_global=256, topk_effective=16)

    def test_fwd_sq256_topk64_phase2(self):
        self._run_fwd(sq_global=256, topk_effective=64)

    def test_bwd_sq64_topk8(self):
        self._run_bwd(sq_global=64, topk_effective=8)

    def test_bwd_sq128_topk16(self):
        self._run_bwd(sq_global=128, topk_effective=16)

    def test_bwd_sq256_topk16(self):
        self._run_bwd(sq_global=256, topk_effective=16)


# ---------------------------------------------------------------------------
# TileLang CP: Full-layer precision (output + all grads + all param grads)
# ---------------------------------------------------------------------------
class TestTileLangCSALayerCP(unittest.TestCase):
    """Full CSA layer: CP+TileLang vs non-CP+TileLang, compare everything.

    Tests the complete _forward_cp pipeline with TileLang enabled:
    - Branch B1a: TileLang loss path (training)
    - Branch B1d: TileLang fwd-only path (eval)
    - Branch D1: TileLangCSAIndexerLossAutoScaler backward
    - Compressor CP path: local project -> all-gather -> global pool -> slice
    """

    COS_THRESHOLD = 0.999

    def setUp(self):
        paddle.enable_compat(scope={"tilelang"}, silent=True)

    def _build_tilelang_config(
        self,
        dsa_index_topk=16,
        loss_coeff=0.0,
        use_sparse_loss=False,
    ):
        return types.SimpleNamespace(
            num_attention_heads=8,
            v_head_dim=64,
            hidden_size=256,
            q_lora_rank=64,
            qk_pos_emb_head_dim=32,
            csa_window_size=64,
            csa_compress_ratios=[4],
            csa_dense_mode=False,
            dsa_index_n_heads=16,
            dsa_index_head_dim=32,
            dsa_index_topk=dsa_index_topk,
            dsa_indexer_loss_coeff=loss_coeff,
            dsa_indexer_use_sparse_loss=use_sparse_loss,
            csa_tilelang_enable_indexer=True,
            csa_tilelang_enable_sparse_attn=False,
            csa_tilelang_backend="attention_paddle_compat",
            csa_indexer_backend="tilelang",
            init_method=None,
            init_method_std=0.02,
            layernorm_epsilon=1e-5,
            num_hidden_layers=1,
        )

    def _run(self, sq_global, dsa_index_topk, loss_coeff, use_sparse_loss):
        sq_local = sq_global // CP_SIZE
        b, head_dim, hidden_size, q_lora_rank = 2, 64, 256, 64

        config = self._build_tilelang_config(
            dsa_index_topk=dsa_index_topk,
            loss_coeff=loss_coeff,
            use_sparse_loss=use_sparse_loss,
        )

        # Reference: full sequence, no CP, TileLang enabled
        paddle.seed(2026)
        csa_ref = _build_csa(config, compress_ratio=4, head_dim=head_dim)
        csa_ref.cp_group = None
        csa_ref.cp_size = 1
        csa_ref.cp_rank = 0
        csa_ref.cp_enabled = False

        # Under test: local slice, CP enabled, TileLang enabled
        paddle.seed(2026)
        csa_cp = _build_csa(config, compress_ratio=4, head_dim=head_dim)
        csa_cp.cp_group = CP_GROUP
        csa_cp.cp_size = CP_SIZE
        csa_cp.cp_rank = CP_RANK
        csa_cp.cp_enabled = True

        if loss_coeff > 0:
            csa_ref.train()
            csa_cp.train()

        # Inputs
        paddle.seed(1000)
        query_full = paddle.randn([b, sq_global, 8, head_dim], dtype=DTYPE)
        key_full = paddle.randn([b, sq_global, 1, head_dim], dtype=DTYPE)
        x_full = paddle.randn([b, sq_global, hidden_size], dtype=DTYPE)
        qr_full = paddle.randn([b, sq_global, q_lora_rank], dtype=DTYPE)

        # Side A: full-sequence reference
        q_a = query_full.clone()
        q_a.stop_gradient = False
        k_a = key_full.clone()
        k_a.stop_gradient = False
        x_a = x_full.clone()
        x_a.stop_gradient = False
        qr_a = qr_full.clone()
        qr_a.stop_gradient = False
        out_a = csa_ref.forward(q_a, k_a, k_a, None, x=x_a, qr=qr_a)
        out_a.sum().backward()

        # Side B: CP local slice
        s, e = CP_RANK * sq_local, (CP_RANK + 1) * sq_local
        q_b = query_full[:, s:e].clone()
        q_b.stop_gradient = False
        k_b = key_full[:, s:e].clone()
        k_b.stop_gradient = False
        x_b = x_full[:, s:e].clone()
        x_b.stop_gradient = False
        qr_b = qr_full[:, s:e].clone()
        qr_b.stop_gradient = False
        out_b = csa_cp.forward(q_b, k_b, k_b, None, x=x_b, qr=qr_b)
        out_b.sum().backward()

        # All-reduce param grads
        for p in csa_cp.parameters():
            if p.grad is not None:
                g = p.grad.contiguous()
                dist.all_reduce(g, group=CP_GROUP)
                paddle.assign(g, p.grad)

        # Compensate indexer params (auto-scaler normalizes by 1/sq_local)
        if loss_coeff > 0 and csa_cp.indexer is not None:
            for p in csa_cp.indexer.parameters():
                if p.grad is not None:
                    paddle.assign(p.grad / CP_SIZE, p.grad)

        # --- Assertions ---
        # Forward output
        fwd_cos = _cosine_sim(out_b, out_a[:, s:e])
        self.assertGreater(
            fwd_cos,
            self.COS_THRESHOLD,
            f"[R{CP_RANK}] output cos={fwd_cos:.6f}",
        )

        # Input grads
        for name, g_cp, g_ref in [
            ("dQ", q_b.grad, q_a.grad[:, s:e]),
            ("dK", k_b.grad, k_a.grad[:, s:e]),
            ("dX", x_b.grad, x_a.grad[:, s:e]),
        ]:
            if g_cp is None or g_ref is None:
                continue
            cos = _cosine_sim(g_cp, g_ref)
            self.assertGreater(
                cos,
                self.COS_THRESHOLD,
                f"[R{CP_RANK}] {name} cos={cos:.6f}",
            )

        # All parameter grads
        ref_params = dict(csa_ref.named_parameters())
        cp_params = dict(csa_cp.named_parameters())
        for pname in sorted(ref_params.keys()):
            g_ref = ref_params[pname].grad
            g_cp = cp_params[pname].grad
            if g_ref is None or g_cp is None:
                continue
            # Skip all-zero grads (e.g. ape in eval mode): cosine is undefined
            if g_ref.abs().max().item() == 0 and g_cp.abs().max().item() == 0:
                continue
            cos = _cosine_sim(g_cp, g_ref)
            self.assertGreater(
                cos,
                self.COS_THRESHOLD,
                f"[R{CP_RANK}] d({pname}) cos={cos:.6f}",
            )

    # --- Fwd-only (eval, no loss; covers TileLang B1d path) ---
    def test_fwd_only_sq128(self):
        self._run(
            sq_global=128,
            dsa_index_topk=16,
            loss_coeff=0.0,
            use_sparse_loss=False,
        )

    def test_fwd_only_sq256(self):
        self._run(
            sq_global=256,
            dsa_index_topk=16,
            loss_coeff=0.0,
            use_sparse_loss=False,
        )

    def test_fwd_only_sq512(self):
        self._run(
            sq_global=512,
            dsa_index_topk=32,
            loss_coeff=0.0,
            use_sparse_loss=True,
        )

    # --- Phase 2 loss (topk_eff = n_compressed; covers B1a + D1 backward) ---
    def test_phase2_sq128(self):
        self._run(
            sq_global=128,
            dsa_index_topk=16,
            loss_coeff=1.0,
            use_sparse_loss=False,
        )

    def test_phase2_sq256(self):
        self._run(
            sq_global=256,
            dsa_index_topk=16,
            loss_coeff=1.0,
            use_sparse_loss=False,
        )

    def test_phase2_sq256_large_topk(self):
        self._run(
            sq_global=256,
            dsa_index_topk=64,
            loss_coeff=1.0,
            use_sparse_loss=False,
        )

    # --- Phase 3 loss (topk_eff = min(topk, n_comp); sparse selection) ---
    def test_phase3_sq128(self):
        self._run(
            sq_global=128,
            dsa_index_topk=8,
            loss_coeff=1.0,
            use_sparse_loss=True,
        )

    def test_phase3_sq256(self):
        self._run(
            sq_global=256,
            dsa_index_topk=16,
            loss_coeff=1.0,
            use_sparse_loss=True,
        )

    def test_phase3_sq512(self):
        self._run(
            sq_global=512,
            dsa_index_topk=32,
            loss_coeff=1.0,
            use_sparse_loss=True,
        )

    # --- Edge: topk >= n_compressed in Phase 3 (collapses to Phase 2) ---
    def test_phase3_topk_ge_ncomp(self):
        self._run(
            sq_global=128,
            dsa_index_topk=128,
            loss_coeff=1.0,
            use_sparse_loss=True,
        )

    # --- Small loss coeff (gradient scaling) ---
    def test_small_loss_coeff(self):
        self._run(
            sq_global=256,
            dsa_index_topk=16,
            loss_coeff=0.1,
            use_sparse_loss=True,
        )

    # --- Large sequence ---
    def test_long_seq_sq1024(self):
        self._run(
            sq_global=1024,
            dsa_index_topk=32,
            loss_coeff=1.0,
            use_sparse_loss=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
