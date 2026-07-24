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

"""Multi-card unit tests for DSA Indexer sequence_parallel path.

Tests verify that DSAIndexer.forward_before_topk correctly gathers
sequence-parallel sharded inputs and produces identical results to
the non-SP (batch-first full-sequence) path.

Run with:
    python -m paddle.distributed.launch --gpus 0,1 \
        tests/fleet/multi_card_tests/transformer/test_dsa_attention_sp.py
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.dsa_attention import (
    DSAIndexer,
    DSAIndexerSublayersSpec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig
from paddleformers.fleet.utils import init_method_normal

# ---------------------------------------------------------------------------
# Module-level initialization
# ---------------------------------------------------------------------------
TP_SIZE = None


def setUpModule():
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": TP_SIZE,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
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
    initialize_fleet(strategy=strategy)


# ---------------------------------------------------------------------------
# Stub layers (BiasedLinear, LayerNormStub) - same as single-card tests
# ---------------------------------------------------------------------------
class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        if x.dtype != self.linear.weight.dtype:
            x = x.cast(self.linear.weight.dtype)
        return self.linear(x), self.linear.bias


class LayerNormStub(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size=None,
        eps=None,
        normalized_shape=None,
        epsilon=None,
        **kwargs,
    ):
        super().__init__()
        size = hidden_size if hidden_size is not None else normalized_shape
        self.eps = (
            eps
            if eps is not None
            else (epsilon if epsilon is not None else 1e-5)
        )
        self.weight = paddle.nn.Parameter(paddle.ones([size]))
        self.bias = paddle.nn.Parameter(paddle.zeros([size]))

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = x.var(axis=-1, keepdim=True, unbiased=False)
        x = (x - mean) / paddle.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
def _create_dsa_config(sequence_parallel=True):
    config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=256,
        num_attention_heads=2,
    )
    config.num_key_value_heads = 2
    config.head_dim = 128
    config.q_lora_rank = 64
    config.kv_lora_rank = 64
    config.qk_nope_head_dim = 32
    config.qk_rope_head_dim = 32
    config.v_head_dim = 64
    config.multi_latent_attention = True

    # RoPE (use yarn for consistency with single-card tests)
    config.rope_type = "yarn"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.rotary_scaling_factor = 40.0
    config.original_max_position_embeddings = 4096
    config.beta_fast = 32.0
    config.beta_slow = 1.0
    config.mscale = 1.0
    config.mscale_all_dim = 0.0
    config.apply_rope_fusion = False

    # DSA Indexer fields
    config.dsa_index_n_heads = 2
    config.dsa_index_head_dim = 128
    config.dsa_index_topk = 8
    config.dsa_indexer_rotary_interleaved = False

    # Parallel
    config.sequence_parallel = sequence_parallel
    config.tensor_model_parallel_size = TP_SIZE

    # Other
    config.init_method = init_method_normal(0.02)
    config.rms_norm_eps = 1e-5

    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDSAIndexerSequenceParallel(unittest.TestCase):
    """Test DSAIndexer.forward_before_topk with real sequence_parallel gather."""

    @classmethod
    def setUpClass(cls):
        model_parallel_cuda_manual_seed(42)
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def _build_indexer(self, config):
        sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        indexer = DSAIndexer(
            config=config,
            sublayers_spec=sublayers,
            layer_number=1,
            pg_collection=self.pg_collection,
        )
        # Convert to bf16 for rotate_activation
        indexer.wq_b = indexer.wq_b.to(dtype="bfloat16")
        indexer.wk = indexer.wk.to(dtype="bfloat16")
        indexer.k_norm = indexer.k_norm.to(dtype="bfloat16")
        return indexer

    def test_sp_output_matches_non_sp(self):
        """Sequence-parallel forward should produce same output as non-SP forward.

        Strategy:
        1. Create full-sequence inputs [b, s, h] (same on all ranks via broadcast)
        2. Run non-SP forward (sequence_parallel=False) to get reference output
        3. Shard inputs to seq-first [s/TP, b, h] for this rank
        4. Run SP forward and verify outputs match reference
        """
        b, s = 2, 16
        config_sp = _create_dsa_config(sequence_parallel=True)
        config_nosp = _create_dsa_config(sequence_parallel=False)

        # Build two indexers with identical weights
        indexer_sp = self._build_indexer(config_sp)
        indexer_nosp = self._build_indexer(config_nosp)

        # Sync weights: copy sp -> nosp (they share the same init seed)
        indexer_nosp.load_dict(indexer_sp.state_dict())

        # Create full-sequence inputs (broadcast from rank 0)
        if dist.get_rank() == 0:
            full_hidden = paddle.randn([b, s, config_sp.hidden_size]).cast(
                "bfloat16"
            )
            full_q_latent = paddle.randn([b, s, config_sp.q_lora_rank]).cast(
                "bfloat16"
            )
        else:
            full_hidden = paddle.zeros([b, s, config_sp.hidden_size]).cast(
                "bfloat16"
            )
            full_q_latent = paddle.zeros([b, s, config_sp.q_lora_rank]).cast(
                "bfloat16"
            )

        dist.broadcast(full_hidden, src=0)
        dist.broadcast(full_q_latent, src=0)

        # --- Non-SP reference (batch-first full-sequence) ---
        q_ref, k_ref, w_ref = indexer_nosp.forward_before_topk(
            full_hidden, full_q_latent
        )

        # --- SP path (seq-first sharded) ---
        # Shard along sequence dim: [b, s, h] -> [s, b, h] -> shard -> [s/TP, b, h]
        tp_rank = dist.get_rank()
        shard_size = s // TP_SIZE
        # Transpose to seq-first then shard
        hidden_sf = full_hidden.transpose([1, 0, 2])  # [s, b, h]
        q_latent_sf = full_q_latent.transpose([1, 0, 2])  # [s, b, h]
        hidden_shard = hidden_sf[
            tp_rank * shard_size : (tp_rank + 1) * shard_size
        ].clone()
        q_latent_shard = q_latent_sf[
            tp_rank * shard_size : (tp_rank + 1) * shard_size
        ].clone()

        q_sp, k_sp, w_sp = indexer_sp.forward_before_topk(
            hidden_shard, q_latent_shard
        )

        # Compare outputs (should be identical since gather reconstructs full sequence)
        np.testing.assert_allclose(
            q_sp.cast("float32").numpy(),
            q_ref.cast("float32").numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"[Rank {tp_rank}] SP query output differs from non-SP reference",
        )
        np.testing.assert_allclose(
            k_sp.cast("float32").numpy(),
            k_ref.cast("float32").numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"[Rank {tp_rank}] SP key output differs from non-SP reference",
        )
        np.testing.assert_allclose(
            w_sp.cast("float32").numpy(),
            w_ref.cast("float32").numpy(),
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"[Rank {tp_rank}] SP weights output differs from non-SP reference",
        )

    def test_sp_output_shapes(self):
        """SP forward should produce batch-first outputs with full sequence length."""
        b, s = 2, 16
        config_sp = _create_dsa_config(sequence_parallel=True)
        indexer = self._build_indexer(config_sp)

        tp_rank = dist.get_rank()
        shard_size = s // TP_SIZE

        # seq-first sharded input [s/TP, b, h]
        hidden_shard = paddle.randn(
            [shard_size, b, config_sp.hidden_size]
        ).cast("bfloat16")
        q_latent_shard = paddle.randn(
            [shard_size, b, config_sp.q_lora_rank]
        ).cast("bfloat16")

        q, k, weights = indexer.forward_before_topk(
            hidden_shard, q_latent_shard
        )

        # Output should be batch-first with FULL sequence length
        self.assertEqual(
            list(q.shape),
            [b, s, config_sp.dsa_index_n_heads, config_sp.dsa_index_head_dim],
        )
        self.assertEqual(list(k.shape), [b, s, config_sp.dsa_index_head_dim])
        self.assertEqual(
            list(weights.shape), [b, s, config_sp.dsa_index_n_heads]
        )


class TestDSAttentionPgCollectionNone(unittest.TestCase):
    """Test DSAttention.__init__ with pg_collection=None uses real mpu process groups."""

    def test_pg_collection_none_uses_mpu(self):
        """When pg_collection is None, DSAttention should get pg from use_mpu_process_groups()."""
        from paddleformers.fleet.transformer.dsa_attention import (
            DSAttention,
            DSAttentionSublayersSpec,
        )
        from paddleformers.fleet.transformer.enums import AttnMaskType

        config = _create_dsa_config(sequence_parallel=False)
        config.dsa_indexer_loss_coeff = None
        sublayers = DSAIndexerSublayersSpec(
            linear_wq_b=BiasedLinear,
            linear_wk=BiasedLinear,
            k_norm=LayerNormStub,
            linear_weights_proj=BiasedLinear,
        )
        dsa_sublayers_spec = DSAttentionSublayersSpec(
            indexer=LayerSpec(
                layer=DSAIndexer,
                sublayers_spec=sublayers,
            ),
        )
        qk_hd = config.qk_nope_head_dim + config.qk_rope_head_dim
        model = DSAttention(
            config=config,
            sublayers_spec=dsa_sublayers_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=qk_hd**-0.5,
            pg_collection=None,  # Trigger the fallback
        )
        # Should have gotten a real ProcessGroupCollection
        self.assertIsNotNone(model.pg_collection)
        self.assertIsInstance(model.pg_collection, ProcessGroupCollection)
        # TP group should have nranks == TP_SIZE
        self.assertEqual(model.pg_collection.tp.nranks, TP_SIZE)


if __name__ == "__main__":
    unittest.main()
