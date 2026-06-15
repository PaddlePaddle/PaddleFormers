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

"""Distributed tests for qk_norm_type (tensor parallel coverage).

This test verifies that SelfAttention with qk_norm_type='per_layer' and tensor
parallelism can run correctly, covering the TP gather/scatter branches.

The per_layer qk_norm_type normalizes Q and K across all heads jointly, which
requires TP gather/scatter operations when TP > 1.

Launch:
    python -m paddle.distributed.launch --gpus="0,1,2,3" \n        tensor_parallel/test_qk_norm_type.py
"""

from __future__ import annotations

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import DotProductAttention
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Test dimensions
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 64
SEED = 123


def _set_random_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _build_config(
    tp_size: int = 1,
    sp: bool = False,
    qk_norm_type: str = "per_head",
    use_qk_norm: bool = True,
) -> TransformerConfig:
    """Build a TransformerConfig for qk_norm_type testing."""
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        use_bias=False,
        attention_bias=False,
        attention_dropout=0.0,
        softmax_type="vanilla",
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sp,
        deterministic_mode=True,
        gated_attention=False,
        use_qk_norm=use_qk_norm,
        qk_norm_type=qk_norm_type,
    )


def _build_attn(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection | None = None,
) -> SelfAttention:
    """Build a SelfAttention module."""
    sublayers_spec = SelfAttentionSublayersSpec(
        qkv_proj=ColumnParallelLinear,
        core_attention=DotProductAttention,
        o_proj=RowParallelLinear,
        q_norm=WrappedPaddleNorm,
        k_norm=WrappedPaddleNorm,
    )

    return SelfAttention(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=pg_collection,
    )


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

TENSOR_PARALLEL = 4


class TestQKNormTypeDistributed(unittest.TestCase):
    """Distributed tests for qk_norm_type with tensor parallelism."""

    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment once for all tests."""
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": TENSOR_PARALLEL,
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
        initialize_fleet(strategy)
        print(f"Rank {dist.get_rank()} / {dist.get_world_size()} initialized.")

    def setUp(self):
        self.tp_size = TENSOR_PARALLEL
        self.seed = SEED

    def _check_gpu_forward(self, sp: bool, qk_norm_type: str, use_qk_norm: bool = True):
        """Test that forward produces correct shape and dtype."""
        _set_random_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)

        config = _build_config(
            tp_size=self.tp_size,
            sp=sp,
            qk_norm_type=qk_norm_type,
            use_qk_norm=use_qk_norm,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        attn = _build_attn(config, pg_collection=pg_collection)

        sp_size = self.tp_size if sp else 1

        if sp:
            hidden_states = paddle.randn([SEQ_LENGTH // sp_size, MICRO_BATCH_SIZE, HIDDEN_SIZE])
        else:
            hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        output, bias = attn(hidden_states, attention_mask=None)

        self.assertEqual(output.ndim, 3, f"Output should be 3D, got {output.ndim}D")

        if sp:
            self.assertEqual(output.shape[0], SEQ_LENGTH // sp_size)
            self.assertEqual(output.shape[1], MICRO_BATCH_SIZE)
            self.assertEqual(output.shape[2], HIDDEN_SIZE)
        else:
            self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
            self.assertEqual(output.shape[1], SEQ_LENGTH)
            self.assertEqual(output.shape[2], HIDDEN_SIZE)

        self.assertEqual(output.dtype, hidden_states.dtype)
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "Output contains NaN or Inf",
        )

    def test_per_layer_forward_no_sp(self):
        """Test per_layer qk_norm_type forward without SP."""
        self._check_gpu_forward(sp=False, qk_norm_type="per_layer", use_qk_norm=True)

    def test_per_layer_forward_with_sp(self):
        """Test per_layer qk_norm_type forward with SP."""
        self._check_gpu_forward(sp=True, qk_norm_type="per_layer", use_qk_norm=True)

    def test_no_qk_norm_forward_no_sp(self):
        """Test forward without QK normalization and without SP."""
        self._check_gpu_forward(sp=False, qk_norm_type="per_head", use_qk_norm=False)

    def test_no_qk_norm_forward_with_sp(self):
        """Test forward without QK normalization and with SP."""
        self._check_gpu_forward(sp=True, qk_norm_type="per_head", use_qk_norm=False)


if __name__ == "__main__":
    unittest.main()
