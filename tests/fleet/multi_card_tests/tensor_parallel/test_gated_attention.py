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

"""Distributed precision tests for Gated Attention (tensor parallel correctness).

This test verifies that SelfAttention with gated_attention=True and tensor
parallelism produces the same outputs and gradients as the single-device (TP=1)
baseline, following the pattern of test_gated_delta_net.py.

Launch:
    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        tensor_parallel/test_gated_attention.py
"""

from __future__ import annotations

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddleformers.fleet.parallel_state as ps
from paddleformers.fleet.process_groups_config import ProcessGroupCollection
from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddleformers.fleet.tensor_parallel.mappings import (
    _gather_along_first_dim,
    _gather_along_last_dim,
)
from paddleformers.fleet.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddleformers.fleet.transformer.dot_product_attention import (
    DotProductAttention,
)
from paddleformers.fleet.transformer.enums import AttnMaskType
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Test dimensions
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = (
    4  # MHA (can change to GQA by setting < NUM_ATTENTION_HEADS)
)
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 64
SEED = 123
INPUT_SEED = 42

# Derived: QKV output dimension (before TP split)
Q_DIM = NUM_ATTENTION_HEADS * HEAD_DIM
KV_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM
# With gated attention: qkv_proj output = Q + Gate + K + V
GATE_DIM = Q_DIM
QKV_GATE_TOTAL = Q_DIM + GATE_DIM + 2 * KV_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_random_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _build_config(
    tp_size: int = 1, sp: bool = False, gated: bool = True
) -> TransformerConfig:
    """Build a TransformerConfig suitable for Gated Attention testing."""
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
        gated_attention=gated,
        use_qk_norm=True,
    )


def _build_attn(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection | None = None,
    tp_group=None,
) -> SelfAttention:
    """Build a SelfAttention module with gated attention."""
    sublayers_spec = SelfAttentionSublayersSpec(
        qkv_proj=ColumnParallelLinear,
        core_attention=DotProductAttention,
        o_proj=RowParallelLinear,
        q_norm=WrappedPaddleNorm,
        k_norm=WrappedPaddleNorm,
    )

    kwargs = {}
    if pg_collection is not None:
        kwargs["pg_collection"] = pg_collection
    elif tp_group is not None:
        kwargs["pg_collection"] = ProcessGroupCollection(tp=tp_group, cp=None)

    return SelfAttention(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        **kwargs,
    )


def _gather_attn_params(
    attn: SelfAttention, tp_group, tp_size
) -> dict[str, paddle.Tensor]:
    """Gather all TP-sharded parameter gradients to full shape for comparison."""
    gathered = {}
    for name, param in attn.named_parameters():
        if param.grad is None:
            gathered[name] = None
            continue

        grad = param.grad

        # ColumnParallel (qkv_proj): sharded along last dim (output dim)
        if "qkv_proj" in name and "weight" in name:
            grad = _gather_along_last_dim(grad, tp_group)
        elif "qkv_proj" in name and "bias" in name:
            grad = _gather_along_last_dim(grad.unsqueeze(0), tp_group).squeeze(
                0
            )
        # RowParallel (o_proj): weight sharded along first dim (input dim)
        elif "o_proj" in name and "weight" in name:
            grad = _gather_along_first_dim(grad, tp_group)
        # o_proj bias: NOT sharded (full on each rank), keep as-is
        # q_norm, k_norm: NOT sharded, keep as-is

        gathered[name] = grad

    return gathered


def _gather_output(
    output: paddle.Tensor,
    config: TransformerConfig,
    tp_group,
) -> paddle.Tensor:
    """Gather output tensor from TP ranks."""
    if config.sequence_parallel:
        return _gather_along_first_dim(output, tp_group)
    return output


# ---------------------------------------------------------------------------
# Baseline: single-device (TP=1) forward + backward
# ---------------------------------------------------------------------------


def _run_baseline(seed: int):
    """Run gated SelfAttention on a single device (TP=1)."""
    _set_random_seed(seed)

    config = _build_config(tp_size=1, sp=False, gated=True)
    tp1_group = dist.new_group([dist.get_rank()])
    attn = _build_attn(config, tp_group=tp1_group)

    _set_random_seed(INPUT_SEED)
    hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
    hidden_states.stop_gradient = False

    output, output_bias = attn(hidden_states, attention_mask=None)
    output.sum().backward()

    return (
        output.detach(),
        hidden_states.grad.detach(),
        attn,
    )


# ---------------------------------------------------------------------------
# Distributed: TP forward + backward
# ---------------------------------------------------------------------------


def _run_distributed(
    seed: int,
    tp_size: int,
    sp: bool,
    output_baseline: paddle.Tensor,
    input_grad_baseline: paddle.Tensor,
    attn_baseline: SelfAttention,
):
    """Run gated SelfAttention with TP and compare against baseline."""
    _set_random_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    config = _build_config(tp_size=tp_size, sp=sp, gated=True)

    tp_group = ps.get_tensor_model_parallel_group()
    tp_rank = ps.get_tensor_model_parallel_rank()
    sp_size = tp_size if sp else 1

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp"]
    )

    attn_dist = _build_attn(config, pg_collection=pg_collection)
    register_sequence_parallel_allreduce_hooks(attn_dist, 1, False)

    # --- Load baseline weights into distributed model ---
    baseline_sd = {}
    for name, param in attn_baseline.named_parameters():
        baseline_sd[name] = param.detach()

    with paddle.no_grad():
        for name, param in attn_dist.named_parameters():
            full_param = baseline_sd[name]

            # ColumnParallel (qkv_proj): shard along last dim
            if "qkv_proj" in name and "weight" in name:
                chunk_size = full_param.shape[-1] // tp_size
                param.set_value(
                    full_param[
                        :, tp_rank * chunk_size : (tp_rank + 1) * chunk_size
                    ]
                )
            elif "qkv_proj" in name and "bias" in name:
                chunk_size = full_param.shape[0] // tp_size
                param.set_value(
                    full_param[
                        tp_rank * chunk_size : (tp_rank + 1) * chunk_size
                    ]
                )
            # RowParallel (o_proj): weight sharded along first dim
            elif "o_proj" in name and "weight" in name:
                chunk_size = full_param.shape[0] // tp_size
                param.set_value(
                    full_param[
                        tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :
                    ]
                )
            # o_proj bias, q_norm, k_norm: NOT sharded, copy as-is
            else:
                param.set_value(full_param.clone())

    # --- Generate input ---
    _set_random_seed(INPUT_SEED)
    hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

    if sp:
        hidden_states = hidden_states.transpose([1, 0, 2])  # [b,s,h] -> [s,b,h]
        sp_seg = SEQ_LENGTH // sp_size
        hidden_states = hidden_states[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]

    hidden_states = hidden_states.contiguous()
    hidden_states.stop_gradient = False

    output_dist, _ = attn_dist(hidden_states, attention_mask=None)
    output_dist.sum().backward()

    # --- Gather and compare output ---
    output_gathered = _gather_output(output_dist, config, tp_group)
    if sp:
        output_gathered = output_gathered.transpose([1, 0, 2])

    assert paddle.all(~paddle.isnan(output_baseline)).item(), (
        "Baseline output has NaN"
    )
    assert paddle.all(~paddle.isnan(output_gathered)).item(), (
        "Distributed output has NaN"
    )
    assert paddle.all(~paddle.isinf(output_baseline)).item(), (
        "Baseline output has Inf"
    )
    assert paddle.all(~paddle.isinf(output_gathered)).item(), (
        "Distributed output has Inf"
    )

    atol, rtol = 5e-4, 5e-4
    assert paddle.allclose(
        output_gathered, output_baseline, atol=atol, rtol=rtol
    ).item(), (
        f"Output mismatch: max_diff="
        f"{(output_gathered - output_baseline).abs().max().item():.6e}"
    )

    # --- Gather and compare input gradient ---
    if sp:
        input_grad_gathered = _gather_along_first_dim(
            hidden_states.grad, tp_group
        )
        input_grad_gathered = input_grad_gathered.transpose([1, 0, 2])
    else:
        input_grad_gathered = hidden_states.grad

    assert paddle.all(~paddle.isnan(input_grad_baseline)).item(), (
        "Baseline input grad has NaN"
    )
    assert paddle.all(~paddle.isnan(input_grad_gathered)).item(), (
        "Distributed input grad has NaN"
    )

    assert paddle.allclose(
        input_grad_gathered, input_grad_baseline, atol=atol, rtol=rtol
    ).item(), (
        f"Input grad mismatch: max_diff="
        f"{(input_grad_gathered - input_grad_baseline).abs().max().item():.6e}"
    )

    # --- Gather and compare parameter gradients ---
    baseline_grads = {}
    for name, param in attn_baseline.named_parameters():
        baseline_grads[name] = (
            param.grad.detach() if param.grad is not None else None
        )

    dist_grads = _gather_attn_params(
        attn_dist, tp_group=tp_group, tp_size=tp_size
    )

    for name in baseline_grads:
        if baseline_grads[name] is None or dist_grads.get(name) is None:
            continue
        b_grad = baseline_grads[name]
        d_grad = dist_grads[name]
        if list(b_grad.shape) != list(d_grad.shape):
            continue
        assert paddle.allclose(d_grad, b_grad, atol=atol, rtol=rtol).item(), (
            f"Grad mismatch for {name}: max_diff="
            f"{(d_grad - b_grad).abs().max().item():.6e}"
        )

    print(f"  [PASS] Gated Attention TP={tp_size}, SP={sp}")


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

TENSOR_PARALLEL = 4


class TestGatedAttentionDistributed(unittest.TestCase):
    """Distributed precision tests for Gated Attention with tensor parallelism."""

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

    # --- Forward shape tests ---

    def _check_gpu_forward(self, sp: bool):
        """Test that forward produces correct shape and dtype."""
        _set_random_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)

        config = _build_config(tp_size=self.tp_size, sp=sp, gated=True)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp"]
        )
        attn = _build_attn(config, pg_collection=pg_collection)

        sp_size = self.tp_size if sp else 1

        if sp:
            hidden_states = paddle.randn(
                [SEQ_LENGTH // sp_size, MICRO_BATCH_SIZE, HIDDEN_SIZE]
            )
        else:
            hidden_states = paddle.randn(
                [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
            )

        output, bias = attn(hidden_states, attention_mask=None)

        self.assertEqual(
            output.ndim, 3, f"Output should be 3D, got {output.ndim}D"
        )

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

    def _test_gpu_forward_no_sp(self):
        """Forward shape and dtype without sequence parallelism."""
        self._check_gpu_forward(sp=False)

    def _test_gpu_forward_with_sp(self):
        """Forward shape and dtype with sequence parallelism."""
        self._check_gpu_forward(sp=True)

    # --- Precision correctness tests ---

    def _check_tp_precision(self, sp: bool):
        """Compare TP output/grads against single-device baseline."""
        output_baseline, input_grad_baseline, attn_baseline = _run_baseline(
            self.seed
        )
        _run_distributed(
            self.seed,
            self.tp_size,
            sp=sp,
            output_baseline=output_baseline,
            input_grad_baseline=input_grad_baseline,
            attn_baseline=attn_baseline,
        )

    def _test_tp_no_sp(self):
        """TP correctness without sequence parallelism."""
        self._check_tp_precision(sp=False)

    def _test_tp_with_sp(self):
        """TP correctness with sequence parallelism."""
        self._check_tp_precision(sp=True)

    def test_all_cases(self):
        self._test_gpu_forward_no_sp()
        self._test_gpu_forward_with_sp()
        self._test_tp_no_sp()
        self._test_tp_with_sp()


if __name__ == "__main__":
    unittest.main()
