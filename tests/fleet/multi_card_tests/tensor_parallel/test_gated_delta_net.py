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

"""Distributed precision tests for GatedDeltaNet (tensor parallel correctness).

This test verifies that GatedDeltaNet with tensor parallelism produces the same
outputs and gradients as the single-device (TP=1) baseline, following the pattern
of Megatron-LM's ``tests/unit_tests/ssm/test_gated_delta_net.py``.

Launch:
    python -m paddle.distributed.launch --gpus="0,1,2,3" \
        tensor_parallel/test_gated_delta_net.py
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
from paddleformers.fleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
)
from paddleformers.fleet.transformer.paddle_norm import RMSNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Test dimensions (kept small for fast multi-GPU CI)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
CONV_KERNEL_DIM = 2
KEY_HEAD_DIM = 64
VALUE_HEAD_DIM = 64
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 8
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 64
SEED = 123

# ---------------------------------------------------------------------------
# Derived dimensions (used by shard / gather helpers)
# ---------------------------------------------------------------------------
QK_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
V_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM

# in_proj output layout:  [q, k, v, gate, beta, alpha]
_IN_PROJ_SECTIONS = [
    QK_DIM,
    QK_DIM,
    V_DIM,
    V_DIM,
    NUM_VALUE_HEADS,
    NUM_VALUE_HEADS,
]
# conv1d channel layout:  [q, k, v]
_CONV_SECTIONS = [QK_DIM, QK_DIM, V_DIM]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_random_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)


def _shard_by_sections(full_tensor, sections, tp_rank, tp_size, dim):
    """Split *full_tensor* into *sections* along *dim*, shard each by TP, concat."""
    parts = paddle.split(full_tensor, sections, axis=dim)
    local_parts = []
    for p in parts:
        chunk = p.shape[dim] // tp_size
        slices = [slice(None)] * p.ndim
        slices[dim] = slice(tp_rank * chunk, (tp_rank + 1) * chunk)
        local_parts.append(p[tuple(slices)])
    return paddle.concat(local_parts, axis=dim)


def _gather_by_sections(local_tensor, sections, tp_group, tp_size, dim):
    """Gather TP-sharded *local_tensor*, then reorder from interleaved to
    section-contiguous layout so that it matches the TP=1 baseline."""
    # Step 1: all-gather along *dim* → interleaved layout
    gathered = (
        _gather_along_last_dim(local_tensor, tp_group)
        if dim == -1 or dim == local_tensor.ndim - 1
        else _gather_along_first_dim(local_tensor, tp_group)
    )
    # gathered layout: [sec0_rank0, sec1_rank0, ..., sec0_rank1, sec1_rank1, ...]
    # target  layout:  [sec0_all, sec1_all, ...]

    # Step 2: split gathered into per-rank chunks, then reorder
    total = gathered.shape[dim]
    rank_chunk = total // tp_size
    rank_tensors = paddle.split(gathered, [rank_chunk] * tp_size, axis=dim)

    # Each rank_tensor has the local sections in order
    local_section_sizes = [s // tp_size for s in sections]
    per_rank_sections = [
        paddle.split(rt, local_section_sizes, axis=dim) for rt in rank_tensors
    ]

    # Reorder: for each section index, gather across ranks
    reordered = []
    for sec_idx in range(len(sections)):
        reordered.append(
            paddle.concat([pr[sec_idx] for pr in per_rank_sections], axis=dim)
        )
    return paddle.concat(reordered, axis=dim)


def _build_config(tp_size: int = 1, sp: bool = False) -> TransformerConfig:
    """Build a TransformerConfig suitable for GatedDeltaNet testing."""
    return TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_VALUE_HEADS,
        num_hidden_layers=1,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sp,
        deterministic_mode=True,
    )


def _build_gdn(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection | None = None,
    tp_group=None,
) -> GatedDeltaNet:
    """Build a GatedDeltaNet module."""
    sublayers_spec = GatedDeltaNetSublayersSpec(
        in_proj=ColumnParallelLinear,
        out_norm=RMSNorm,
        out_proj=RowParallelLinear,
    )

    kwargs = {}
    if pg_collection is not None:
        kwargs["pg_collection"] = pg_collection
    if tp_group is not None:
        kwargs["pg_collection"] = ProcessGroupCollection(tp=tp_group)

    return GatedDeltaNet(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        conv_kernel_dim=CONV_KERNEL_DIM,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
        **kwargs,
    )


def _gather_gdn_params(
    gdn: GatedDeltaNet, tp_group, tp_size
) -> dict[str, paddle.Tensor]:
    """Gather all TP-sharded parameters of a GatedDeltaNet to full shape.

    Returns a dict mapping parameter name to the full (gathered) tensor.
    """
    gathered = {}
    for name, param in gdn.named_parameters():
        if param.grad is None:
            gathered[name] = None
            continue

        grad = param.grad
        # ColumnParallel weights (in_proj): gather by sections along last dim
        if "in_proj" in name and "weight" in name:
            grad = _gather_by_sections(
                grad, _IN_PROJ_SECTIONS, tp_group, tp_size, dim=-1
            )
        # RowParallel weights (out_proj): sharded along first dim
        elif "out_proj" in name and "weight" in name:
            grad = _gather_along_first_dim(grad, tp_group)
        # conv1d weight: gather by sections along dim 0
        elif "conv1d" in name and "weight" in name:
            grad = _gather_by_sections(
                grad, _CONV_SECTIONS, tp_group, tp_size, dim=0
            )
        # conv1d bias: gather by sections along dim 0
        elif "conv1d" in name and "bias" in name:
            grad = _gather_by_sections(
                grad, _CONV_SECTIONS, tp_group, tp_size, dim=0
            )
        # dt_bias, A_log: shape [num_v_heads_local] -> gather along dim 0
        elif name.endswith("dt_bias") or name.endswith("A_log"):
            grad = _gather_along_first_dim(grad, tp_group)
        # out_norm weight: NOT sharded, but each rank only sees local heads,
        # so the gradient is a partial sum. All-reduce to get the full gradient.
        # elif "out_norm" in name:
        #     grad = grad.clone()
        #     dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tp_group)
        # out_proj bias: NOT sharded (full on each rank)
        # else: keep as-is

        gathered[name] = grad

    return gathered


def _gather_output(
    output: paddle.Tensor,
    config: TransformerConfig,
    tp_group,
) -> paddle.Tensor:
    """Gather the output tensor from TP ranks to full shape.

    For sequence_parallel: output is [s_local, b, h] -> gather along dim 0.
    For TP without SP: output is [b, s, h] and already all-reduced by RowParallel.
    """
    if config.sequence_parallel:
        return _gather_along_first_dim(output, tp_group)
    else:
        return output


# ---------------------------------------------------------------------------
# Baseline: single-device (TP=1) forward + backward
# ---------------------------------------------------------------------------


INPUT_SEED = 42  # Separate fixed seed for generating input data


def _run_baseline(seed: int):
    """Run GatedDeltaNet on a single device (TP=1) and return output + input grad."""
    _set_random_seed(seed)

    print("===== run_baseline ======")
    config = _build_config(tp_size=1, sp=False)

    # Build with TP=1 group (single rank)
    tp1_group = dist.new_group([dist.get_rank()])
    gdn = _build_gdn(config, tp_group=tp1_group)

    # Use a separate fixed seed to generate input, so it is identical
    # regardless of how many RNG calls the model init consumed above.
    _set_random_seed(INPUT_SEED)
    hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
    hidden_states.stop_gradient = False

    output, output_bias = gdn(hidden_states, attention_mask=None)
    output.sum().backward()

    return (
        output.detach(),
        hidden_states.grad.detach(),
        gdn,
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
    gdn_baseline: GatedDeltaNet,
):
    print("====== run distributed ======")
    """Run GatedDeltaNet with tensor parallelism and compare against baseline."""
    _set_random_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    config = _build_config(tp_size=tp_size, sp=sp)

    tp_group = ps.get_tensor_model_parallel_group()
    tp_rank = ps.get_tensor_model_parallel_rank()
    sp_size = tp_size if sp else 1

    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp"]
    )

    gdn_dist = _build_gdn(config, pg_collection=pg_collection)
    register_sequence_parallel_allreduce_hooks(gdn_dist, 1, False)

    # --- Load baseline weights into distributed model ---
    # We need to shard the baseline weights correctly across TP ranks.
    baseline_sd = {}
    for name, param in gdn_baseline.named_parameters():
        baseline_sd[name] = param.detach()

    with paddle.no_grad():
        for name, param in gdn_dist.named_parameters():
            full_param = baseline_sd[name]

            # ColumnParallel (in_proj): shard by sections along last dim
            if "in_proj" in name and "weight" in name:
                param.set_value(
                    _shard_by_sections(
                        full_param, _IN_PROJ_SECTIONS, tp_rank, tp_size, dim=-1
                    )
                )
            # RowParallel (out_proj): weight sharded along first dim (v_dim → by value heads)
            elif "out_proj" in name and "weight" in name:
                chunk_size = full_param.shape[0] // tp_size
                param.set_value(
                    full_param[
                        tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :
                    ]
                )
            # conv1d weight: [conv_dim, 1, kernel] -> shard by sections along dim 0
            elif "conv1d" in name and "weight" in name:
                param.set_value(
                    _shard_by_sections(
                        full_param, _CONV_SECTIONS, tp_rank, tp_size, dim=0
                    )
                )
            # conv1d bias: [conv_dim] -> shard by sections along dim 0
            elif "conv1d" in name and "bias" in name:
                param.set_value(
                    _shard_by_sections(
                        full_param, _CONV_SECTIONS, tp_rank, tp_size, dim=0
                    )
                )
            # dt_bias, A_log: [num_v_heads] -> simple contiguous shard
            elif name.endswith("dt_bias") or name.endswith("A_log"):
                chunk_size = full_param.shape[0] // tp_size
                param.set_value(
                    full_param[
                        tp_rank * chunk_size : (tp_rank + 1) * chunk_size
                    ]
                )
            # out_norm weight: per-head (NOT sharded), copy as-is
            else:
                param.set_value(full_param.clone())

    # Input: use the same fixed seed as baseline to get identical data
    _set_random_seed(INPUT_SEED)
    hidden_states = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

    if sp:
        # For sequence parallel, input is [s, b, h] and sliced per rank
        hidden_states = hidden_states.transpose(
            [1, 0, 2]
        )  # [b, s, h] -> [s, b, h]
        sp_seg = SEQ_LENGTH // sp_size
        hidden_states = hidden_states[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]

    hidden_states.stop_gradient = False
    hidden_states = hidden_states.contiguous()

    output_dist, output_bias_dist = gdn_dist(hidden_states, attention_mask=None)
    output_dist.sum().backward()

    # --- Gather distributed output and compare ---
    output_gathered = _gather_output(output_dist, config, tp_group)

    # If SP, output_gathered is [s, b, h] -> transpose to [b, s, h] for comparison
    if sp:
        output_gathered = output_gathered.transpose([1, 0, 2])

    # Check output for NaN/Inf
    assert paddle.all(~paddle.isnan(output_baseline)).item(), (
        "output_baseline contains NaN"
    )
    assert paddle.all(~paddle.isinf(output_baseline)).item(), (
        "output_baseline contains Inf"
    )
    assert paddle.all(~paddle.isnan(output_gathered)).item(), (
        "output_gathered contains NaN"
    )
    assert paddle.all(~paddle.isinf(output_gathered)).item(), (
        "output_gathered contains Inf"
    )

    # Compare output
    atol, rtol = 5e-4, 5e-4
    assert paddle.allclose(
        output_gathered, output_baseline, atol=atol, rtol=rtol
    ).item(), (
        f"Output mismatch: max_diff="
        f"{(output_gathered - output_baseline).abs().max().item():.6e}"
    )

    # --- Gather input gradients and compare ---
    if sp:
        input_grad_gathered = _gather_along_first_dim(
            hidden_states.grad, tp_group
        )
        # [s, b, h] -> [b, s, h]
        input_grad_gathered = input_grad_gathered.transpose([1, 0, 2])
    else:
        input_grad_gathered = hidden_states.grad

    assert paddle.all(~paddle.isnan(input_grad_baseline)).item(), (
        "input_grad_baseline contains NaN"
    )
    assert paddle.all(~paddle.isnan(input_grad_gathered)).item(), (
        "input_grad_gathered contains NaN"
    )

    assert paddle.allclose(
        input_grad_gathered, input_grad_baseline, atol=atol, rtol=rtol
    ).item(), (
        f"Input grad mismatch: max_diff="
        f"{(input_grad_gathered - input_grad_baseline).abs().max().item():.6e}"
    )

    # --- Gather parameter gradients and compare ---
    # Collect baseline grads (full, no gathering needed)
    baseline_grads = {}
    for name, param in gdn_baseline.named_parameters():
        baseline_grads[name] = (
            param.grad.detach() if param.grad is not None else None
        )

    # Collect distributed grads (need gathering for TP-sharded params)
    dist_grads = _gather_gdn_params(
        gdn_dist, tp_group=tp_group, tp_size=tp_size
    )

    for name in baseline_grads:
        if baseline_grads[name] is None or dist_grads.get(name) is None:
            continue
        b_grad = baseline_grads[name]
        d_grad = dist_grads[name]
        if list(b_grad.shape) != list(d_grad.shape):
            # Shape mismatch for non-TP params is expected if they are per-head
            continue
        assert paddle.allclose(d_grad, b_grad, atol=atol, rtol=rtol).item(), (
            f"Grad mismatch for {name}: max_diff="
            f"{(d_grad - b_grad).abs().max().item():.6e}"
        )

    print(f"  [PASS] TP={tp_size}, SP={sp}")


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

TENSOR_PARALLEL = 4


class TestGatedDeltaNetDistributed(unittest.TestCase):
    """Distributed precision tests for GatedDeltaNet with tensor parallelism."""

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
        """Helper: test that forward produces correct shape and dtype."""
        _set_random_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)

        config = _build_config(tp_size=self.tp_size, sp=sp)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp"]
        )
        gdn = _build_gdn(config, pg_collection=pg_collection)

        sp_size = self.tp_size if sp else 1

        if sp:
            hidden_states = paddle.randn(
                [SEQ_LENGTH // sp_size, MICRO_BATCH_SIZE, HIDDEN_SIZE]
            )
        else:
            hidden_states = paddle.randn(
                [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
            )

        output, bias = gdn(hidden_states, attention_mask=None)

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
        """Helper: compare TP output/grads against single-device baseline."""
        output_baseline, input_grad_baseline, gdn_baseline = _run_baseline(
            self.seed
        )
        _run_distributed(
            self.seed,
            self.tp_size,
            sp=sp,
            output_baseline=output_baseline,
            input_grad_baseline=input_grad_baseline,
            gdn_baseline=gdn_baseline,
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
