# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import paddle
from paddle import Tensor, framework

try:
    from paddlefleet_ops import deep_gemm as paddlefleet_deep_gemm
except (ImportError, RuntimeError):
    pass
try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None
import paddle.distributed as dist
from paddle.autograd.py_layer import PyLayer

from paddleformers.fleet.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from paddleformers.fleet.training.global_vars import get_global_training_logs
from paddleformers.fleet.utils import get_pg_size

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle.distributed.communication.group import Group


_USE_ACCURACY_COMPATIBLE_KERNEL = (
    os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"
)


def use_accuracy_compatible_kernel() -> bool:
    """Unified switch for accuracy-compatible (Megatron-aligned) numeric paths.

    Controlled via the ``FLAGS_use_accuracy_compatible_kernel`` environment
    variable. When enabled, modules switch to fp32-accumulating / Torch-aligned
    kernels at the cost of throughput.
    """
    return _USE_ACCURACY_COMPATIBLE_KERNEL


class AutoSBHistoryTracker:
    """只统计 warmup 阶段连续 MoE auto-subbatch forward 起点的显存下降。"""

    def __init__(self):
        self.step_idx = 0
        self.forward_count = 0
        self.backward_count = 0
        self._last_forward_free = None
        self.max_delta = 0
        self.prev_total_steps = 0
        self.prev_max_delta = 0

    def in_warmup(self) -> bool:
        return self.backward_count == 0

    def record_forward(self, available_free: int):
        if self.in_warmup():
            if self._last_forward_free is not None:
                self.max_delta = max(
                    self.max_delta,
                    self._last_forward_free - available_free,
                    0,
                )
            self._last_forward_free = available_free
            self.step_idx += 1
        self.forward_count += 1

    def record_backward(self) -> bool:
        self.backward_count += 1
        if self.forward_count > 0 and self.backward_count == self.forward_count:
            self._new_iteration()
            return True
        return False

    def _new_iteration(self):
        if self.step_idx > 0:
            self.prev_total_steps = self.step_idx
            self.prev_max_delta = max(self.prev_max_delta, self.max_delta)
        self.step_idx = 0
        self.forward_count = 0
        self.backward_count = 0
        self._last_forward_free = None
        self.max_delta = 0

    def should_degrade(self, available_free: int) -> bool:
        predicted_need = self.predicted_need_for_remaining()
        return (
            self.in_warmup()
            and predicted_need > 0
            and available_free < predicted_need
        )

    def predicted_need_for_remaining(self) -> int:
        """预测当前 warmup 从当前 forward 起还需要的显存。返回 0 表示无法预测（冷启动）。"""
        if self.prev_total_steps == 0 or self.prev_max_delta == 0:
            return 0
        remaining = self.prev_total_steps - self.step_idx + 1
        if remaining <= 0:
            return 0
        predicted = self.prev_max_delta * remaining
        # 安全系数 1.2x + 128MB 余量
        predicted = int(predicted * 1.2) + 128 * 1024 * 1024
        return predicted


_AutoSBHistory = AutoSBHistoryTracker()


def get_auto_sb_history():
    return _AutoSBHistory


def _unpermute_scatter(
    permuted_tokens: paddle.Tensor, sorted_indices: paddle.Tensor, restore_shape
) -> paddle.Tensor:
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    output_tokens.scatter_(
        index=sorted_indices, updates=permuted_tokens, overwrite=False
    )
    return output_tokens


def _unpermute_fp32_accum(
    permuted_tokens: paddle.Tensor, sorted_indices: paddle.Tensor, restore_shape
) -> paddle.Tensor:
    output_tokens = paddle.zeros(restore_shape, dtype="float32")
    output_tokens.scatter_(
        index=sorted_indices,
        updates=permuted_tokens.cast("float32"),
        overwrite=False,
    )
    return output_tokens.cast(permuted_tokens.dtype)


class ApplyPermutedProbs(PyLayer):
    """tokens * probs with fp32-accumulated probs gradient."""

    @staticmethod
    def forward(ctx, permuted_tokens, permuted_probs):
        ctx.input_dtype = permuted_tokens.dtype
        ctx.save_for_backward(permuted_tokens, permuted_probs)
        return permuted_tokens * permuted_probs.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        permuted_tokens, permuted_probs = ctx.saved_tensor()
        grad_tokens = grad_output * permuted_probs.unsqueeze(-1)
        grad_probs = (
            permuted_tokens.cast("float32") * grad_output.cast("float32")
        ).sum(axis=-1)
        return grad_tokens.cast(ctx.input_dtype), grad_probs.cast(
            permuted_probs.dtype
        )


def barrier_ep(ep_group):
    """barrier_ep"""
    paddle.distributed.barrier(ep_group)


def permute(
    tokens,
    routing_map,
    num_out_tokens: int | None = None,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (paddle.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.cast(paddle.bool).T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        paddle.arange(num_tokens).unsqueeze(0).expand([num_experts, -1])
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(axis=0, index=sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: paddle.Tensor,
    sorted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    routing_map: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        sorted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        routing_map (paddle.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"

    if probs is not None:
        assert routing_map is not None, (
            "Mask must be provided to permute the probs."
        )
        permuted_probs = probs.T.contiguous().masked_select(
            routing_map.T.contiguous().cast(paddle.bool)
        )
        if use_accuracy_compatible_kernel():
            permuted_tokens = ApplyPermutedProbs.apply(
                permuted_tokens, permuted_probs
            )
        else:
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    if use_accuracy_compatible_kernel():
        output_tokens = _unpermute_fp32_accum(
            permuted_tokens, sorted_indices, restore_shape
        )
    else:
        output_tokens = _unpermute_scatter(
            permuted_tokens, sorted_indices, restore_shape
        )

    return output_tokens


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx: Any,
        output_shape: list,
        input: Tensor,
        out_split_sizes: list | None = None,
        in_split_sizes: list | None = None,
        group: Group = None,
    ) -> Tensor:  # type: ignore
        """
        All-to-all communication in the group.
        Args:
            ctx (Any): Context object.
            output_shape (list): Output shape.
            input (Tensor): Input tensor.
            out_split_sizes (list): Output split sizes.
            in_split_sizes (list): Input split sizes.
            group (Group): The group object.
        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_shape = input.shape
        ctx.out_split_sizes = out_split_sizes
        ctx.in_split_sizes = in_split_sizes

        # return input
        if dist.get_world_size(group) <= 1:
            return input

        output = paddle.empty(
            output_shape, dtype=input.dtype, requires_grad=True
        )
        paddle.distributed.barrier(group)
        task = dist.alltoall_single(
            output,
            input,
            out_split_sizes=out_split_sizes,
            in_split_sizes=in_split_sizes,
            sync_op=False,
            group=group,
        )
        task.wait()

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[Tensor]:
        """
        Aggregates gradient information from all input tensors into a single tensor.
        Args:
            ctx (Any): The context object used to store information that needs to be passed.
            *grad_output (Tensor): A list of input tensors whose gradients are to be aggregated.
        Returns:
            tuple[Tensor]: A tuple containing a tensor that holds the gradients of all input tensors.
        """
        # return grad_output
        paddle.distributed.barrier(ctx.group)
        return _AllToAll.apply(
            ctx.input_shape,
            *grad_output,
            ctx.in_split_sizes,
            ctx.out_split_sizes,
            ctx.group,
        )


class RandomSTE(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        if dist.get_world_size() <= 1:
            return paddle.randn(x.shape).cast(x.dtype)
        else:
            with get_cuda_rng_tracker().fork(
                get_expert_parallel_rng_tracker_name()
            ):
                return paddle.randn(x.shape).cast(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return paddle.zeros(ctx.x_shape, dtype=ctx.x_dtype)


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)


def global_moe_balance_training_logs_enabled():
    logs = get_global_training_logs()
    if logs is None:
        return False

    is_enabled = getattr(logs, "is_moe_balance_logs_enabled", None)
    return callable(is_enabled) and is_enabled()


def log_moe_losses(layer_number, aux_loss=None, z_loss=None):
    if not global_moe_balance_training_logs_enabled():
        return
    logs = get_global_training_logs()
    if logs is None or not hasattr(logs, "update"):
        return

    log = {}
    if aux_loss is not None:
        aux_loss = aux_loss.detach()
        log["aux_loss"] = aux_loss
        if layer_number is not None:
            log[f"aux_loss_layer_{layer_number}"] = aux_loss

    if z_loss is not None:
        z_loss = z_loss.detach()
        log["zloss"] = z_loss
        if layer_number is not None:
            log[f"zloss_layer_{layer_number}"] = z_loss

    logs.update(**log)


def _all_gather_local_tokens(local_tokens_per_expert, group):
    local_tokens_per_expert = local_tokens_per_expert.reshape([-1])
    if group is None or get_pg_size(group) <= 1 or not group.is_member():
        return local_tokens_per_expert.reshape([1, -1])

    if local_tokens_per_expert.place.is_cpu_place():
        gathered = []
        dist.all_gather_object(
            gathered,
            local_tokens_per_expert.tolist(),
            group=group,
        )
        return paddle.to_tensor(
            gathered,
            dtype=local_tokens_per_expert.dtype,
            place=paddle.CPUPlace(),
        ).reshape([get_pg_size(group), -1])

    output_shape = local_tokens_per_expert.shape
    output_shape[0] *= get_pg_size(group)
    output = paddle.empty(output_shape, dtype=local_tokens_per_expert.dtype)
    dist.stream.all_gather(
        output,
        local_tokens_per_expert,
        group=group,
        use_calc_stream=True,
    )
    return output.reshape([get_pg_size(group), -1])


def _log_summary(key, layer_number, summary_data):
    logs = get_global_training_logs()
    if logs is None or not hasattr(logs, "update"):
        return

    summary_data = summary_data.detach()
    if summary_data.numel() == 0:
        return

    summary_data = summary_data.astype("float32")
    max_value = float(paddle.max(summary_data).item())
    min_value = float(paddle.min(summary_data).item())
    var_value = float(paddle.var(summary_data).item())
    median_value = float(paddle.median(summary_data).item())
    mean_value = float(paddle.mean(summary_data).item())
    max_mean_ratio = max_value / mean_value if mean_value != 0 else 1.0
    min_mean_ratio = min_value / mean_value if mean_value != 0 else 1.0

    prefix = f"{key}_layer_{layer_number}"
    logs.update(
        **{
            f"{prefix}_max": max_value,
            f"{prefix}_min": min_value,
            f"{prefix}_var": var_value,
            f"{prefix}_median": median_value,
            f"{prefix}_mean": mean_value,
            f"{prefix}_max_mean_ratio": max_mean_ratio,
            f"{prefix}_min_mean_ratio": min_mean_ratio,
        }
    )


def _log_tokens_per_expert(layer_number, key, summary_data, count):
    count = count.reshape([1]).astype("float32")
    count = paddle.ones_like(count) if count.item() == 0 else count
    avg_data = summary_data.astype("float32") / count

    _log_summary(f"{key}_avg", layer_number, avg_data)
    _log_summary(key, layer_number, summary_data)


def _log_local_tokens_per_card(layer_number, local_tokens_by_rank):
    card_totals = local_tokens_by_rank.sum(axis=1)
    _log_summary(
        "local_tokens_per_card",
        layer_number,
        card_totals,
    )


def log_moe_balance(
    layer_number,
    moe_group,
    num_experts_per_tok,
    tokens_per_expert,
):
    """Log fixed-topk MoE balance summaries from dispatched expert counts."""
    if tokens_per_expert is None:
        return

    with paddle.no_grad():
        if not isinstance(tokens_per_expert, paddle.Tensor):
            tokens_per_expert = paddle.to_tensor(
                tokens_per_expert,
                dtype="int64",
                place=paddle.CPUPlace(),
            )
        else:
            tokens_per_expert = tokens_per_expert.detach()
            if not tokens_per_expert.place.is_cpu_place():
                # Moving a GPU tensor to CPU here introduces a synchronization
                # point, so this path is slower and can affect training
                # performance.
                tokens_per_expert = tokens_per_expert.cpu()

        local_tokens_by_rank = _all_gather_local_tokens(
            tokens_per_expert,
            moe_group,
        )
        topk = max(int(num_experts_per_tok or 1), 1)
        summary = local_tokens_by_rank.reshape([-1])
        count = summary.astype("float32").sum().reshape([1]) / topk

        _log_tokens_per_expert(
            layer_number,
            "tokens_per_expert",
            summary.clone() if isinstance(summary, paddle.Tensor) else summary,
            count,
        )
        _log_local_tokens_per_card(layer_number, local_tokens_by_rank)


def is_tensor(data):
    """Check if data is a tensor"""
    return isinstance(data, (paddle.Tensor, paddle.base.core.eager.Tensor))


def detach_and_requires_grad_(*args):
    """Detach tensors and preserve their requires_grad settings"""
    ret = [a.detach() if is_tensor(a) else a for a in args]
    for r, a in zip(ret, args):
        if is_tensor(a):
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    """
    In manual_backward, in order to preserve the local computation graph for temporary backward computation,
    we need to clone the output of manual_backward. This clone operation essentially doesn't need the value
    of the output, but rather needs to obtain the computation graph attached to the output.

    However, calling paddle.clone would perform an unnecessary data copy.
    FakeClone avoids this data copy and achieves the goal of extracting the computation graph.
    """

    @staticmethod
    def forward(ctx, input):
        """Forward pass"""
        if input.is_contiguous():
            fake_output = paddle.Tensor()
            fake_output.get_tensor()._share_data_nocheck_with(
                input.get_tensor()
            )
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: list[Any]):
    """
    Args:
        f(callable)
        args(*Any)
    Returns
        bw_f(callable): manual backward fn
        out(List[Tensor]): output of f(*args)
    """
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True  # turn on grad trace so we can manual backward

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [
        FakeClone.apply(a) if is_tensor(a) else a for a in detached_args
    ]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [
        FakeClone.apply(o) for o in out if o is not None
    ]  # do not cache stop_gradient output

    for o in out_cached:
        o._clear_dataptr()  # free mem
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        grad, out_cached = zip(
            *[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient]
        )

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)

        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if is_tensor(t)])

    return bwd_f, out


class FilterScores(PyLayer):
    @staticmethod
    def forward(ctx, probs, indices):
        topk_scores = paddle._C_ops._run_custom_op(
            "filter_scores", probs, indices
        )[0]
        ctx.save_for_backward(indices)
        return topk_scores

    @staticmethod
    def backward(ctx, grad_topk_scores):
        (indices,) = ctx.saved_tensor()
        grads = paddle._C_ops._run_custom_op(
            "filter_scores_grad",
            indices,
            grad_topk_scores,
        )
        grad_probs = grads[0]
        return grad_probs, None


def fused_expert_parallel_TC_topk_router_metadata(
    dispatched_indices,
    expert_frequency_offset,
    K,
):
    return paddle._C_ops._run_custom_op(
        "router_metadata", dispatched_indices, expert_frequency_offset, K
    )


def count_cumsum(
    dispatched_indices,
    num_experts_per_device,
    do_cumsum,
):
    return paddle._C_ops._run_custom_op(
        "count_cumsum",
        dispatched_indices,
        num_experts_per_device,
        do_cumsum,
    )


def filter_scores(
    dispatched_probs,
    dispatched_indices,
):
    return FilterScores.apply(dispatched_probs, dispatched_indices)


def k_grouped_bf16_gemm_tn_contiguous_aligned(a, b, d, ks, ks_tensor, c):
    ALIGNMENT = paddlefleet_deep_gemm.get_mk_alignment_for_contiguous_layout()

    # Compute padded sizes using tensor ops
    padded_ks_tensor = ((ks_tensor + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    padded_sizes_list = padded_ks_tensor.tolist()

    def pad_grouped_tensor(tensor, ks_tensor, padded_ks_tensor):
        """
        Vectorized padding for grouped tensors.
        Eliminates for-loops and uses a single global index assignment.
        """
        total_unpadded = ks_tensor.sum().item()
        total_padded = padded_ks_tensor.sum().item()

        # 1. Compute start offsets for both source and destination
        # We use cumsum to find where each group begins
        src_offsets = paddle.cat([paddle.tensor([0]), ks_tensor.cumsum(0)[:-1]])
        dst_offsets = paddle.cat(
            [paddle.tensor([0]), padded_ks_tensor.cumsum(0)[:-1]]
        )

        # 2. Calculate the "shift" required for every single element
        # diff represents how much further each group moves in the padded tensor
        diff = dst_offsets - src_offsets

        # 3. Create a map of indices from source to destination
        # Repeat the shift amount for every element in that group
        element_shifts = paddle.repeat_interleave(diff, ks_tensor)
        src_indices = paddle.arange(total_unpadded, device=tensor.device)
        dst_indices = src_indices + element_shifts

        # 4. Allocate and scatter
        padded_tensor = paddle.zeros(
            (total_padded, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # Single vectorized assignment
        padded_tensor[dst_indices] = tensor

        del (
            src_offsets,
            dst_offsets,
            diff,
            element_shifts,
            src_indices,
            dst_indices,
        )
        return padded_tensor

    # Vectorized pad
    a_padded = pad_grouped_tensor(a, ks_tensor, padded_ks_tensor)
    b_padded = pad_grouped_tensor(b, ks_tensor, padded_ks_tensor)

    paddlefleet_deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        a=a_padded,
        b=b_padded,
        d=d,
        ks=padded_sizes_list,
        ks_tensor=padded_ks_tensor,
        c=c,
    )

    del a_padded, b_padded


def sort_chunks_by_idxs(
    input: paddle.Tensor,
    split_sizes: paddle.Tensor,
    sorted_idxs: paddle.Tensor,
    probs: paddle.Tensor | None = None,
    fused: bool = False,
):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = paddle.split(input, split_sizes.tolist(), axis=0)
    output = paddle.cat([input[i] for i in sorted_idxs.tolist()], axis=0)
    # TODO: support probs is not None
    permuted_probs = None
    return output, permuted_probs


def all_gather_group(input, group=None, axis=0):
    """Perform collective all-gather operation across a process group with axis control.

    Functional Behavior:
      - Aggregates input tensors from all processes in the specified group
      - Supports concatenation along arbitrary dimensions (axis parameter)
      - Optimizes for axis=0 via direct shape expansion to avoid concatenation overhead

    Args:
        input (Tensor):        Local tensor to be gathered (shape: [..., D, ...])
        group (ProcessGroup):  Communication group (defaults to model parallel group)
        axis (int):            Concatenation dimension (default=0)

    Returns:
        Tensor: Concatenated tensor combining inputs from all processes:
                - When axis=0: shape [D*N, ...] (N = group size)
                - Otherwise:   shape [..., D*N, ...] along specified axis
    """
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    # TODO: Support only axis != 0
    assert axis == 0
    output_shape[axis] = output_shape[axis] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.all_gather(output, input, group=group, use_calc_stream=True)
    return output


def reduce_scatter_group(input, group=None):
    """Perform reduce-scatter collective operation across a process group.

    Functional Behavior:
      - Aggregates (sums) input tensors across all processes in the group
      - Scatters the reduced result equally to all participants
      - Operates along the first dimension (axis=0) of the input tensor

    Args:
        input (Tensor):        Local tensor to reduce (shape: [N*K, ...] where N=group_size)
        group (ProcessGroup): Communication group (defaults to model parallel group)

    Returns:
        Tensor: Scattered portion of reduced tensor with shape [K, ...]
    """
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    assert input.shape[0] % parallelism == 0, (
        f"Input sequence length {input.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(
        output, input, op=dist.ReduceOp.SUM, group=group, use_calc_stream=True
    )
    return output


class AllGatherGroupOp(paddle.autograd.PyLayer):
    """
    Perform group allgather.
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """Forward pass: All-Gather operation
        Args:
            input (Tensor):  Partitioned tensor with shape [s/n, b, h]
                            The 's' dimension is distributed across devices
            group (ProcessGroup): Model parallel process group,
                                uses global group by default
        Returns:
            Tensor: Assembled tensor after All-Gather with shape [s, b, h],
                   containing full parameter from all devices
        """
        paddle.distributed.barrier(group)
        ctx.group = group
        return all_gather_group(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        """Backward pass: Reduce-Scatter operation
        Args:
            grad (Tensor): Full gradient tensor with shape [s, b, h]
        Returns:
            Tensor: Scattered gradient with shape [s/n, b, h],
                   distributing reduced gradients to each device
        """
        paddle.distributed.barrier(ctx.group)
        return reduce_scatter_group(grad, group=ctx.group)
