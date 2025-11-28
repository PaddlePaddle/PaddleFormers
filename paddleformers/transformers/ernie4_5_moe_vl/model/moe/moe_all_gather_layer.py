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


"""
moe_layer_all_gather
"""

import inspect
from typing import Callable, Dict, List, Optional, Tuple

import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group, _get_global_group
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn.functional import (
    build_src_rank_and_local_expert_id,
    expand_modality_expert_id,
    moe_gate_dispatch_partial_nosoftmaxtopk,
)
from paddle.incubate.tensor.manipulation import async_offload

from paddleformers.peft.lora.lora_quantization_layers import QuantizationLoRALinear
from paddleformers.utils.log import logger

from ..distributed.common_dist_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    all_gather_group,
    get_async_loader,
    hack_offload_wait,
    reduce_scatter_group,
)
from .moe_layer import MOELayer, manual_backward


def allgather_async(input, group=None):
    """Perform asynchronous All-Gather operation for model parallelism.

    Args:
        input (Tensor):        Local tensor to gather (shape: [N, ...])
        group (ProcessGroup): Model parallel group (default: auto-detected)

    Returns:
        tuple: (output_tensor, communication_task)
            output_tensor: Pre-allocated buffer with shape [N*K, ...] (K=group_size)
            communication_task: Paddle communication task handle for synchronization
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone(), None
    output_shape = input.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = dist.stream.all_gather(output, input, group=group, use_calc_stream=False, sync_op=False)
    return output, task


def reduce_scatter_async(input, group=None):
    """Perform asynchronous reduce-scatter operation for distributed training.

    Args:
        input (Tensor):        Local tensor to reduce (shape: [N*K, ...], N=group_size)
        group (ProcessGroup): Communication group (default: model parallel group)

    Returns:
        tuple: (output_tensor, communication_task)
            output_tensor: Scattered tensor portion with shape [K, ...]
            communication_task: Handle for synchronizing the async operation
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone(), None
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), f"Input sequence length {input.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    task = dist.stream.reduce_scatter(
        output,
        input,
        op=dist.ReduceOp.SUM,
        group=group,
        use_calc_stream=False,
        sync_op=False,
    )
    return output, task


class AllGatherAsync(PyLayer):
    """
    Perform async allgather.
    """

    @staticmethod
    def forward(ctx, input, *fn_args, group=None, fn=None, is_first_fwd=False):
        """Forward pass with integrated communication-computation overlap.

        Args:
            ctx: PyLayer context object
            input (Tensor): Sharded input tensor [s/n, b, h]
            *fn_args: Arguments for custom forward function
            group: Model parallel process group
            fn: Custom forward function to execute after communication
            is_first_fwd: Flag indicating first forward pass in sequence

        Returns:
            tuple: (gathered_tensor, ...custom_forward_outputs)
        """
        ctx.group = group
        if dist.get_world_size(group) <= 1:
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (input,) + fn_out
        out, task = allgather_async(input, group=group)
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task and task.wait()
        return (out,) + fn_out

    @staticmethod
    def backward(ctx, grad, *fn_out_grads):
        """Backward pass with gradient synchronization.

        Args:
            ctx: PyLayer context with stored communication group
            grad (Tensor): Full gradient tensor [s, b, h]
            *fn_out_grads: Gradients from custom forward outputs

        Returns:
            tuple: (scattered_grad, ...custom_arg_grads)
        """
        if dist.get_world_size(ctx.group) <= 1:
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad,) + fn_args_grads

        grad, task = reduce_scatter_async(grad, group=ctx.group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task and task.wait()
        return (grad,) + fn_args_grads


class ReshardCombineWeight(PyLayer):
    """
    Perform weights transform.
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """Converts expert-partitioned weights to sequence-partitioned format.

        Args:
            ctx: PyLayer context object
            input (Tensor): Expert-wise partitioned weights [Seq, k] where:
                            - Non-local experts are zeroed out
                            - Seq: sequence dimension (may be sharded)
                            - k: expert capacity
            group (ProcessGroup): Model parallel group (default:)

        Returns:
            Tensor: Sequence-wise partitioned weights [Seq/n, k] via reduce-scatter
        """

        ctx.mask = input == 0.0
        ctx.group = group
        return reduce_scatter_group(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        """Reconstructs expert-partitioned gradients from sequence-wise gradients.

        Args:
            grad (Tensor): Sequence-wise partitioned gradients [Seq/n, k]

        Returns:
            Tensor: Expert-wise partitioned gradients [Seq, k] with zeros for
                   non-local experts
        """
        gathered = all_gather_group(grad, group=ctx.group)
        return gathered.masked_fill(
            ctx.mask,
            0.0,
        )


class AlltoAllSmart(paddle.autograd.PyLayer):
    """
    Perform dispatch inputs alltoall.
    """

    @staticmethod
    def forward(
        ctx,
        *inputs,
        router_loss_fn: Optional[Callable],
        forward_func_dict: Optional[Dict[int, Callable]],
        local_expert_id=None,
        send_rank_global=None,
        recv_rank_global=None,
        num_local_experts=None,
        capacity=None,
        use_padding=True,
        expert_num_global=None,
        is_first_fwd=None,
        group=None,
        recv_size=None,
        send_counts=None,
        recv_counts=None,
        send_counts_num=None,
        recv_counts_num=None,
    ):
        """Implements batched point-to-point communication with expert computation overlap.

        Functional Behavior:
          - Performs distributed All-to-All communication with variable message sizes
          - Overlaps expert forward computation with communication operations
          - Calculates router loss for dynamic expert selection
          - Handles padding/compression for irregular tensor shapes

        Key Operations:
          1. Prepare communication buffers based on send/recv counts
          2. Launch asynchronous All-to-All operations
          3. Execute expert forward functions in parallel with communication
          4. Calculate routing loss and prepare gradient masks

        Args:
            ctx: PyLayer context object
            *inputs: Variable-length expert inputs (Tensor[...])
            router_loss_fn: Routing loss calculator function
            forward_func_dict: Expert-specific forward functions {expert_id: callable}
            local_expert_id: Tensor indicating local expert assignments
            send_rank_global: Global ranks for sending data
            recv_rank_global: Global ranks for receiving data
            num_local_experts: Number of experts per device
            capacity: Maximum tokens per expert
            use_padding: Enable padding for fixed-size buffers
            expert_num_global: Global expert count
            is_first_fwd: Flag for activation checkpointing
            group: Process group for communication
            recv_size: Precomputed receive buffer size
            send_counts: Per-expert send counts [num_local_experts, world_size]
            recv_counts: Per-expert recv counts [num_local_experts, world_size]
            send_counts_num: Aggregated send expert
            recv_counts_num: Aggregated recv counts per expert

        Returns:
            tuple: (output_tensor, router_loss, gradient_mask)
        """
        if group is None:
            group = _get_global_group()
        router_loss_args = inputs[num_local_experts:]
        inputs = inputs[:num_local_experts]

        ctx.group = group
        ctx.use_padding = use_padding
        ctx.num_local_experts = num_local_experts
        ctx.input_shape = [i.shape if i is not None else None for i in inputs]

        this_rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        capacity = len(send_rank_global) // world_size // num_local_experts
        ctx.capacity = capacity
        assert len(local_expert_id) == len(recv_rank_global), (
            len(local_expert_id),
            len(recv_rank_global),
        )

        for i in inputs:
            if i is not None:
                input_dtype = i.dtype
                input_shape = i.shape
                break
        else:
            raise RuntimeError("all inputs are None")

        output = paddle.zeros([recv_size] + input_shape[1:], dtype=input_dtype)
        output_ptr = 0

        tasks = []
        dummy_input = paddle.empty([0] + input_shape[1:], dtype=input_dtype)
        ctx.dummy_input = dummy_input
        ctx.bw_funcs = {}

        for i_local_expert in range(num_local_experts):
            send_count = send_counts[i_local_expert]
            recv_count = recv_counts[i_local_expert]
            assert len(recv_count) == len(send_count) == (world_size), (
                len(recv_count),
                len(send_count),
            )

            if send_counts_num[i_local_expert] > 0:
                input_local_expert = inputs[i_local_expert].slice((0,), 0, send_counts_num[i_local_expert])
                if forward_func_dict is not None:
                    input_local_expert.stop_gradient = False
                    bwf, (input_local_expert,) = manual_backward(
                        forward_func_dict[i_local_expert],
                        is_first_fwd,
                        input_local_expert,
                    )
                    ctx.bw_funcs[i_local_expert] = bwf

                if input_local_expert is None:
                    input_local_expert = dummy_input
                input_local_expert.stop_gradient = True
            else:
                input_local_expert = dummy_input
            if recv_counts_num[i_local_expert] > 0:
                # When FLAGS_use_stride_kernel=0, tensor.slice(...) returns a
                # new tensor instead of a view, causing in-place assignment to fail.
                # tensor._slice ensures it always returns a view.
                # See:
                #   https://github.com/PaddlePaddle/Paddle/blob/release/3.1/paddle/phi/core/dense_tensor_impl.cc#L299
                output_local_expert = output._slice(output_ptr, (output_ptr + recv_counts_num[i_local_expert]))
            else:
                output_local_expert = dummy_input

            output_ptr += recv_counts_num[i_local_expert]

            if group.nranks <= 1:
                output_local_expert[:] = input_local_expert[:]
            else:
                tasks.append(
                    dist.stream.alltoall_single(
                        output_local_expert,
                        input_local_expert,
                        recv_count,
                        send_count,
                        group=group,
                        sync_op=False,
                        use_calc_stream=False,
                    )
                )
        ctx.router_loss_bwfn, (router_loss,) = manual_backward(router_loss_fn, is_first_fwd, *router_loss_args)
        with paddle.no_grad():
            recv_mask = (recv_rank_global == this_rank).astype(send_rank_global.dtype)
            if ctx.use_padding:
                recv_mask_alltoall_out = (
                    recv_mask.reshape([-1, num_local_experts, capacity]).transpose([1, 0, 2]).reshape([-1])
                )
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.view([num_local_experts, -1, capacity])
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
            else:
                recv_mask_alltoall_out = recv_mask.split(expert_num_global)  # h->d copy break overlap
                recv_mask_alltoall_out = [
                    recv_mask_alltoall_out[(iexpert % world_size) * num_local_experts + (iexpert // world_size)]
                    for iexpert in range(world_size * num_local_experts)
                ]
                alltoall_shape = [i.shape[0] for i in recv_mask_alltoall_out]

                recv_mask_alltoall_out = paddle.concat(recv_mask_alltoall_out, 0)
                distributed_input_to_alltoall_out = paddle.maximum(
                    (recv_mask_alltoall_out.cumsum() - 1).astype(recv_mask_alltoall_out.dtype),
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = distributed_input_to_alltoall_out.split(alltoall_shape)

                distributed_input_to_alltoall_out = paddle.concat(
                    [
                        distributed_input_to_alltoall_out[
                            (iexpert % num_local_experts) * world_size + (iexpert // num_local_experts)
                        ]
                        for iexpert in range(world_size * num_local_experts)
                    ],
                    0,
                )

        distributed_input_to_alltoall_out.stop_gradient = True
        for t in tasks:
            t and t.wait()
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        return output, router_loss, distributed_input_to_alltoall_out

    @staticmethod
    def backward(
        ctx,
        out_grad,
        d_routerloss,
        _,  # scatter-idx no grad
    ):
        """Performs distributed gradient propagation for expert-parallel models.

        Functional Behavior:
          - Distributes output gradients via reverse All-to-All communication
          - Computes expert-specific gradients using stored backward functions
          - Aggregates routing loss gradients

        Key Operations:
          1. Prepare gradient buffers based on forward pass metadata
          2. Execute reverse All-to-All communication
          3. Apply expert-specific backward computations
          4. Combine gradients from all sources

        Args:
            ctx: Context object storing forward pass information
            out_grad (Tensor): Gradient from downstream layers
            d_routerloss (Tensor): Routing loss gradient
            _: Ignored placeholder

        Returns:
            tuple: Combined gradients (expert gradients + router loss gradients)
        """

        grads = [paddle.zeros(s, dtype=out_grad.dtype) if s is not None else None for s in ctx.input_shape]
        assert len(grads) == ctx.num_local_experts
        out_ptr = 0
        tasks = []
        tmp_g = []
        send_counts_num = ctx.send_counts.sum(-1)
        recv_counts_num = ctx.recv_counts.sum(-1)
        out_grad = out_grad.contiguous()
        for i_local_expert in range(ctx.num_local_experts):
            send_count = ctx.send_counts[i_local_expert]
            recv_count = ctx.recv_counts[i_local_expert]
            if recv_counts_num[i_local_expert] > 0:
                out_g = out_grad.slice((0,), out_ptr, out_ptr + recv_counts_num[i_local_expert])
            else:
                out_g = ctx.dummy_input  # paddle.empty([0,]+out_grad.shape[1:], dtype=out_grad.dtype)
            if send_counts_num[i_local_expert] > 0:
                # When FLAGS_use_stride_kernel=0, tensor.slice(...) returns a
                # new tensor instead of a view, causing in-place assignment to fail.
                # tensor._slice ensures it always returns a view.
                # See:
                #   https://github.com/PaddlePaddle/Paddle/blob/release/3.1/paddle/phi/core/dense_tensor_impl.cc#L299
                g = grads[i_local_expert]._slice(0, send_counts_num[i_local_expert])
            else:
                g = ctx.dummy_input
            tmp_g.append(g)
            out_ptr += recv_counts_num[i_local_expert]
            if ctx.group.nranks <= 1:
                g[:] = out_g[:]
            else:
                task = dist.stream.alltoall_single(
                    g,
                    out_g,
                    send_count,
                    recv_count,
                    group=ctx.group,
                    sync_op=False,
                    use_calc_stream=False,
                )
                tasks.append(task)
        router_fn_args_grad = ctx.router_loss_bwfn(d_routerloss)

        for i_local_expert, t in enumerate(tasks):
            t and t.wait()
            send_cnt = send_counts_num[i_local_expert]
            if send_cnt > 0 and ctx.bw_funcs:
                (g,) = ctx.bw_funcs[i_local_expert](tmp_g[i_local_expert])
                grads[i_local_expert][:send_cnt] = g

        grads = [g for g in grads if g is not None]
        return tuple(grads) + tuple(router_fn_args_grad)


class AlltoAllSmartXPU(paddle.autograd.PyLayer):
    """
    Perform dispatch inputs alltoall. (XPU VERSION)
    """

    @staticmethod
    def forward(
        ctx,
        *inputs,
        router_loss_fn: Optional[Callable],
        forward_func_dict: Optional[Dict[int, Callable]],
        local_expert_id=None,
        send_rank_global=None,
        recv_rank_global=None,
        num_local_experts=None,
        capacity=None,
        use_padding=True,
        expert_num_global=None,
        is_first_fwd=None,
        group=None,
        recv_size=None,
        send_counts=None,
        recv_counts=None,
        send_counts_num=None,
        recv_counts_num=None,
    ):
        if group is None:
            group = _get_global_group()
        router_loss_args = inputs[num_local_experts:]
        inputs = inputs[:num_local_experts]

        ctx.group = group
        ctx.use_padding = use_padding
        ctx.num_local_experts = num_local_experts
        ctx.input_shape = [i.shape if i is not None else None for i in inputs]
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.send_counts_num = send_counts_num
        ctx.recv_counts_num = recv_counts_num

        world_size = dist.get_world_size(group)
        this_rank = dist.get_rank(group)
        if use_padding and capacity is None:
            capacity = len(send_rank_global) // world_size // num_local_experts

        for i in inputs:
            if i is not None:
                input_dtype = i.dtype
                input_shape = i.shape
                break
        else:
            first_expert = forward_func_dict[0]
            input_dtype = first_expert.down_proj.weight.dtype
            hidden_size = first_expert.down_proj.weight.shape[1]
            input_shape = [0, hidden_size]

        dummy_input = paddle.empty([0] + input_shape[1:], dtype=input_dtype)
        ctx.dummy_input = dummy_input
        ctx.bw_funcs = {}

        processed_inputs = []
        no_tokens_expert_outputs = []

        for i_local_expert in range(num_local_experts):
            if send_counts_num[i_local_expert] > 0:
                input_local_expert = inputs[i_local_expert].slice((0,), 0, send_counts_num[i_local_expert])
                if forward_func_dict is not None:
                    input_local_expert.stop_gradient = False
                    bwf, (processed_input,) = manual_backward(
                        forward_func_dict[i_local_expert],
                        is_first_fwd,
                        input_local_expert,
                    )
                    ctx.bw_funcs[i_local_expert] = bwf
                    processed_input.stop_gradient = True
                else:
                    processed_input = input_local_expert
                processed_inputs.append(processed_input)
            elif forward_func_dict is not None:
                expert_func = forward_func_dict[i_local_expert]
                fake_chunk = paddle.zeros(
                    [1, expert_func.down_proj.weight.shape[1]],
                    dtype=expert_func.down_proj.weight.dtype,
                )
                if expert_func.training:
                    fake_chunk.stop_gradient = False

                _, (expert_out,) = manual_backward(expert_func, is_first_fwd, fake_chunk)

                no_tokens_expert_outputs.append(expert_out * 0.0)

        all_processed_inputs = paddle.concat(processed_inputs, axis=0) if processed_inputs else dummy_input

        if no_tokens_expert_outputs:
            if all_processed_inputs.shape[0] > 0:
                all_processed_inputs[0] = all_processed_inputs[0] + sum(no_tokens_expert_outputs)
            else:
                router_loss_args = list(router_loss_args)
                router_loss_args[0] = router_loss_args[0] + sum(no_tokens_expert_outputs).mean() * 0.0

        in_tensors_by_rank = [[] for _ in range(world_size)]
        processed_input_ptr = 0
        for i_local_expert in range(num_local_experts):
            num_tokens = send_counts_num[i_local_expert]
            if num_tokens > 0:
                expert_input = all_processed_inputs.slice([0], processed_input_ptr, processed_input_ptr + num_tokens)
                processed_input_ptr += num_tokens
                splits = expert_input.split(send_counts[i_local_expert].tolist(), axis=0)
                for j_rank in range(world_size):
                    in_tensors_by_rank[j_rank].append(splits[j_rank])

        in_tensor_list = [paddle.concat(tensors, 0) if tensors else dummy_input for tensors in in_tensors_by_rank]

        all_to_all_input = paddle.concat(in_tensor_list, 0)
        send_counts_for_api = [t.shape[0] for t in in_tensor_list]

        recv_counts_tensor = paddle.to_tensor(recv_counts)
        recv_counts_for_api = [int(recv_counts_tensor[:, j_rank].sum()) for j_rank in range(world_size)]
        temp_output = paddle.empty([recv_size.item()] + input_shape[1:], dtype=input_dtype)

        if group.nranks <= 1:
            task = None
            if all_to_all_input.shape[0] > 0:
                temp_output[:] = all_to_all_input[:]
        else:
            task = dist.stream.alltoall_single(
                temp_output,
                all_to_all_input,
                recv_counts_for_api,
                send_counts_for_api,
                group=group,
                sync_op=False,
                use_calc_stream=False,
            )

        ctx.router_loss_bwfn, (router_loss,) = manual_backward(router_loss_fn, is_first_fwd, *router_loss_args)
        with paddle.no_grad():
            recv_mask = (recv_rank_global == this_rank).astype(send_rank_global.dtype)
            if ctx.use_padding:
                recv_mask_alltoall_out = (
                    recv_mask.reshape([-1, num_local_experts, capacity]).transpose([1, 0, 2]).reshape([-1])
                )
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.view([num_local_experts, -1, capacity])
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
            else:
                recv_mask_alltoall_out = recv_mask.split(expert_num_global)
                recv_mask_alltoall_out = [
                    recv_mask_alltoall_out[(iexpert % world_size) * num_local_experts + (iexpert // world_size)]
                    for iexpert in range(world_size * num_local_experts)
                ]
                alltoall_shape = [i.shape[0] for i in recv_mask_alltoall_out]
                recv_mask_alltoall_out = paddle.concat(recv_mask_alltoall_out, 0)
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = distributed_input_to_alltoall_out.split(alltoall_shape)
                distributed_input_to_alltoall_out = paddle.concat(
                    [
                        distributed_input_to_alltoall_out[
                            (iexpert % num_local_experts) * world_size + (iexpert // num_local_experts)
                        ]
                        for iexpert in range(world_size * num_local_experts)
                    ],
                    0,
                )

        distributed_input_to_alltoall_out.stop_gradient = True

        if task is not None:
            task.wait()

        temp_output_splits_by_src_rank = temp_output.split(recv_counts_for_api, 0)
        chunks_by_expert = [[] for _ in range(num_local_experts)]
        for j_rank in range(world_size):
            data_from_j = temp_output_splits_by_src_rank[j_rank]
            expert_chunks_from_j = data_from_j.split(recv_counts[:, j_rank].tolist(), 0)
            for i_expert in range(num_local_experts):
                chunks_by_expert[i_expert].append(expert_chunks_from_j[i_expert])

        output_chunks = []
        for i_expert in range(num_local_experts):
            if recv_counts_num[i_expert] > 0:
                output_chunks.append(paddle.concat(chunks_by_expert[i_expert], 0))
        output = paddle.concat(output_chunks, 0) if output_chunks else dummy_input

        return output, router_loss, distributed_input_to_alltoall_out

    @staticmethod
    def backward(
        ctx,
        out_grad,
        d_routerloss,
        _,  # scatter-idx no grad
    ):
        world_size = dist.get_world_size(ctx.group)
        num_local_experts = ctx.num_local_experts
        dummy_input = ctx.dummy_input
        out_grad = out_grad.contiguous()

        send_counts_bw = ctx.recv_counts
        send_counts_num_bw = ctx.recv_counts_num
        in_tensors_by_rank_bw = [[] for _ in range(world_size)]
        grad_ptr = 0
        for i_expert in range(num_local_experts):
            num_tokens = send_counts_num_bw[i_expert]
            if num_tokens > 0:
                expert_grad = out_grad.slice([0], grad_ptr, grad_ptr + num_tokens)
                grad_ptr += num_tokens
                splits = expert_grad.split(send_counts_bw[i_expert].tolist(), 0)
                for j_rank in range(world_size):
                    in_tensors_by_rank_bw[j_rank].append(splits[j_rank])
        in_tensor_list_bw = [
            paddle.concat(tensors, 0) if tensors else dummy_input for tensors in in_tensors_by_rank_bw
        ]

        all_to_all_grad_input = paddle.concat(in_tensor_list_bw, 0)
        send_counts_bw_for_api = [t.shape[0] for t in in_tensor_list_bw]

        recv_counts_bw = ctx.send_counts
        recv_counts_tensor_bw = paddle.to_tensor(recv_counts_bw)
        recv_counts_bw_for_api = [int(recv_counts_tensor_bw[:, j_rank].sum()) for j_rank in range(world_size)]
        total_output_grad_size = int(ctx.send_counts_num.sum())
        temp_grad_output = paddle.empty([total_output_grad_size] + list(out_grad.shape[1:]), dtype=out_grad.dtype)

        if ctx.group.nranks <= 1:
            task = None
            if all_to_all_grad_input.shape[0] > 0:
                temp_grad_output[:] = all_to_all_grad_input[:]
        else:
            task = dist.stream.alltoall_single(
                temp_grad_output,
                all_to_all_grad_input,
                recv_counts_bw_for_api,
                send_counts_bw_for_api,
                group=ctx.group,
                sync_op=False,
                use_calc_stream=False,
            )

        router_fn_args_grad = ctx.router_loss_bwfn(d_routerloss)

        if task is not None:
            task.wait()

        temp_grad_output_splits = temp_grad_output.split(recv_counts_bw_for_api, 0)
        grad_chunks_by_expert = [[] for _ in range(num_local_experts)]
        for j_rank in range(world_size):
            data_from_j = temp_grad_output_splits[j_rank]
            expert_chunks_from_j = data_from_j.split(recv_counts_bw[:, j_rank].tolist(), 0)
            for i_expert in range(num_local_experts):
                grad_chunks_by_expert[i_expert].append(expert_chunks_from_j[i_expert])

        grads = [paddle.zeros(s, dtype=out_grad.dtype) if s is not None else None for s in ctx.input_shape]
        for i_expert in range(num_local_experts):
            num_tokens = ctx.send_counts_num[i_expert]
            if num_tokens > 0:
                reconstructed_grad = paddle.concat(grad_chunks_by_expert[i_expert], 0)
                if i_expert in ctx.bw_funcs:
                    (final_grad,) = ctx.bw_funcs[i_expert](reconstructed_grad)
                else:
                    final_grad = reconstructed_grad
                if grads[i_expert] is not None:
                    grads[i_expert][:num_tokens] = final_grad

        grads = [g for g in grads if g is not None]
        return tuple(grads) + tuple(router_fn_args_grad)


# Conditionally select the AlltoAllSmart implementation
if paddle.is_compiled_with_xpu():
    AlltoAllSmart = AlltoAllSmartXPU  # noqa: F811


class MOEAllGatherLayerV2(MOELayer):
    """
    MoE Layer with allgather implement.
    """

    def __init__(
        self,
        gate: nn.Layer,
        experts: List[nn.Layer],
        layer_idx,
        shared_experts: Optional[List[nn.Layer]] = None,
        group: Group = None,
        recompute=False,
        k=2,
        enable_reverse_token_drop=False,
        all_to_all_dropout=0,
        group_experts=False,
        use_expert_out_alltoall=True,  #
        use_padding=True,
        dense_token_type=3,  # considerd as dense tokens (no moe)
        moe_statics=None,
        moe_num_experts=None,
    ):
        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            k,
            all_to_all_dropout,
            group_experts,
            moe_statics,
            moe_num_experts,
        )
        self.enable_reverse_token_drop = enable_reverse_token_drop
        self.is_allgather_moe_layer = True
        self.use_padding = use_padding

        # 全局 gate gather
        self.send_rank = None
        self.local_expert_id = None
        self.dense_experts = None
        self.dense_token_type = dense_token_type
        self.capacity_tensor = None
        self.use_expert_out_alltoall = use_expert_out_alltoall
        logger.info(
            f"uisng MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "  # false
            f"use_padding={use_padding}, enable_reverse_token_drop={self.enable_reverse_token_drop}"  # true false
        )
        self.two = paddle.to_tensor(2, dtype=paddle.float32)
        self.zero = paddle.to_tensor(0, dtype=paddle.float32)

    def forward(
        self,
        input: paddle.Tensor,
        token_type_ids=None,
        use_dense_expert=False,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements forward pass for Mixture-of-Experts (MoE) layer with distributed communication.

        Core Functionality:
          - Processes input through gating network to determine expert assignments
          - Performs distributed All-to-All communication for expert computation
          - Combines expert outputs and calculates routing loss

        Key Features:
          1. Supports both dense and sparse expert computation modes
          2. Implements fused gating and dispatch for performance optimization
          3. Handles sequence length padding/unpadding for irregular inputs
          4. Enables communication-computation overlap through asynchronous operations

        Args:
            input (Tensor): Input tensor of shape [seq_len, hidden_dim]
            token_type_ids: Optional segmentation markers for heterogeneous inputs
            use_dense_expert: Flag to enable dense expert computation bypass

        Returns:
            tuple: (
                combined_output: Aggregated expert outputs [seq_len, hidden_dim],
                combine_weights: Expert combination coefficients,
                router_loss: Calculated router balancing loss
            )
        """
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None

        assert len(input.shape) == 2, f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        dispatch_token_type_ids = None
        global_dense_expert_mask = None
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1].reshape([-1])
            dispatch_token_type_ids = token_type_ids
            if self.config.sequence_parallel:
                hcg = fleet.get_hybrid_communicate_group()
                rank = hcg.get_model_parallel_rank()
                interval = token_type_ids.shape[0] // hcg.get_model_parallel_world_size()
                token_type_ids = token_type_ids.slice([0], rank * interval, (rank + 1) * interval)
                token_type_ids.stop_gradient = True

            if use_dense_expert:
                global_dense_expert_mask = dispatch_token_type_ids == self.dense_token_type

        assert self.gate is not None
        if hasattr(self, "rng") and self.rng.random() < self.all_to_all_dropout:
            orig_shape_2 = input.shape
            output = self.forward_experts(input)
            output += self.gate.weight.sum() * 0.0  # hack for grad
            output = output.reshape(orig_shape or orig_shape_2)  # [e*1,c,m]
            return output, None, 0
        (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_no_token_drop,
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits, gate_prob),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        ) = self.fused_gate_and_dispatch(input, token_type_ids, global_dense_expert_mask)
        seqlen_this_mp = input.shape[0]
        if len(scatter_index_rev):
            recv_rank_local = scatter_index_rev // seqlen_this_mp
        else:
            recv_rank_local = scatter_index_rev

        if self.use_padding:
            if self.send_rank is None:
                capacity = self.gate.get_capacity(input.shape[0] * self.config.moe_world_size)
                self.send_rank = (
                    paddle.arange(self.config.moe_world_size)
                    .repeat_interleave(capacity * self.num_local_experts)
                    .astype("int32")  # cap
                )
                self.local_expert_id = (
                    paddle.arange(self.num_local_experts)
                    .repeat_interleave(capacity)
                    .tile(self.config.moe_world_size)
                    .astype(self.send_rank.dtype)
                )
            recv_rank, recv_rank_task = allgather_async(recv_rank_local, group=self.config.moe_group)
            send_rank = self.send_rank
            local_expert_id = self.local_expert_id

        else:
            all_expert_num = sum(expert_num_global_list)
            # 非常慢
            if self.config.moe_group.nranks > 1:
                recv_rank = paddle.empty([all_expert_num], dtype=recv_rank_local.dtype)
                # 非常慢
                recv_rank_task = dist.stream.alltoall_single(
                    recv_rank,
                    recv_rank_local.tile(self.config.moe_world_size),
                    [
                        sum(expert_num_global_list[i * self.num_local_experts : (i + 1) * self.num_local_experts])
                        for i in range(self.config.moe_world_size)
                    ],  # output-size
                    [len(recv_rank_local)] * self.config.moe_world_size,  # input-size
                    group=self.config.moe_group,
                    sync_op=False,
                    use_calc_stream=False,
                )
            else:
                recv_rank_task = None
                recv_rank = recv_rank_local.tile(self.config.moe_world_size)

            # send_rank_cpu = np.concatenate(
            #     [
            #         np.full([j], i // self.num_local_experts, dtype="int32")
            #         for i, j in enumerate(expert_num_global_list)
            #     ],
            #     0,
            # )
            # local_expert_id_cpu = np.concatenate(
            #     [
            #         np.full([j], i % self.num_local_experts, dtype="int32")
            #         for i, j in enumerate(expert_num_global_list)
            #     ],
            #     0,
            # )
            # gpu_ids = paddle.to_tensor(np.stack([send_rank_cpu, local_expert_id_cpu], 0), place="gpu")
            # send_rank, local_expert_id = gpu_ids.unbind(0)
            send_rank, local_expert_id = build_src_rank_and_local_expert_id(
                expert_num_global, expert_num_global_list, self.num_local_experts
            )

        if not self.use_expert_out_alltoall:
            expert_outs = (
                recompute(self.forward_experts, *dispatched_input)
                if self.recompute and self.training
                else self.forward_experts(*dispatched_input)
            )
            expert_outs = paddle.concat([e for e in expert_outs if e is not None], axis=0)  # [e*c,m]
            expert_out_to_combine = AllGatherGroupOp.apply(expert_outs, group=self.config.moe_group)  # for test
            router_loss2 = self.calc_router_loss_and_logging(
                router_loss,
                gate_logits,
                gate_prob,
                gate_logits_mm,
                gate_prob_mm,
                local_combine_weights,
                expert_num_global_no_token_drop,
                token_type_ids,
                dispatch_token_type_ids,
            )
        else:
            recv_rank_task and recv_rank_task.wait()  # wait for recv_rank

            world_size = dist.get_world_size(self.config.moe_group)
            this_rank = dist.get_rank(self.config.moe_group)

            recv_size = paddle.count_nonzero(recv_rank == dist.get_rank(self.config.moe_group))
            recv_size = paddle.maximum(recv_size, paddle.ones([], dtype=recv_size.dtype))

            recv_size_cpu, recv_size_task = async_offload(recv_size, get_async_loader())

            send_rank_this_rank = paddle.count_nonzero(send_rank == this_rank)

            send_rank_this_rank_cpu, send_rank_this_rank_task = async_offload(send_rank_this_rank, get_async_loader())

            recv_rank[recv_rank == -1] = world_size
            send_recv_count_global = paddle.scatter_nd_add(
                paddle.zeros(
                    [self.num_local_experts, world_size + 1, world_size + 1],
                    dtype="int32",
                ),
                paddle.stack([local_expert_id, send_rank, recv_rank], -1),
                paddle.ones([len(send_rank)], dtype="int32"),
            )  # [num_local_experts, world_size + 1 , world_size + 1]
            send_counts_cpu = send_recv_count_global[:, this_rank, :-1].numpy()
            recv_counts_cpu = send_recv_count_global[:, :-1, this_rank].numpy()
            send_counts_num_cpu = send_counts_cpu.sum(-1)
            recv_counts_num_cpu = recv_counts_cpu.sum(-1)

            dispatched_input = self.forward_experts(*dispatched_input)

            if recv_size_task is not None:
                recv_size_task.cpu_wait()
            if send_rank_this_rank_task is not None:
                send_rank_this_rank_task.cpu_wait()

            input_size = sum([len(i) if i is not None else 0 for i in dispatched_input])
            if self.use_padding or input_size > 1:
                assert send_rank_this_rank_cpu.item() == input_size, (
                    send_rank,
                    [len(i) if i is not None else 0 for i in dispatched_input],
                )

            expert_out_to_combine, router_loss2, distributed_input_to_alltoall_out = AlltoAllSmart.apply(
                *dispatched_input,
                router_loss,
                gate_logits,
                gate_prob,
                gate_logits_mm,
                gate_prob_mm,
                local_combine_weights,
                expert_num_global_no_token_drop,
                token_type_ids,
                dispatch_token_type_ids,
                forward_func_dict=None,
                router_loss_fn=self.calc_router_loss_and_logging,
                local_expert_id=local_expert_id,
                send_rank_global=send_rank,
                recv_rank_global=recv_rank,
                num_local_experts=self.num_local_experts,
                capacity=dispatched_input[0].shape[1] if self.use_padding else None,
                use_padding=self.use_padding,
                expert_num_global=expert_num_global_list,
                is_first_fwd=not framework._dygraph_tracer()._has_grad,
                group=self.config.moe_group,
                recv_size=recv_size_cpu,
                send_counts=send_counts_cpu,
                recv_counts=recv_counts_cpu,
                send_counts_num=send_counts_num_cpu,
                recv_counts_num=recv_counts_num_cpu,
            )
            # /origin input -> distributed input/ => /origin-input -> alltoall out -input/
            local_scatter_index = distributed_input_to_alltoall_out[local_scatter_index]
            local_scatter_index.stop_gradient = True
        # global -> local
        combined_output = self.combine_expert_output(expert_out_to_combine, local_combine_weights, local_scatter_index)

        if self.shared_experts is not None:
            shared_out = self.shared_experts(input)
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.reshape(orig_shape[:-1] + [combined_output.shape[-1]])

        return combined_output, local_combine_weights, router_loss2, gate_logits

    def fused_gate_logits_process_fused(self, gate_logits_lm, gate_logits_mm=None, token_type_ids=None):
        """Process gating logits for expert selection in Mixture-of-Experts (MoE) layers.

        Core Functionality:
        - Transforms raw gating logits into expert selection weights and IDs
        - Supports both grouped and standard expert selection modes
        - Handles bias correction for improved expert load balancing

        Args:
            gate_logits_lm (Tensor): Raw gating scores of shape [batch_size, total_experts]

        Returns:
            tuple: (
                lm_weight_and_expert_id: Combined tensor containing selection weights
                       and expert IDs [batch_size, 2*top_k],
                prob_flat: Flattened expert probabilities [batch_size, total_experts]
            )
        """
        top_k = self.k
        num_expert_per_rank_per_modality = gate_logits_lm.shape[-1] // self.config.moe_world_size
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape([gate_logits_lm.shape[0], top_k, -1])
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, axis=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias:
                prob_lm_ = prob_lm + self.moe_statics.e_score_correction_bias[0].detach()
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, axis=-1)

        if self.use_correction_bias:
            batch_idx = paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        expert_id_lm = expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=(num_expert_per_rank_per_modality if token_type_ids is not None else 0),
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = paddle.concat([weight_lm, expert_id_lm.astype("float32")], -1)

        if token_type_ids is None or gate_logits_mm is None:
            return (
                lm_weight_and_expert_id,
                prob_lm.reshape([prob_lm.shape[0], -1]),
                None,
            )

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias:
            prob_mm_ = prob_mm + self.moe_statics.e_score_correction_bias[1].detach()
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, axis=-1)
        if self.use_correction_bias:
            batch_idx = paddle.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = paddle.concat([weight_mm, expert_id_mm.astype("float32")], -1)
        weight_and_expert = paddle.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id,
            mm_weight_and_expert_id,
        )
        return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

    def fused_gate_and_dispatch(self, input, token_type_ids=None, global_dense_expert_mask=None):
        """Implements fused expert gating and token dispatch logic for Mixture-of-Experts (MoE) layers.

        Core Functionality:
          - Computes expert selection probabilities and routing weights
          - Performs distributed token-to-expert assignment
          - Handles communication and synchronization in model-parallel environments

        Args:
            input (Tensor): Input tensor of shape [seq_len, hidden_dim]

        Returns:
            tuple: (
                dispatched_input: Expert-assigned tokens [num_experts, capacity, hidden_dim],
                global_hidden_states: Full sequence representations,
                local_combine_weights: Local expert combination weights,
                expert_num_global_notrunc: Global expert token counts (without capacity truncation),
                expert_num_global: Actual expert token counts,
                expert_num_global_list: Per-expert token counts,
                local_scatter_index: Local token reorganization indices,
                scatter_index_rev: Reverse scattering indices,
                router_loss: Calculated routing loss,
                gate_outputs: Raw gating network outputs,
                expert_num_local: Local expert utilization counts
            )
        """
        seqlen, d_model = input.shape
        args = ()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        router_loss = paddle.zeros([1], dtype="float32")
        router_loss.stop_gradient = False
        top_k = self.k

        def build_weights_and_expert_id(input):
            nonlocal token_type_ids, args
            logits, capacity, router_loss = self.gate(input, *args, transform_weight=False)
            if self.config.multimodel_experts:
                gate_logits_lm, gate_logits_mm = logits.chunk(2, axis=-1)
            else:
                gate_logits_lm, gate_logits_mm = logits, None

            weigth_and_expert, gate_prob_lm, gate_prob_mm = self.fused_gate_logits_process_fused(
                gate_logits_lm,
                gate_logits_mm,
                token_type_ids if global_dense_expert_mask is None else None,
            )
            weigth_and_expert = AllGatherGroupOp.apply(weigth_and_expert, group=self.config.moe_group)
            return (
                weigth_and_expert,
                gate_logits_lm,
                gate_logits_mm,
                gate_prob_lm,
                gate_prob_mm,
            )

        capacity = self.gate.get_capacity(input.shape[0]) * self.world_size
        (
            global_hidden_states,
            combine_weights_and_expert_id,
            gate_logits_lm,
            gate_logits_mm,
            gate_prob_lm,
            gate_prob_mm,
        ) = AllGatherAsync.apply(
            input,
            input,
            fn=build_weights_and_expert_id,
            group=self.config.moe_group,
            is_first_fwd=not framework._dygraph_tracer()._has_grad,
        )
        combine_weights_unnorm, expert_id = combine_weights_and_expert_id.chunk(2, axis=-1)
        expert_id = expert_id.cast("int32")
        expert_id.stop_gradient = True
        num_experts = (
            sum(self.config.moe_num_experts)
            if isinstance(self.config.moe_num_experts, (tuple, list))
            else self.config.moe_num_experts
        )  # all-experts = 96
        if global_dense_expert_mask is not None:
            combine_weights_unnorm[global_dense_expert_mask] = 0.0
            expert_id[global_dense_expert_mask] = num_experts
            num_experts += 1

        if "reverse_token_drop" in inspect.signature(moe_gate_dispatch_partial_nosoftmaxtopk).parameters:
            compat_kwargs = {"reverse_token_drop": self.enable_reverse_token_drop}
        else:
            compat_kwargs = {}

        # Disable AMP because:
        # - combine_weights_unnorm is fp32, global_hidden_states is bf16
        # - AMP O2 would upcast global_hidden_states to fp32, making dispatched_input fp32
        # - This is a data movement op with no computation, so upcasting is unnecessary
        with paddle.amp.auto_cast(False):
            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,  # input -> dispatched_input
                scatter_index_rev,  # dispatch-input -> input
                expert_num_global,
                expert_num_local,
            ) = moe_gate_dispatch_partial_nosoftmaxtopk(
                global_hidden_states,
                combine_weights_unnorm,
                expert_id,
                top_k,
                capacity,
                num_experts,
                self.use_padding,
                expert_start_index=self.num_local_experts * self.config.moe_rank,
                expert_end_index=self.num_local_experts * (self.config.moe_rank + 1),
                **compat_kwargs,
            )

        if self.use_correction_bias:
            if self.gate.config.multimodel_experts:
                # MLLM
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += expert_num_local[self.gate.experts_type_mask[i]].detach()
            else:
                # LLM
                self.moe_statics.expert_usage[0] += expert_num_local.detach()

        # When use unpad , `moe_ops_partial` output likes `scatter_index_rev==[]`.
        if scatter_index_rev.ndim == 0:
            assert not self.use_padding
            scatter_index_rev = paddle.empty([0], dtype=scatter_index_rev.dtype)

        dispatched_input.stop_gradient = False
        combine_weights_unnorm.stop_gradient = False
        scatter_index.stop_gradient = True
        expert_num_global.stop_gradient = True
        expert_num_global_notrunc = expert_num_global
        self.capacity_tensor = paddle.to_tensor(capacity, dtype=expert_num_global.dtype)
        expert_num_global = paddle.minimum(expert_num_global, self.capacity_tensor)

        if global_dense_expert_mask is not None:
            expert_num_global = expert_num_global[:-1]
            expert_num_local = expert_num_local[:-1]
            expert_num_global_notrunc = expert_num_global_notrunc[:-1]

        scatter_index = scatter_index.transpose([1, 0])  # [k,s] ->[s,k]

        last_local_expert = self.num_local_experts * self.config.moe_rank
        expert_offset_global = expert_num_global.cumsum()

        loader = get_async_loader()
        expert_num_global_list, offload_task = async_offload(expert_num_global, loader)
        if self.use_padding:
            offset = last_local_expert * capacity
        else:
            offset = expert_offset_global[last_local_expert - 1] if self.config.moe_rank > 0 else 0
        local_combine_weights_unnorm = ReshardCombineWeight.apply(
            combine_weights_unnorm.contiguous(), group=self.config.moe_group
        )
        local_scatter_index = ReduceScatterGroupOp.apply(
            paddle.where(
                combine_weights_unnorm > 0.0,
                scatter_index + offset,
                scatter_index,
            ),
            group=self.config.moe_group,
        )
        if self.gate.norm_gate_logits:
            local_combine_weights = local_combine_weights_unnorm / paddle.clip(
                local_combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            local_combine_weights = local_combine_weights_unnorm
        local_combine_weights = local_combine_weights.cast(dispatched_input.dtype)
        if self.use_padding:
            dispatched_input = dispatched_input.reshape([self.num_local_experts, -1, d_model])
            dispatched_input = dispatched_input.unbind(0)
        else:
            s = self.num_local_experts * self.config.moe_rank
            e = self.num_local_experts * (self.config.moe_rank + 1)
            expert_num_local = expert_num_local.tolist()[s:e]
            expert_num_local_valid = [i for i in expert_num_local if i > 0]
            valid_pos = [j for j, i in enumerate(expert_num_local) if i > 0]
            if expert_num_local_valid:
                dispatched_input_list = dispatched_input.split(expert_num_local_valid)
                dispatched_input = [None] * len(expert_num_local)
                for p, t in zip(valid_pos, dispatched_input_list):
                    dispatched_input[p] = t
            else:
                dispatched_input = [dispatched_input] + ([None] * (len(expert_num_local) - 1))

        scatter_index.stop_gradient = True
        scatter_index_rev.stop_gradient = True
        if offload_task is not None:
            hack_offload_wait(offload_task)
        expert_num_global_list = expert_num_global_list.tolist()

        return (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_notrunc,  # for auxloss calculation.
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits_lm, gate_prob_lm),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        )

    def forward_experts(self, *dispatched_input):
        """Execute expert model computations in sequence for Mixture-of-Experts (MoE) layer.

        Core Functionality:
          - Distributes dispatched tokens to local expert models
          - Handles empty expert inputs with zero-initialized fallback
          - Maintains gradient flow for expert outputs
          - Aggregates outputs from all active experts

        Args:
            *dispatched_input: Variable-length expert-specific input tensors

        Returns:
            list: Expert output tensors (None for inactive experts)

        Implementation Details:
          1. Processes valid expert inputs through corresponding expert models
          2. Generates dummy inputs for inactive experts to preserve model structure
          3. Aggregates dummy outputs to first active expert to maintain gradient flow
        """
        expert_outputs = []
        assert isinstance(self.experts, nn.LayerList), type(self.experts)

        no_tokens_expert_outputs = []
        if not self.multimodal_experts:
            true_experts = self.experts[self.rank * self.num_local_experts : (self.rank + 1) * self.num_local_experts]
        else:
            true_experts = []
            for i, num in enumerate(self.num_local_multimodal_experts):
                current_modal_experts = self.experts[
                    self.multimodal_expert_index[i] : self.multimodal_expert_index[i + 1]
                ]
                true_experts.extend(current_modal_experts[self.rank * num : (self.rank + 1) * num])

        assert len(dispatched_input) == len(true_experts), (
            len(dispatched_input),
            len(true_experts),
        )

        # experts_weights = []
        for iexpert, chunk in enumerate(dispatched_input):
            if chunk is None:
                # QuantizationLoRALinear can not call `.weight`.
                if not isinstance(true_experts[iexpert].up_proj, QuantizationLoRALinear):
                    input_shape = [
                        1,
                        true_experts[iexpert].down_proj.weight.shape[1],
                    ]
                    input_dtype = true_experts[iexpert].down_proj.weight.dtype
                else:
                    input_shape = [
                        1,
                        true_experts[iexpert].down_proj.lora_B.shape[1],
                    ]
                    input_dtype = true_experts[iexpert].down_proj.lora_B.dtype

                chunk = paddle.zeros(
                    input_shape,
                    input_dtype,
                )
                if true_experts[iexpert].training:
                    chunk.stop_gradient = False
                expert_out = true_experts[iexpert](chunk.contiguous())
                no_tokens_expert_outputs.append(expert_out * 0.0)  # mutiply 0.0 to zero out and grad

                expert_outputs.append(None)
                continue

            expert_out = true_experts[iexpert](chunk.contiguous())
            expert_outputs.append(expert_out)

        # if self.config.moe_layer_feed_fake_token and len(no_tokens_expert_outputs) > 0:
        if len(no_tokens_expert_outputs) > 0:
            first_has_tokens_idx = 0
            for idx, expert_out in enumerate(expert_outputs):
                if expert_out is not None:
                    first_has_tokens_idx = idx
                    break
            for idx, expert_out in enumerate(no_tokens_expert_outputs):
                expert_outputs[first_has_tokens_idx] += expert_out

        return expert_outputs

    def calc_router_loss_and_logging(
        self,
        router_loss,
        gate_logits,
        gate_prob,
        gate_logits_mm,
        gate_prob_mm,
        combine_weights,
        dispatch_mask,
        token_type_ids,
        dispatch_token_type_ids,
    ):
        """Calculate and aggregate router auxiliary loss for Mixture-of-Experts training.

        Core Functionality:
        - Computes expert load balancing loss to prevent expert under-utilization
        - Integrates multiple loss components from different routing stages
        - Maintains gradient flow for routing mechanism optimization

        Args:
            router_loss (Tensor): Accumulated router loss tensor
            gate_logits (Tensor): Raw gating network outputs [batch_size, num_experts]
            gate_prob (Tensor): Activated gating probabilities [batch_size, num_experts]
            combine_weights (Tensor): Expert combination weights [batch_size, top_k]
            dispatch_mask (Tensor): Token dispatch mask indicating expert assignments

        Returns:
            Tensor: Updated router loss with new auxiliary components
        """
        dispatch_mask_3d = dispatch_mask.reshape([self.config.moe_world_size, -1])
        if token_type_ids is not None and self.gate.config.moe_use_hard_gate:
            # MLLM
            if not self.gate.weight.stop_gradient:
                dispatch_tokens_mask = dispatch_token_type_ids == 0 if dispatch_token_type_ids is not None else None
                lm_tokens_mask = (token_type_ids == 0).astype(gate_prob.dtype)
                # hard code
                lm_experts = (
                    self.gate.num_experts[0]
                    if isinstance(self.gate.num_experts, (tuple, list))
                    else self.gate.num_experts
                )
                dispatch_mask_lm = dispatch_mask_3d[:, : lm_experts // self.config.moe_world_size].reshape([-1])
                router_loss += self._calc_router_loss(
                    dispatch_mask_lm,
                    gate_logits * lm_tokens_mask.unsqueeze(-1),
                    gate_prob * lm_tokens_mask.unsqueeze(-1),
                    self.gate.num_experts_list[0],
                    self.group_experts,
                    self.layer_idx,
                    0,  # ortholoss
                    lm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="lm",
                )
            else:
                router_loss += self.zero * gate_logits[0, 0] * gate_prob[0, 0]
            if gate_prob_mm is not None:
                mm_tokens_mask = (token_type_ids == 1).astype(gate_prob_mm.dtype)
                dispatch_tokens_mask = dispatch_token_type_ids == 1 if dispatch_token_type_ids is not None else None
                dispatch_mask_mm = dispatch_mask_3d[
                    :, self.gate.num_experts[0] // self.config.moe_world_size :
                ].reshape([-1])

                router_loss += self._calc_router_loss(
                    dispatch_mask_mm,
                    gate_logits_mm * mm_tokens_mask.unsqueeze(-1),
                    gate_prob_mm * mm_tokens_mask.unsqueeze(-1),
                    self.gate.num_experts_list[1],
                    False,
                    self.layer_idx,
                    1,
                    mm_tokens_mask,
                    dispatch_tokens_mask,
                    prefix="mm",
                )

        else:
            # LLM
            router_loss += self._calc_router_loss(
                dispatch_mask,
                gate_logits,
                gate_prob,
                self.gate.num_experts_tensor,
                self.group_experts,
                self.layer_idx,
                0,
                paddle.ones([gate_prob.shape[0]], "bool"),
                paddle.ones([self.gate.config.moe_world_size * gate_prob.shape[0]], "bool"),
                prefix="lm",
            )

        return router_loss
