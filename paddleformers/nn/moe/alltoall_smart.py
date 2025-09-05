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

from paddleformers.transformers.ernie4_5.distributed.common_dist_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    all_gather_group,
    get_async_loader,
    hack_offload_wait,
    reduce_scatter_group,
)

from .utils import manual_backward

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
                input_local_expert = inputs[i_local_expert].slice(
                    (0,), 0, send_counts_num[i_local_expert]
                )
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
                output_local_expert = output._slice(
                    output_ptr, (output_ptr + recv_counts_num[i_local_expert])
                )
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
        ctx.router_loss_bwfn, (router_loss,) = manual_backward(
            router_loss_fn, is_first_fwd, *router_loss_args
        )
        with paddle.no_grad():
            recv_mask = (recv_rank_global == this_rank).astype(send_rank_global.dtype)
            if ctx.use_padding:
                recv_mask_alltoall_out = (
                    recv_mask.reshape([-1, num_local_experts, capacity])
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.view(
                        [num_local_experts, -1, capacity]
                    )
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
            else:
                recv_mask_alltoall_out = recv_mask.split(
                    expert_num_global
                )  # h->d copy break overlap
                recv_mask_alltoall_out = [
                    recv_mask_alltoall_out[
                        (iexpert % world_size) * num_local_experts
                        + (iexpert // world_size)
                    ]
                    for iexpert in range(world_size * num_local_experts)
                ]
                alltoall_shape = [i.shape[0] for i in recv_mask_alltoall_out]

                recv_mask_alltoall_out = paddle.concat(recv_mask_alltoall_out, 0)
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.split(alltoall_shape)
                )

                distributed_input_to_alltoall_out = paddle.concat(
                    [
                        distributed_input_to_alltoall_out[
                            (iexpert % num_local_experts) * world_size
                            + (iexpert // num_local_experts)
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

        grads = [
            paddle.zeros(s, dtype=out_grad.dtype) if s is not None else None
            for s in ctx.input_shape
        ]
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
                out_g = out_grad.slice(
                    (0,), out_ptr, out_ptr + recv_counts_num[i_local_expert]
                )
            else:
                out_g = (
                    ctx.dummy_input
                )  # paddle.empty([0,]+out_grad.shape[1:], dtype=out_grad.dtype)
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
            input_dtype = first_expert.up_gate_proj.weight.dtype
            hidden_size = first_expert.up_gate_proj.weight.shape[0]
            input_shape = [0, hidden_size]

        dummy_input = paddle.empty([0] + input_shape[1:], dtype=input_dtype)
        ctx.dummy_input = dummy_input
        ctx.bw_funcs = {}

        processed_inputs = []
        no_tokens_expert_outputs = []

        for i_local_expert in range(num_local_experts):
            if send_counts_num[i_local_expert] > 0:
                input_local_expert = inputs[i_local_expert].slice(
                    (0,), 0, send_counts_num[i_local_expert]
                )
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
                    [1, expert_func.up_gate_proj.weight.shape[0]],
                    dtype=expert_func.up_gate_proj.weight.dtype,
                )
                if expert_func.training:
                    fake_chunk.stop_gradient = False

                _, (expert_out,) = manual_backward(
                    expert_func, is_first_fwd, fake_chunk
                )

                no_tokens_expert_outputs.append(expert_out * 0.0)

        all_processed_inputs = (
            paddle.concat(processed_inputs, axis=0) if processed_inputs else dummy_input
        )

        if no_tokens_expert_outputs:
            if all_processed_inputs.shape[0] > 0:
                all_processed_inputs[0] = all_processed_inputs[0] + sum(
                    no_tokens_expert_outputs
                )
            else:
                router_loss_args = list(router_loss_args)
                router_loss_args[0] = (
                    router_loss_args[0] + sum(no_tokens_expert_outputs).mean() * 0.0
                )

        in_tensors_by_rank = [[] for _ in range(world_size)]
        processed_input_ptr = 0
        for i_local_expert in range(num_local_experts):
            num_tokens = send_counts_num[i_local_expert]
            if num_tokens > 0:
                expert_input = all_processed_inputs.slice(
                    [0], processed_input_ptr, processed_input_ptr + num_tokens
                )
                processed_input_ptr += num_tokens
                splits = expert_input.split(
                    send_counts[i_local_expert].tolist(), axis=0
                )
                for j_rank in range(world_size):
                    in_tensors_by_rank[j_rank].append(splits[j_rank])

        in_tensor_list = [
            paddle.concat(tensors, 0) if tensors else dummy_input
            for tensors in in_tensors_by_rank
        ]

        all_to_all_input = paddle.concat(in_tensor_list, 0)
        send_counts_for_api = [t.shape[0] for t in in_tensor_list]

        recv_counts_tensor = paddle.to_tensor(recv_counts)
        recv_counts_for_api = [
            int(recv_counts_tensor[:, j_rank].sum()) for j_rank in range(world_size)
        ]
        temp_output = paddle.empty(
            [recv_size.item()] + input_shape[1:], dtype=input_dtype
        )

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

        ctx.router_loss_bwfn, (router_loss,) = manual_backward(
            router_loss_fn, is_first_fwd, *router_loss_args
        )
        with paddle.no_grad():
            recv_mask = (recv_rank_global == this_rank).astype(send_rank_global.dtype)
            if ctx.use_padding:
                recv_mask_alltoall_out = (
                    recv_mask.reshape([-1, num_local_experts, capacity])
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.view(
                        [num_local_experts, -1, capacity]
                    )
                    .transpose([1, 0, 2])
                    .reshape([-1])
                )
            else:
                recv_mask_alltoall_out = recv_mask.split(expert_num_global)
                recv_mask_alltoall_out = [
                    recv_mask_alltoall_out[
                        (iexpert % world_size) * num_local_experts
                        + (iexpert // world_size)
                    ]
                    for iexpert in range(world_size * num_local_experts)
                ]
                alltoall_shape = [i.shape[0] for i in recv_mask_alltoall_out]
                recv_mask_alltoall_out = paddle.concat(recv_mask_alltoall_out, 0)
                distributed_input_to_alltoall_out = paddle.maximum(
                    recv_mask_alltoall_out.cumsum() - 1,
                    paddle.zeros([1], dtype=recv_mask_alltoall_out.dtype),
                )
                distributed_input_to_alltoall_out = (
                    distributed_input_to_alltoall_out.split(alltoall_shape)
                )
                distributed_input_to_alltoall_out = paddle.concat(
                    [
                        distributed_input_to_alltoall_out[
                            (iexpert % num_local_experts) * world_size
                            + (iexpert // num_local_experts)
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
            paddle.concat(tensors, 0) if tensors else dummy_input
            for tensors in in_tensors_by_rank_bw
        ]

        all_to_all_grad_input = paddle.concat(in_tensor_list_bw, 0)
        send_counts_bw_for_api = [t.shape[0] for t in in_tensor_list_bw]

        recv_counts_bw = ctx.send_counts
        recv_counts_tensor_bw = paddle.to_tensor(recv_counts_bw)
        recv_counts_bw_for_api = [
            int(recv_counts_tensor_bw[:, j_rank].sum()) for j_rank in range(world_size)
        ]
        total_output_grad_size = int(ctx.send_counts_num.sum())
        temp_grad_output = paddle.empty(
            [total_output_grad_size] + list(out_grad.shape[1:]), dtype=out_grad.dtype
        )

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
            expert_chunks_from_j = data_from_j.split(
                recv_counts_bw[:, j_rank].tolist(), 0
            )
            for i_expert in range(num_local_experts):
                grad_chunks_by_expert[i_expert].append(expert_chunks_from_j[i_expert])

        grads = [
            paddle.zeros(s, dtype=out_grad.dtype) if s is not None else None
            for s in ctx.input_shape
        ]
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
    AlltoAllSmart = AlltoAllSmartXPU