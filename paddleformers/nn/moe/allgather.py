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
    task = dist.stream.all_gather(
        output, input, group=group, use_calc_stream=False, sync_op=False
    )
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

