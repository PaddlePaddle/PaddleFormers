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

# Refer to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


import paddle
import paddle.distributed as dist
from paddle.distributed.communication.reduce_scatter import _reduce_scatter_base

from paddleformers.fleet.parallel_state import get_global_memory_buffer
from paddleformers.fleet.tensor_parallel.utils import split_tensor_along_last_dim
from paddleformers.fleet.utils import get_tensor_model_parallel_group_if_none


def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    assert group is not None, "group should not be None"

    # Bypass the function if we are using only 1 GPU.
    if group.world_size == 1:
        return input_

    # All-reduce.
    paddle.distributed.all_reduce(input_.contiguous(), group=group)

    return input_


def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    assert group is not None, "group should not be None"

    world_size = len(group.ranks)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    rank = group.rank
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    assert group is not None, "group should not be None"

    world_size = group.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.shape[0]
    assert dim_size % world_size == 0, "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = group.rank
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_, group):
    """Gather tensors and concatenate along the last dimension."""

    world_size = group.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.shape)
    dim_size[0] = dim_size[0] * world_size

    tensor_list = []
    dist.all_gather(tensor_list, input_.contiguous(), group=group)

    output = paddle.concat(tensor_list, dim=-1).contiguous()

    return output


def _reduce_scatter_along_last_dim(input_, group):
    """Reduce-scatter tensors on the last dimension."""

    world_size = group.world_size
    target_shape = list(input_.shape)
    assert (
        target_shape[-1] % world_size == 0
    ), f"input_.shape[-1] {target_shape[-1]} should be divisible by world_size {world_size}"
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = paddle.split(input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1)
    concat_tensor = paddle.concat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor, group=group).reshape(target_shape)
    return output


def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (paddle.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        paddle.Tensor: Gathered tensor.
    """

    assert group is not None, "group should not be None"
    world_size = group.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output_tensor_list = []
    dist.all_gather(output_tensor_list, input_.contiguous(), group=group)
    output = paddle.concat(output_tensor_list, dim=0)

    return output


def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (paddle.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    assert group is not None, "group should not be None"
    world_size = group.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.shape)
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = paddle.empty(dim_size, dtype=input_.dtype)
        _reduce_scatter_base(output, input_.contiguous(), group=group)
    else:
        rank = group.rank
        input_tensor_list = list(paddle.split(input_, input_split_sizes, dim=0))

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(input_tensor_list[rank].shape, input_.dtype, "mpu")
        else:
            output = paddle.empty_like(input_tensor_list[rank])
        paddle.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output


class _CopyToModelParallelRegion(paddle.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _reduce(grad_output, ctx.group)


class _ReduceFromModelParallelRegion(paddle.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        if group is None or group.nranks <= 1:
            return input_
        return _reduce(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        if group is None or group.nranks <= 1:
            return input_
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output


class _ScatterToModelParallelRegion(paddle.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _gather_along_last_dim(grad_output, ctx.group)


class _GatherFromModelParallelRegion(paddle.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _split_along_last_dim(grad_output, ctx.group)


class _ScatterToSequenceParallelRegion(paddle.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        return _split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _gather_along_first_dim(grad_output, ctx.group)


class _GatherFromSequenceParallelRegion(paddle.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def symbolic(
        graph,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        if group is None:
            return input_
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(
                grad_output,
                ctx.group,
                ctx.output_split_sizes,
                ctx.use_global_buffer,
            )
        else:
            assert ctx.output_split_sizes is None
            return _split_along_first_dim(grad_output, ctx.group)


class _ReduceScatterToSequenceParallelRegion(paddle.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group, input_split_sizes=None, use_global_buffer=False):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, group, input_split_sizes=None, use_global_buffer=False):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        ctx.input_split_sizes = input_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        input_split_sizes = ctx.input_split_sizes
        use_global_buffer = ctx.use_global_buffer
        return _gather_along_first_dim(grad_output, ctx.group, input_split_sizes, use_global_buffer)


class _AllGatherFromTensorParallelRegion(paddle.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _reduce_scatter_along_last_dim(grad_output, ctx.group)


class _ReduceScatterToTensorParallelRegion(paddle.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        if group is None:
            return input_
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        if ctx.group is None:
            return grad_output
        return _gather_along_last_dim(grad_output, ctx.group)


class _AllToAll(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = paddle.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes), *list(input.size()[1:])],
                dtype=input.dtype,
            )
        dist.all_to_all_single(
            output,
            input,
            in_split_sizes=input_split_sizes,
            out_split_sizes=output_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.input_split_sizes,
                ctx.output_split_sizes,
            ),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_, group=None, is_expert=False):
    """Wrapper for autograd function: forward: copy, backward allreduce"""
    group = get_tensor_model_parallel_group_if_none(group, is_expert)
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_, group=None, is_expert=False):
    """Wrapper for autograd function: forward: all reduce, backward copy"""
    group = get_tensor_model_parallel_group_if_none(group, is_expert)
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward: split <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromModelParallelRegion.apply(input_, group)


def scatter_to_sequence_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: split, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_,
        group,
        tensor_parallel_output_grad,
        output_split_sizes,
        use_global_buffer,
    )


def reduce_scatter_to_sequence_parallel_region(input_, group=None, input_split_sizes=None, use_global_buffer=False):
    """Wrapper for autograd function: forward: RS, backward AG <first dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToSequenceParallelRegion.apply(input_, group, input_split_sizes, use_global_buffer)


def all_gather_last_dim_from_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward RS <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _AllGatherFromTensorParallelRegion.apply(input_, group)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward AG: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToTensorParallelRegion.apply(input_, group)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """Wrapper for autograd function"""
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)


def all_to_all_sp2hp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (paddle.Tensor):
            The input tensor which has been distributed along the sequence
            dimension.
        group (paddle.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        paddle.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    assert (
        input_.shape[-1] % world_size
    ), f"input last dim ({input_.shape[-1]}) must be divisible by world size ({world_size})"

    split_tensors = paddle.split(input_, num_or_sections=world_size, dim=1)
    concat_tensor = paddle.cat(split_tensors, dim=0)
    output = all_to_all(group, concat_tensor)
    return output


def all_to_all_hp2sp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (paddle.Tensor):
            The input tensor which has been distributed along the hidden
            dimension.
        group (paddle.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        paddle.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    input_exchanged = all_to_all(group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    assert (
        input_reshaped.shape[0] % world_size == 0
    ), f"input first dim ({input_reshaped.shape[0]}) must be divisible by world size ({world_size})"
    split_tensors = paddle.split(input_reshaped, num_or_sections=world_size, dim=0)
    output = paddle.concat(split_tensors, dim=-1)
    return output
