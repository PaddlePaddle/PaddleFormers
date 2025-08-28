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
Common distributed utils.
"""

from typing import Any, Callable, List, Optional

import paddle
from paddle import distributed as dist
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.incubate.tensor.manipulation import create_async_load


def get_hcg():
    """
    Get hybrid communicate group.
    """
    return fleet.get_hybrid_communicate_group()


def scatter_axis(input, group=None, axis=0):
    """
    Uniformly splits the `input` along dimension 0 across model parallel groups.
    This API is not related to `distributed.scatter`.

    Args:
        input: Input tensor to be split
        group: Communication group for parallel processing (default: model parallel group)
        axis: Dimension along which to split (default: 0)

    Returns:
        A slice of the input tensor corresponding to this rank's portion
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    rank = group.rank
    seq_len = input.shape[axis]
    assert seq_len % parallelism == 0, (
        f"Input sequence length {seq_len} can't be divided exactly" f" by sequence parallelism {parallelism}"
    )
    interval = seq_len // parallelism
    input = paddle.slice(input, axes=[axis], starts=[interval * rank], ends=[interval * (rank + 1)])
    # slice uses stride, so we maintain the memory of whole input, use assign to free the whole input
    # which can avoid OOM.
    input = paddle.assign(input)
    return input


class ReduceScatterGroupOp(PyLayer):
    """
    Perform group reduce scatter.
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """Forward pass: Reduce-Scatter operation
        Args:
            input (Tensor):  Input tensor with shape [s, b, h].
                            The 's' dimension will be split across model parallel group.
            group (ProcessGroup): Model parallel process group,
                                uses global group by default.
        Returns:
            Tensor: Output tensor after Reduce-Scatter with shape [s/n, b, h],
                   each device holds partial data of the original input.
        """
        ctx.group = group
        return reduce_scatter_group(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        """Backward pass: All-Gather operation
        Args:
            grad (Tensor): Upstream gradient with shape [s/n, b, h]
        Returns:
            Tensor: Full gradient after All-Gather with restored shape [s, b, h],
                   aggregating gradients from all devices in model parallel group.
        """
        return all_gather_group(grad, group=ctx.group)


class AllGatherGroupOp(PyLayer):
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
        return reduce_scatter_group(grad, group=ctx.group)


def get_async_loader():
    """get_async_loader"""
    global async_loader
    if not hasattr(fleet.fleet, "_hcg"):
        if async_loader is None:
            async_loader = create_async_load()
        return async_loader

    hcg = get_hcg()
    if not hasattr(hcg, "async_loader"):
        hcg.async_loader = create_async_load()
    return hcg.async_loader


def hack_offload_wait(task):
    """hack_offload_wait"""
    task.cpu_wait()


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
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    if axis == 0:
        output_shape[axis] = output_shape[axis] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        dist.stream.all_gather(output, input, group=group, use_calc_stream=True)
        return output
    outputs = [paddle.empty(output_shape, dtype=input.dtype) for _ in range(parallelism)]
    dist.stream.all_gather(outputs, input, group=group, use_calc_stream=True)
    output = paddle.concat(outputs, axis=axis)
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
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), f"Input sequence length {input.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(output, input, op=dist.ReduceOp.SUM, group=group, use_calc_stream=True)
    return output


class ScatterOp(PyLayer):
    """
    Each rank slices its own portion from the **same** sequence (uniformly split).
    During backward pass, gradients from all ranks are aggregated to restore
    the mp (model parallelism) synchronization state.
    The inverse operation is `GatherOp`.

    input: Tensor [S,*]

    Note: Not related to `distributed.scatter`.
    """

    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        """forward"""
        ctx.axis = axis
        ctx.group = group
        return scatter_axis(input, axis=axis, group=ctx.group)

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return all_gather_group(grad, axis=ctx.axis, group=ctx.group)


def detach_and_requires_grad_(*args):
    """
    Detach tensors while preserving their requires_grad status.

    Args:
        args: Input tensors

    Returns:
        list: Detached tensors
    """
    ret = [a.detach() if a is not None else None for a in args]
    for r, a in zip(ret, args):
        if a is not None:
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    """
    Fake clone operation that preserves computation graph without data copy.
    """

    @staticmethod
    def forward(ctx, input):
        """
        Create fake clone of input tensor.

        Args:
            input: Input tensor

        Returns:
            Tensor: Fake cloned tensor
        """
        if input.is_contiguous():
            fake_output = paddle.empty_like(input)
            input._share_buffer_to(fake_output)
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for fake clone.

        Args:
            grad_output: Gradient of output

        Returns:
            Tensor: Gradient of input
        """
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: List[Any]):
    """
    Perform manual backward pass with gradient tracing control.

    Args:
        f: Function to execute
        is_first_fwd: Whether this is the first forward pass
        args: Arguments for the function

    Returns:
        tuple: (backward function, function outputs)
    """
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True  # turn on grad trace so we can manual backward

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [FakeClone.apply(a) if a is not None else None for a in detached_args]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [FakeClone.apply(o) for o in out if o is not None]  # do not cache stop_gradient output

    for o in out_cached:
        o._clear_dataptr()  # free mem
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        # out 中的 stop_graident 参数，也会收到 gradient，在这里过滤掉
        grad, out_cached = zip(*[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient])

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)
        # out, grad = zip(*[(o, g) for o, g in zip(out, grad) if g is not None])
        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if t is not None])

    return bwd_f, out

def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
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
    token_indices = paddle.arange(num_tokens).unsqueeze(0).expand([num_experts, -1])
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
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    output_tokens.put_along_axis_(
        axis=0,
        indices=sorted_indices.unsqueeze(1).expand([-1, hidden]),
        values=permuted_tokens,
        reduce="add",
        include_self=True,
    )
    return output_tokens