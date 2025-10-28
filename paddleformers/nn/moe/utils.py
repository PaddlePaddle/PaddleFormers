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

from typing import Any, Callable, List, Union

import paddle
from paddle import distributed as dist
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.group import _get_global_group
from paddle.incubate.tensor.manipulation import create_async_load

from ...utils.log import logger


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
    output = paddle.cat(outputs, axis=axis)
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


def _parse_moe_group(
    moe_group: str,
) -> Union[str, paddle.distributed.communication.group.Group]:
    """Parse and initialize the MoE (Mixture of Experts) communication group.

    Converts string representation of MoE group into actual process group
    for distributed expert parallelism.

    Args:
        moe_group (str): Specifies the type of parallel group to use for MoE.
            Supported values:
            - "data" or "dp": Data parallel group
            - "mp", "model" or "tp": Model parallel group
            - "dummy": Dummy group for single process
            - "none", "world" or "all": Global communication group

    Returns:
        Union[str, paddle.distributed.communication.group.Group]:
            The corresponding process group object, or dummy group string.
            Returns dummy group for single-process case.
    """
    moe_group = moe_group.lower()
    assert moe_group in {
        "data",
        "dp",
        "mp",
        "tp",
        "model",
        "dummy",
        "none",
        "world",
        "all",
    }, f"moe-group not supported, got: {moe_group}"
    logger.info(f"using moe-group: {moe_group}")
    if moe_group in {"data", "dp"}:
        moe_group = fleet.get_hybrid_communicate_group().get_data_parallel_group()
    elif moe_group in {"mp", "model", "tp"}:
        try:
            moe_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            # (LiuTing): multi-gpu but tp=1
            # need use dummy group for `moe_gate_dispatch_partial_nosoftmaxtopk` kernel.
            if moe_group.nranks <= 1:
                moe_group = paddle.distributed.communication.group.Group(0, None, [0])
        except:
            # (LiuTing): just single-gpu
            moe_group = paddle.distributed.communication.group.Group(0, None, [0])

    elif moe_group in {"dummy"}:
        dummy_group = paddle.distributed.communication.group.Group(0, None, [0])
        moe_group = dummy_group
    else:
        moe_group = _get_global_group()

    return moe_group
