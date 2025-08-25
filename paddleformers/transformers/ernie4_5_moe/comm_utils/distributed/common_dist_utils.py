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
Common distributed utils
"""

import paddle
import paddle.nn.functional as F
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    ColumnSequenceParallelLinear,
    GatherOp,
    ReduceScatterOp,
    RowSequenceParallelLinear,
    ScatterOp,
    all_gather,
    mark_as_sequence_parallel_parameter,
    scatter,
)
from paddle.incubate.tensor.manipulation import create_async_load

# from ..refined_recompute.utils import RefinedRecomputeFunction

__all__ = [
    "get_hcg",
    "_parallel_matmul",
    "scatter_axis",
    "mp_slice",
    "all_gather_varlen",
    "ColumnParallelLinear",
    "ColumnSequenceParallelLinear",
    "RowParallelLinear",
    "RowSequenceParallelLinear",
    "GatherOp",
    "ScatterOp",
    "mark_as_sequence_parallel_parameter",
    "RRColumnSequenceParallelLinear",
    "RRRowSequenceParallelLinear",
    "AllGatherVarlenOp",
    "sequence_parallel_sparse_mask_labels",
    "get_async_loader",
    "hack_offload_wait",
    "hack_reload_wait",
    "all_gather_group",
    "reduce_scatter_group",
]


def get_hcg():
    """
    Get hybrid communicate group.
    """
    return fleet.get_hybrid_communicate_group()


def _parallel_matmul(
    x,
    y,
    bias=None,
    transpose_y=False,
    tensor_parallel_degree=1,
    tensor_parallel_output=True,
    fuse_linear=False,
):
    """
    Performs parallel matrix multiplication with tensor model parallelism support.

    Args:
        x (paddle.Tensor): Input tensor with shape [batch_size, seq_len, hidden_size]
        y (Union[paddle.Tensor, EagerParamBase]): Weight matrix which can be:
            - Regular tensor
            - Distributed parameter in tensor parallel mode
        bias (Optional[paddle.Tensor]): Optional bias tensor
        transpose_y (bool): Whether to transpose the 'y' matrix before multiplication
        tensor_parallel_degree (int): Degree of tensor model parallelism (default: 1)
        tensor_parallel_output (bool): Whether to keep output in tensor parallel format
            or gather across devices (default: True)
        fuse_linear (bool): Whether to use fused linear operation for optimization

    Returns:
        paddle.Tensor

    Raises:
        AssertionError: If tensor parallel is enabled but weight is not distributed
        AttributeError: If called without distributed.launch context
    """
    if tensor_parallel_degree > 1:
        if isinstance(y, paddle.base.framework.EagerParamBase):
            assert y.is_distributed
        # if not running under distributed.launch, it will raise AttributeError: 'Fleet' object has no attribute '_hcg'
        pg = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        input_parallel = paddle.distributed.collective._c_identity(x, group=pg)

        if transpose_y:
            logits = paddle.matmul(input_parallel, y, transpose_y=True)
            if bias is not None:
                logits += bias
        else:
            if fuse_linear:
                logits = paddle.incubate.nn.functional.fused_linear(input_parallel, y, bias)
            else:
                logits = F.linear(input_parallel, y, bias)

        if tensor_parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(logits, group=pg)

    else:
        if fuse_linear:
            logits = paddle.incubate.nn.functional.fused_linear(x, y, bias, transpose_weight=transpose_y)
        else:
            logits = paddle.matmul(x, y, transpose_y=transpose_y)
            if bias is not None:
                logits += bias
        return logits


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


def mp_slice(x, indices=None, group=None, axis=0):
    """
    Slices tensor `x` along dimension 0 according to `indices` without communication.

    Args:
        x: Input tensor to be sliced
        indices: List of indices defining how to slice the tensor
        group: Communication group for parallel processing (default: model parallel group)
        axis: Dimension along which to slice (default: 0)

    Returns:
        A slice of the input tensor corresponding to this rank's portion
    """
    if indices is None:
        return scatter(x, group, axis)
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return x
    rank = group.rank
    assert len(indices) == parallelism, (len(indices), parallelism)
    indices = F.pad(paddle.to_tensor(indices).cumsum(0), [1, 0])
    input = paddle.slice(x, axes=[axis], starts=[indices[rank]], ends=[indices[rank + 1]])
    input = paddle.assign(input)
    return input


def all_gather_varlen(input, indices, group=None, axis=0, sync_op=True):
    """
    Variable-length version of `all_gather` that behaves similarly to `distributed.all_gather`.

    Args:
        input: Local tensor to be gathered
        indices: List of sizes from each rank indicating how much to gather from each
        group: Communication group for parallel processing (default: model parallel group)
        axis: Dimension along which to gather (only 0 is supported)
        sync_op: Whether to synchronize the operation

    Returns:
        A concatenated tensor containing all gathered data
    """
    assert axis == 0, "only support axis=0"
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    input_sizes = [len(input)] * parallelism
    output_sizes = indices
    out = paddle.empty([sum(indices)] + input.shape[1:], dtype=input.dtype)
    task = dist.stream.alltoall_single(
        out,
        paddle.concat([input] * parallelism, 0) if len(input) else input,  # TODO: check this
        output_sizes,  # input-size
        input_sizes,
        group=group,
        sync_op=sync_op,
        use_calc_stream=sync_op,
    )
    task.wait()
    return out


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


class RRColumnSequenceParallelLinear(ColumnSequenceParallelLinear):
    """
    ColumnSequenceParallelLinear with refined recompute.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=None,
        gather_output=True,
        fuse_matmul_bias=False,
        mp_group=None,
        use_rr=False,
        name=None,
    ):
        """
        Initializes a ColumnSequenceParallelLinear module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            weight_attr (ParamAttr, optional): The parameter attribute for the learnable
                weight matrix. Default: None.
            has_bias (bool, optional): Whether the layer uses a bias. By default, it is set to False.
                If ``has_bias`` is set to False, no bias term is used. If ``has_bias`` is set to True,
                a bias vector is used. Default: None, which means inherit the value of `has_bias`
                from the current instance's `has_bias`.
            gather_output (bool, optional): Whether to gather all outputs from all ranks during forward pass.
                Default: True. If True, all outputs from all ranks are gathered during forward pass, which
                makes sure that each example's output is produced only once. If False, all outputs are
                produced on each rank separately, and the outputs from different ranks may overlap.
                This can save communication time but may cause slower convergence. Default: True.
            fuse_matmul_bias (bool, optional): Whether to fuse matmul and bias into one op. Default: False.
            mp_group (paddle.distributed.Group, optional): The group for model parallel. Default: None.
            use_rr (bool, optional): Whether to use refined rcompute. Default: False.
            name (str, optional): Name for the instance to use in tracebacks. Default: None.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            weight_attr=weight_attr,
            has_bias=has_bias,
            gather_output=gather_output,
            fuse_matmul_bias=fuse_matmul_bias,
            mp_group=mp_group,
            name=name,
        )

        # self._rr_column_ln = RefinedRecomputeFunction() if use_rr else None
        self._rr_column_ln = None
        if self.weight.is_distributed:
            self.weight.split_axis = 1
        if has_bias and self.bias.is_distributed:
            self.bias.split_axis = 0

    def forward(self, x):
        """
        Forward pass function that computes the product of the input tensor and model parameters.

        Args:
            x (paddle.Tensor): Input tensor with shape (batch_size, seq_len, hidden_size) or (batch_size, hidden_size).
            If sequence parallel is True, the shape is (seq_len, batch_size, hidden_size).

        Returns:
            paddle.Tensor: Returns a tensor with shape (batch_size, seq_len, hidden_size) or (batch_size, hidden_size).
            If sequence parallel is True, the shape is (seq_len, batch_size, hidden_size).
        """
        # sequence parallelism is same as model parallelism
        # if sequence parallel is true, input shape is [s, b, h]
        # else input shape is [b, s, h]
        if self.is_mp:
            input_parallel = AllGatherOp.apply(x)
        else:
            input_parallel = x

        if self._rr_column_ln is not None and self.training:  # in eval mode, do not use refined recompute
            output = self._rr_column_ln(
                self.linear,
                x=input_parallel,
                weight=self.weight,
                bias=self.bias,
            )
        else:
            output = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        return output


class RRRowSequenceParallelLinear(RowSequenceParallelLinear):
    """
    RowSequenceParallelLinear with refined recompute.
    """

    def __init__(
        self,
        in_features,
        out_features,
        weight_attr=None,
        has_bias=True,
        input_is_parallel=False,
        fuse_matmul_bias=False,
        mp_group=None,
        use_rr=False,
        name=None,
    ):
        """
        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            weight_attr (ParamAttr, optional): The parameter attribute for the learnable
                weight matrix. Defaults to None. If it is None, the system will
                generate a default Attribute object.
            has_bias (bool, optional): Whether the layer uses a bias term. Defaults to True.
            input_is_parallel (bool, optional): Whether the input is parallel. Defaults to False.
            fuse_matmul_bias (bool, optional): Whether to fuse matmul and bias into one kernel. Defaults to False.
            mp_group (Group, optional): Model parallel group. Defaults to None.
            use_rr (bool, optional): Whether to use refined rr. Defaults to False.
            name (str, optional): Name of the layer. Defaults to None.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            weight_attr=weight_attr,
            has_bias=has_bias,
            input_is_parallel=input_is_parallel,
            fuse_matmul_bias=fuse_matmul_bias,
            mp_group=mp_group,
            name=name,
        )

        # self._rr_row_ln = RefinedRecomputeFunction() if use_rr else None
        self._rr_row_ln = None

        if self.weight.is_distributed:
            self.weight.split_axis = 0

    def forward(self, x):
        """
        Forward pass function that computes the product of the input tensor and model parameters.

        Args:
            x (paddle.Tensor): Input tensor with shape (batch_size, in_features).

        Returns:
            paddle.Tensor: Returns a tensor with shape (batch_size, out_features).
        """
        input_parallel = x
        if self.is_mp:
            if self.mp_scale is not None:
                bias = self.mp_scale(self.bias, self.world_size)
            else:
                bias = None

            def linear_reduce_scatter(input, weight, bias=None, name=None):
                output = self.linear(input, weight=weight, bias=bias, name=name)
                return ReduceScatterOp.apply(output)

            if self._rr_row_ln is not None and self.training:  # in eval mode, do not use refined recompute
                output_ = self._rr_row_ln(
                    linear_reduce_scatter,
                    input_parallel,
                    self.weight,
                    bias=bias,
                    name=self._name,
                )
            else:
                output_ = linear_reduce_scatter(input_parallel, self.weight, bias=bias, name=self._name)

            # if self.bias is not none, sequence parallel will use
            # register_hook to all_reduce self.bias
            if bias is None and self.bias is not None:
                output = output_ + self.bias
            else:
                output = output_
        else:
            output = self.linear(input_parallel, self.weight, self.bias, name=self._name)
        return output


class AllGatherVarlenOp(PyLayer):
    """
    A custom PyLayer that performs variable-length allgather operation.

    This operation handles tensors with different shapes across ranks by:
    1. Gathering shape information from all ranks
    2. Padding tensors to maximum size
    3. Performing allgather
    4. Reconstructing the original variable-length tensors
    """

    @staticmethod
    def forward(ctx, input):
        """Forward pass for variable-length allgather operation.

        Args:
            ctx: PyLayer context for saving state
            input (Tensor): Input tensor to be gathered (may have different sizes across ranks)

        Returns:
            Tensor: Concatenated output from all ranks with original lengths
        """
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

        shape0 = paddle.to_tensor([input.shape[0]])
        shape0_all = paddle.empty(shape=[group.nranks], dtype=shape0.dtype)
        dist.stream.all_gather(shape0_all, shape0, group=group, use_calc_stream=True)
        shape0_all = shape0_all.numpy()
        max_shape0 = shape0_all.max()

        indices = []
        for idx, s in enumerate(shape0_all):
            offset = idx * max_shape0
            indices.extend(list(range(offset, offset + s)))
        indices = paddle.to_tensor(indices)

        padding = max_shape0 - input.shape[0]

        ctx.shape0 = input.shape[0]
        ctx.max_shape0 = max_shape0
        ctx.shape0_all = shape0_all
        ctx.padding = padding
        ctx.indices = indices

        if padding > 0:
            input_shape = input.shape
            input_shape[0] = padding
            padding_tensor = paddle.empty(shape=input_shape, dtype=input.dtype)
            input = paddle.concat([input, padding_tensor], axis=0)
        output = all_gather(input)
        output = paddle.gather(output, indices, axis=0)

        return output

    @staticmethod
    def backward(ctx, grad):
        """Backward pass for variable-length allgather operation.

        Args:
            ctx: PyLayer context with saved state
            grad (Tensor): Gradient flowing back through the graph

        Returns:
            Tensor: Scattered gradient with original variable lengths
        """
        input_shape = grad.shape
        input_shape[0] = ctx.max_shape0 * ctx.shape0_all.shape[0]
        output = paddle.zeros(shape=input_shape, dtype=grad.dtype)

        # grad = paddle.put_along_axis(output, ctx.indices, grad, axis=0)
        grad = paddle.scatter(output, ctx.indices, grad)
        grad = scatter(grad)

        if ctx.padding > 0:
            grad = grad[: ctx.shape0]
        return grad


def sequence_parallel_sparse_mask_labels(labels, ignore_label=-100):
    """
    Processes sparse labels in sequence parallel training by gathering non-ignored labels across all ranks.

    This function handles the case where labels may contain ignored values (typically -100) by:
    1. Distributing labels across model parallel ranks
    2. Identifying and gathering only valid (non-ignored) labels
    3. Performing a variable-length allgather operation to collect all valid labels

    Args:
        labels (paddle.Tensor): The input label tensor which may contain ignore_label values.
                              Shape should be compatible with model parallel distribution.
        ignore_label (int, optional): The value used to indicate labels that should be ignored.
                                     Defaults to -100 (common convention in NLP tasks).

    Returns:
        tuple: Contains two elements:
            - labels_all_gather (paddle.Tensor): Concatenated tensor of all non-ignored labels
                                               from all model parallel ranks.
            - tgt_index (paddle.Tensor): Indices of the non-ignored labels in the local rank's
                                        portion of the original labels tensor.

    Note:
        - This function assumes sequence parallel training is being used.
        - If a rank has no valid labels (all ignored), it will still contribute one dummy label
          (index 0) to maintain consistency in the distributed computation.
        - The returned tgt_index can be used to reconstruct the original label positions.
    """
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    labels = labels.flatten()
    labels_local = paddle.split(labels, group.nranks)[group.rank]

    tgt_index = paddle.nonzero(labels_local != ignore_label).reshape([-1])
    if tgt_index.numel() == 0:
        tgt_index = paddle.to_tensor([0])

    labels_local_gather = paddle.gather(labels_local, tgt_index, axis=0)
    labels_all_gather = AllGatherVarlenOp.apply(labels_local_gather)
    return labels_all_gather, tgt_index


async_loader = None


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


def hack_reload_wait(task):
    """hack_offload_wait"""
    task.cuda_wait()


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
