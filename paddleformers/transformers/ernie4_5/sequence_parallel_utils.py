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

import hashlib

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet

from .distributed.common_dist_utils import (
    all_gather_group,
    all_gather_varlen,
    mp_slice,
    reduce_scatter_group,
    scatter_axis,
)

if not hasattr(paddle.Tensor, "contiguous"):

    def contiguous(self):
        """
        Make the tensor contiguous.
        """
        return self

    paddle.Tensor.contiguous = contiguous


if not hasattr(paddle.Tensor, "_md5sum"):

    def _md5sum(self):
        """
        Calculate the md5sum of the Tensor.
        """
        numpy_array = np.array(self)
        array_bytes = numpy_array.tobytes()
        return hashlib.md5(array_bytes).hexdigest()

    paddle.Tensor._md5sum = _md5sum


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        input,
        group,
        output_split_sizes=None,
        input_split_sizes=None,
    ):
        """
        All-to-all communication in the group

        Args:
            ctx (Any): Context object.
            input (Tensor): Input tensor.
            group (Group): The group object.

        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        # return input
        if dist.get_world_size(group) <= 1:
            return input
        if input_split_sizes is None and output_split_sizes is None:
            output = paddle.empty_like(input)
            task = dist.stream.alltoall_single(output, input, None, None, group, True, True)
            task.wait()
        else:
            out_sizes = [sum(output_split_sizes)]
            out_sizes.extend(input.shape[1:])
            output = paddle.empty(out_sizes, dtype=input.dtype)
            task = dist.stream.alltoall_single(
                output, input, output_split_sizes, input_split_sizes, group, sync_op=False
            )
            task.wait()
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """
        all-to-all backward

        """
        # return grad_output
        if ctx.input_split_sizes is None and ctx.output_split_sizes is None:
            return _AllToAll.apply(*grad_output, ctx.group)
        else:
            return _AllToAll.apply(*grad_output, ctx.group, ctx.input_split_sizes, ctx.output_split_sizes)


class AllGatherVarlenOpV2(PyLayer):
    """
    Custom PyLayer for variable-length all-gather operation with autograd support.
    """

    @staticmethod
    def forward(ctx, input, indices, axis=0, group=None):
        """forward"""
        ctx.axis = axis
        ctx.group = group
        ctx.indices = indices
        return all_gather_varlen(input, indices, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return mp_slice(grad, ctx.indices, axis=ctx.axis, group=ctx.group)


class SliceVarlenOp(PyLayer):
    """
    Each rank slices a variable-length portion from the **same** sequence.
    During backward pass, gradients from all ranks are aggregated to restore
    the mp (model parallelism) synchronization state.

    This is the variable-length version of `ScatterOp`. The inverse operation is `VarlenGatherOp`.

    Args:
        input: Tensor [S,*]
        indices: Slice lengths for each rank
        minimum_size: If slice is empty, return `minimum_size` dummy elements.
    Returns:
        Sliced Tensor
    """

    @staticmethod
    def forward(
        ctx,
        input,
        indices,
        group=None,
    ):
        """forward"""
        ctx.indices = indices
        ctx.group = group
        ret = mp_slice(input, indices, group=ctx.group)
        return ret

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return all_gather_varlen(grad, axis=ctx.axis, group=ctx.group)


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


SliceOp = ScatterOp  # `ScatterOp` similar to Sclice


class GatherOp(PyLayer):
    """
    input shape: [s/n, b, h], n is mp parallelism
    after forward shape: [s, b, h]
    Behavior is similar to `AllGather`, but gradients will not be aggregated in backward, from MP asynchronous state to MP synchronous state.
    """

    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        """forward"""
        ctx.axis = axis
        ctx.group = group
        return all_gather_group(input, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return scatter_axis(grad, axis=ctx.axis, group=ctx.group)


class AllGatherOp(PyLayer):
    """
    input shape: [s/n, b, h], n is mp parallelism
    after forward shape: [s, b, h]
    The behavior is similar to `AllGather`, and the gradients will be aggregated in backward. After AllGather, it is still in MP asynchronous state.
    """

    @staticmethod
    def forward(ctx, input, group=None):
        """forward"""
        ctx.group = group
        return all_gather_group(input, group=group)

    # grad shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def backward(ctx, grad):
        """backward"""
        return reduce_scatter_group(grad, group=ctx.group)


class AllGatherVarlenOp(PyLayer):
    """the shape of allgather can be not same for each rank"""

    @staticmethod
    def forward(ctx, input, group=None):
        """forward"""
        hcg = fleet.get_hybrid_communicate_group()
        if group is None:
            group = hcg.get_model_parallel_group()

        shape0 = paddle.to_tensor([input.shape[0]])
        shape0_all = paddle.empty(shape=[group.nranks], dtype=shape0.dtype)
        dist.stream.all_gather(shape0_all, shape0, group=group, use_calc_stream=True)
        shape0_all = shape0_all.numpy()
        max_shape0 = shape0_all.max()

        indices = []
        for idx, s in enumerate(shape0_all):
            offset = idx * max_shape0
            indices.append(list(range(offset, offset + s)))
        indices = np.concatenate(indices, axis=0)
        indices = indices.reshape([-1] + [1] * (len(input.shape) - 1))
        indices = paddle.to_tensor(indices, dtype=paddle.int32)

        padding = max_shape0 - input.shape[0]

        ctx.shape0 = input.shape[0]
        ctx.max_shape0 = max_shape0
        ctx.shape0_all = shape0_all
        ctx.padding = padding
        ctx.indices = indices
        ctx.group = group

        if padding > 0:
            input_shape = input.shape
            input_shape[0] = padding
            padding_tensor = paddle.empty(shape=input_shape, dtype=input.dtype)
            input = paddle.concat([input, padding_tensor], axis=0)
        output = all_gather_group(input, group)
        output = paddle.take_along_axis(output, indices, axis=0)

        return output

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        input_shape = grad.shape
        input_shape[0] = ctx.max_shape0 * ctx.shape0_all.shape[0]
        output = paddle.zeros(shape=input_shape, dtype=grad.dtype)

        grad = paddle.scatter(output, ctx.indices, grad)

        grad = scatter_axis(grad, ctx.group)

        if ctx.padding > 0:
            grad = grad[: ctx.shape0]
        return grad


def sequence_parallel_sparse_mask_labels(labels, ignore_label=-100):
    """allgather sparse label and return sparse idx"""
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    # parallelism = group.nranks
    labels = labels.flatten()
    labels_local = paddle.split(labels, group.nranks)[group.rank]

    tgt_index = paddle.nonzero(labels_local != ignore_label).squeeze()
    if tgt_index.numel() == 0:
        tgt_index = paddle.to_tensor([0])

    tgt_index = tgt_index.reshape([-1]).astype(paddle.int32)
    labels_local_gather = paddle.take_along_axis(labels_local, tgt_index, axis=0)
    labels_all_gather = AllGatherVarlenOp.apply(labels_local_gather)
    return labels_all_gather, tgt_index.reshape([-1, 1])


###################################################
#                                                 #
#        Modified Parallel Linear Operator        #
#                                                 #
###################################################


def mark_as_sequence_parallel_parameter(parameter):
    parameter.sequence_parallel = True


class MPScale(PyLayer):
    @staticmethod
    def forward(ctx, x, mp_degree):
        """forward"""
        out = paddle.scale(x, 1.0 / mp_degree)
        return out

    @staticmethod
    def backward(ctx, dout):
        """backward"""
        return dout
