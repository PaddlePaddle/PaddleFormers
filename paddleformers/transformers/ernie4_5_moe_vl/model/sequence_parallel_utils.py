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
from paddle.autograd import PyLayer

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


###################################################
#                                                 #
#        Modified Parallel Linear Operator        #
#                                                 #
###################################################


def mark_as_sequence_parallel_parameter(parameter):
    parameter.sequence_parallel = True
