# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet

####################################################
#                                                  #
#        Distributed Communication Operator        #
#                                                  #
####################################################


def get_hcg():
    """
    get the hybrid comm group from fleet
    """
    return fleet.get_hybrid_communicate_group()


def scatter(input):
    hcg = get_hcg()
    group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    rank = group.rank
    seq_len = input.shape[0]
    assert (
        seq_len % parallelism == 0
    ), "Input sequence length {} can't be divided exactly by sequence parallelism {}".format(seq_len, parallelism)
    interval = seq_len // parallelism
    input = paddle.slice(input, axes=[0], starts=[interval * rank], ends=[interval * (rank + 1)])
    input = paddle.assign(input)
    return input


def all_gather(input):
    hcg = get_hcg()
    group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    output_shape = input.shape
    output_shape[0] = output_shape[0] * parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    group.process_group.all_gather(input, output).wait()
    return output


###################################################
#                                                 #
#        Modified Parallel Linear Operator        #
#                                                 #
###################################################


class AllGatherVarlenOp(PyLayer):
    """the shape of allgather can be not same for each rank"""

    @staticmethod
    def forward(ctx, input):
        """Forward pass."""
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
            indices.append(list(range(offset, offset + s)))
        indices = np.concatenate(indices, axis=0)
        indices = indices.reshape([-1] + [1] * (len(input.shape) - 1))
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
        output = paddle.take_along_axis(output, indices, axis=0)

        return output

    @staticmethod
    def backward(ctx, grad):
        """Backward pass."""
        input_shape = grad.shape
        input_shape[0] = ctx.max_shape0 * ctx.shape0_all.shape[0]
        output = paddle.zeros(shape=input_shape, dtype=grad.dtype)

        grad = paddle.scatter(output, ctx.indices, grad)
        grad = scatter(grad)

        if ctx.padding > 0:
            grad = grad[: ctx.shape0]
        return grad


def sequence_parallel_sparse_mask_labels(labels, ignore_label=-100):
    """allgather sparse label and return sparse idx"""
    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_model_parallel_group()
    labels = labels.flatten()
    labels_local = paddle.split(labels, group.nranks)[group.rank]

    tgt_index = paddle.nonzero(labels_local != ignore_label).squeeze()
    # NOTE(hehuang): There will be at least one label in each rank.
    if tgt_index.numel() == 0:
        tgt_index = paddle.to_tensor([0])

    tgt_index = tgt_index.reshape([-1])
    labels_local_gather = paddle.take_along_axis(labels_local, tgt_index, axis=0)
    labels_all_gather = AllGatherVarlenOp.apply(labels_local_gather)
    return labels_all_gather, tgt_index.reshape([-1, 1])
