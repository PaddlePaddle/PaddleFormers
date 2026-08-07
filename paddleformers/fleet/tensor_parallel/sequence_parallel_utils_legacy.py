# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Legacy sequence parallel utilities migrated from ernie_core.models.sequence_parallel_utils.

These implementations are needed by PaddleFleet when gpt_model_use_experimental_version
is enabled, to avoid importing from ernie_core at runtime.
"""

import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet


def _scatter(input, group=None, axis=0):
    if group is None:
        if not hasattr(fleet.fleet, "_hcg"):
            return input.clone()
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    rank = group.rank
    seq_len = input.shape[axis]
    assert seq_len % parallelism == 0, (
        f"Input sequence length {seq_len} can't be divided exactly"
        f" by sequence parallelism {parallelism}"
    )
    interval = seq_len // parallelism
    input = paddle.slice(
        input,
        axes=[axis],
        starts=[interval * rank],
        ends=[interval * (rank + 1)],
    )
    input = paddle.assign(input)
    return input


def _all_gather(input, group=None, axis=0):
    if group is None:
        if not hasattr(fleet.fleet, "_hcg"):
            return input.clone()
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = list(input.shape)
    if axis == 0:
        output_shape[axis] = output_shape[axis] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        dist.stream.all_gather(output, input, group=group, use_calc_stream=True)
        return output
    outputs = [
        paddle.empty(list(input.shape), dtype=input.dtype)
        for _ in range(parallelism)
    ]
    dist.stream.all_gather(outputs, input, group=group, use_calc_stream=True)
    output = paddle.concat(outputs, axis=axis)
    return output


def _reduce_scatter(input, group=None):
    if group is None:
        if not hasattr(fleet.fleet, "_hcg"):
            return input.clone()
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = list(input.shape)
    assert input.shape[0] % parallelism == 0, (
        f"Input sequence length {input.shape[0]} can't be divided exactly"
        f" by sequence parallelism {parallelism}"
    )
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(
        output, input, op=dist.ReduceOp.SUM, group=group, use_calc_stream=True
    )
    return output


class ScatterOpLegacy(PyLayer):
    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return _scatter(input, axis=axis, group=ctx.group)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, axis=ctx.axis, group=ctx.group)


class GatherOpLegacy(PyLayer):
    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.axis = axis
        ctx.group = group
        return _all_gather(input, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        return _scatter(grad, axis=ctx.axis, group=ctx.group)


class AllGatherOpLegacy(PyLayer):
    @staticmethod
    def forward(ctx, input, axis=0, group=None):
        ctx.group = group
        ctx.axis = axis
        return _all_gather(input, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter(grad, group=ctx.group)


class ReduceScatterOpLegacy(PyLayer):
    @staticmethod
    def forward(ctx, input, group=None):
        ctx.group = group
        return _reduce_scatter(input, group=group)

    @staticmethod
    def backward(ctx, grad):
        return _all_gather(grad, group=ctx.group)
