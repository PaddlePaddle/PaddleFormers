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

import functools
import logging
from contextlib import contextmanager

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager as batch_isend_irecv_coalescing_manager,
)
from paddle.nn import functional as F

from paddleformers.trainer.plugins.timer import get_timers

logger = logging.getLogger(__name__)


def scatter(input, group=None, axis=0):
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
    input = paddle.assign(input)
    return input


def mp_slice(x, indices=None, group=None, axis=0):
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
        (paddle.concat([input] * parallelism, 0) if len(input) else input),
        output_sizes,
        input_sizes,
        group=group,
        sync_op=sync_op,
        use_calc_stream=sync_op,
    )
    task.wait()
    return out


def scatter_varlen(x, recv_tensor, indices, src_rank, group, sync_op=True):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if rank == src_rank:
        in_split_size = indices
    else:
        x = paddle.empty([], dtype=recv_tensor.dtype)
        in_split_size = [0] * world_size
    out_split_size = [indices[rank] if i == src_rank else 0 for i in range(world_size)]
    task = dist.stream.alltoall_single(
        recv_tensor,
        x,
        out_split_size,
        in_split_size,
        group=group,
        sync_op=sync_op,
        use_calc_stream=sync_op,
    )
    task.wait()


def all_gather(input, group=None, axis=0):
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


def reduce_scatter(input, group=None):
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


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx={}):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        assert len(arg_idx) == len(axis), "Number of batching args and number of batching dims should match."

        inps = [args[i] for i in arg_idx]
        axis_width = [inp.shape[d] for inp, d in zip(inps, axis)]
        assert len(set(axis_width)) == 1, "Batch sizes should be kept equal."

        inp_axis = {inp: d for inp, d in zip(inps, axis)}

        axis_width = axis_width[0]
        if axis_width < bs:
            return f(*args, **kwargs)

        outs = []
        for slice_at in np.arange(0, axis_width, bs):
            _args = []
            for i, inp in enumerate(args):
                if i in same_arg_idx:
                    assert (
                        i > same_arg_idx[i]
                    ), f"expect i > same_arg_idx[i], but got i: {i} and same_arg_idx[i]: {same_arg_idx[i]}"
                    _args.append(_args[same_arg_idx[i]])
                elif i in arg_idx:
                    inp = inp.slice(
                        [inp_axis[inp]],
                        [slice_at],
                        [min(inp.shape[inp_axis[inp]], slice_at + bs)],
                    )
                    _args.append(inp)
                else:
                    _args.append(inp)
            if use_recompute:
                out = paddle.distributed.fleet.utils.recompute(f, *_args, **kwargs)
            else:
                out = f(*_args, **kwargs)
            outs.append(out)

        return paddle.concat(outs, out_idx)

    return wrapper


def gather_varlen(input, dst, group, offload_pp_data_chunk_size=0, all_shape_and_dtype=None):
    if dist.get_world_size(group) <= 1:
        return input
    if group is None:
        group = dist.collective._get_global_group()

    shape_and_dtype = (None, None) if input is None else (input.shape, input.dtype)
    if all_shape_and_dtype is None:
        all_shape_and_dtype = []
        dist.all_gather_object(all_shape_and_dtype, shape_and_dtype, group=group)
    assert any(s is not None for s, _ in all_shape_and_dtype), all_shape_and_dtype

    any_shape = None
    shape0_all = []
    for s, d in all_shape_and_dtype:
        if s is not None and any_shape is None:
            any_shape = s
        elif s is not None and any_shape is not None:
            assert any_shape[1:] == s[1:], f"{any_shape[1:]} != {s[1:]}"
        shape0_all.append(s if s is not None else 0)

    output = []
    if offload_pp_data_chunk_size > 0:
        assert (group.nranks >= offload_pp_data_chunk_size) and (group.nranks % offload_pp_data_chunk_size == 0), (
            f"group.nranks {group.nranks} must be greater than offload_pp_data_chunk_size {offload_pp_data_chunk_size} "
            f"and group.nranks % offload_pp_data_chunk_size == 0"
        )
        if group.ranks[group.rank] == dst:
            num_sub_group = group.nranks // offload_pp_data_chunk_size
            for sub_group_idx in range(num_sub_group):
                start = sub_group_idx * offload_pp_data_chunk_size
                end = start + offload_pp_data_chunk_size
                tasks = []
                output_ptr = len(output)
                with batch_isend_irecv_coalescing_manager(group, tasks):
                    for src in range(start, end):
                        if all_shape_and_dtype[src][0] is None or all_shape_and_dtype[src][0][0] == 0:
                            pass
                        elif src != group.rank:
                            recv_tensor = paddle.empty(
                                all_shape_and_dtype[src][0],
                                dtype=all_shape_and_dtype[src][1],
                            )
                            output.append(recv_tensor)
                            task = dist.irecv(recv_tensor, group.ranks[src], group=group)
                            tasks.append(task)
                        else:
                            output.append(input)
                    for task in tasks:
                        task.wait()
                for i in range(output_ptr, len(output)):
                    output[i] = output[i].pin_memory()
        else:
            num_sub_group = group.nranks // offload_pp_data_chunk_size
            for sub_group_idx in range(num_sub_group):
                start = sub_group_idx * offload_pp_data_chunk_size
                end = start + offload_pp_data_chunk_size
                tasks = []
                with batch_isend_irecv_coalescing_manager(group, tasks):
                    for _ in range(1):
                        if group.rank in list(range(start, end)) and input is not None and input.shape[0] != 0:
                            task = dist.isend(input, dst, group=group)
                            tasks.append(task)
                for task in tasks:
                    task.wait()
    else:
        if group.ranks[group.rank] == dst:
            tasks = []
            with batch_isend_irecv_coalescing_manager(group, tasks):
                for src in range(group.nranks):
                    if all_shape_and_dtype[src][0] is None:
                        pass
                    elif src != group.rank:
                        recv_tensor = paddle.empty(
                            all_shape_and_dtype[src][0],
                            dtype=all_shape_and_dtype[src][1],
                        )
                        output.append(recv_tensor)
                        task = dist.irecv(recv_tensor, group.ranks[src], group=group)
                        tasks.append(task)
                    else:
                        output.append(input)
            for task in tasks:
                task.wait()
        else:
            tasks = []
            with batch_isend_irecv_coalescing_manager(group, tasks):
                for _ in range(1):
                    if input is not None:
                        task = dist.isend(input, dst, group=group)
                        tasks.append(task)
            for task in tasks:
                task.wait()

        if len(output) != 0:
            output = paddle.concat(output, 0)
    return output


@contextmanager
def profile(name, use_event=True):
    if get_timers() is not None:
        get_timers()(name, use_event=use_event).start()
    yield
    if get_timers() is not None:
        get_timers()(name, use_event=use_event).stop()
