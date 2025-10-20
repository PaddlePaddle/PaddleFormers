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

"""This module provides some utilities used in training process"""

import functools
import hashlib
import logging
from contextlib import contextmanager

import numpy as np
import paddle
from paddle import distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager as batch_isend_irecv_coalescing_manager,
)
from paddle.nn import functional as F
from paddleformers.trainer.plugins.timer import get_timers

logger = logging.getLogger(__name__)


def md5(tensor):
    """debug use"""
    numpy_array = tensor.numpy()
    array_bytes = numpy_array.tobytes()
    return hashlib.md5(array_bytes).hexdigest()


class ZLossOp(PyLayer):
    """Z Loss OP."""

    @staticmethod
    def forward(ctx, logits, z_loss_lambda=0, group=None):
        """Z loss forward."""
        max_logits = logits.max(axis=-1, keepdim=True)
        if group is not None:
            dist.all_reduce(max_logits, op=dist.ReduceOp.MAX, group=group)
        exp_logits = (logits - max_logits).exp()
        sum_exp_logits = exp_logits.sum(axis=-1, keepdim=True)
        if group is not None:
            dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)
        log_z = sum_exp_logits.log() + max_logits
        z_loss = z_loss_lambda * log_z.square()

        logits_grad = 2 * z_loss_lambda * log_z * exp_logits / sum_exp_logits
        ctx.save_for_backward(logits_grad)
        return z_loss

    @staticmethod
    def backward(ctx, grad):
        """Z loss backward."""
        logits_grad = grad * ctx.saved_tensor()[0]
        return logits_grad


class PrintOp(PyLayer):
    """debug use"""

    # input shape: [s, b, h], n is mp parallelism
    # after forward shape: [s/n, b, h]
    @staticmethod
    def forward(ctx, x, name):
        """doc"""
        ctx.name = name
        logger.info(f"{ctx.name}: {md5(x)[:5]} {x.abs().mean(-1)}")
        return x

    @staticmethod
    def backward(ctx, x):
        """doc"""
        logger.info(f"grad@{ctx.name}: {md5(x)[:5]} {x.abs().mean(-1)}")
        return x


def scatter(input, group=None, axis=0):
    """
    在MP 间按照第 0 维对`input`进行均匀切分。
    这个API 跟`distributed.scatter`并没有什么关系
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
    # slice use stride, so we maintain the memory of whole input, use assign to free the whole input
    # which can avoid OOM.
    input = paddle.assign(input)
    return input


def mp_slice(x, indices=None, group=None, axis=0):
    """
    对 tensor `x`按照第 0 维,根据`indices`切分。没有通信。
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
    支持变长输入版本`all_gather`, 行为类似`distributed.all_gather`
    `indices`: gather sizes from each rank
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
        paddle.concat([input] * parallelism, 0) if len(input) else input,  # 很好奇为什么 `paddle.tile` 不能指定axis
        output_sizes,  # input-size
        input_sizes,
        group=group,
        sync_op=sync_op,
        use_calc_stream=sync_op,
    )
    task.wait()
    return out


def scatter_varlen(x, recv_tensor, indices, src_rank, group, sync_op=True):
    """
    行为等价于`distributed.scatter` 但是接受变长输入。其中各 rank 的接受长度由`indices`指定。
    Args:
        x (_type_): 大tensor
        recv_tensor (_type_): 接收tensor
        indices (_type_): dim0 维scatter切分
        src_rank (_type_): _description_
        group (_type_): _description_
        sync_op (bool, optional): _description_. Defaults to True.
    """
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
        out_split_size,  # input-size
        in_split_size,
        group=group,
        sync_op=sync_op,
        use_calc_stream=sync_op,
    )
    task.wait()


def all_gather(input, group=None, axis=0):
    """all_gather"""
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
    """reduce_scatter"""
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


def subbatch(f, arg_idx, axis, bs, out_idx, use_recompute=False, same_arg_idx=None):
    """Converts a function to one that applies to subbatch of an input
    dimension.
    Args:
        f(Callable): original function.
        arg_idx([int]): indices of the inputs to be subbatched.
        axis([int]): index of the dimension to be subbatched.
        bs(int): subbatch size.
        out_idx(int): index of the output dimension that needs stacking
        same_arg_idx(dict), optional: index of same arg mapping. e.g {1: 0} means arg[1] == arg[0],
                            we assign _args[1] = _args[0] avoiding slice repeatly.
    Returns:
        converted function.
    """
    if same_arg_idx is None:
        same_arg_idx = {}

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
                    inp = inp.slice([inp_axis[inp]], [slice_at], [min(inp.shape[inp_axis[inp]], slice_at + bs)])
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
    """
    允许每个卡 shape 不一样的 gather, 行为类似`distributed.gather`, Feat. Guoxia
    """
    if dist.get_world_size(group) <= 1:
        return input
    if group is None:
        # Note: Maybe group is pipe_parallel_group for pp_need_data
        # but I need to pass CI
        # hcg = dist.fleet.get_hybrid_communicate_group()
        # group = hcg.get_pipe_parallel_group()
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
            # recv
            num_sub_group = group.nranks // offload_pp_data_chunk_size
            for sub_group_idx in range(num_sub_group):
                start = sub_group_idx * offload_pp_data_chunk_size
                end = start + offload_pp_data_chunk_size
                tasks = []
                output_ptr = len(output)
                with batch_isend_irecv_coalescing_manager(group, tasks):
                    for src in range(start, end):
                        if all_shape_and_dtype[src][0] is None or all_shape_and_dtype[src][0][0] == 0:
                            # output.append(paddle.empty([0] + any_shape[1:], dtype=d))
                            # nothing to do
                            pass
                        elif src != group.rank:
                            recv_tensor = paddle.empty(all_shape_and_dtype[src][0], dtype=all_shape_and_dtype[src][1])
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
            # send
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
            # recv
            tasks = []
            with batch_isend_irecv_coalescing_manager(group, tasks):
                for src in range(group.nranks):
                    if all_shape_and_dtype[src][0] is None:
                        # output.append(paddle.empty([0] + any_shape[1:], dtype=d))
                        # nothing to do
                        pass
                    elif src != group.rank:
                        recv_tensor = paddle.empty(all_shape_and_dtype[src][0], dtype=all_shape_and_dtype[src][1])
                        output.append(recv_tensor)
                        task = dist.irecv(recv_tensor, group.ranks[src], group=group)
                        tasks.append(task)
                    else:
                        output.append(input)
            for task in tasks:
                task.wait()
        else:
            # send
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
    """profile gurad"""
    if get_timers() is not None:
        get_timers()(name, use_event=use_event).start()
    yield
    if get_timers() is not None:
        get_timers()(name, use_event=use_event).stop()


def test_sharding_bandwidth(buf_size_mb=2048, warmup_num=10, test_num=50):
    """
    test_sharding_bandwidth
    """
    import time

    import paddle
    from paddle.distributed import fleet

    hcg = fleet.get_hybrid_communicate_group()
    group = hcg.get_sharding_parallel_group()
    nranks = group.nranks
    if nranks <= 1:
        return

    buf_size = int(buf_size_mb * 1024 * 1024)
    align = 256 * nranks
    buf_size = (buf_size + align - 1) // align * align
    assert buf_size % (2 * nranks) == 0
    buf_size = buf_size // 2
    buf_size_per_rank = buf_size // nranks
    rank = paddle.distributed.get_rank(group)

    nbytes = buf_size * 4
    x_rs = paddle.zeros([buf_size], dtype=paddle.float32)
    x_rs_slice = x_rs._slice(rank * x_rs.size // nranks, (rank + 1) * x_rs.size // nranks)

    x_ag = x_rs._slice(0, buf_size // 2)
    x_ag_slice = x_ag._slice(rank * x_ag.size // nranks, (rank + 1) * x_ag.size // nranks)

    rs_func = lambda: paddle.distributed.stream.reduce_scatter(
        x_rs_slice, x_rs, group=group, sync_op=True, use_calc_stream=True
    )
    ag_func = lambda: paddle.distributed.stream.all_gather(
        x_ag, x_ag_slice, group=group, sync_op=True, use_calc_stream=True
    )

    def test_func(func):
        """
        test_func
        """
        for _ in range(warmup_num):
            func()
        paddle.distributed.barrier(group)
        paddle.device.synchronize()
        start_t = time.time()
        for _ in range(test_num):
            func()
        paddle.device.synchronize()
        end_t = time.time()
        cost_t = (end_t - start_t) / test_num
        return cost_t

    coeff = (nranks - 1) / nranks * nbytes / 1e9
    rs_busbw = coeff / test_func(rs_func)
    ag_busbw = coeff / test_func(ag_func) / 2
    print(f"Sharding Bandwidth Test (GB/s): ReduceScatter {rs_busbw} , AllGather {ag_busbw}", flush=True)
