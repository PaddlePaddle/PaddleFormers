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

import paddle
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.distributed.communication.batch_isend_irecv import (
    _coalescing_manager as batch_isend_irecv_coalescing_manager,
)


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
