# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed.fleet as fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    HybridParallelOptimizer,
)
from paddle.distributed.fleet.model import PipelineParallel

from ....transformers.model_utils import unwrap_optimizer
from ....utils.log import logger

try:
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizerV2,
    )
except:
    DygraphShardingOptimizerV2 = None


from paddle.distributed.communication.reduce import ReduceOp

from .common import get_moe_sharding_group


def shard(node_model_state, model, optimizer):
    assert DygraphShardingOptimizerV2 is not None
    split_infos = collect_split_info(optimizer, model)
    group = node_model_state.group
    cur_rank = max(group.rank, 0)

    def split_func(k, v):
        param_name = k[1]
        opt_name = k[-1]
        assert param_name in split_infos, f"param_name {param_name}, split_infos{split_infos}"
        is_beta = is_bata(opt_name)
        index, padded_size, buffer_size, has_slice_grad = split_infos[param_name]

        if not is_beta:
            v = pad_tensor(k, v, padded_size)

        def get_slice(v, begin, end):
            if is_beta:
                return v
            return slice_tensor(v, begin, end)

        assert buffer_size % group.nranks == 0, f"buffer_size {buffer_size} group.nranks {group.nranks}"
        buffer_slice = buffer_size // group.nranks

        # has slice grad in cur rank
        if has_slice_grad:
            assert index < (cur_rank + 1) * buffer_slice
            assert index + padded_size > cur_rank * buffer_slice

        offset = buffer_slice - index % buffer_slice
        tensors = []
        tensors.append((index // buffer_slice, get_slice(v, 0, min(offset, padded_size))))

        left_size = padded_size - offset

        if left_size > 0:
            for _ in range((left_size + buffer_slice - 1) // buffer_slice):
                end = min(offset + buffer_slice, padded_size)
                assert end <= buffer_size
                tensors.append(((offset + index) // buffer_slice, get_slice(v, offset, end)))
                offset = end

        return tensors

    node_model_state.split_items(split_func).flatten_key()

    def filter_func(k):
        names, rank = k
        assert rank < group.nranks
        return rank == cur_rank

    # reshard
    node_model_state.reshard(filter_func)
    node_model_state.drop_rank()
    return node_model_state


def restore(node_model_state, model, optimizer):
    # evenly distribute param
    node_model_state.even_distribute()
    param_shapes = {k: v.shape for (k, v) in model.state_dict().items()}

    def merge_func(k, v):
        structure_name = k[0]
        opt_name = k[-1]
        assert structure_name in param_shapes, structure_name
        tensor_list = [e[1] for e in v]
        # do not merge beta acc
        if is_bata(opt_name):
            return tensor_list[0]
        shape = param_shapes[structure_name]
        return merge_tensors(k, tensor_list, shape)

    node_model_state.collapse_key().merge_items(merge_func)
    return node_model_state


def merge_tensors(k, tensor_list, shape):
    assert len(tensor_list) > 0
    if len(tensor_list) == 1:
        t = tensor_list[0]
    else:
        assert len(tensor_list[0].shape) == 1
        t = paddle.concat(x=tensor_list, axis=0)
    tensor_size = np.prod(shape)
    padded_size = t._numel()
    assert padded_size >= tensor_size, f"{k} padded_size {padded_size} tensor_size {tensor_size}"
    t = t._slice(0, tensor_size)
    t.get_tensor()._set_dims(shape)
    return t


def pad_tensor(k, tensor, padded_size):
    tensor_shape = tensor.shape
    tensor_size = np.prod(tensor_shape)
    assert tensor_size <= padded_size, f"{k} tensor_size {tensor_size} padded_size {padded_size}"
    t = paddle.zeros([padded_size], dtype=tensor.dtype)
    tensor.flatten_()
    t[0:tensor_size] = tensor
    tensor.get_tensor()._set_dims(tensor_shape)
    return t


def slice_tensor(tensor, begin, end):
    return tensor[begin:end]


def collect_split_info(optimizer, model, only_return_lengths=False):
    split_infos = {}

    def gather_infos(comm_buffer):
        for (k, v) in comm_buffer._sharding_param_grad_view.items():
            index = v._index
            padded_size = v._padded_size
            buffer_size = v._param_buffer._numel()
            has_slice_grad = v._slice_grad is not None
            if only_return_lengths:
                if v._param_begin < v._param_end:
                    split_infos[k] = v._param_end - v._param_begin
                else:
                    split_infos[k] = None
            else:
                split_infos[k] = (index, padded_size, buffer_size, has_slice_grad)

    if isinstance(model, PipelineParallel) and model._sharding_comm_overlap > 0:
        optimizer = unwrap_optimizer(optimizer, HybridParallelOptimizer)
        assert optimizer is not None
        # dalayed comm_overlap_hook register
        model.register_sharding_comm_overlap_hook(optimizer)
        for (k, v) in model._chunk_2_comm_buffers.items():
            for comm_buffer in v:
                gather_infos(comm_buffer)

    else:
        optimizer = unwrap_optimizer(optimizer, DygraphShardingOptimizerV2)
        assert optimizer is not None
        for comm_buffer in optimizer._comm_buffer_list:
            gather_infos(comm_buffer)

    assert len(split_infos) > 0
    return split_infos


def is_matched_optimizer_state_dict(opt_state_dict, optimizer, model, hcg=None, need_allgather=True):
    split_infos = collect_split_info(optimizer, model, only_return_lengths=True)
    master_weights = opt_state_dict.get("master_weights", None)

    def get_matched_length(name):
        if master_weights and name in master_weights:
            tensor = master_weights[name]
        else:
            moment_name = name + "_moment1_0"
            if moment_name not in opt_state_dict:
                return None

            tensor = opt_state_dict[moment_name]
            if isinstance(tensor, (list, tuple)):
                assert len(tensor) == 2, tensor
                assert isinstance(tensor[0], str), tensor[0]
                tensor = tensor[1]
        shape = tensor.shape
        assert len(shape) == 1, shape
        length = shape[0]
        return length

    is_matched = 1
    for k, length in split_infos.items():
        matched_length = get_matched_length(k)
        if length != matched_length:
            is_matched = 0
            break

    if need_allgather:
        if hcg is None:
            hcg = fleet.get_hybrid_communicate_group()
        sharding_group = hcg.get_sharding_parallel_group()
        moe_sharding_group = get_moe_sharding_group(hcg)
        for group in [sharding_group, moe_sharding_group]:
            if group is not None and group.nranks > 1:
                x = paddle.to_tensor([is_matched], dtype=paddle.int32)
                paddle.distributed.stream.all_reduce(
                    x, op=ReduceOp.MIN, group=group, sync_op=True, use_calc_stream=True
                )
                is_matched = int(x.numpy()[0])
        global_is_matched = is_matched
        group = hcg.get_sharding_parallel_group()
        if group is not None and group.nranks > 1:
            x = paddle.to_tensor([is_matched], dtype=paddle.int32)
            paddle.distributed.stream.all_reduce(x, op=ReduceOp.MIN, group=group, sync_op=True, use_calc_stream=True)
            global_is_matched = int(x.numpy()[0])
    else:
        global_is_matched = is_matched

    global_is_matched = True if global_is_matched else False
    logger.info(f"Sharding reshard checkpoint: local_match = {is_matched} , global_match = {global_is_matched}")
    return global_is_matched


def is_bata(name):
    if "_beta1_pow_acc_" in name:
        return True
    if "_beta2_pow_acc_" in name:
        return True
    return False
