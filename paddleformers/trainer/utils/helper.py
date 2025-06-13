# Copyright 2020-present the HuggingFace Inc. team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers

import os
from typing import Any, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.parallel import sync_params_buffers

from ...utils.log import logger
from ...utils.nested import nested_broadcast_tensor_with_empty  # noqa: F401
from ...utils.nested import (
    nested_broadcast_tensor,
    nested_empty_tensor,
    nested_reduce_tensor,
)

__all__ = [
    "distributed_concat",
    "paddle_pad_and_concatenate",
    "nested_concat",
    "nested_detach",
    "nested_numpify",
    "nested_truncate",
]


def distributed_concat(tensor: Any, num_total_examples: Optional[int] = None) -> Any:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        output_tensors = []
        dist.all_gather(output_tensors, tensor)
        output_tensors = [t if len(t.shape) > 0 else t.reshape_([-1]) for t in output_tensors]
        concat = paddle.concat(output_tensors, axis=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def paddle_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return paddle.concat((tensor1, tensor2), axis=0)

    # raise ValueError("Error")
    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tuple(
        tensor1.shape[2:]
    )

    # Now let's fill the result tensor
    # result = tensor1.new_full(new_shape, padding_index)
    result = paddle.full(new_shape, padding_index, dtype=tensor1.dtype)

    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    # Now let's fill the result tensor
    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, paddle.Tensor):
        return paddle_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    t = tensors.cpu()
    if t.dtype == paddle.float16:
        t = t.cast(paddle.float32)
    return t.cpu().numpy()


def nested_truncate(tensors, limit):
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_truncate(t, limit) for t in tensors)
    return tensors[:limit]


def distributed_isfile(filename):
    """Check all machine nodes. return False if no machine have such file."""
    trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    if trainers_num <= 1:
        return os.path.isfile(filename)
    else:
        local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
        file_count = paddle.zeros([1], dtype="int64")
        if local_rank == 0 and os.path.isfile(filename):
            file_count += 1

        paddle.distributed.all_reduce(file_count)
        return file_count >= 1


def distributed_file(filename):
    trainers_num = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    if trainers_num <= 1:
        return filename
    else:
        local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
        found_file = paddle.to_tensor([2**20], dtype="int64")
        if local_rank == 0 and os.path.isfile(filename):
            found_file = paddle.to_tensor([paddle.distributed.get_rank()], dtype="int64")

        tensor_list = []
        paddle.distributed.all_gather(tensor_list, found_file)
        src = paddle.min(paddle.concat(tensor_list)).item()

        file_object_list = [None]
        if paddle.distributed.get_rank() == src:
            file_object_list = [open(filename, "rb").read()]

        paddle.distributed.broadcast_object_list(file_object_list, src=src)
        file_object = file_object_list[0]

        if local_rank == 0 and not os.path.isfile(filename):
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

            with open(filename, "wb") as f:
                f.write(file_object)

        paddle.distributed.barrier()

        return filename


def broadcast_dp_optimizer(state_dict):
    if paddle.distributed.get_world_size() <= 1:
        return state_dict

    logger.info("Start broadcast optimizer in data parallel group.")
    try:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        src_rank = hcg.get_data_parallel_group_src_rank()
        process_rank = paddle.distributed.get_rank()
        # Don't broadcast optimizer for dp rank is 1.
        if dp_group.nranks <= 1:
            return state_dict
    except:
        dp_group = None
        src_rank = 0
        process_rank = paddle.distributed.get_rank()

    if process_rank == src_rank:
        if state_dict is None:
            logger.warning(
                f"Your local rank {paddle.distributed.get_rank()} must have a state_dict. dp_rank:{process_rank}, src_rank:{src_rank}"
            )
        fake_state_dict = [nested_reduce_tensor(state_dict)]
    else:
        if state_dict is not None:
            logger.warning(
                f"Your local rank {paddle.distributed.get_rank()}  are forbidden to have a state_dict. dp_rank:{process_rank}, src_rank:{src_rank}"
            )
        fake_state_dict = [None]

    paddle.distributed.broadcast_object_list(
        fake_state_dict,
        src=src_rank,
        group=dp_group,
    )
    fake_state_dict = fake_state_dict[0]
    if process_rank != src_rank:
        state_dict = nested_empty_tensor(fake_state_dict)

    state_dict = nested_broadcast_tensor(state_dict, src=src_rank, group=dp_group)

    return state_dict


def broadcast_moe_optimizer(state_dict, model_state_dict=None, broadcast_dp=True):
    try:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        src_rank = hcg.get_data_parallel_group_src_rank()
        data_parallel_rank = hcg.get_data_parallel_rank()
        # Don't broadcast optimizer for dp rank is 1.
        if dp_group.nranks <= 1:
            return state_dict
    except:
        dp_group = None
        src_rank = 0
        data_parallel_rank = dist.get_rank()

    def _filter_sync_optimizer_state(model_state_dict, opt_state_dict):
        # get sync name
        sync_vname = []
        for k, v in model_state_dict.items():
            if not getattr(v, "no_sync", False):
                sync_vname.append(v.name)

        filter_opt_state_dict = {"master_weights": {}}
        filter_opt_state_dict["LR_Scheduler"] = opt_state_dict.get("LR_Scheduler", {})
        for op_k, op_v in opt_state_dict.items():
            if op_k not in ["master_weights", "LR_Scheduler"]:
                for sync_v in sync_vname:
                    if op_k.startswith(sync_v):
                        filter_opt_state_dict[op_k] = op_v
                        break
            elif op_k == "master_weights":
                for k, v in op_v.items():
                    for sync_v in sync_vname:
                        if k.startswith(sync_v):
                            filter_opt_state_dict["master_weights"][k] = v
        return filter_opt_state_dict

    def _broadcast_moe_optimizer_state(state_dict):
        # boardcast_keys
        base_state_dict = {"master_weights": {}}
        buf = [
            {i: j.shape for i, j in state_dict.items() if i not in ["master_weights", "LR_Scheduler"]},
            {i: j.shape for i, j in state_dict["master_weights"].items()},
            {"LR_Scheduler": state_dict.get("LR_Scheduler", {})},
        ]

        dist.broadcast_object_list(buf, src=src_rank, group=dp_group)
        # logger.info(f"moe-optimizer-gather-keys{buf}")
        for k, s in buf[0].items():
            v = state_dict.get(k, paddle.zeros(s, "float32")).cuda()
            v.name = k
            # k = k.replace("_fp32_master_0", "")
            dist.broadcast(v, src=src_rank, group=dp_group)
            logger.info(f"broadcast moe optimizer {k} from {src_rank}")
            base_state_dict[k] = v.cpu()
        for k, s in buf[1].items():
            v = state_dict["master_weights"].get(k, paddle.zeros(s, "float32")).cuda()
            v.name = k
            dist.broadcast(v, src=src_rank, group=dp_group)
            logger.info(f"broadcast moe optimizer-master_weights {k} from {src_rank}")
            base_state_dict["master_weights"][k] = v.cpu()
        base_state_dict.update(buf[2])
        return base_state_dict

    if broadcast_dp:
        filter_opt_state_dict = _filter_sync_optimizer_state(model_state_dict, state_dict)
        base_state_dict = broadcast_dp_optimizer(filter_opt_state_dict)
    else:
        base_state_dict = _broadcast_moe_optimizer_state(state_dict)

    if data_parallel_rank > 0:
        master_weight = state_dict.pop("master_weights", {})
        base_state_dict.update(state_dict)
        if master_weight:
            if "master_weights" in base_state_dict:
                base_state_dict["master_weights"].update(master_weight)
            else:
                base_state_dict["master_weights"] = master_weight
        state_dict = base_state_dict
        del base_state_dict
    return state_dict


def broadcast_dataset_rank0_model(model):
    if paddle.distributed.get_world_size() <= 1:
        return

    logger.info("Start broadcast model in sharding group or data parallel group.")
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()
    if sharding_group.nranks > 1:
        sync_params_buffers(
            model,
            sharding_group,
            hcg.get_sharding_parallel_group_src_rank(),
            is_model_parallel=False,
            fuse_params=False,
        )
    if dp_group.nranks > 1:
        sync_params_buffers(
            model,
            dp_group,
            hcg.get_data_parallel_group_src_rank(),
            is_model_parallel=False,
            fuse_params=False,
        )
