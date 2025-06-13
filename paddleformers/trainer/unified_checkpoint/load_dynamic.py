# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Unified Checkpoint Dynamic Loading Functions"""

import copy
import json
import os
import sys

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from ...peft import LoRAModel, PrefixModelForCausalLM
from ...transformers.model_utils import _load_state_dict_into_model
from ...transformers.utils import device_guard, is_safetensors_available
from ...utils.env import (
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_OPTIMIZER_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
)
from ...utils.log import logger
from ...utils.nested import nested_copy

if is_safetensors_available():
    if sys.platform.startswith("win"):
        from safetensors import safe_open
    else:
        from ...utils.safetensors import fast_safe_open as safe_open

from .utils import (
    FP32_MASTER,
    get_expected_state_dict,
    mapping_optimizer_tp_actions,
    optimizer_non_scaler_name,
    optimizer_scalar_name,
    select_model_weight_index,
    update_master_weight_status,
)

__all__ = ["load_unified_checkpoint_dynamically", "load_unified_optimizer_dynamically"]


def create_send_table(file_keyname_mappings, file_machine_mappings):
    send_table = {}
    global_rank = dist.get_rank()
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    for filename, keys in file_keyname_mappings.items():
        machine = file_machine_mappings[filename][0]
        is_src = (global_rank // local_device_count) == machine
        for i, key in enumerate(keys):
            if is_src and local_rank == i % local_device_count:
                send_table[key] = global_rank
    dispatch_list = []
    dist.all_gather_object(dispatch_list, send_table)
    send_table = {}
    for dl in dispatch_list:
        send_table.update(dl)
    return send_table


def create_dispatch_table(args, model, file_keyname_mappings, file_machine_mappings):
    """Create dispatch table for dynamically loading state dict.

    Args:
        args
    """

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # Create tensor receive table, contains {"key0": [global_rank, tp_rank], "key1": [global_rank, tp_rank]}
    dispatch_list = []
    recv_table = {}
    if args.dataset_rank == 0:
        state_dict = get_expected_state_dict(model)
        for (k, v) in state_dict.items():
            if hasattr(v, "is_distributed") and v.is_distributed:
                recv_table[k] = [(dist.get_rank(), tp_rank)]
            else:
                recv_table[k] = [(dist.get_rank(), -1)]

    # Gather receive table in global group.
    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for key, value in dl.items():
            if key not in recv_table:
                recv_table[key] = value
            else:
                recv_table[key] += value

    # Create send table, to decide which worker to send the key. Contains {"key0:" global_rank, "key1": global_rank, ...}
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)

    return send_table, recv_table


def create_optimizer_dispatch_table(
    args,
    model,
    optimizer,
    file_keyname_mappings,
    file_machine_mappings,
    struct2static_name_mappings,
    is_master_weights=False,
    typename_set=None,
):
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    tp_rank = tp_group.rank

    # Create receive table, contains {"param_key0": [global_rank, tp_rank], "param_key1": [global_rank, tp_rank]}
    dispatch_list = []
    recv_table = {}
    if args.data_parallel_rank == 0:
        state_dict = get_expected_state_dict(model)
        for (k, v) in state_dict.items():
            if sharding_group.nranks > 1:
                static_name = struct2static_name_mappings[k]
                param_rank = param2rank.get(static_name, None)
                if param_rank != sharding_rank:
                    continue
            if is_master_weights:
                if hasattr(v, "is_distributed") and v.is_distributed:
                    recv_table[k] = [(dist.get_rank(), tp_rank)]
                else:
                    recv_table[k] = [(dist.get_rank(), -1)]
            else:
                for typename in typename_set:
                    type_key = k + "/" + typename
                    if typename in optimizer_non_scaler_name:
                        if hasattr(v, "is_distributed") and v.is_distributed:
                            recv_table[type_key] = [(dist.get_rank(), tp_rank)]
                        else:
                            recv_table[type_key] = [(dist.get_rank(), -1)]
                    else:
                        recv_table[type_key] = [(dist.get_rank(), -1)]

    dist.all_gather_object(dispatch_list, recv_table)
    recv_table = {}
    for dl in dispatch_list:
        for k, v in dl.items():
            if k not in recv_table:
                recv_table[k] = v
            else:
                recv_table[k] += v

    # Create send table, to decide which worker to send the key. Contains {"param_key0:" 0, "param_key1": 1, ...}
    send_table = create_send_table(file_keyname_mappings, file_machine_mappings)
    return send_table, recv_table


def get_file_mappings(index, resume_from_checkpoint):
    file_keyname_mappings = {}
    for k, v in index["weight_map"].items():
        if v not in file_keyname_mappings:
            file_keyname_mappings[v] = []
        file_keyname_mappings[v].append(k)
    for k in file_keyname_mappings.keys():
        file_keyname_mappings[k] = sorted(file_keyname_mappings[k])

    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    global_rank = dist.get_rank()
    file_machine_mappings = {}
    for filename in file_keyname_mappings.keys():
        if local_rank == 0 and os.path.exists(os.path.join(resume_from_checkpoint, filename)):
            file_machine_mappings[filename] = [global_rank // local_device_count]
    file_machine_list = []
    dist.all_gather_object(file_machine_list, file_machine_mappings)
    file_machine_mappings = {}
    for mappings in file_machine_list:
        for k, v in mappings.items():
            if k not in file_machine_mappings:
                file_machine_mappings[k] = v
            else:
                file_machine_mappings[k] += v
    return file_keyname_mappings, file_machine_mappings


def distributed_send_recv(
    state_dict,
    tp_actions,
    send_table,
    recv_table,
    resume_from_checkpoint,
    file_keyname_mappings,
    file_machine_mappings,
):

    local_device_count = int(os.getenv("PADDLE_LOCAL_SIZE"))
    global_rank = dist.get_rank()
    for filename in file_keyname_mappings.keys():
        machine = file_machine_mappings[filename][0]
        is_src = global_rank // local_device_count == machine
        if is_src:
            f = safe_open(os.path.join(resume_from_checkpoint, filename), framework="np")

        for key in file_keyname_mappings[filename]:
            recv_info = recv_table[key]
            recv_ranklist = [a for (a, _) in recv_info]
            if is_src and global_rank == send_table[key]:
                py_safe_slice_ = f.get_slice(key)
                # send
                if key in tp_actions:
                    weight = tp_actions[key](py_safe_slice_)
                    # copy weight to GPU
                    for j in range(len(weight)):
                        with device_guard():
                            weight[j] = paddle.Tensor(weight[j], zero_copy=True)
                        weight[j] = weight[j]._copy_to(paddle.framework._current_expected_place(), False)

                    for recv_rank, split_index in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight[split_index]
                        else:
                            dist.stream.send(weight[split_index], dst=recv_rank)
                else:
                    # no need to tp split
                    weight = py_safe_slice_[:]
                    with device_guard():
                        weight = paddle.Tensor(weight, zero_copy=True)
                    weight = weight._copy_to(paddle.framework._current_expected_place(), False)
                    for recv_rank, _ in recv_info:
                        if recv_rank == global_rank:
                            state_dict[key] = weight
                        else:
                            dist.stream.send(weight, dst=recv_rank)

            if global_rank != send_table[key] and global_rank in recv_ranklist:
                dist.stream.recv(state_dict[key], src=send_table[key])

        if is_src:
            f.__exit__(None, None, None)

    return state_dict


def load_unified_checkpoint_dynamically(args, model, resume_from_checkpoint, safe_serialization=False):
    index_filename = select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=False)
    index_filename = os.path.join(resume_from_checkpoint, index_filename)

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    # `file_keyname_mappings` indicates which keys each file contains. For example, {"model-00001-of-00002.safetensors": ["llama.embed_tokens.weight", "llama.layers.0.self_attn.q_proj.weight", ...]}
    # `file_machine_mappings` indicates the machine where the files appear. For example, {"model-00001-of-00002.safetensors": [machine_0, machine_1], "model-00002-of-00002.safetensors": [machine_0]}
    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    logger.debug("Creating dispatch table for unified checkpoint load ...")
    # Get send_table and recv_table. The send table indicates which workers are responsible for sending tensors, and the recv table indicates which workers should receive the tensors.
    send_table, recv_table = create_dispatch_table(
        args,
        model,
        file_keyname_mappings,
        file_machine_mappings,
    )

    # Get all the keys that are splited by tensor parallelism.
    all_tp_keys = set()
    for k, v in recv_table.items():
        if v[0][1] != -1:
            all_tp_keys.add(k)

    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    if len(all_tp_keys) == 0:
        tp_actions = {}
    else:
        # Get corresponding tensor parallel actions.
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(
                set(all_tp_keys), is_split=True, ignore_error=True, config=config_revise
            )
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)

    logger.debug("Distributed send recv for state dict load ...")
    # Distribute the checkpoint tensor dynamically, using the `send_table` and `recv_table` we create before.
    state_dict = distributed_send_recv(
        get_expected_state_dict(model),
        tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )
    dist.barrier()
    logger.debug("Setting state dict into model ...")
    model_to_load_state_dict = model.state_dict()
    error_msgs = _load_state_dict_into_model(model, state_dict, "", model_to_load_state_dict)
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        raise RuntimeError(f"Error(s) in loading dynamic state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def load_unified_optimizer_dynamically(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    optim_state_dict = nested_copy(optimizer.state_dict())
    if "master_weights" in optim_state_dict.keys():
        optim_state_dict.pop("master_weights")

    if safe_serialization:
        index_filename, index_filename_mw = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME
    else:
        index_filename, index_filename_mw = PADDLE_OPTIMIZER_INDEX_NAME, PADDLE_MASTER_WEIGHTS_INDEX_NAME

    with open(os.path.join(resume_from_checkpoint, index_filename), "r") as f:
        index = json.loads(f.read())

    # `file_keyname_mappings` indicates which keys each file contains. For example, {"optimizer-00001-of-00002.safetensors": ["llama.embed_tokens.weight/moment1_0", "llama.layers.1.mlp.gate_proj.weight/moment1_0", ...]}
    # `file_machine_mappings` indicates the machine where the files appear. For example, {"optimizer-00001-of-00002.safetensors": [machine_0, machine_1], "optimizer-00002-of-00002.safetensors": [machine_0]}
    file_keyname_mappings, file_machine_mappings = get_file_mappings(index, resume_from_checkpoint)

    has_master_weights = index["master_weights"]
    # update has_master_weights and index_filename_master_weights
    # 1. if the master weights exists, only has_master_weights is set True and load master weights when needed
    # 2. if master weights does not exist, convert model weights to master weights when needed
    has_master_weights, index_filename_mw = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )

    if has_master_weights:
        with open(os.path.join(resume_from_checkpoint, index_filename_mw), "r") as f:
            index_mw = json.loads(f.read())
        file_keyname_mappings_mw, file_machine_mappings_mw = get_file_mappings(index_mw, resume_from_checkpoint)

    # Get optimizer param type name, like moment1_0, moment2_0, beta1_pow_acc_0.
    typename_set = set()
    for key in index["weight_map"].keys():
        _, typename = key.split("/")
        typename_set.add(typename)

    model_state_dict = get_expected_state_dict(model)
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}
    static2struct_name_mappings = {v.name: k for k, v in model_state_dict.items()}
    # Get send_table and recv_table. The send table indicates which workers are responsible for sending tensors, and the recv table indicates which workers should receive the tensors.
    send_table, recv_table = create_optimizer_dispatch_table(
        args,
        model,
        optimizer,
        file_keyname_mappings,
        file_machine_mappings,
        struct2static_name_mappings,
        is_master_weights=False,
        typename_set=typename_set,
    )
    if has_master_weights:
        send_table_mw, recv_table_mw = create_optimizer_dispatch_table(
            args,
            model,
            optimizer,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
            struct2static_name_mappings,
            is_master_weights=True,
        )

    # Initialize optimizer state dict.
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    if sharding_group.nranks > 1:
        param2rank = optimizer._param2rank
    optim_state_dict_mw = {}

    def check_optimizer_param(parameter):
        if sharding_group.nranks > 1:
            param_rank = param2rank.get(parameter.name, None)
            if param_rank != sharding_group.rank:
                return False
        if parameter.stop_gradient:
            return False
        return True

    optimizer_keys_with_shape = []
    if isinstance(optimizer._parameter_list[0], dict):
        for param_group in optimizer._parameter_list:
            # If parameter groups are set, there must be `params` key. This is guaranteed by the optimizer's initialization code.
            for parameter in param_group["params"]:
                if check_optimizer_param(parameter):
                    optimizer_keys_with_shape.append((parameter.name, parameter.shape))
    else:
        for parameter in optimizer._parameter_list:
            if check_optimizer_param(parameter):
                optimizer_keys_with_shape.append((parameter.name, parameter.shape))

    # see how to change
    for static_name, shape in optimizer_keys_with_shape:
        k = static2struct_name_mappings[static_name]
        for typename in typename_set:
            new_k = k + "/" + typename
            if typename in optimizer_scalar_name:
                optim_state_dict[new_k] = paddle.empty([1], dtype="float32")
            else:
                optim_state_dict[new_k] = paddle.empty(shape, dtype="float32")
        if has_master_weights:
            optim_state_dict_mw[k] = paddle.empty(shape, dtype="float32")

    # Get all the keys that are splited by tensor parallelism.
    all_tp_keys = set()
    for k, v in recv_table.items():
        structure_name, typename = k.split("/")
        if typename in optimizer_non_scaler_name:
            if v[0][1] != -1:
                all_tp_keys.add(structure_name)

    # Get corresponding tensor parallel actions.
    config_revise = copy.deepcopy(model.config)
    config_revise.tensor_parallel_rank = None
    if len(all_tp_keys) == 0:
        tp_actions = {}
    else:
        if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
            tp_actions = model._get_tensor_parallel_convert_actions(
                set(all_tp_keys), is_split=True, ignore_error=True, config=config_revise
            )
        else:
            tp_actions = model.get_tensor_parallel_convert_actions(config_revise, all_tp_keys, ignore_error=True)
    optimizer_keys = list(index["weight_map"].keys())
    optimizer_tp_actions = mapping_optimizer_tp_actions(tp_actions, optimizer_keys)
    if has_master_weights:
        optimizer_tp_actions.update(tp_actions)

    # Distribute the optimizer checkpoint dynamically, using the `send_table` and `recv_table` we create before.
    optim_state_dict = distributed_send_recv(
        optim_state_dict,
        optimizer_tp_actions,
        send_table,
        recv_table,
        resume_from_checkpoint,
        file_keyname_mappings,
        file_machine_mappings,
    )
    dist.barrier()
    if has_master_weights:
        optim_state_dict_mw = distributed_send_recv(
            optim_state_dict_mw,
            optimizer_tp_actions,
            send_table_mw,
            recv_table_mw,
            resume_from_checkpoint,
            file_keyname_mappings_mw,
            file_machine_mappings_mw,
        )
        dist.barrier()

    # Rename optimizer state dict.
    for key in list(optim_state_dict.keys()):
        if key == "LR_Scheduler":
            continue
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            if model_state_dict[key_name[0]].dtype != paddle.float32:
                key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        optim_state_dict[key_name] = optim_state_dict.pop(key)
        optim_state_dict[key_name].name = key_name

    if has_master_weights:
        optim_state_dict["master_weights"] = {}
        for key in list(optim_state_dict_mw.keys()):
            static_name = struct2static_name_mappings[key]
            optim_state_dict["master_weights"][static_name] = optim_state_dict_mw.pop(key)
            optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    if args.data_parallel_rank == 0:
        return optim_state_dict
    return None
