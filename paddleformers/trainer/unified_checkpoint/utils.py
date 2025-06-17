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
"""Unified Checkpoint Utility Functions"""

import copy
import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from ...peft import LoRAModel, PrefixModelForCausalLM
from ...trainer.utils.helper import distributed_isfile
from ...transformers.model_utils import (
    PretrainedModel,
    clean_model_class_name,
    get_parameter_dtype,
    unwrap_model,
)
from ...transformers.utils import dtype_byte_size
from ...utils.distributed import distributed_allgather, distributed_gather
from ...utils.env import (
    BETA1_KEYNAME,
    BETA2_KEYNAME,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_PEFT_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PAST_KEY_VALUES_FILE_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from ...utils.log import logger
from ...utils.nested import flatten_list
from ...utils.tools import get_env_device
from ..trainer_utils import ExplicitEnum, ShardingOption

FP32_MASTER = "fp32_master_0"
optimizer_scalar_name = [
    "beta1_pow_acc_0",
    "beta2_pow_acc_0",
]
optimizer_non_scaler_name = [
    "moment1_0",
    "moment2_0",
    "velocity_0",
]  # to be added


DEST_PLACE = paddle.CPUPlace()
if paddle.device.is_compiled_with_cuda():
    DEST_PLACE = paddle.CUDAPinnedPlace()


class UnifiedCheckpointOption(ExplicitEnum):
    """
    "- skip_save_model_weight: do not save model weights when the masters weight exist\n"
    "- master_weight_compatible: 1. if the master weights exist, only load when needed\n"
    "                            2. if master weights does not exist, convert model weights to master weights when needed\n"
    "- async_save: enable asynchronous saving checkpoints to disk\n"
    "- enable_all_options: enable all optimization configurations\n"
    """

    SKIP_SAVE_MODEL_WEIGHT = "skip_save_model_weight"
    MASTER_WEIGHT_COMPATIBLE = "master_weight_compatible"
    REMOVE_MASTER_WEIGHT = "remove_master_weight"
    ASYNC_SAVE = "async_save"
    IGNORE_MERGE_OPTIMIZER = "ignore_merge_optimizer"


def unwrap_optimizer(optimizer):
    while hasattr(optimizer, "_inner_opt") or hasattr(optimizer, "_optim"):
        if hasattr(optimizer, "_inner_opt"):
            optimizer = optimizer._inner_opt
        if hasattr(optimizer, "_optim"):
            optimizer = optimizer._optim
    return optimizer


def is_need_master_weight(optimizer, is_fp16_or_bp16):
    optimizer = unwrap_optimizer(optimizer)
    if hasattr(optimizer, "_multi_precision"):
        return optimizer._multi_precision and is_fp16_or_bp16
    else:
        return False


def update_master_weight_status(args, optimizer, has_master_weight, safe_serialization):
    if is_need_master_weight(optimizer, is_fp16_or_bp16=(args.fp16 or args.bf16)):
        if not has_master_weight:
            if (
                UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value in args.unified_checkpoint_config
                or UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value in args.unified_checkpoint_config
            ):
                index_filename_master_weights = (
                    PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
                )
                has_master_weight = True
                logger.warning(
                    "The unified checkpoint does not contain master weight, "
                    "the model weight will be loaded as master weight."
                )
            else:
                raise ValueError(
                    "Can't find a valid unified master weight checkpoint,"
                    f"add '{UnifiedCheckpointOption.MASTER_WEIGHT_COMPATIBLE.value}'"
                    f" or '{UnifiedCheckpointOption.REMOVE_MASTER_WEIGHT.value}' into 'unified_checkpoint_config' to "
                    "load model checkpoint as master weight"
                )
        else:
            has_master_weight = True
            index_filename_master_weights = (
                PADDLE_MASTER_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_MASTER_WEIGHTS_INDEX_NAME
            )
            if UnifiedCheckpointOption.SKIP_SAVE_MODEL_WEIGHT.value in args.unified_checkpoint_config:
                index_filename_master_weights = (
                    PADDLE_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_WEIGHTS_INDEX_NAME
                )
    else:
        has_master_weight = False
        index_filename_master_weights = None

    return has_master_weight, index_filename_master_weights


def reduce_master_weights_status(has_master_weights=False):
    """
    Get master_weight status througn tp, pp and sharding group.
    """
    data = paddle.to_tensor([has_master_weights], dtype="int32")

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    sharding_group = hcg.get_sharding_parallel_group()

    if tp_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=tp_group)
    if pp_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=pp_group)
    if sharding_group.nranks > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM, group=sharding_group)

    return data.item() > 0


def select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=True):
    """
    try select model weight index from model weight or master weight index.
    """

    # find model weight index file
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_PEFT_WEIGHTS_INDEX_NAME
    else:
        index_filename = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else PADDLE_WEIGHTS_INDEX_NAME

    index_filename_path = os.path.join(resume_from_checkpoint, index_filename)
    identify_func = os.path.isfile if local else distributed_isfile

    if identify_func(index_filename_path):
        return index_filename
    else:
        index_filename = PADDLE_MASTER_WEIGHTS_INDEX_NAME if not safe_serialization else SAFE_MASTER_WEIGHTS_INDEX_NAME
        index_filename_path = os.path.join(resume_from_checkpoint, index_filename)

        if identify_func(index_filename_path):
            return index_filename
        else:
            raise ValueError("Can't find a valid unified model or master weight checkpoint to load.")


def mapping_optimizer_tp_actions(tp_actions, optimizer_loaded_keys):
    """# convert param.name to
    param.key/moment1_0
    or param.key/beta1_XXX
    or param.key/beta2_XXX
    Args:
        tp_actions (dict): dictionary of tensor parallel actions {key: action}
        optimizer_loaded_keys (list or set): [param.key1/moment1_0, param.key2/beta1_XXX, param.key3/beta2_XXX]
    Returns:
        dict: new dictionary of tensor parallel actions {key: action}
    """
    new_actions = {}
    for key in optimizer_loaded_keys:
        key_base, typename = key.split("/")
        if typename in optimizer_non_scaler_name and key_base in tp_actions:
            new_actions[key] = tp_actions[key_base]
    return new_actions


def get_expected_state_dict(model_to_save, **kwargs):
    """
    Get trainable state_dict of model_to_save.
    """
    model_to_save = unwrap_model(model_to_save)

    if isinstance(model_to_save, PretrainedModel):
        state_dict = model_to_save.state_dict()
    elif isinstance(model_to_save, LoRAModel):
        concat_additional_adapter = kwargs.get("concat_additional_adapter", False)
        concat_init_lora = model_to_save.lora_config.loraga and concat_additional_adapter
        state_dict = model_to_save.get_trainable_state_dict(concat_init_lora=concat_init_lora)
    elif isinstance(model_to_save, PrefixModelForCausalLM):
        state_dict = model_to_save.prefix_encoder.state_dict()

    return state_dict


def get_expected_keys(args, sharded_metadata, model, optimizer, is_master_weights=False):
    hcg = fleet.get_hybrid_communicate_group()
    sharding_group = hcg.get_sharding_parallel_group()
    sharding_rank = sharding_group.rank
    in_sharding_parallel_model = sharding_group.nranks > 1
    if in_sharding_parallel_model:
        params2rank = optimizer._param2rank

    model_state_dict = get_expected_state_dict(model)
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}

    expected_keys = []
    for key in list(sharded_metadata["all_optimizer_keys"]):
        key_name = key.split("/")[0]
        if is_master_weights and key_name in model_state_dict and model_state_dict[key_name].dtype == paddle.float32:
            continue

        if args.use_expert_parallel and args.data_parallel_rank > 0:
            if key_name in model_state_dict and not getattr(model_state_dict[key_name], "no_sync", False):
                continue

        static_name = struct2static_name_mappings.get(key_name, None)

        if in_sharding_parallel_model:
            params_rank = params2rank.get(static_name, None)
            if params_rank == sharding_rank:
                expected_keys.append(key)
        else:
            if static_name is not None:
                expected_keys.append(key)
    expected_keys = set(expected_keys)

    loaded_keys = sharded_metadata["all_optimizer_keys"]
    missing_keys = expected_keys - set(loaded_keys)
    if len(missing_keys) > 0:
        raise ValueError(f"optimizer missing weights keys: {missing_keys}")

    return expected_keys


def get_optimizer_shard_files(optimizer_path, index_filename):
    """
    For a given model:
    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    For the description of each arg, see [`PretrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """

    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a optimizer index ({index_filename}) in {optimizer_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_optimizer_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    sharded_metadata["master_weights"] = index.get("master_weights", False)

    file_map = {file: set() for file in shard_filenames}
    for weight, file in index["weight_map"].items():
        file_map[file].add(weight)

    sharded_metadata["file_map"] = file_map

    # First, let's deal with local folder.
    # TODO: if optimizer_path is a folder, we should check if the optimizer is already cached or not.
    if os.path.isdir(optimizer_path):
        shard_filenames = [os.path.join(optimizer_path, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata


def generate_base_static_name(vname):
    """
    Return base static name and specific type name, like [embedding_0.w_0, moment1_0]
    """
    if FP32_MASTER in vname:
        vname = vname.split("_" + FP32_MASTER + "_")
        return vname[0], vname[1]
    else:
        # Directly deal with type names, for example: moe_gate_1_moment1_0.
        type_names = optimizer_scalar_name + optimizer_non_scaler_name
        for name in type_names:
            if name in vname:
                a = vname.split(name)[0][:-1]
                b = name
                return a, b


def merge_large_tensor_parallel(tensor, tp_group, tp_action, dst_rank, is_dst):
    """
    Move large tensor merge process to CPU, in order to avoid OOM.
    """
    num_rows = tensor.shape[0]
    num_splits = 4
    parts = np.array_split(np.arange(num_rows), num_splits)
    splits = [len(part) for part in parts]
    split_parts = np.insert(np.cumsum(splits), 0, 0)
    split_tensors = []
    for i in range(num_splits):
        if get_env_device() == "xpu":
            ret = distributed_allgather(tensor[split_parts[i] : split_parts[i + 1], :], group=tp_group, offload=False)
        else:
            ret = distributed_gather(
                tensor[split_parts[i] : split_parts[i + 1], :], dst=dst_rank, group=tp_group, offload=False
            )
        # Copy to CPUPlace temporarily, may lower speed.
        if ret is not None:
            ret = [t.cpu() for t in ret]
        split_tensors.append(ret)
    concat_tensors = []
    if is_dst:
        for i in range(tp_group.nranks):
            tmp = []
            for j in range(num_splits):
                tmp.append(split_tensors[j][i])
            concat_tensors.append(paddle.concat(tmp))
        tensor = tp_action(concat_tensors)
    else:
        tensor = None
    return tensor


def merge_tensor_parallel_with_shard(state_dict, tp_actions, all_filter_keys):
    """
    Merge tensor parallel according to tp_actions, used for model weight.
    """
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    # filter actions for pipeline mode
    if hcg.get_pipe_parallel_group().nranks > 1:
        filter_keys = set([y for x in all_filter_keys for y in x])
        for key in list(tp_actions.keys()):
            if key not in filter_keys:
                tp_actions.pop(key)

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            key = filter_keys[i]
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            mp_moe = getattr(tensor, "mp_moe", False)
            if key in tp_actions and not mp_moe:
                # Get tensor size
                tensor_bytes = tensor.numel().item() * dtype_byte_size(tensor.dtype) * tp_group.nranks
                if tensor_bytes >= 5 * 1024 * 1024 * 1024:  # temporarily set 5GB as threshold
                    tensor = merge_large_tensor_parallel(tensor, tp_group, tp_actions[key], j, is_dst)
                else:
                    if get_env_device() == "xpu":
                        ret = distributed_allgather(tensor, group=tp_group, offload=False)
                    else:
                        ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                    action = tp_actions.pop(key)
                    tensor = action(ret) if is_dst else None
            else:
                if is_dst:
                    tensor = tensor._copy_to(DEST_PLACE, False) if tensor.place.is_cpu_place() else tensor
                else:
                    tensor = None

            if is_dst:
                state_dict_to_save[key] = tensor

    if len(tp_actions) > 0:
        for x in tp_actions.keys():
            logger.debug(f"key <{x}> need to merge tensor parallel but we can't find in model state.")

    return state_dict_to_save


def merge_tensor_parallel_for_optimizer(state_dict, model_state_dict, tp_actions, all_filter_keys):
    """
    Merge tensor parallel according to tp_actions, used for master_weight and optimizer weight.
    """
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    tp_rank = tp_group.rank

    state_dict_to_save = {}
    max_key_len = max([len(_) for _ in all_filter_keys])
    for i in range(max_key_len):
        for j, filter_keys in enumerate(all_filter_keys):
            is_dst = tp_rank == j
            if i > len(filter_keys) - 1:
                continue
            # get base model key
            model_key = filter_keys[i].split("/")[0]
            if filter_keys[i] not in state_dict:
                continue
            tensor = state_dict[filter_keys[i]]
            mp_moe = getattr(model_state_dict[model_key], "mp_moe", False)
            if model_key in tp_actions and not mp_moe:
                # for example: beta1, beta2
                if tensor.numel().item() == 1:
                    if is_dst:
                        tensor = tensor._copy_to(DEST_PLACE, False) if not tensor.place.is_cpu_place() else tensor
                    else:
                        tensor = None
                else:
                    # Get tensor size
                    tensor_bytes = tensor.numel().item() * dtype_byte_size(tensor.dtype) * tp_group.nranks
                    if tensor_bytes >= 5 * 1024 * 1024 * 1024:  # temporarily set 5GB as threshold
                        tensor = merge_large_tensor_parallel(tensor, tp_group, tp_actions[model_key], j, is_dst)
                    else:
                        if get_env_device() == "xpu":
                            ret = distributed_allgather(tensor, group=tp_group, offload=False)
                        else:
                            ret = distributed_gather(tensor, dst=j, group=tp_group, offload=False)
                        action = tp_actions[model_key]
                        tensor = action(ret) if is_dst else None
            else:
                if is_dst:
                    tensor = tensor._copy_to(DEST_PLACE, False) if not tensor.place.is_cpu_place() else tensor
                else:
                    tensor = None

            if is_dst:
                state_dict_to_save[filter_keys[i]] = tensor

    return state_dict_to_save


def filter_params(model_to_save, state_dict, args, is_optimizer=False):
    """
    Group according to the size of the tensor, aiming to make the weight size
    stored on each device as equal as possible.
    """
    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()

    tp_size = tp_group.nranks
    tp_rank = tp_group.rank

    # for pure sharding or pure pp
    if tp_size <= 1:
        return [list(state_dict.keys())]

    filter_tensor_list = [[] for _ in range(tp_size)]
    is_master_weights = False

    model_state_dict = get_expected_state_dict(model_to_save)
    if tp_rank == 0:
        quant = False
        if args.ckpt_quant_stage != "O0":
            quant = True
        tensor_bytes_dict = {}
        for (k, v) in state_dict.items():
            # master weight has same key as model weight
            if not is_master_weights and k in model_state_dict:
                is_master_weights = True

            weight_key = k.split("/")[0]
            model_v = model_state_dict[weight_key] if is_optimizer else v
            mp_moe = getattr(model_v, "mp_moe", False)
            if not mp_moe:
                if not quant or not is_optimizer:
                    if hasattr(model_v, "is_distributed") and model_v.is_distributed:
                        tensor_bytes_dict[k] = v.numel().item() * tp_size * dtype_byte_size(v.dtype)
                    else:
                        tensor_bytes_dict[k] = v.numel().item() * dtype_byte_size(v.dtype)
                else:
                    if weight_key not in tensor_bytes_dict:
                        tensor_bytes_dict[weight_key] = 0

                    if hasattr(model_v, "is_distributed") and model_v.is_distributed:
                        tensor_bytes_dict[weight_key] += v.numel().item() * tp_size * dtype_byte_size(v.dtype)
                    else:
                        tensor_bytes_dict[weight_key] += v.numel().item() * dtype_byte_size(v.dtype)

        filter_tensor_list = []
        current_block = []
        current_block_size = 0
        total_size = 0

        max_shard_size = (sum(tensor_bytes_dict.values()) + tp_size - 1) // tp_size

        for index, (key, weight_size) in enumerate(tensor_bytes_dict.items()):
            # If this weight is going to tip up over the maximal size, we split.
            # if current_block_size + weight_size > max_shard_size:
            if total_size + weight_size > max_shard_size * (len(filter_tensor_list) + 1) or (
                len(tensor_bytes_dict) - index < (tp_size - len(filter_tensor_list))
            ):
                # fix if the first param is large than max_shard_size
                if len(current_block) > 0:
                    filter_tensor_list.append(current_block)
                current_block = []
                current_block_size = 0

            if not quant or not is_optimizer or is_master_weights:
                current_block.append(key)
            else:
                current_block.append(key + "/" + MOMENT1_KEYNAME)
                current_block.append(key + "/" + MOMENT2_KEYNAME)
                current_block.append(key + "/" + BETA1_KEYNAME)
                current_block.append(key + "/" + BETA2_KEYNAME)

            current_block_size += weight_size
            total_size += weight_size

        filter_tensor_list.append(current_block)
        if len(filter_tensor_list) < tp_size:
            filter_tensor_list.extend([[] for i in range(tp_size - len(filter_tensor_list))])

    dist.broadcast_object_list(
        filter_tensor_list,
        src=hcg.get_model_parallel_group_src_rank(),
        group=tp_group,
    )

    # deal with expert parameters in model parallel group.
    for (k, v) in state_dict.items():
        weight_key = k.split("/")[0]
        model_v = model_state_dict[weight_key] if is_optimizer else v
        mp_moe = getattr(model_v, "mp_moe", False)
        if mp_moe:
            filter_tensor_list[tp_rank].append(k)

    final_filter_tensor_list = []
    dist.all_gather_object(final_filter_tensor_list, filter_tensor_list[tp_rank], group=tp_group)

    return final_filter_tensor_list


def get_sharded_file_name(args, file_name, is_optimizer=False):
    """
    Get safetensors file name for saving.
    """
    if not is_optimizer:
        sd_degree = args.sharding_parallel_degree if args.sharding_parallel_degree > 1 else 1
        size = sd_degree if args.use_expert_parallel else args.dataset_world_size
        shard_file = file_name.replace(
            ".pdparams",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//size:05d}.pdparams",
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//size:05d}.safetensors",
        )
    else:
        hcg = fleet.get_hybrid_communicate_group()
        dp_group = hcg.get_data_parallel_group()
        size = dp_group.nranks if not args.use_expert_parallel else 1
        shard_file = file_name.replace(
            ".pdparams", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//size:05d}.pdparams"
        )
        shard_file = shard_file.replace(
            ".safetensors",
            f"-{args.logical_process_index + 1:05d}-of-{args.world_size//size:05d}.safetensors",
        )
        shard_file = shard_file.replace(
            ".pdopt", f"-{args.logical_process_index + 1:05d}-of-{args.world_size//size:05d}.pdopt"
        )
    return shard_file


def get_sharded_index(
    index_file_list,
    total_size_list,
):
    """
    Save safetensors index json file, including metadata and weight_map.
    """
    local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))
    if local_rank == 0:
        sharded_index_json = {}

        sharded_index_json["metadata"] = {"total_size": sum(total_size_list)}

        weight_map = {}
        for i, _ in enumerate(index_file_list):
            weight_map.update(index_file_list[i])

        sharded_index_json["weight_map"] = weight_map
        return sharded_index_json

    return None


def gather_sharded_object(index_file, total_size, is_optimizer=False, use_expert_parallel=False):
    """
    All gather sharded files list across different groups.
    """
    index_file_list, total_size_list = [], []

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()

    logger.info(
        f"Unified checkpoint: generating sharded_index json files for {'optimizer or master weight' if is_optimizer else 'model weight'}."
    )

    if tp_group.nranks > 1:
        dist.all_gather_object(index_file_list, index_file, tp_group)
        dist.all_gather_object(total_size_list, total_size, tp_group)
    if pp_group.nranks > 1:
        pp_index_file_list = []
        pp_total_size_list = []
        dist.all_gather_object(
            pp_index_file_list, index_file_list if len(index_file_list) > 0 else index_file, pp_group
        )
        dist.all_gather_object(
            pp_total_size_list, total_size_list if len(total_size_list) > 0 else total_size, pp_group
        )
        index_file_list = pp_index_file_list
        total_size_list = pp_total_size_list

    index_file_list = flatten_list(index_file_list)
    total_size_list = flatten_list(total_size_list)

    # for pure sharding
    if len(index_file_list) == 0 and len(total_size_list) == 0:
        index_file_list = [index_file]
        total_size_list = [total_size]

    if use_expert_parallel:
        data_group = hcg.get_data_parallel_group()
        if data_group.nranks > 1:
            data_index_file_list = []
            data_total_size_list = []
            dist.all_gather_object(data_index_file_list, index_file_list, data_group)
            dist.all_gather_object(data_total_size_list, total_size_list, data_group)
            index_file_list = flatten_list(data_index_file_list)
            total_size_list = flatten_list(data_total_size_list)

    if is_optimizer:
        sharding_group = hcg.get_sharding_parallel_group()
        if sharding_group.nranks > 1:
            sharding_index_file_list = []
            sharding_total_size_list = []
            dist.all_gather_object(sharding_index_file_list, index_file_list, sharding_group)
            dist.all_gather_object(sharding_total_size_list, total_size_list, sharding_group)
            index_file_list = flatten_list(sharding_index_file_list)
            total_size_list = flatten_list(sharding_total_size_list)

    return index_file_list, total_size_list


def rename_shard_file(args, shard_file, file_name):
    """
    Rename shard file when using expert_parallel.
    """
    assert args.use_expert_parallel, "only expert_parallel need to use this function"

    shard_file_list = []

    hcg = fleet.get_hybrid_communicate_group()
    tp_group = hcg.get_model_parallel_group()
    pp_group = hcg.get_pipe_parallel_group()
    data_group = hcg.get_data_parallel_group()

    if tp_group.nranks > 1:
        dist.all_gather_object(shard_file_list, shard_file, tp_group)
    if pp_group.nranks > 1:
        pp_shard_file_list = []
        dist.all_gather_object(
            pp_shard_file_list, shard_file_list if len(shard_file_list) > 0 else shard_file, pp_group
        )
        shard_file_list = flatten_list(pp_shard_file_list)
    if data_group.nranks > 1:
        data_shard_file_list = []
        dist.all_gather_object(
            data_shard_file_list, shard_file_list if len(shard_file_list) > 0 else shard_file, data_group
        )
        shard_file_list = flatten_list(data_shard_file_list)

    new_index = shard_file_list.index(shard_file)
    sd_degree = args.sharding_parallel_degree if args.sharding_parallel_degree > 1 else 1
    shard_file = file_name.replace(
        ".pdparams",
        f"-{new_index + 1:05d}-of-{args.world_size//sd_degree:05d}.pdparams",
    )
    shard_file = shard_file.replace(
        ".safetensors",
        f"-{new_index + 1:05d}-of-{args.world_size//sd_degree:05d}.safetensors",
    )
    return shard_file


def save_prefix_past_key_value(model_to_save, save_directory):
    """
    Used only for PrefixModelForCausalLM.
    """
    past_key_value = model_to_save.prefix_encoder(model_to_save.prefix_tokens.unsqueeze(0).expand([1, -1]))
    past_key_value = past_key_value.reshape(
        [
            model_to_save.prefix_config.num_prefix_tokens,
            2,
            model_to_save.prefix_config.num_hidden_layers,
            model_to_save.num_heads,
            model_to_save.head_dim,
        ]
    )
    past_key_value = paddle.transpose(past_key_value, perm=[2, 1, 3, 0, 4]).cpu().numpy()
    np.save(os.path.join(save_directory, PAST_KEY_VALUES_FILE_NAME), past_key_value)


def is_sharding_split_param_mode(args):
    return (
        args.sharding_parallel_degree > 1
        and ShardingOption.SHARD_OP in args.sharding
        and "split_param" in args.sharding_parallel_config
    )


def save_model_config(model_to_save, save_directory):
    """
    Save model config.
    """

    def save_config(model_to_save):
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.dtype = str(dtype).split(".")[1]
        config_to_save = copy.deepcopy(model_to_save.config)

        if config_to_save.tensor_parallel_degree > 1:
            # do we need to change?
            config_to_save.tensor_parallel_degree = 1

        return config_to_save

    # Save prefix model past_key_values
    if isinstance(model_to_save, PrefixModelForCausalLM):
        save_prefix_past_key_value(model_to_save, save_directory)
        model_to_save.prefix_config.save_pretrained(save_directory)
    if isinstance(model_to_save, LoRAModel):
        model_to_save.lora_config.save_pretrained(save_directory)

    # save the config
    config_to_save = save_config(model_to_save)
    # Attach architecture to the config
    if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
        config_to_save.architectures = [clean_model_class_name(model_to_save.model.__class__.__name__)]
    else:
        config_to_save.architectures = [clean_model_class_name(model_to_save.__class__.__name__)]

    config_to_save.save_pretrained(save_directory)
    # save generation config
    if model_to_save.can_generate():
        model_to_save.generation_config.save_pretrained(save_directory)


def filter_sync_parameters(model_state_dict, optim_state_dict=None, master_weights=None, is_model_weight=True):
    """Filter sync parameters under expert parallel mode."""

    hcg = fleet.get_hybrid_communicate_group()
    dp_group = hcg.get_data_parallel_group()
    dp_rank = dp_group.rank if dp_group.nranks > 1 else 0

    if is_model_weight:
        for key in list(model_state_dict.keys()):
            if dp_rank > 0 and not getattr(model_state_dict[key], "no_sync", False):
                model_state_dict.pop(key)
    else:
        no_sync_kname = []
        for k, v in model_state_dict.items():
            if getattr(v, "no_sync", False):
                no_sync_kname.append(k)

        for key in list(optim_state_dict.keys()):
            model_key = key.split("/")[0]
            if dp_rank > 0 and model_key not in no_sync_kname:
                optim_state_dict.pop(key)

        if master_weights is not None:
            for key in list(master_weights.keys()):
                if dp_rank > 0 and key not in no_sync_kname:
                    master_weights.pop(key)
