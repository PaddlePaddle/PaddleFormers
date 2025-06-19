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
"""Unfied checkpoint locally loading functions."""

import gc
import json
import os

import paddle
from tqdm.auto import tqdm

from ...peft import LoRAModel, PrefixModelForCausalLM
from ...transformers.model_utils import (
    _load_state_dict_into_model,
    faster_set_state_dict,
    load_state_dict,
)
from ...transformers.utils import get_checkpoint_shard_files
from ...utils.env import (
    PADDLE_MASTER_WEIGHTS_INDEX_NAME,
    PADDLE_OPTIMIZER_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
)
from ...utils.log import logger
from ...utils.nested import nested_copy
from .sharding_split_param_utils import load_unified_optimizer_split_param
from .utils import (
    FP32_MASTER,
    get_expected_keys,
    get_expected_state_dict,
    get_optimizer_shard_files,
    is_sharding_split_param_mode,
    mapping_optimizer_tp_actions,
    select_model_weight_index,
    update_master_weight_status,
)

__all__ = ["load_unified_checkpoint_locally", "load_unified_optimizer_locally"]


def load_unified_checkpoint_locally(args, model, resume_from_checkpoint: str, safe_serialization=False):
    """
    Only dataset_rank == 0 or using expert parallel can enter this function.
    """
    index_filename = select_model_weight_index(model, resume_from_checkpoint, safe_serialization, local=True)

    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    loaded_keys = sharded_metadata["all_checkpoint_keys"]

    model_state_dict = get_expected_state_dict(model)
    # If using expert parallel, when dp_rank > 0, need to modify the expected_keys here.
    if not args.use_expert_parallel or (args.use_expert_parallel and args.data_parallel_rank == 0):
        expected_keys = set(list(model_state_dict.keys()))
    else:
        expected_keys = set()
        for key in model_state_dict.keys():
            if getattr(model_state_dict[key], "no_sync", False):
                expected_keys.add(key)
    missing_keys = expected_keys - set(loaded_keys)

    use_fast_set = True
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        use_fast_set = False

    if len(missing_keys) > 0:
        raise ValueError(f"missing_keys: {missing_keys}")

    def _remove_unused_keys(
        state_dict,
        model_state_dict,
    ):
        unused_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        for unused_key in unused_keys:
            del state_dict[unused_key]
        return unused_keys

    # This should always be a list but, just to be sure.
    if not isinstance(resolved_archive_file, list):
        resolved_archive_file = [resolved_archive_file]

    error_msgs = []

    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

    for shard_file in resolved_archive_file:
        # TODO: check if  no expected_keys in shard_file, then don't load it
        if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
            continue

        pre_tensor_parallel_split = False
        if shard_file.endswith(".safetensors") and model.config.tensor_parallel_degree > 1:
            pre_tensor_parallel_split = True
            assert loaded_keys is not None, "loaded_keys is not None."
            if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                tp_actions = model._get_tensor_parallel_convert_actions(
                    set(loaded_keys), is_split=True, ignore_error=True
                )
            else:
                tp_actions = model.get_tensor_parallel_convert_actions(model.config, loaded_keys, ignore_error=True)
        # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
        state_dict = load_state_dict(
            shard_file, tp_actions if pre_tensor_parallel_split else None, expected_keys, device="expected"
        )

        if not pre_tensor_parallel_split:
            # Since we load all keys but we only need one of pipeline stages
            _ = _remove_unused_keys(state_dict, model_state_dict)

        if model.config.tensor_parallel_degree > 1 and not pre_tensor_parallel_split:
            logger.info("Converting state_dict to Tensor Parallel Format")
            # ignore error for multi shard, since only parts of data
            state_dict = model.convert_tensor_parallel(
                None, model.config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
            )

        if use_fast_set:
            error_msgs += faster_set_state_dict(model, state_dict, strict_dtype=False)
        else:
            error_msgs += _load_state_dict_into_model(model, state_dict, "")

        # force memory release
        del state_dict
        # gc.collect()

    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if " but the expected shape is" in error_msg:
            error_msg += (
                "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            )
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")


def load_unified_optimizer_locally(args, model, optimizer, resume_from_checkpoint, safe_serialization=False):
    if not safe_serialization:
        index_filename, index_filename_master_weights = (
            PADDLE_OPTIMIZER_INDEX_NAME,
            PADDLE_MASTER_WEIGHTS_INDEX_NAME,
        )
    else:
        index_filename, index_filename_master_weights = SAFE_OPTIMIZER_INDEX_NAME, SAFE_MASTER_WEIGHTS_INDEX_NAME

    with open(os.path.join(resume_from_checkpoint, index_filename), "r") as f:
        index = json.loads(f.read())

    ckpt_quant_stage = "O0"
    if "ckpt_quant_stage" in index:
        ckpt_quant_stage = index["ckpt_quant_stage"]

    # Special process with split param.
    if is_sharding_split_param_mode(args):
        returned_optim_state_dict = load_unified_optimizer_split_param(
            args, model, optimizer, resume_from_checkpoint, ckpt_quant_stage
        )
        return returned_optim_state_dict

    # init and get optimizer LR_Scheduler
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    model_state_dict = get_expected_state_dict(model)
    model_keys = list(model_state_dict.keys())
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}  # get optimizer param mappings

    expected_keys = get_expected_keys(args, sharded_metadata, model, optimizer)

    # This should always be a list but, just to be sure.
    if not isinstance(resolved_archive_file, list):
        resolved_archive_file = [resolved_archive_file]

    if len(resolved_archive_file) > 1:
        resolved_archive_file = tqdm(resolved_archive_file, desc="Loading optimizer shards")

    # update has_master_weights and index_filename_master_weights
    # 1. if the master weight exists, only has_master_weights is set True and loaded when needed
    # 2. if master weight does not exist, convert model weight to master weight when needed
    has_master_weights, index_filename_master_weights = update_master_weight_status(
        args, optimizer, has_master_weights, safe_serialization
    )

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}

        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, index_filename_master_weights),
        )

        expected_keys_mw = get_expected_keys(args, sharded_metadata_mw, model, optimizer, is_master_weights=True)
        if not isinstance(resolved_archive_file_mw, list):
            resolved_archive_file_mw = [resolved_archive_file_mw]
        if len(resolved_archive_file_mw) > 1:
            resolved_archive_file_mw = tqdm(resolved_archive_file_mw, desc="Loading master weights shards")

    def load_resolved_archive_file(
        resolved_archive_file, sharded_metadata, expected_keys, is_master_weights=False, ckpt_quant_stage="O0"
    ):
        returned_state_dict = {}
        # load optimizer
        for shard_file in resolved_archive_file:
            # TODO: check if no expected_keys in shard_file, then don't load it
            if expected_keys.isdisjoint(sharded_metadata["file_map"][os.path.split(shard_file)[-1]]):
                continue

            if shard_file.endswith(".safetensors"):
                # assert model_keys is not None, "model_keys is None." TODO: correct the assert
                if model.config.tensor_parallel_degree > 1:
                    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
                        tp_actions = model._get_tensor_parallel_convert_actions(
                            model_keys, is_split=True, ignore_error=True
                        )
                    else:
                        tp_actions = model.get_tensor_parallel_convert_actions(
                            model.config, model_keys, ignore_error=True
                        )
                    if not is_master_weights:
                        tp_actions = mapping_optimizer_tp_actions(tp_actions, expected_keys)

                    # Here we use expected_keys to optimize weights loading for pipeline model. Only works for safetensors
                    state_dict = load_state_dict(
                        shard_file,
                        tp_actions,
                        expected_keys,
                        device="expected",
                        ckpt_quant_stage=ckpt_quant_stage,
                    )
                else:
                    # for pipeline model, we don't need to use tp_actions
                    state_dict = load_state_dict(
                        shard_file,
                        None,
                        expected_keys,
                        device="expected",
                        ckpt_quant_stage=ckpt_quant_stage,
                    )

            returned_state_dict.update(state_dict)
            # force memory release
            del state_dict
            gc.collect()
        return returned_state_dict

    state_dict_optim = load_resolved_archive_file(
        resolved_archive_file, sharded_metadata, expected_keys, ckpt_quant_stage=ckpt_quant_stage
    )
    if has_master_weights:
        state_dict_master_weight = load_resolved_archive_file(
            resolved_archive_file_mw, sharded_metadata_mw, expected_keys_mw, is_master_weights=True
        )
    # rename optimizer param
    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        model_weight_key = key_name[0]
        static_name = struct2static_name_mappings[model_weight_key]
        if has_master_weights:
            if model_state_dict[model_weight_key].dtype != paddle.float32:
                key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        returned_optim_state_dict[key_name] = state_dict_optim.pop(key)
        returned_optim_state_dict[key_name].name = key_name

    if has_master_weights:
        for key in list(state_dict_master_weight.keys()):
            static_name = struct2static_name_mappings[key]
            returned_optim_state_dict["master_weights"][static_name] = state_dict_master_weight.pop(key)
            # master weight cast (only in remove_master_weight)
            if returned_optim_state_dict["master_weights"][static_name].dtype != paddle.float32:
                returned_optim_state_dict["master_weights"][static_name] = paddle.cast(
                    returned_optim_state_dict["master_weights"][static_name], dtype=paddle.float32
                )
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])

    return returned_optim_state_dict
