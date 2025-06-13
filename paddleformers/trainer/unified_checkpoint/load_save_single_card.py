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
"""Save and load single card checkpoint for Unified Checkpoint"""

import gc
import json
import os

import paddle

from ...peft import LoRAModel, PrefixModelForCausalLM
from ...transformers.model_utils import _load_state_dict_into_model, load_state_dict
from ...transformers.utils import (
    dtype_byte_size,
    get_checkpoint_shard_files,
    is_safetensors_available,
)
from ...utils.env import (
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_OPTIMIZER_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from ...utils.log import logger
from ...utils.nested import nested_copy

if is_safetensors_available():
    from safetensors.numpy import save_file as safe_save_file

from .utils import (
    FP32_MASTER,
    generate_base_static_name,
    get_expected_state_dict,
    get_optimizer_shard_files,
    save_model_config,
)

__all__ = [
    "load_single_card_checkpoint",
    "load_single_card_optimizer",
    "save_single_card_checkpoint",
    "save_single_card_optimizer",
]


def save_file_sync(state_dict, path):
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            state_dict[k] = state_dict.pop(k).cpu().numpy()
    safe_save_file(state_dict, path, metadata={"format": "np"})


def save_single_card_checkpoint(model_to_save, output_dir):
    """Save checkpoint for non-distributed environment."""

    state_dict = get_expected_state_dict(model_to_save, concat_additional_adapter=True)
    if isinstance(model_to_save, LoRAModel) or isinstance(model_to_save, PrefixModelForCausalLM):
        weight_filename = "peft_model-00001-of-00001.safetensors"
        index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME
    else:
        weight_filename = "model-00001-of-00001.safetensors"
        index_filename = SAFE_WEIGHTS_INDEX_NAME
    # get index json
    index_weight_file = {}
    total_size = 0
    for key, weight in state_dict.items():
        index_weight_file[key] = weight_filename
        total_size += weight.numel().item() * dtype_byte_size(weight.dtype)
    sharded_index_json = {}
    sharded_index_json["metadata"] = {"total_size": total_size}
    sharded_index_json["weight_map"] = index_weight_file
    if isinstance(model_to_save, LoRAModel):
        sharded_index_json["type"] = "lora"
    elif isinstance(model_to_save, PrefixModelForCausalLM):
        sharded_index_json["type"] = "ptuning"

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, index_filename)
    with open(path, "w") as f:
        json.dump(sharded_index_json, f, indent=4)

    # save checkpoint, do no support asynchronous save for single card currently.
    logger.warning("Asynchronous saving is not supported for single card environment currently.")
    save_file_sync(state_dict, path=os.path.join(output_dir, weight_filename))

    save_model_config(model_to_save, output_dir)


def save_single_card_optimizer(model, optimizer, output_dir):
    """ "Save optimizer for non-distributed environment."""
    # Split into optimizer params and master weights.
    optim_state_dict = nested_copy(optimizer.state_dict())
    master_weights = None
    if "master_weights" in optim_state_dict.keys():
        master_weights = optim_state_dict.pop("master_weights")
    if "LR_Scheduler" in optim_state_dict.keys():
        optim_state_dict.pop("LR_Scheduler")

    static2struct_name_mappings = {}
    state_dict = get_expected_state_dict(model)
    fp32_weight = {}
    for k, v in state_dict.items():
        static2struct_name_mappings[v.name] = k
        if master_weights is not None and v.dtype == paddle.float32:
            fp32_weight[k] = v

    # rename optimizer param
    for key in list(optim_state_dict.keys()):
        static_name, type_name = generate_base_static_name(key)
        new_name = static2struct_name_mappings[static_name] + "/" + type_name
        optim_state_dict[new_name] = optim_state_dict.pop(key)
    if master_weights is not None:
        for key in list(master_weights.keys()):
            master_weights[static2struct_name_mappings[key]] = master_weights.pop(key)
        master_weights.update(fp32_weight)

    # save index json
    index_optimizer_file, index_master_weight_file = {}, {}
    total_optim_size, total_master_weight_size = 0, 0
    for key, weight in optim_state_dict.items():
        index_optimizer_file[key] = "optimizer-00001-of-00001.safetensors"
        total_optim_size += weight.numel().item() * dtype_byte_size(weight.dtype)
    if master_weights is not None:
        for key, weight in master_weights.items():
            index_master_weight_file[key] = "master_weights-00001-of-00001.safetensors"
            total_master_weight_size += weight.numel().item() * dtype_byte_size(weight.dtype)
    path = os.path.join(output_dir, SAFE_OPTIMIZER_INDEX_NAME)
    master_path = os.path.join(output_dir, SAFE_MASTER_WEIGHTS_INDEX_NAME)
    with open(path, "w") as f:
        has_master_weights = master_weights is not None
        json.dump(
            {
                "metadata": {"total_size": total_optim_size},
                "weight_map": index_optimizer_file,
                "master_weights": has_master_weights,
            },
            f,
            indent=4,
        )
    if master_weights is not None:
        with open(master_path, "w") as f:
            json.dump(
                {"metadata": {"total_size": total_master_weight_size}, "weight_map": index_master_weight_file},
                f,
                indent=4,
            )

    # save optimizer state dict
    save_file_sync(optim_state_dict, path=os.path.join(output_dir, "optimizer-00001-of-00001.safetensors"))
    if master_weights is not None:
        save_file_sync(master_weights, path=os.path.join(output_dir, "master_weights-00001-of-00001.safetensors"))


def load_single_card_checkpoint(model, resume_from_checkpoint: str):
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        index_filename = SAFE_PEFT_WEIGHTS_INDEX_NAME
    else:
        index_filename = SAFE_WEIGHTS_INDEX_NAME
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        pretrained_model_name_or_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, index_filename),
    )

    loaded_keys = sharded_metadata["all_checkpoint_keys"]
    model_state_dict = get_expected_state_dict(model)
    expected_keys = set(list(model_state_dict.keys()))
    missing_keys = expected_keys - set(loaded_keys)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {missing_keys}")

    state_dict = load_state_dict(resolved_archive_file[0], None, expected_keys)
    error_msgs = _load_state_dict_into_model(model, state_dict, "")
    del state_dict
    gc.collect()

    if error_msgs:
        raise RuntimeError(f"Error(s) in loading state dict for {model.__class__.__name__}:\n\t{error_msgs}")


def load_single_card_optimizer(model, optimizer, resume_from_checkpoint: str):
    returned_optim_state_dict = nested_copy(optimizer.state_dict())

    resolved_archive_file, sharded_metadata = get_optimizer_shard_files(
        optimizer_path=resume_from_checkpoint,
        index_filename=os.path.join(resume_from_checkpoint, SAFE_OPTIMIZER_INDEX_NAME),
    )
    has_master_weights = True if sharded_metadata["master_weights"] else False

    model_state_dict = get_expected_state_dict(model)
    struct2static_name_mappings = {k: v.name for k, v in model_state_dict.items()}
    expected_keys = sharded_metadata["all_optimizer_keys"]

    if has_master_weights:
        returned_optim_state_dict["master_weights"] = {}
        resolved_archive_file_mw, sharded_metadata_mw = get_optimizer_shard_files(
            optimizer_path=resume_from_checkpoint,
            index_filename=os.path.join(resume_from_checkpoint, SAFE_MASTER_WEIGHTS_INDEX_NAME),
        )
        expected_keys_mw = sharded_metadata_mw["all_optimizer_keys"]

    state_dict_optim = load_state_dict(resolved_archive_file[0], None, expected_keys)
    if has_master_weights:
        state_dict_optim_mw = load_state_dict(resolved_archive_file_mw[0], None, expected_keys_mw)

    for key in list(state_dict_optim.keys()):
        key_name = key.split("/")
        static_name = struct2static_name_mappings[key_name[0]]
        if has_master_weights:
            if model_state_dict[key_name[0]].dtype != paddle.float32:
                key_name = "_".join([static_name, FP32_MASTER, key_name[1]])
            else:
                key_name = "_".join([static_name, key_name[1]])
        else:
            key_name = "_".join([static_name, key_name[1]])
        returned_optim_state_dict[key_name] = state_dict_optim.pop(key)
        returned_optim_state_dict[key_name].name = key_name
    if has_master_weights:
        for key in list(state_dict_optim_mw.keys()):
            static_name = struct2static_name_mappings[key]
            returned_optim_state_dict["master_weights"][static_name] = state_dict_optim_mw.pop(key)
            returned_optim_state_dict["master_weights"][static_name].name = "_".join([static_name, FP32_MASTER])
    return returned_optim_state_dict
