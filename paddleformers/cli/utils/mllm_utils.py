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
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from paddleformers.utils.log import logger

_MULTIMODEL_KEY_REGISTRY: Dict[str, MultiModelKeys] = {}
_ALL_MODULES = ["vision", "aligner", "llm"]


class MLLMModelMapping:
    qwen2_5_vl = "qwen2_5_vl"
    qwen3_vl = "qwen3_vl"
    qwen3_vl_moe = "qwen3_vl_moe"
    paddleocr_vl = "paddleocr_vl"
    ernie4_5_moe_vl = "ernie4_5_moe_vl"


@dataclass
class ModelKeys:
    model_dtype: str

    embedding: Optional[str] = None
    module_list: Optional[str] = None
    lm_head: Optional[str] = None

    q_proj: Optional[str] = None
    k_proj: Optional[str] = None
    v_proj: Optional[str] = None
    o_proj: Optional[str] = None
    mlp: Optional[str] = None


@dataclass
class MultiModelKeys(ModelKeys):
    llm: Union[str, List[str]] = field(default_factory=list)
    aligner: Union[str, List[str]] = field(default_factory=list)
    vision: Union[str, List[str]] = field(default_factory=list)

    def __post_init__(self):
        for key in ["llm", "aligner", "vision"]:
            v = getattr(self, key)
            if isinstance(v, str):
                setattr(self, key, [v])
            elif v is None:
                setattr(self, key, [])


def register_multimodel_keys(multimodel_key: ModelKeys, *, exist_ok: bool = False) -> None:
    model_dtype = multimodel_key.model_dtype
    if not exist_ok and model_dtype in _MULTIMODEL_KEY_REGISTRY:
        raise ValueError(f"The `{model_dtype}` has already been registered.")
    _MULTIMODEL_KEY_REGISTRY[model_dtype] = multimodel_key


def get_multimodel_target_modules(model_type: Optional[str]) -> Optional[Union[ModelKeys, MultiModelKeys]]:
    if not model_type:
        return None
    return _MULTIMODEL_KEY_REGISTRY.get(model_type)


def get_multimodel_lora_target_modules(model, target_modules, freeze_config):

    model_type = model.config.model_type

    multimodel_keys = get_multimodel_target_modules(model_type)
    if not multimodel_keys:
        logger.warning(
            f"No MultiModelKeys registered for {model_type}. "
            f"'freeze_config' only supports MLLM models and will not take effect here."
        )
        return target_modules

    prefix_to_module = {}

    for module in _ALL_MODULES:
        prefixes = (
            multimodel_keys.get(module, [])
            if isinstance(multimodel_keys, dict)
            else getattr(multimodel_keys, module, [])
        )
        for p in prefixes:
            prefix_to_module[p] = module

    sorted_prefixes = sorted(prefix_to_module.keys(), key=len, reverse=True)
    active_freeze_config = {m for m in _ALL_MODULES if f"freeze_{m}" in freeze_config}

    multimodel_target_modules = []
    removed_info = defaultdict(list)

    for tm in target_modules:
        remove_module = None

        for prefix in sorted_prefixes:
            if prefix in tm:
                remove_module = prefix_to_module[prefix]
                break

        if remove_module and remove_module in active_freeze_config:
            removed_info[remove_module].append(tm)
        else:
            multimodel_target_modules.append(tm)

    if removed_info:
        log_info = [f"LoRA target modules filtered by [{freeze_config}]:"]
        for module, keys in removed_info.items():
            log_info.append(f"+ [{module}] removed {len(keys)} targets:")
            log_info.extend([f"  - {k}" for k in keys])
        logger.info("\n".join(log_info))

    return multimodel_target_modules


def freeze_model_parameters(model, freeze_config):

    if not (hasattr(model, "config") and hasattr(model.config, "model_type")):
        logger.warning("Model has no config.model_type, skip freezing.")
        return
    model_type = model.config.model_type

    multimodel_keys = get_multimodel_target_modules(model_type)
    if not multimodel_keys:
        logger.warning(
            f"No MultiModelKeys registered for {model_type}. "
            f"'freeze_config' only supports MLLM models and will not take effect here."
        )
        return

    prefix_to_module = {}

    for module in _ALL_MODULES:
        prefixes = (
            multimodel_keys.get(module, [])
            if isinstance(multimodel_keys, dict)
            else getattr(multimodel_keys, module, [])
        )
        for p in prefixes:
            prefix_to_module[p] = module

    sorted_prefixes = sorted(prefix_to_module.keys(), key=len, reverse=True)
    active_freeze_config = {m for m in _ALL_MODULES if f"freeze_{m}" in freeze_config}
    full_pattern = re.compile("^(" + "|".join(re.escape(p) for p in sorted_prefixes) + ")")

    frozen_keys = defaultdict(list)

    for name, param in model.named_parameters():
        match = full_pattern.match(name)

        if match:
            matched_prefix = match.group()
            module_name = prefix_to_module[matched_prefix]
            if module_name in active_freeze_config:
                param.stop_gradient = True
                frozen_keys[module_name].append(name)
        else:
            param.stop_gradient = False

    if frozen_keys:
        active_modules = ", ".join([f"freeze_{k}" for k in sorted(frozen_keys.keys())])
        total_count = sum(len(k) for k in frozen_keys.values())

        log_info = [f"Freeze Config: {active_modules} || Total Frozen Keys: {total_count}"]

        for module_name, keys in frozen_keys.items():
            patterns = sorted({re.sub(r"\.\d+\.", ".$LAYEY_ID.", k) for k in keys})
            log_info.append(f"+ [{module_name}] ({len(keys)} keys)")
            log_info.extend([f"  - {p}" for p in patterns])
        logger.info("\n".join(log_info))


register_multimodel_keys(
    MultiModelKeys(
        model_dtype=MLLMModelMapping.qwen2_5_vl,
        aligner="model.visual.merger",
        llm=["model.language_model", "lm_head"],
        vision="model.visual",
    )
)
register_multimodel_keys(
    MultiModelKeys(
        model_dtype=MLLMModelMapping.qwen3_vl,
        aligner=["model.visual.merger", "model.visual.deepstack_merger_list"],
        llm=["model.language_model", "lm_head"],
        vision="model.visual",
    )
)
register_multimodel_keys(
    MultiModelKeys(
        model_dtype=MLLMModelMapping.qwen3_vl_moe,
        aligner=["model.visual.merger", "model.visual.deepstack_merger_list"],
        llm=["model.language_model", "lm_head"],
        vision="model.visual",
    )
)

register_multimodel_keys(
    MultiModelKeys(
        model_dtype=MLLMModelMapping.paddleocr_vl,
        aligner=["mlp_AR"],
        llm=["model", "lm_head"],
        vision="visual",
    )
)

register_multimodel_keys(
    MultiModelKeys(
        model_dtype=MLLMModelMapping.ernie4_5_moe_vl,
        aligner="resampler_model",
        llm=["model", "lm_head", "mlp", "self_attn"],
        vision="vision_model",
    )
)
