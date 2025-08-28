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

import copy
import gc
import math
import os
import re
import tempfile
from collections import OrderedDict
from functools import partial
from typing import Dict, Union

import aistudio_sdk
import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import (
    ColumnParallelLinear,
    PipelineLayer,
    RowParallelLinear,
)

from ...transformers import linear_utils
from ...transformers.conversion_utils import ConversionMixin
from ...transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    _load_state_dict_into_model,
    dtype_guard,
    load_state_dict,
)
from ...transformers.utils import get_checkpoint_shard_files, weight_name_suffix
from ...utils.distributed import distributed_allgather, distributed_gather
from ...utils.env import LORA_WEIGHTS_NAME, SAFE_PEFT_WEIGHTS_INDEX_NAME
from ...utils.log import logger
from ...utils.tools import get_env_device
from .lora_config import LoRAAutoConfig, LoRAConfig


def get_lora_layers():
    try:
        if get_env_device() == "xpu":
            # If paddle_xpu is not installed, just use PaddleFormers's native lora layers
            from paddle_xpu.layers.nn.lora_layers import (
                XPUColumnParallelLoRALinear as ColumnParallelLoRALinear,
            )
            from paddle_xpu.layers.nn.lora_layers import (
                XPUColumnSequenceParallelLoRALinear as ColumnSequenceParallelLoRALinear,
            )
            from paddle_xpu.layers.nn.lora_layers import XPULoRALinear as LoRALinear
            from paddle_xpu.layers.nn.lora_layers import (
                XPURowParallelLoRALinear as RowParallelLoRALinear,
            )
            from paddle_xpu.layers.nn.lora_layers import (
                XPURowSequenceParallelLoRALinear as RowSequenceParallelLoRALinear,
            )

            from .lora_layers import LoRAConv2D
        else:
            raise ImportError  # Force to use the fallback if not XPU
    except ImportError:
        from .lora_layers import (
            ColumnParallelLoRALinear,
            ColumnSequenceParallelLoRALinear,
            LoRAConv2D,
            LoRALinear,
            RowParallelLoRALinear,
            RowSequenceParallelLoRALinear,
        )

    return {
        "ColumnParallelLoRALinear": ColumnParallelLoRALinear,
        "ColumnSequenceParallelLoRALinear": ColumnSequenceParallelLoRALinear,
        "LoRAConv2D": LoRAConv2D,
        "LoRALinear": LoRALinear,
        "RowParallelLoRALinear": RowParallelLoRALinear,
        "RowSequenceParallelLoRALinear": RowSequenceParallelLoRALinear,
    }


lora_layers = get_lora_layers()
ColumnParallelLoRALinear = lora_layers["ColumnParallelLoRALinear"]
ColumnSequenceParallelLoRALinear = lora_layers["ColumnSequenceParallelLoRALinear"]
LoRAConv2D = lora_layers["LoRAConv2D"]
LoRALinear = lora_layers["LoRALinear"]
RowParallelLoRALinear = lora_layers["RowParallelLoRALinear"]
RowSequenceParallelLoRALinear = lora_layers["RowSequenceParallelLoRALinear"]

from ...quantization.quantization_linear import (
    ColumnParallelQuantizationLinear,
    QuantizationLinear,
    RowParallelQuantizationLinear,
)
from .lora_quantization_layers import (
    ColumnParallelQuantizationLoRALinear,
    QuantizationLoRALinear,
    RowParallelQuantizationLoRALinear,
)

AVAILABLE_LAYERS = [
    ColumnParallelLoRALinear,
    ColumnSequenceParallelLoRALinear,
    LoRAConv2D,
    LoRALinear,
    RowParallelLoRALinear,
    RowSequenceParallelLoRALinear,
    ColumnParallelQuantizationLoRALinear,
    QuantizationLoRALinear,
    RowParallelQuantizationLoRALinear,
]


class LoRAModel(nn.Layer):
    # TODO:lugimzzz support restore in following PR
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        LoRALinear: nn.Linear,
        LoRAConv2D: nn.Conv2D,
        # ColumnParallelLoRALinear: ColumnParallelLinear,
        # RowParallelLoRALinear: RowParallelLinear,
        # QuantizationLoRALinear: QuantizationLinear,
    }

    def __init__(self, model, lora_config: LoRAConfig) -> None:
        super().__init__()
        self.quantized = False
        self.lora_config = lora_config
        self.lora_split_mapping = {}
        if self.lora_config.dtype is None:
            self.lora_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.lora_config.dtype):
            self.model = self.get_lora_model(model, lora_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            self.is_pipelinemodel = True
            self.model._single_to_pp_mapping = None
        if (self.lora_config.tensor_parallel_degree > 1 or self.is_pipelinemodel) and (
            self.lora_config.lora_use_mixer or self.lora_config.use_mora
        ):
            raise NotImplementedError("lora_use_mixer or mora is not supported in tensor parallel mode.")
        if self.lora_config.tensor_parallel_degree != self.model.config.tensor_parallel_degree:
            self.lora_config.tensor_parallel_degree = self.model.config.tensor_parallel_degree
            logger.warning(
                f"Reset tensor_parallel_degree of lora_config to {self.model.config.tensor_parallel_degree}."
            )

        self.forward = self.model.forward
        if lora_config.loraga:
            self.loraga_init_dict = {}
            self.reinit_base_model = False

        logger.info("Mark only lora and trainable_module as trainable.")
        self.mark_only_lora_as_trainable()

    def add_lora_split_mapping(self, module_name, is_column=False):
        self.lora_split_mapping[module_name] = is_column

    def _get_tensor_parallel_mappings(self, config, is_split=True):

        from ...transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        rename_lora_split_mapping = {}
        if issubclass(type(self.model), PipelineLayer):
            # rename lora_split_mapping
            prefixes = self.model.get_sequential_name_prefixes()
            keys = self.lora_split_mapping.keys()
            first_key = ""
            for k in keys:
                first_key = k
                break
            first_key = first_key.split(".")
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            for k in keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, set origin name as {'.'.join(single_name)}"
                            )
                    else:
                        raise ValueError(f"Please check! {k} is not a valid key.")
                else:
                    idx = name_splited[0]
                    # for normal pp layer name
                    if idx.isdigit():
                        single_name = [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif "shared_layers" in idx:
                        single_name = ["ernie"]
                        single_name.extend(k.split("shared_layers.embed_weight_share.")[1:])
                    else:
                        raise ValueError(f"Unexpected key: {k} for pp lora layer.")
                rename_lora_split_mapping[".".join(single_name)] = self.lora_split_mapping[k]

        lora_split_mapping = (
            rename_lora_split_mapping if issubclass(type(self.model), PipelineLayer) else self.lora_split_mapping
        )

        def get_tensor_parallel_split_mappings():
            final_actions = {}
            for key, is_col in lora_split_mapping.items():
                final_actions[key] = partial(fn, is_column=is_col)

            return final_actions

        mappings = get_tensor_parallel_split_mappings()

        return mappings

    @classmethod
    def from_pretrained(cls, model, lora_path, **kwargs):
        lora_config = kwargs.pop("lora_config", None)
        # init lora config & lora model
        if not isinstance(lora_config, LoRAConfig):
            lora_config = LoRAConfig.from_pretrained(lora_path)
        # define a new variable to conserve original lora_config.tensor_parallel_degree value which will update while initializing lora model
        lora_config_tensor_parallel_degree = lora_config.tensor_parallel_degree
        lora_model = cls(model, lora_config)

        lora_model_index_file = os.path.join(lora_path, SAFE_PEFT_WEIGHTS_INDEX_NAME)
        if os.path.exists(lora_model_index_file):
            # load safetensors format file.
            resolved_archieve_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path=lora_path,
                index_filename=lora_model_index_file,
            )
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
            expected_keys = set(lora_model.get_trainable_state_dict().keys())
            missing_keys = expected_keys - set(loaded_keys)
            if len(missing_keys) > 0:
                raise ValueError(f"missing_keys: {missing_keys}")

            error_msgs = []
            for shard_file in resolved_archieve_file:
                pre_tensor_parallel_split = False
                if model.config.tensor_parallel_degree > 1:
                    pre_tensor_parallel_split = True
                    tp_actions = lora_model._get_tensor_parallel_convert_actions(loaded_keys, is_split=True)
                state_dict = load_state_dict(
                    shard_file,
                    tp_actions if pre_tensor_parallel_split else None,
                    expected_keys,
                )
                error_msgs += _load_state_dict_into_model(lora_model, state_dict, "")
                del state_dict
                gc.collect()

            if len(error_msgs) > 0:
                error_msg = "\n\t".join(error_msgs)
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {lora_model.__class__.__name__}:\n\t{error_msg}"
                )

            return lora_model

        # define lora weight name
        if lora_config_tensor_parallel_degree > 1:
            lora_weight_name = _add_variant(LORA_WEIGHTS_NAME, f"tp{model.config.tensor_parallel_rank:0>2d}")
        else:
            lora_weight_name = LORA_WEIGHTS_NAME

        # load and set lora weight parameter
        lora_weight_path = os.path.join(lora_path, lora_weight_name)
        if os.path.exists(lora_weight_path):
            # load lora weight parameter
            lora_state_dict = paddle.load(lora_weight_path, return_numpy=True)
            logger.info(f"Loading the LoRA weights from {lora_weight_path}")

            if (
                lora_config_tensor_parallel_degree > 1
                and lora_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{lora_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge LoRA weights first."
                )

            # convert parameters to tensor parallel for mp model
            if lora_config_tensor_parallel_degree <= 1 and model.config.tensor_parallel_degree > 1:
                lora_state_dict = lora_model._convert_tensor_parallel(lora_state_dict=lora_state_dict)

            # set lora state dict
            lora_model.set_state_dict(lora_state_dict)
        else:
            logger.error(f"LoRA weights not found under {lora_path}, creating LoRA weights from scratch")

        return lora_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )

        model_state_dict = self.model.state_dict()
        if self.lora_config.loraga:

            def process_split_and_assign(name, concat_tensor, init_dict, state_dict):
                if "lora_A" in name:
                    axis = 1
                else:
                    axis = 0
                if isinstance(concat_tensor, np.ndarray):
                    final_lora, init_lora = np.split(concat_tensor, 2, axis=axis)
                    init_lora = paddle.to_tensor(init_lora)
                else:
                    final_lora, init_lora = paddle.split(concat_tensor, 2, axis=axis)
                if "lora_B" in name:
                    init_lora *= -1
                init_dict[name] = init_lora
                state_dict[name] = final_lora
                return init_lora

            for name in state_dict.keys():
                if "lora_A" in name:
                    concat_lora_A = state_dict[name]
                    init_loraA = process_split_and_assign(
                        name, concat_lora_A, init_dict=self.loraga_init_dict, state_dict=state_dict
                    )

                    loraB_name = name.replace("lora_A", "lora_B")
                    concat_lora_B = state_dict[loraB_name]
                    init_loraB = process_split_and_assign(
                        loraB_name, concat_lora_B, init_dict=self.loraga_init_dict, state_dict=state_dict
                    )

                    base_name = name.replace("lora_A", "weight")
                    if not self.reinit_base_model:
                        # Reinit base model
                        offset = init_loraA._copy_to(
                            paddle.framework._current_expected_place(), False
                        ) @ init_loraB._copy_to(paddle.framework._current_expected_place(), False)
                        ori_weight = model_state_dict[base_name]
                        model_state_dict[base_name].set_value(ori_weight - self.lora_config.scaling * offset)
        del model_state_dict
        gc.collect()
        self.model.set_state_dict(state_dict)
        logger.info("Load lora weight successfully")

    def _merge_trainable_tensor_parallel(self, trainable_state_dict):
        trainable_name_action_mappings = self._get_tensor_parallel_convert_actions(
            trainable_state_dict.keys(), is_split=False
        )

        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        mp_group = hcg.get_model_parallel_group()
        is_dst = paddle.distributed.get_rank(mp_group) == 0

        for key in trainable_state_dict:
            tensor = trainable_state_dict[key]
            if key in trainable_name_action_mappings:
                if get_env_device() == "xpu":
                    ret = distributed_allgather(tensor, group=mp_group, offload=True)
                else:
                    ret = distributed_gather(tensor, group=mp_group, offload=True)
                action = trainable_name_action_mappings[key]
                if key in self.lora_split_mapping and not self.lora_split_mapping[key] and "_scale" in key and is_dst:
                    ret = paddle.to_tensor(ret)
                    tensor = paddle.max(ret, axis=0)
                else:
                    tensor = action(ret) if is_dst else None
                trainable_state_dict[key] = tensor
            else:
                trainable_state_dict[key] = tensor.cpu().numpy() if is_dst else None

        return trainable_state_dict

    def _get_tensor_parallel_convert_actions(self, loaded_keys, is_split=True, ignore_error=False, config=None):
        if config is None:
            config = self.model.config
        specific_name_action_mappings = self._get_tensor_parallel_mappings(config, is_split=is_split)
        name_action_mappings = self.model._get_tensor_parallel_mappings(config, is_split=is_split)
        state_keys_map = ConversionMixin._resolve_prefix_keys(
            name_action_mappings.keys(), self.model.state_dict().keys(), ignore_error=ignore_error
        )
        for k, v in state_keys_map.items():
            if v in loaded_keys:
                specific_name_action_mappings[v] = name_action_mappings[k]
        return specific_name_action_mappings

    def _convert_tensor_parallel(self, lora_state_dict):
        lora_name_action_mappings = self._get_tensor_parallel_convert_actions(lora_state_dict.keys(), is_split=True)

        for name, action in lora_name_action_mappings.items():
            if name in lora_state_dict:
                tensor = lora_state_dict.pop(name)
                lora_state_dict[name] = action(tensor)
            else:
                logger.warning(f"{name} not found in lora_state_dict!")
        return lora_state_dict

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        save_model_config = kwargs.get("save_model_config", True)

        if self.is_pipelinemodel:
            self.model._single_to_pp_mapping = None
        if self.is_pipelinemodel and merge_tensor_parallel and self.lora_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Pipeline parallelism does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )

        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        if isinstance(self.lora_config, LoRAConfig):
            lora_config_to_save = LoRAConfig(**self.lora_config.to_dict())
        else:
            lora_config_to_save = LoRAAutoConfig(**self.lora_config.to_dict())

        trainable_state_dict = self.get_trainable_state_dict(concat_init_lora=lora_config_to_save.loraga)

        if merge_tensor_parallel and lora_config_to_save.tensor_parallel_degree > 1:
            trainable_state_dict = self._merge_trainable_tensor_parallel(trainable_state_dict)
            if not is_main_process:
                logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                return
            if variant is not None and "tp" in variant:
                variant = "_".join([x for x in variant.split("_") if "tp" not in x])
            lora_config_to_save.tensor_parallel_degree = -1
        else:
            if lora_config_to_save.tensor_parallel_degree > 1:
                if variant is None:
                    variant = weight_name_suffix()

        # save lora weight
        lora_weight_name = _add_variant(LORA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, lora_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save lora config
        if is_main_process:
            lora_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, lora_config):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lora_module = None
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                pissa=lora_config.pissa,
                bias_attr=False if module.bias is None else None,
                use_quick_lora=lora_config.use_quick_lora,
                lora_use_mixer=lora_config.lora_use_mixer,
                use_mora=lora_config.use_mora,
                mp_moe=getattr(module.weight, "mp_moe", False),
                is_distributed=getattr(module.weight, "is_distributed", False),
                lorapro=lora_config.lorapro,
            )
        elif isinstance(module, nn.Conv2D):
            lora_module = LoRAConv2D(
                in_channels=module._in_channels,
                out_channels=module._out_channels,
                kernel_size=module._kernel_size,
                stride=module._stride,
                padding=module._padding,
                dilation=module._dilation,
                groups=module._groups,
                padding_mode=module._padding_mode,
                data_format=module._data_format,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                bias_attr=module._bias_attr,
            )
        elif isinstance(module, ColumnParallelLinear):
            # recover the original output_features
            output_features = module.weight.shape[1] * module.world_size
            lora_module = ColumnParallelLoRALinear(
                in_features=module.weight.shape[0],
                out_features=output_features,
                gather_output=module.gather_output,
                has_bias=module.bias is not None,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                pissa=lora_config.pissa,
                lora_A_weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
                use_quick_lora=lora_config.use_quick_lora,
            )
            # Lora column parallel will spilt lora B matrix
            self.add_lora_split_mapping(module_name + ".lora_B", is_column=True)

            # for lora qat
            if self.lora_config.do_qat:
                self.add_lora_split_mapping(module_name + ".weight_quanter._scale", is_column=True)
                self.add_lora_split_mapping(module_name + ".activation_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter.quanter._scale", is_column=False)
        elif isinstance(module, RowParallelLinear):
            # recover the original output_features
            lora_module = RowParallelLoRALinear(
                in_features=module.weight.shape[0] * module.world_size,
                out_features=module.weight.shape[1],
                has_bias=module.bias is not None,
                input_is_parallel=module.input_is_parallel,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                pissa=lora_config.pissa,
                use_quick_lora=lora_config.use_quick_lora,
            )
            # Lora column parallel will spilt lora A matrix
            self.add_lora_split_mapping(module_name + ".lora_A", is_column=False)

            # for lora qat
            if self.lora_config.do_qat:
                self.add_lora_split_mapping(module_name + ".weight_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter.quanter._scale", is_column=False)
        elif isinstance(module, linear_utils.ColumnSequenceParallelLinear):
            # recover the original output_features
            output_features = module.weight.shape[1] * module.world_size
            lora_module = ColumnSequenceParallelLoRALinear(
                in_features=module.weight.shape[0],
                out_features=output_features,
                gather_output=module.gather_output,
                has_bias=module.bias is not None,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                lora_A_weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.KaimingUniform(negative_slope=math.sqrt(5), nonlinearity="leaky_relu")
                ),
                use_quick_lora=lora_config.use_quick_lora,
            )
            # Lora column parallel will spilt lora B matrix
            self.add_lora_split_mapping(module_name + ".lora_B", is_column=True)

            # for lora qat
            if self.lora_config.do_qat:
                self.add_lora_split_mapping(module_name + ".weight_quanter._scale", is_column=True)
                self.add_lora_split_mapping(module_name + ".activation_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter.quanter._scale", is_column=False)
        elif isinstance(module, linear_utils.RowSequenceParallelLinear):
            # recover the original output_features
            lora_module = RowSequenceParallelLoRALinear(
                in_features=module.weight.shape[0] * module.world_size,
                out_features=module.weight.shape[1],
                has_bias=module.bias is not None,
                input_is_parallel=module.input_is_parallel,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                rslora=lora_config.rslora,
                lora_plus_scale=lora_config.lora_plus_scale,
                use_quick_lora=lora_config.use_quick_lora,
            )
            # Lora column parallel will spilt lora A matrix
            self.add_lora_split_mapping(module_name + ".lora_A", is_column=False)

            # for lora qat
            if self.lora_config.do_qat:
                self.add_lora_split_mapping(module_name + ".weight_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter._scale", is_column=False)
                self.add_lora_split_mapping(module_name + ".activation_quanter.quanter._scale", is_column=False)
        elif isinstance(module, QuantizationLinear):
            lora_module = QuantizationLoRALinear(module, lora_config)
        elif isinstance(module, ColumnParallelQuantizationLinear):
            lora_module = ColumnParallelQuantizationLoRALinear(module, lora_config)
            # Lora column parallel will spilt lora B matrix
            self.add_lora_split_mapping(module_name + ".lora_B", is_column=True)
        elif isinstance(module, RowParallelQuantizationLinear):
            lora_module = RowParallelQuantizationLoRALinear(module, lora_config)
            # Lora row parallel will spilt lora A matrix
            self.add_lora_split_mapping(module_name + ".lora_A", is_column=False)
        if lora_module is None:
            raise ValueError(
                f"LoRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear or paddleformers.transformers.sequence_utils. {module}({module_name} {type(module).__name__}) is not supported。"
            )
        if getattr(lora_module, "weight", None) is not None:
            lora_module.weight = module.weight
            if module.bias is not None:
                lora_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lora_module)

    def _find_and_restore_module(self, module_name):
        parent_module = self.model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        original_model_class = self.restore_layer_map[module.__class__]
        original_module = original_model_class(in_features=module.weight.shape[0], out_features=module.weight.shape[1])
        original_module.weight = module.weight
        if module.bias is not None:
            original_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], original_module)

    def get_trainable_state_dict(self, concat_init_lora=False):
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            # get lora parameter & QAT scale parameter
            if not weight.stop_gradient or "activation_quanter" in name or "weight_quanter" in name:
                if concat_init_lora:
                    if "lora_A" in name:
                        trainable_state_dict[name] = paddle.concat([weight, self.loraga_init_dict[name]], axis=1)
                    else:
                        trainable_state_dict[name] = paddle.concat([weight, -self.loraga_init_dict[name]], axis=0)
                else:
                    trainable_state_dict[name] = weight

        return trainable_state_dict

    def print_trainable_parameters(self) -> None:
        freeze_numel = 0
        trainable_numel = 0
        for _, weight in self.model.state_dict().items():
            if weight.stop_gradient:
                freeze_numel += np.prod(weight.shape)
            else:
                trainable_numel += np.prod(weight.shape)
        logger.debug(
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel + trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel + trainable_numel):.2%}"
        )

    def mark_only_lora_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if (
                isinstance(layer, LoRALinear)
                or isinstance(layer, LoRAConv2D)
                or isinstance(layer, ColumnParallelLoRALinear)
                or isinstance(layer, RowParallelLoRALinear)
                or isinstance(layer, ColumnSequenceParallelLoRALinear)
                or isinstance(layer, RowSequenceParallelLoRALinear)
                or (QuantizationLoRALinear is not None and isinstance(layer, QuantizationLoRALinear))
                or (
                    ColumnParallelQuantizationLoRALinear is not None
                    and isinstance(layer, ColumnParallelQuantizationLoRALinear)
                )
                or (
                    RowParallelQuantizationLoRALinear is not None
                    and isinstance(layer, RowParallelQuantizationLoRALinear)
                )
            ):
                for name, weight in layer.state_dict().items():
                    if self.lora_config.trainable_bias in ["lora", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "lora" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.lora_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.lora_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.lora_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_lora_model(self, model: Union[PretrainedModel, nn.Layer], lora_config: LoRAConfig):

        if lora_config.target_modules is None:
            return model
        if isinstance(lora_config.target_modules, str):
            lora_config.target_modules = [lora_config.target_modules]

        for target_module in lora_config.target_modules:
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, lora_config)
        return model

    def restore_original_model(self):
        # make sure W and lora weights are not merged before we restore the original model

        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, LoRALinear):
                self._find_and_restore_module(layer_name)
            elif (
                isinstance(layer, ColumnParallelLoRALinear)
                or isinstance(layer, ColumnSequenceParallelLoRALinear)
                or isinstance(layer, LoRAConv2D)
                or isinstance(layer, RowParallelLoRALinear)
                or isinstance(layer, RowSequenceParallelLoRALinear)
                or (QuantizationLoRALinear is not None and isinstance(layer, QuantizationLoRALinear))
                or (
                    ColumnParallelQuantizationLoRALinear is not None
                    and isinstance(layer, ColumnParallelQuantizationLoRALinear)
                )
                or (
                    RowParallelQuantizationLoRALinear is not None
                    and isinstance(layer, RowParallelQuantizationLoRALinear)
                )
            ):
                raise NotImplementedError(f"{layer} restoration is not supported yet.")
        return self.model

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Layer's logic
        except AttributeError:
            return getattr(self.model, name)

    def train(self):
        self.training = True
        self.model.training = True
        for layer in self.model.sublayers():
            layer.training = True
            layer.train()

    def eval(self):
        self.training = False
        self.model.training = False
        for layer in self.model.sublayers():
            layer.training = False
            layer.eval()

    def save_to_aistudio(
        self,
        repo_id,
        private=True,
        license="Apache License 2.0",
        exist_ok=True,
        subfolder=None,
        merge_tensor_parallel=False,
        **kwargs
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            token (str): Your token for the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "Apache License 2.0".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            merge_tensor_parallel (bool): Whether to merge the tensor parallel weights. Defaults to False.
        """
        res = aistudio_sdk.hub.create_repo(repo_id=repo_id, private=private, license=license, **kwargs)
        if "error_code" in res:
            if res["error_code"] == 10003 and exist_ok:
                logger.info(
                    f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                )
            else:
                logger.error(
                    f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                )
        else:
            logger.info(f"Successfully created repo {repo_id}")

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir, merge_tensor_parallel=merge_tensor_parallel)

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            for filename in os.listdir(save_dir):
                res = aistudio_sdk.hub.upload(
                    repo_id=repo_id, path_or_fileobj=os.path.join(save_dir, filename), path_in_repo=filename, **kwargs
                )
                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")

    def disable_lora(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.disable_lora = True

    def enable_lora(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.disable_lora = False

    def merge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.merge()

    def unmerge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lora_layer) for lora_layer in AVAILABLE_LAYERS):
                layer.unmerge()
