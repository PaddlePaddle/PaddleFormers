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
import os
import re
import tempfile
from collections import OrderedDict
from typing import Dict, List, Union

import aistudio_sdk
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn

from ...transformers import AutoConfig
from ...transformers.conversion_utils import ConversionMixin
from ...transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    _load_state_dict_into_model,
    dtype_guard,
    load_state_dict,
)
from ...transformers.utils import get_checkpoint_shard_files, weight_name_suffix
from ...utils.env import LORA_WEIGHTS_NAME, SAFE_PEFT_WEIGHTS_INDEX_NAME
from ...utils.log import logger
from .lora_config import LoRAAutoConfig
from .lora_layers import LoRALinear


class LoRAAutoLinear(LoRALinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_quick_lora: bool = False,
        rslora: bool = False,
        lora_plus_scale: float = 1.0,
        pissa: bool = False,
        lora_use_mixer: bool = False,
        use_mora: bool = False,
        **kwargs
    ):
        self.use_intermediate_api = kwargs.pop("use_intermediate_api", False)
        self.weight_dist_attr = kwargs.pop("weight_dist_attr", None)
        self.parallelize_plan = kwargs.pop("parallelize_plan", None)
        self._auto_dist_config = {"mp_config": {"parallelize_plan": {}}}
        super().__init__(
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            use_quick_lora,
            rslora,
            lora_plus_scale,
            pissa,
            lora_use_mixer,
            use_mora,
            **kwargs,
        )
        if self.use_intermediate_api:
            self.process_intermediate_api()
        else:
            self.process_base_api()

    def process_intermediate_api(self):
        if self.parallelize_plan is not None:
            if isinstance(self.parallelize_plan, dist.ColWiseParallel):
                self._auto_dist_config["mp_config"]["parallelize_plan"] = {"lora_B": dist.ColWiseParallel()}
            elif isinstance(self.parallelize_plan, dist.RowWiseParallel):
                self._auto_dist_config["mp_config"]["parallelize_plan"] = {"lora_A": dist.RowWiseParallel()}

    def process_base_api(self):
        if self.weight_dist_attr is not None:
            process_mesh = self.weight_dist_attr[0]
            placements = self.weight_dist_attr[1]
            if process_mesh is None or placements is None:
                return
            mp_index = process_mesh.dim_names.index("mp")
            self.weight = dist.shard_tensor(self.weight, process_mesh, placements)
            if placements[mp_index] == dist.Shard(1):
                # this layer is column_parallel linear
                self.lora_B = dist.shard_tensor(self.lora_B, process_mesh, placements)
            elif placements[mp_index] == dist.Shard(0):
                # this layer is Rowise_parallel linear
                self.lora_A = dist.shard_tensor(self.lora_A, process_mesh, placements)

    def auto_dist_config(self, prefix=""):
        if prefix != "":
            assert prefix.endswith(".")
        final_config = {"mp_config": {"parallelize_plan": {}}}
        if self._auto_dist_config["mp_config"] is not None:
            for k, v in self._auto_dist_config["mp_config"]["parallelize_plan"].items():
                if final_config["mp_config"] is None:
                    final_config["mp_config"]["parallelize_plan"] = {f"{prefix}{k}": v}
                else:
                    final_config["mp_config"]["parallelize_plan"][f"{prefix}{k}"] = v
        return final_config


lora_layers = {
    "LoRAAutoLinear": LoRAAutoLinear,
}
LoRAAutoLinear = lora_layers["LoRAAutoLinear"]
AVAILABLE_LAYERS = [
    LoRAAutoLinear,
]


class LoRAAutoModel(nn.Layer):
    # TODO:lugimzzz support restore in following PR
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        LoRAAutoLinear: nn.Linear,
    }

    def __init__(self, model, lora_config: LoRAAutoConfig) -> None:
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(lora_config.base_model_name_or_path)
        self.quantized = False
        self.lora_config = lora_config
        if self.lora_config.dtype is None:
            self.lora_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.lora_config.dtype):
            self.model = self.get_lora_model(model, lora_config)
        if (self.lora_config.tensor_parallel_degree > 1 or self.lora_config.pipeline_parallel_degree > 1) and (
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

    @classmethod
    def from_pretrained(cls, model, lora_path, **kwargs):
        lora_config = kwargs.pop("lora_config", None)
        # init lora config & lora model
        if not isinstance(lora_config, LoRAAutoConfig):
            lora_config = LoRAAutoConfig.from_pretrained(lora_path)
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

            def process_split_and_assign(name, concat_tensor, axis, init_dict, state_dict):
                if isinstance(concat_tensor, np.ndarray):
                    final_lora, init_lora = np.split(concat_tensor, 2, axis=axis)
                    init_lora = paddle.to_tensor(init_lora)
                else:
                    final_lora, init_lora = paddle.split(concat_tensor, 2, axis=axis)
                init_dict[name] = init_lora
                state_dict[name] = final_lora
                return init_lora

            for name in state_dict.keys():
                if "lora_A" in name:
                    concat_lora_A = state_dict[name]
                    init_loraA = process_split_and_assign(
                        name, concat_lora_A, axis=1, init_dict=self.loraga_init_dict, state_dict=state_dict
                    )

                    loraB_name = name.replace("lora_A", "lora_B")
                    concat_lora_B = state_dict[loraB_name]
                    init_loraB = process_split_and_assign(
                        loraB_name, concat_lora_B, axis=0, init_dict=self.loraga_init_dict, state_dict=state_dict
                    )

                    base_name = name.replace("lora_A", "weight")
                    if not self.reinit_base_model:
                        # Reinit base model
                        offset = init_loraA.cuda() @ init_loraB.cuda()
                        ori_weight = model_state_dict[base_name]
                        model_state_dict[base_name].set_value(ori_weight - self.lora_config.scaling * offset)
        del model_state_dict
        gc.collect()
        self.model.set_state_dict(state_dict)
        logger.info("Load lora weight successfully")

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
        if self.quantized and merge_tensor_parallel and self.lora_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Quantized strategy does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )
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

    def _find_and_replace_module(self, model, module_name, lora_config, enable_lora, layer_parallelize_plan):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lora_module = None
        if isinstance(module, nn.Linear):
            lora_module = LoRAAutoLinear(
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
                use_intermediate_api=lora_config.use_intermediate_api,
                weight_dist_attr=tuple((module.weight.process_mesh, module.weight.placements)),
                parallelize_plan=layer_parallelize_plan,
            )
        if lora_module is None:
            raise ValueError(
                f"LoRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear or paddleformers.transformers.sequence_utils. {module}({module_name} {type(module).__name__}) is not supported。"
            )
        if getattr(lora_module, "quant_weight", None) is not None:
            lora_module.quant_weight = module.quant_weight
            if getattr(lora_module, "weight_scale", None) is not None:
                lora_module.weight_scale = module.weight_scale
            if getattr(lora_module, "qweight_scale", None) is not None:
                lora_module.qweight_scale = module.qweight_scale
            if getattr(lora_module, "double_weight_scale", None) is not None:
                lora_module.double_weight_scale = module.double_weight_scale
            if getattr(lora_module, "quant_sacle_offset", None) is not None:
                lora_module.quant_sacle_offset = module.quant_sacle_offset
        else:
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
                        trainable_state_dict[name] = paddle.concat([weight, self.loraga_init_dict[name]], axis=0)
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
            if isinstance(layer, LoRAAutoLinear):
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

    def get_lora_model(self, model: Union[PretrainedModel, nn.Layer], lora_config: LoRAAutoConfig):

        if lora_config.target_modules is None:
            return model
        elif isinstance(lora_config.target_modules, str):
            target_modules = [lora_config.target_modules]
            if lora_config.enable_lora_list is None or (
                isinstance(lora_config.enable_lora_list, List)
                and all(isinstance(item, bool) for item in lora_config.enable_lora_list)
            ):
                enable_lora_list = [lora_config.enable_lora_list]
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `str`, `enable_lora_list` must be `None` or `List[bool]`"
                )
        else:
            target_modules = lora_config.target_modules
            if lora_config.enable_lora_list is None:
                enable_lora_list = [None for _ in range(len(target_modules))]
            elif isinstance(lora_config.enable_lora_list, List):
                enable_lora_list = lora_config.enable_lora_list
                if len(enable_lora_list) != len(target_modules):
                    raise TypeError(
                        f"Invalid lora_config.enable_lora_list value: {lora_config.enable_lora_list}. Since lora_config.target_modules is `List[str]`, `enable_lora_list` should have the same length as `target_modules`"
                    )
                for enable_lora in enable_lora_list:
                    if not (
                        enable_lora is None
                        or (isinstance(enable_lora, List) and all(isinstance(item, bool) for item in enable_lora))
                    ):
                        raise TypeError(
                            f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or  `List[Optional[List[bool]]]`"
                        )
            else:
                raise TypeError(
                    f"Invalid `enable_lora_list` value: {lora_config.enable_lora_list}. Since `target_modules` is `List[str]`, `enable_lora_list` must be `None` or `List[Optional[List[bool]]]`"
                )

        def _match_layer(module_name, parallelize_plan):
            # Match the layer to a plan.
            # Will return the plan if the layer hits one, otherwise return None.
            for key, plan in parallelize_plan.items():
                # Find some plan for specific parameter, such as
                # "lm_head.weight": ColWiseParallel()
                # Only support weight or bias.
                if key.endswith(".weight"):
                    key = key.replace(".weight", "")
                elif key.endswith(".bias"):
                    key = key.replace(".bias", "")
                re_find = re.match(key, module_name)
                if key == module_name or (
                    re_find is not None and int(re_find.end()) - int(re_find.start()) == len(module_name)
                ):
                    return plan

        if lora_config.use_intermediate_api:
            assert hasattr(
                model, "auto_dist_config"
            ), "train lora_model requires auto_dist_config when use intermediate api"
            auto_dist_config = model.auto_dist_config()
            if auto_dist_config["mp_config"] is not None:
                mp_parallelize_plan = auto_dist_config["mp_config"]["parallelize_plan"]
        for target_module, enable_lora in zip(target_modules, enable_lora_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    layer_parallelize_plan = None
                    if lora_config.use_intermediate_api:
                        layer_parallelize_plan = _match_layer(module_name, mp_parallelize_plan)
                    self._find_and_replace_module(model, module_name, lora_config, enable_lora, layer_parallelize_plan)
        return model

    def merge_auto_dist_configs(self, configs):
        """
        Merged all auto dist configs into one config.
        configs is a list of config,every config is a dict,which means a model auto_dist_config.
        [
            {
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },{
                mp_config (dict): {
                    "parallelize_plan": dict, the plan to shard the layer.
                }
                pp_config (dict): {
                    "split_spec": OrderedDict|dict|str|list(str), The pipeline parallel split point.
                    "global_spec": str|list(str), make the output tensor of specific layers on global mesh.
                }
            },....
        ]
        """
        assert isinstance(configs, (dict, list))
        if isinstance(configs, dict):
            return configs
        final_config = {
            "mp_config": None,
            "sp_config": None,
            "pp_config": None,
        }
        for config in configs:
            if "mp_config" in config and config["mp_config"] is not None:
                if final_config["mp_config"] is None:
                    final_config["mp_config"] = config["mp_config"]
                else:
                    for k, v in config["mp_config"]["parallelize_plan"].items():
                        assert (
                            k not in final_config["mp_config"]["parallelize_plan"].keys()
                        ), f"sublayer mp_config should be a subset of model but got sublayer config {config['mp_config']} and model config {final_config['mp_config']}."
                        final_config["mp_config"]["parallelize_plan"][k] = v
            if "sp_config" in config and config["sp_config"] is not None:
                if final_config["sp_config"] is None:
                    final_config["sp_config"] = config["sp_config"]
                else:
                    for k, v in config["sp_config"]["parallelize_plan"].items():
                        assert (
                            k not in final_config["sp_config"]["parallelize_plan"].keys()
                        ), f"sublayer sp_config should be a subset of model but got sublayer config {config['sp_config']} and model config {final_config['sp_config']}."
                        final_config["sp_config"]["parallelize_plan"][k] = v
            if "pp_config" in config and config["pp_config"] is not None:

                def process_spec(spec_name):
                    if isinstance(config["pp_config"][spec_name], str):
                        config["pp_config"][spec_name] = [config["pp_config"][spec_name]]
                        if final_config["pp_config"] is None:
                            final_config["pp_config"] = config["pp_config"]
                        elif config["pp_config"][spec_name] not in final_config["pp_config"][spec_name]:
                            final_config["pp_config"][spec_name] += config["pp_config"][spec_name]
                    elif isinstance(config["pp_config"][spec_name], (tuple, list)):
                        if final_config["pp_config"] is None:
                            final_config["pp_config"] = config["pp_config"]
                        elif config["pp_config"][spec_name] not in final_config["pp_config"][spec_name]:
                            final_config["pp_config"][spec_name] += config["pp_config"][spec_name]

                process_spec("split_spec")
                process_spec("global_spec")

        if final_config["pp_config"] is not None:
            if len(final_config["pp_config"]["split_spec"]) == 1:
                final_config["pp_config"]["split_spec"] = final_config["pp_config"]["split_spec"][0]
            elif len(final_config["pp_config"]["split_spec"]) > 1:
                final_config["pp_config"]["split_spec"] = list(set(final_config["pp_config"]["split_spec"]))
            if len(final_config["pp_config"]["global_spec"]) > 1:
                final_config["pp_config"]["global_spec"] = list(set(final_config["pp_config"]["global_spec"]))
        return final_config

    def _generate_auto_dist_config(self, auto_dist_degree):
        merged_config = {
            "sp_config": None,
            "mp_config": None,
            "pp_config": None,
        }
        layer_name = []
        for name, layer in self.named_sublayers(include_self=True):
            if hasattr(layer, "auto_dist_config"):
                if name != "":
                    prefix = name + "."
                else:
                    prefix = ""
                layer_config = layer.auto_dist_config(prefix)
                merged_config = self.merge_auto_dist_configs([merged_config, layer_config])
                layer_name.append(name)
                # for _, deeper_layer in layer.named_sublayers():
                #     if hasattr(deeper_layer, "auto_dist_config"):
                #         # mask all `auto_dist_config` methods in deeper layer
                #         deeper_layer.auto_dist_config = lambda x: {}
        final_config = {
            "dp_config": None,
            "mp_config": None,
            "pp_config": None,
        }
        if "tensor_parallel" in auto_dist_degree and auto_dist_degree["tensor_parallel"]:
            merged_config["mp_config"] is not None
            final_config["mp_config"] = merged_config["mp_config"]

        if "sequence_parallel" in auto_dist_degree and auto_dist_degree["sequence_parallel"]:
            merged_config["sp_config"] is not None
            final_config["mp_config"] = merged_config["sp_config"]

        if "pipeline_parallel" in auto_dist_degree and auto_dist_degree["pipeline_parallel"]:
            merged_config["pp_config"] is not None
            final_config["pp_config"] = merged_config["pp_config"]
            if final_config["pp_config"]["global_spec"] is not None:
                temp_specs_name = final_config["pp_config"]["global_spec"]
                for spec_name_i in temp_specs_name:
                    for spec_name_j in temp_specs_name:
                        if spec_name_i != spec_name_j and spec_name_i in spec_name_j:
                            final_config["pp_config"]["global_spec"].remove(spec_name_i)
                            break

            if final_config["pp_config"]["split_spec"] is not None:
                temp_specs_name = final_config["pp_config"]["split_spec"]
                for spec_name_i in temp_specs_name:
                    for spec_name_j in temp_specs_name:
                        if spec_name_i != spec_name_j and spec_name_i in spec_name_j:
                            final_config["pp_config"]["split_spec"].remove(spec_name_i)
                            break

        if "data_sharding_parallel" in auto_dist_degree and auto_dist_degree["data_sharding_parallel"]:
            # to avoid a circular import
            from paddleformers.trainer.trainer_utils import ShardingOption

            level = 0
            if "sharding" in auto_dist_degree and auto_dist_degree["sharding"] is not None:
                sharding = auto_dist_degree["sharding"]
                if ShardingOption.SHARD_OP in sharding:
                    level = 1
                if ShardingOption.SHARD_GRAD_OP in sharding:
                    level = 2
                if ShardingOption.FULL_SHARD in sharding:
                    level = 3
            final_config["dp_config"] = {
                "sharding_level": level,
                "sharding_mesh_dim": auto_dist_degree.get("sharding_mesh_dim", None),
            }

        return final_config

    def restore_original_model(self):
        # make sure W and lora weights are not merged before we restore the original model

        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, LoRAAutoLinear):
                self._find_and_restore_module(layer_name)
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

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()
