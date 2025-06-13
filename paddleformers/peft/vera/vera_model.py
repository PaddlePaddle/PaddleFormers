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
import os
import re
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel import PipelineLayer

from ...transformers.model_utils import PretrainedModel, _add_variant, dtype_guard
from ...utils.env import VERA_WEIGHTS_NAME
from ...utils.log import logger
from .vera_config import VeRAConfig
from .vera_layers import VeRALinear


class VeRAModel(nn.Layer):
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        VeRALinear: nn.Linear,
    }

    def __init__(self, model, vera_config: VeRAConfig) -> None:
        super().__init__()
        self.quantized = False
        self.vera_config = vera_config
        if self.vera_config.dtype is None:
            self.vera_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.vera_config.dtype):
            self.model = self.get_vera_model(model, vera_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            raise NotImplementedError("vera don't support pipeline parallel now")
        if vera_config.tensor_parallel_degree > 1:
            raise NotImplementedError("vera don't support tensor parallel now")
        self.forward = self.model.forward

    @classmethod
    def from_pretrained(cls, model, vera_path, **kwargs):
        vera_config = kwargs.pop("vera_config", None)
        # init vera config & vera model
        if not isinstance(vera_config, VeRAConfig):
            vera_config = VeRAConfig.from_pretrained(vera_path)
        # define a new variable to conserve original vera_config.tensor_parallel_degree value which will update while initializing vera model
        vera_config_tensor_parallel_degree = vera_config.tensor_parallel_degree
        vera_model = cls(model, vera_config)

        vera_weight_name = VERA_WEIGHTS_NAME

        # load and set vera weight parameter
        vera_weight_path = os.path.join(vera_path, vera_weight_name)
        logger.info(f"vera weight path is {vera_weight_path}")
        if os.path.exists(vera_weight_path):
            # load vera weight parameter
            logger.info("vera_weight_path existed, loading vera weight parameter")

            vera_state_dict = paddle.load(vera_weight_path, return_numpy=True)
            logger.info(f"Loading the VeRA weights from {vera_weight_path}")

            if (
                vera_config_tensor_parallel_degree > 1
                and vera_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{vera_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge VeRA weights first."
                )

            # set vera state dict
            vera_model.set_state_dict(vera_state_dict)
        else:
            logger.error(f"VeRA weights not found under {vera_path}, creating VeRA weights from scratch")

        return vera_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load vera weight successfully")

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):

        logger.info("save vera pretrained")
        save_model_config = kwargs.get("save_model_config", True)

        if self.is_pipelinemodel:
            self.model._single_to_pp_mapping = None
        if self.quantized and merge_tensor_parallel and self.vera_config.tensor_parallel_degree > 1:
            merge_tensor_parallel = False
            logger.warning(
                "Quantized strategy does not support merge_tensor_parallel. Set merge_tensor_parallel to False."
            )
        if self.is_pipelinemodel and merge_tensor_parallel and self.vera_config.tensor_parallel_degree > 1:
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

        vera_config_to_save = VeRAConfig(**self.vera_config.to_dict())

        logger.info(f"vera config to save is {vera_config_to_save}")

        trainable_state_dict = self.get_trainable_state_dict()

        # save vera weight
        vera_weight_name = _add_variant(VERA_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, vera_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save vera config
        if is_main_process:
            vera_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, vera_config, enable_vera):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        vera_module = None
        if enable_vera is None:
            if isinstance(module, nn.Linear):
                vera_module = VeRALinear(
                    # pass the base linear module
                    base_linear_module=module,
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                    r=vera_config.r,
                    vera_alpha=vera_config.vera_alpha,
                    vera_dropout=vera_config.vera_dropout,
                    bias_attr=False if module.bias is None else None,
                    pissa_init=vera_config.pissa_init,
                )

        if vera_module is None:
            raise ValueError(
                f"VeRA strategy only supports paddle.nn.Linear or paddle.distributed.fleet.meta_parallel.ColumnParallelLinear. {module}({module_name}) is not supported。"
            )

        if module.bias is not None:
            vera_module.bias = module.bias

        setattr(parent_module, attribute_chain[-1], vera_module)

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

    def get_trainable_state_dict(self):
        trainable_state_dict = OrderedDict()
        for name, weight in self.model.state_dict().items():
            # get vera parameter
            if not weight.stop_gradient:
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
            f"Frozen parameters: {freeze_numel:.2e} || Trainable parameters:{trainable_numel:.2e} || Total parameters:{freeze_numel+trainable_numel:.2e}|| Trainable:{trainable_numel / (freeze_numel+trainable_numel):.2%}"
        )

    def mark_only_vera_as_trainable(self, notfreezeB=False) -> None:
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, VeRALinear):
                for name, weight in layer.state_dict().items():
                    if self.vera_config.trainable_bias in ["vera", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "vera" in name:
                        # notfreezeB=True, vera_b, vera_d, vera_B is trainable
                        # notfreezeB=False, vera_b, vera_d is trainable
                        if "vera_b" in name or "vera_d" in name:
                            weight.stop_gradient = False
                        elif "vera_B" in name and notfreezeB:
                            weight.stop_gradient = False
                        else:
                            weight.stop_gradient = True
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.vera_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.vera_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.vera_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_vera_model(self, model: Union[PretrainedModel, nn.Layer], vera_config: VeRAConfig):

        if vera_config.target_modules is None:
            return model
        elif isinstance(vera_config.target_modules, str):
            target_modules = [vera_config.target_modules]
            enable_vera_list = [None]
        else:
            target_modules = vera_config.target_modules
            enable_vera_list = [None for _ in range(len(target_modules))]

        for target_module, enable_vera in zip(target_modules, enable_vera_list):
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, vera_config, enable_vera)
        return model

    def restore_original_model(self):
        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, VeRALinear):
                self._find_and_restore_module(layer_name)
            else:
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
