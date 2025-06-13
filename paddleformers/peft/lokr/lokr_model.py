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

from ...transformers import AutoConfig, PretrainedModel
from ...transformers.model_utils import _add_variant, dtype_guard
from ...utils.env import LOKR_WEIGHTS_NAME
from ...utils.log import logger
from .lokr_config import LoKrConfig


def get_lokr_layers():
    from .lokr_layers import LoKrLinear

    return {
        "LoKrLinear": LoKrLinear,
    }


lokr_layers = get_lokr_layers()
LoKrLinear = lokr_layers["LoKrLinear"]
AVAILABLE_LAYERS = [
    LoKrLinear,
]


class LoKrModel(nn.Layer):
    restore_layer_map: Dict[nn.Layer, nn.Layer] = {
        LoKrLinear: nn.Linear,
    }

    def __init__(self, model, lokr_config: LoKrConfig) -> None:
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(lokr_config.base_model_name_or_path)
        self.quantized = False
        self.lokr_config = lokr_config
        self.lokr_split_mapping = {}
        if self.lokr_config.dtype is None:
            self.lokr_config.dtype = paddle.get_default_dtype()
        with dtype_guard(self.lokr_config.dtype):
            self.model = self.get_lokr_model(model, lokr_config)
        self.is_pipelinemodel = False
        if issubclass(type(self.model), PipelineLayer):
            raise NotImplementedError("lokr don't support pipeline parallel now")
        if lokr_config.tensor_parallel_degree > 1:
            self.lokr_config.tensor_parallel_degree = -1
            self.model.config.tensor_parallel_degree = -1
            raise NotImplementedError("lokr don't support tensor parallel now")
        # currently tensor_parallel_degree should all be set to -1.
        self.forward = self.model.forward

        logger.info("Mark only lokr and trainable_module as trainable.")
        self.mark_only_lokr_as_trainable()

    @classmethod
    def from_pretrained(cls, model, lokr_path, **kwargs):
        lokr_config = kwargs.pop("lokr_config", None)
        # init lokr config & lokr model
        if not isinstance(lokr_config, LoKrConfig):
            lokr_config = LoKrConfig.from_pretrained(lokr_path)
        # define a new variable to conserve original lora_config.tensor_parallel_degree value which will update while initializing lora model
        lokr_config_tensor_parallel_degree = lokr_config.tensor_parallel_degree
        lokr_model = cls(model, lokr_config)

        # define lokr weight name
        lokr_weight_name = LOKR_WEIGHTS_NAME

        # load and set lokr weight parameter
        lokr_weight_path = os.path.join(lokr_path, lokr_weight_name)
        if os.path.exists(lokr_weight_path):
            # load lokr weight parameter
            lokr_state_dict = paddle.load(lokr_weight_path, return_numpy=True)
            logger.info(f"Loading the LoKR weights from {lokr_weight_path}")

            if (
                lokr_config_tensor_parallel_degree > 1
                and lokr_config_tensor_parallel_degree != model.config.tensor_parallel_degree
            ):
                raise NotImplementedError(
                    f"{lokr_config_tensor_parallel_degree} is not equal to {model.config.tensor_parallel_degree}. Please merge LoKR weights first."
                )
            # set lokr state dict
            lokr_model.set_state_dict(lokr_state_dict)
        else:
            logger.error(f"LoKR weights not found under {lokr_path}, creating LoKR weights from scratch")

        return lokr_model

    def set_state_dict(self, state_dict):
        import warnings

        warnings.filterwarnings(
            action="ignore", message=".*Skip loading for.*", category=Warning, lineno=0, append=False
        )
        self.model.set_state_dict(state_dict)
        logger.info("Load lokr weight successfully")

    def save_pretrained(self, save_directory: str, merge_tensor_parallel: bool = False, **kwargs):
        logger.info("save lokr pretrained")
        save_model_config = kwargs.get("save_model_config", True)

        variant = kwargs.get("variant", None)
        is_main_process = kwargs.get("is_main_process", paddle.distributed.get_rank() == 0)

        assert not os.path.isfile(
            save_directory
        ), f"Saving directory ({save_directory}) should be a directory, not a file"
        os.makedirs(save_directory, exist_ok=True)

        lokr_config_to_save = LoKrConfig(**self.lokr_config.to_dict())
        trainable_state_dict = self.get_trainable_state_dict()

        # save lokr weight
        lokr_weight_name = _add_variant(LOKR_WEIGHTS_NAME, variant)
        weight_filename = os.path.join(save_directory, lokr_weight_name)
        paddle.save(trainable_state_dict, weight_filename)

        # save lokr config
        if is_main_process:
            lokr_config_to_save.save_pretrained(save_directory)
            if save_model_config:
                model_config_to_save = copy.deepcopy(self.model.config)
                if merge_tensor_parallel:
                    model_config_to_save.tensor_parallel_degree = -1
                model_config_to_save.save_pretrained(save_directory)

    def _find_and_replace_module(self, model, module_name, lokr_config):
        parent_module = model
        attribute_chain = module_name.split(".")
        for name in attribute_chain[:-1]:
            parent_module = getattr(parent_module, name)
        module = getattr(parent_module, attribute_chain[-1])
        lokr_module = None
        if isinstance(module, nn.Linear):
            lokr_module = LoKrLinear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                lokr_dim=lokr_config.lokr_dim,
                decompose_both=lokr_config.decompose_both,
                lokr_alpha=lokr_config.lokr_alpha,
                factor=lokr_config.factor,
                bias_attr=False if module.bias is None else None,
            )
        if lokr_module is None:
            raise ValueError("Target LoKr Module not found. LoKr strategy only supports paddle.nn.Linear right now")

        lokr_module.weight = module.weight
        if module.bias is not None:
            lokr_module.bias = module.bias
        setattr(parent_module, attribute_chain[-1], lokr_module)

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
            # get lokr parameter & QAT scale parameter
            if not weight.stop_gradient or "activation_quanter" in name or "weight_quanter" in name:
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

    def mark_only_lokr_as_trainable(self) -> None:
        for _, layer in self.model.named_sublayers():
            if isinstance(layer, LoKrLinear):
                for name, weight in layer.state_dict().items():
                    if self.lokr_config.trainable_bias in ["lokr", "all"] and "bias" in name:
                        weight.stop_gradient = False
                    elif "lokr" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
            else:
                for name, weight in layer.state_dict().items():
                    if self.lokr_config.trainable_bias == "all" and "bias" in name:
                        weight.stop_gradient = False
                    else:
                        weight.stop_gradient = True
        if self.lokr_config.trainable_modules is not None:
            for name, weight in self.model.state_dict().items():
                if any(
                    re.fullmatch(trainable_module, name) for trainable_module in self.lokr_config.trainable_modules
                ):
                    weight.stop_gradient = False

    def get_lokr_model(self, model: Union[PretrainedModel, nn.Layer], lokr_config: LoKrConfig):
        """
        Iterate all base model layers, change target modules to LoKrLayer.
        """
        if lokr_config.target_modules is None:
            return model
        else:
            target_modules = lokr_config.target_modules

        for target_module in target_modules:
            for i in model.named_sublayers():
                module_name = i[0]
                if re.fullmatch(target_module, module_name):
                    self._find_and_replace_module(model, module_name, lokr_config)
        return model

    def restore_original_model(self):
        # make sure W and lokr weights are not merged before we restore the original model
        for layer_name, layer in self.model.named_sublayers():
            if isinstance(layer, LoKrLinear):
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

    def disable_lokr(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.disable_lokr = True

    def enable_lokr(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.disable_lokr = False

    def merge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.merge()

    def unmerge(self):
        for _, layer in self.model.named_sublayers():
            if any(isinstance(layer, lokr_layer) for lokr_layer in AVAILABLE_LAYERS):
                layer.unmerge()

    def get_model_config(
        self,
    ):
        return self.model_config.to_dict()
