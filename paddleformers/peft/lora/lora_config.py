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

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from ...utils.env import LORA_CONFIG_NAME
from ...utils.log import logger


@dataclass
class LoRAConfig:
    """
    This is the configuration class to store the configuration of a [`LoRAModel`].
    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        trainable_modules (`List[str]`): The names of the modules to train when applying Lora.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    trainable_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to train when applying with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    trainable_bias: Optional[str] = field(
        default=None, metadata={"help": "Define trainable bias parameters for the Lora model."}
    )
    enable_lora_list: Optional[Union[List[bool], List[Optional[List[bool]]]]] = field(
        default=None,
        metadata={
            "help": "Provides fine-grained control over `MergedLoRALinear`. If None, `LoRALinear` is used instead."
        },
    )
    tensor_parallel_degree: int = field(default=-1, metadata={"help": "1 for not use tensor parallel"})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})
    head_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "The model multi head dimension.Only for LoRAMergedLinear and ColumnParallelLoRAMergedLinear."
        },
    )
    do_qat: bool = field(default=False, metadata={"help": "Whether the lora model would do quant-aware training"})
    rslora: bool = field(default=False, metadata={"help": "Whether to use RsLoRA"})
    pissa: bool = field(default=False, metadata={"help": "Whether to use Pissa: https://arxiv.org/pdf/2404.02948.pdf"})
    loraga: bool = field(default=False, metadata={"help": "Whether to LoRA-GA"})
    use_mora: bool = field(
        default=False, metadata={"help": "Whether to use MoRA: https://arxiv.org/pdf/2405.12130.pdf"}
    )
    lora_plus_scale: float = field(default=1.0, metadata={"help": "Lora B scale in LoRA+"})
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    use_quick_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use quick lora, The use of Quick LoRa will only take effect when lora_dropout is set to 0."
        },
    )
    lora_use_mixer: bool = field(
        default=False,
        metadata={"help": "Whether to use mos lora."},
    )
    lorapro: bool = field(default=False, metadata={"help": "Whether to use LoRA-PRO"})

    def __post_init__(self):
        if self.use_quick_lora and self.lora_dropout > 0:
            logger.warning(
                "Quick LoRa is enabled, but lora_dropout is set to a non-zero value. "
                "We will automatically set `use_quick_lora` to `False` to avoid potential inconsistencies."
            )
            self.use_quick_lora = False
        if self.merge_weights:
            logger.error(
                "'merge_weights' is deprecated and will be removed in a future version. "
                "Please apply model.merge() or model.unmerge() to merge/unmerge LoRA weight to base model."
            )

    @property
    def scaling(self):
        if not self.rslora and not self.pissa:
            return self.lora_alpha / self.r
        elif self.pissa:
            return 1.0
        else:
            return self.lora_alpha / math.sqrt(self.r)

    @property
    def __dict__(self):
        return asdict(self)

    def to_dict(self):
        return self.__dict__

    def save_pretrained(self, save_directory):
        r"""
        This method saves the configuration of your adapter model in a directory.
        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_dict = self.__dict__
        output_dict["scaling"] = self.scaling
        output_path = os.path.join(save_directory, LORA_CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        This method loads the configuration of your adapter model from a directory.
        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the hub-id where the configuration is saved.
            **kwargs:
                Additional keyword arguments passed along to the child class initialization.
        """
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, LORA_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find lora_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)
        loaded_attributes.pop("scaling", None)

        config = cls(**kwargs)

        for key, value in loaded_attributes.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @classmethod
    def from_json_file(cls, path_json_file):
        r"""
        Loads a configuration file from a json file.
        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object


@dataclass
class LoRAAutoConfig(LoRAConfig):
    use_intermediate_api: bool = field(
        default=False,
        metadata={"help": "Weather to use auto_parallel intermediate api"},
    )
    pipeline_parallel_degree: bool = field(
        default=False,
        metadata={"help": "Weather to use pipeline parallel"},
    )
