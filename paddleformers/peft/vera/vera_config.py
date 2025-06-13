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
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

from ...utils.env import VERA_CONFIG_NAME


@dataclass
class VeRAConfig:
    """
    This is the configuration class to store the configuration of a [`VeRAModel`].
    Args:
        r (`int`): vera attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply vera to.
        trainable_modules (`List[str]`): The names of the modules to train when applying vera.
        vera_alpha (`float`): The alpha parameter for vera scaling.
        vera_dropout (`float`): The dropout probability for vera layers.
    """

    r: int = field(default=8, metadata={"help": "vera attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with vera."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    trainable_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to train when applying with vera."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    vera_alpha: int = field(default=8, metadata={"help": "vera alpha"})
    vera_dropout: float = field(default=0.0, metadata={"help": "vera dropout"})
    trainable_bias: Optional[str] = field(
        default=None, metadata={"help": "Define trainable bias parameters for the vera model."}
    )
    tensor_parallel_degree: int = field(default=-1, metadata={"help": "1 for not use tensor parallel"})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})
    head_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "The model multi head dimension.Only for veraMergedLinear and ColumnParallelveraMergedLinear."
        },
    )
    do_qat: bool = field(default=False, metadata={"help": "Whether the vera model would do quant-aware training"})
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    pissa_init: bool = field(default=False, metadata={"help": "Whether the vera weight initialized by pissa"})

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
        output_path = os.path.join(save_directory, VERA_CONFIG_NAME)

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
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, VERA_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, VERA_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find vera_config.json at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)

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
