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
from typing import Optional

from ...utils.env import PREFIX_CONFIG_NAME


@dataclass
class PrefixConfig:
    prefix_dropout: float = field(default=0.0, metadata={"help": "Prefix projection dropout"})
    num_prefix_tokens: Optional[int] = field(default=None, metadata={"help": "Number of prefix tokens"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    multi_query_group_num: Optional[int] = field(default=None, metadata={"help": "Number of Multi-Query Groups."})
    num_hidden_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer hidden layers"})
    hidden_size: Optional[int] = field(
        default=None, metadata={"help": "The hidden embedding dimension of the transformer model"}
    )
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    prefix_projection_hidden_size: Optional[int] = field(
        default=None, metadata={"help": "The hidden embedding dimension of the transformer model"}
    )
    tensor_parallel_degree: int = field(default=-1, metadata={"help": ("1 for not use tensor parallel")})
    dtype: Optional[str] = field(default=None, metadata={"help": "The data type of tensor"})

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
        output_path = os.path.join(save_directory, PREFIX_CONFIG_NAME)

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
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, PREFIX_CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, PREFIX_CONFIG_NAME)
        else:
            raise ValueError(f"Can't find prefix_config.json at '{pretrained_model_name_or_path}'")

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
