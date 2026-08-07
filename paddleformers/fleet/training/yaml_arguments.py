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


from omegaconf import DictConfig, OmegaConf

_PRESERVED_DICT_CONFIG_KEYS = {"deepep_buffer_configs"}


def _flatten_configs(cfg):
    result = {}

    def recurse(node):
        if isinstance(node, DictConfig):
            for k, v in node.items():
                if isinstance(v, DictConfig):
                    if k in _PRESERVED_DICT_CONFIG_KEYS:
                        result[k] = OmegaConf.to_container(v, resolve=True)
                    else:
                        recurse(v)
                else:
                    result[k] = v

    recurse(cfg)
    return OmegaConf.create(result)


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        configs = OmegaConf.load(f)
        config = _flatten_configs(configs)
        return config
