# !/usr/bin/env python3

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

import argparse
import os

import yaml
from omegaconf import OmegaConf


def get_config(verbose=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", action="store", nargs="+", required=True, help="config files")
    parser.add_argument("--kwargs", action="store", nargs="+", default=[], help="extra k-v configs")
    opt = parser.parse_args()
    configs = [OmegaConf.load(p) for p in opt.configs]
    configs.append(OmegaConf.from_dotlist(opt.kwargs))
    config = OmegaConf.merge(*configs)

    if "env" in config:
        for key, value in OmegaConf.to_object(config.env).items():
            config.env[key] = os.environ.get(key, value)
    OmegaConf.resolve(config)
    if verbose:
        print(
            yaml.dump(
                OmegaConf.to_object(config),
                default_flow_style=False,
                indent=4,
                width=9999,
                allow_unicode=True,
            )
        )
    return config
