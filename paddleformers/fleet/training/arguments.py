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
import dataclasses

from paddleformers.fleet.transformer import TransformerConfig


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="PaddleFleet Arguments", allow_abbrev=False)

    parser.add_argument("--configs", type=str, default=None)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if args.configs is not None:
        from .yaml_arguments import load_yaml

        args = load_yaml(args.configs)

    return args


def core_transformer_config_from_args(args, config_class=None):
    config_class = config_class or TransformerConfig

    kw_args = {}

    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)

    # Translate args to core transformer configuration
    # such as: rename, invert bool value, etc.
    # kw_args["use_xxx"] = args.use_x
    # kw_args["close_offload"] = not args.open_offload

    # Return config.
    return config_class(**kw_args)
