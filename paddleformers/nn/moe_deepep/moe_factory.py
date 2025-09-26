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

import json
from pathlib import Path
from typing import Any, Dict

from ...transformers.configuration_utils import PretrainedConfig
from .modular_moe_layer import ModularMoELayer


class QuickAccessMoEFactory:
    _moe_configs: Dict[str, Dict[str, Any]] = None

    @classmethod
    def _load_moe_configs(cls) -> Dict[str, Dict[str, Any]]:
        if cls._moe_configs is None:
            config_path = Path(__file__).parent / "moe_config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                cls._moe_configs = json.load(f)
        return cls._moe_configs

    @staticmethod
    def create_from_model_name(
        pretrained_config: PretrainedConfig,
    ) -> ModularMoELayer:
        moe_configs = QuickAccessMoEFactory._load_moe_configs()

        model_type = getattr(pretrained_config, "model_type", None)
        if model_type is None:
            raise ValueError("Cannot determine model type from pretrained_config")

        moe_config = moe_configs.get(model_type)
        if moe_config is None:
            raise ValueError(f"No MOE configuration found for model type: {model_type}")

        return ModularMoELayer(pretrained_config, moe_config)
