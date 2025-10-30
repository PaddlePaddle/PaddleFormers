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
from .moe_config import MOE_CONFIG

class QuickAccessMoEFactory:

    @staticmethod
    def create_from_model_name(
        pretrained_config: PretrainedConfig,
    ) -> ModularMoELayer:
        model_type = getattr(pretrained_config, "model_type", None)
        if model_type is None:
            raise ValueError("Cannot determine model type from pretrained_config")

        moe_config = MOE_CONFIG.get(model_type, None)
        if moe_config is None:
            raise ValueError(f"No MoE config found for {model_type}")

        return ModularMoELayer(
            hidden_size=pretrained_config.hidden_size,
            moe_intermediate_size=pretrained_config.moe_intermediate_size,
            num_experts=pretrained_config.get(
                "num_experts", pretrained_config.get("n_routed_experts", pretrained_config.get("moe_num_experts", -1))
            ),
            num_shared_experts=pretrained_config.get(
                "n_shared_experts", pretrained_config.get("moe_num_shared_experts", 0)
            ),
            num_experts_per_tok=pretrained_config.get("num_experts_per_tok", pretrained_config.get("moe_k", -1)),
            norm_topk_prob=pretrained_config.get("norm_topk_prob", True),
            expert_activation=pretrained_config.get("hidden_act", pretrained_config.get("expert_activation", "silu")),
            moe_config=moe_config,
            model_type=model_type,
            pretrained_config=pretrained_config,
        )


__all__ = ["QuickAccessMoEFactory"]
