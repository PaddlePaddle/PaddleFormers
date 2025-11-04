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

from typing import Dict, Optional

import paddle

from .moe_loss import LossConfig, LossRegistry


def get_global_loss_registry():
    if not hasattr(get_global_loss_registry, "_instance"):
        get_global_loss_registry._instance = LossRegistry()
        get_global_loss_registry._instance.register_loss("custom_diversity_loss1", custom_diversity_loss)
        get_global_loss_registry._instance.register_combiner(
            "custom_weighted_sum_combiner1", custom_weighted_sum_combiner
        )
    return get_global_loss_registry._instance


def custom_diversity_loss(
    routing_weights: paddle.Tensor,
    selected_experts: paddle.Tensor,
    gate_logits: Optional[paddle.Tensor] = None,
    **kwargs
) -> paddle.Tensor:
    num_experts = kwargs.get("num_experts", 8)
    expert_counts = paddle.zeros([num_experts])

    for i in range(selected_experts.shape[0]):
        for j in range(selected_experts.shape[1]):
            expert_idx = selected_experts[i, j].item()
            expert_counts[expert_idx] += 1

    uniform_dist = paddle.ones_like(expert_counts) / expert_counts.shape[0]
    expert_probs = expert_counts / (expert_counts.sum() + 1e-8)

    diversity_loss = paddle.nn.functional.kl_div(
        paddle.log(expert_probs + 1e-8), paddle.log(uniform_dist + 1e-8), reduction="sum"
    )

    return diversity_loss


def custom_weighted_sum_combiner(
    self, losses: Dict[str, paddle.Tensor], configs: Dict[str, LossConfig]
) -> paddle.Tensor:
    combined_loss = paddle.to_tensor(0.0)
    for name, loss_value in losses.items():
        config = configs.get(name)
        if config and config.enabled:
            combined_loss += config.weight * loss_value
    return combined_loss
