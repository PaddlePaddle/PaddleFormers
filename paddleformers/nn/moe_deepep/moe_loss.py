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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.communication.group import Group

logger = logging.getLogger(__name__)


class LossType(Enum):
    """损失函数类型枚举"""

    AUXILIARY = "auxiliary"
    Z_LOSS = "z_loss"
    ENTROPY = "entropy"
    SPARSITY = "sparsity"
    DIVERSITY = "diversity"
    CUSTOM = "custom"


@dataclass
class LossConfig:
    """损失函数配置"""

    name: str
    loss_type: LossType
    weight: float = 0.0
    enabled: bool = True
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class LossFunction(Protocol):
    """损失函数协议接口"""

    def __call__(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """计算损失函数"""
        ...


class LossCombiner(Protocol):
    """损失组合器协议接口"""

    def __call__(self, losses: Dict[str, paddle.Tensor], configs: Dict[str, LossConfig]) -> paddle.Tensor:
        """组合多个损失函数"""
        ...


class LossRegistry:
    """损失函数注册器"""

    def __init__(self):
        self._loss_functions: Dict[str, LossFunction] = {}
        self._loss_combiners: Dict[str, LossCombiner] = {}
        self._register_default_losses()
        self._register_default_combiners()

    def _register_default_losses(self):
        """注册默认损失函数"""
        self.register_loss("auxiliary", self._auxiliary_loss)
        self.register_loss("z_loss", self._z_loss)
        self.register_loss("entropy", self._entropy_loss)
        self.register_loss("sparsity", self._sparsity_loss)
        self.register_loss("diversity", self._diversity_loss)

    def _register_default_combiners(self):
        """注册默认损失组合器"""
        self.register_combiner("weighted_sum", self._weighted_sum_combiner)
        self.register_combiner("adaptive_sum", self._adaptive_sum_combiner)
        self.register_combiner("geometric_mean", self._geometric_mean_combiner)

    def register_loss(self, name: str, loss_func: LossFunction):
        """注册损失函数"""
        self._loss_functions[name] = loss_func
        logger.info(f"注册损失函数: {name}")

    def register_combiner(self, name: str, combiner: LossCombiner):
        """注册损失组合器"""
        self._loss_combiners[name] = combiner
        logger.info(f"注册损失组合器: {name}")

    def get_loss(self, name: str) -> Optional[LossFunction]:
        """获取损失函数"""
        return self._loss_functions.get(name)

    def get_combiner(self, name: str) -> Optional[LossCombiner]:
        """获取损失组合器"""
        return self._loss_combiners.get(name)

    def list_losses(self) -> List[str]:
        """列出所有损失函数"""
        return list(self._loss_functions.keys())

    def list_combiners(self) -> List[str]:
        """列出所有损失组合器"""
        return list(self._loss_combiners.keys())

    # 默认损失函数实现
    def _auxiliary_loss(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """标准辅助损失（负载均衡损失）"""
        num_experts = kwargs.get("num_experts", selected_experts.max().item() + 1)
        expert_usage = paddle.zeros([num_experts], dtype=routing_weights.dtype)

        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                expert_idx = selected_experts[i, j].item()
                expert_usage[expert_idx] += routing_weights[i, j]

        expert_usage = expert_usage / selected_experts.shape[0]
        aux_loss = paddle.sum(expert_usage * paddle.log(expert_usage + 1e-8))
        return aux_loss

    def _z_loss(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """标准Z损失"""
        if gate_logits is None:
            return paddle.to_tensor(0.0)
        return paddle.sum(gate_logits**2)

    def _entropy_loss(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """熵损失 - 鼓励路由权重的多样性"""
        return -paddle.sum(routing_weights * paddle.log(routing_weights + 1e-8))

    def _sparsity_loss(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """稀疏性损失 - 鼓励专家选择的稀疏性"""
        num_experts = kwargs.get("num_experts", selected_experts.max().item() + 1)
        expert_usage = paddle.zeros([num_experts])

        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                expert_idx = selected_experts[i, j].item()
                expert_usage[expert_idx] += 1

        return paddle.sum(paddle.abs(expert_usage))

    def _diversity_loss(
        self,
        routing_weights: paddle.Tensor,
        selected_experts: paddle.Tensor,
        gate_logits: Optional[paddle.Tensor] = None,
        **kwargs
    ) -> paddle.Tensor:
        """多样性损失 - 鼓励专家选择的多样性"""
        num_experts = kwargs.get("num_experts", selected_experts.max().item() + 1)
        expert_counts = paddle.zeros([num_experts])

        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                expert_idx = selected_experts[i, j].item()
                expert_counts[expert_idx] += 1

        uniform_dist = paddle.ones_like(expert_counts) / expert_counts.shape[0]
        diversity_loss = paddle.nn.functional.kl_div(
            paddle.log(expert_counts + 1e-8), paddle.log(uniform_dist + 1e-8), reduction="sum"
        )
        return diversity_loss

    # 默认损失组合器实现
    def _weighted_sum_combiner(
        self, losses: Dict[str, paddle.Tensor], configs: Dict[str, LossConfig]
    ) -> paddle.Tensor:
        """加权求和组合"""
        combined_loss = paddle.to_tensor(0.0)
        for name, loss_value in losses.items():
            config = configs.get(name)
            if config and config.enabled:
                combined_loss += config.weight * loss_value
        return combined_loss

    def _adaptive_sum_combiner(
        self, losses: Dict[str, paddle.Tensor], configs: Dict[str, LossConfig]
    ) -> paddle.Tensor:
        """自适应加权求和组合"""
        combined_loss = paddle.to_tensor(0.0)
        enabled_losses = [
            loss for name, loss in losses.items() if configs.get(name, LossConfig("", LossType.CUSTOM)).enabled
        ]

        if len(enabled_losses) > 1:
            loss_std = paddle.std(paddle.stack(enabled_losses))
        else:
            loss_std = paddle.to_tensor(1.0)

        adaptation_factor = 0.1
        for name, loss_value in losses.items():
            config = configs.get(name)
            if config and config.enabled:
                adaptive_weight = config.weight * (1 + adaptation_factor * loss_std)
                combined_loss += adaptive_weight * loss_value

        return combined_loss

    def _geometric_mean_combiner(
        self, losses: Dict[str, paddle.Tensor], configs: Dict[str, LossConfig]
    ) -> paddle.Tensor:
        """几何平均组合"""
        combined_loss = paddle.to_tensor(1.0)
        for name, loss_value in losses.items():
            config = configs.get(name)
            if config and config.enabled and config.weight > 0:
                combined_loss *= (loss_value + 1e-8) ** config.weight
        return combined_loss
