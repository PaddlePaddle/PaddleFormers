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
import os
import paddle.nn.functional as F
from typing import Any, Dict, Optional

import paddle
import paddle.distributed as dist
from paddle import nn

from ...transformers.token_dispatcher import MoEFlexTokenDispatcher
from .moe_communication import (
    DeepEPMoECommunication,
    MoECommunicationInterface,
    StandardMoECommunication,
)
from .moe_expert import MoEExpertInterface, StandardMoEExpert, Qwen2MLP
from .moe_gate import PretrainedMoEGate
from .moe_loss import LossCombiner, LossConfig, LossFunction, LossRegistry, LossType
from ..linear import Linear as GeneralLinear

logger = logging.getLogger(__name__)

# 全局损失注册器实例
loss_registry = LossRegistry()

class Qwen2MoeMLP(Qwen2MLP):
    def __init__(self, config: Qwen2MoeConfig, intermediate_size=None):
        super().__init__(hidden_size=config.hidden_size, intermediate_size=intermediate_size, config=config)
class Qwen3MoeMLP(Qwen2MoeMLP):
    pass

class ModularMoELayer(nn.Layer):
    """
    模块化MoE Layer EP并行实现

    设计理念：
    1. 高度模块化：门控、专家、通信完全解耦
    2. 易于扩展：支持自定义门控策略和专家架构
    """

    def __init__(
        self,
        config: PretrainedConfig,
    ):
        """
        初始化模块化MoE Layer

        Args:
            hidden_size: 隐藏维度
            intermediate_size: 中间维度
            num_experts: 专家数量
            num_experts_per_tok: 每个token选择的专家数
            num_shared_experts: 共享专家数量
            expert_parallel_degree: 专家并行度
            gate_type: 门控类型
            gate_activation: 门控激活函数
            expert_activation: 专家激活函数
            aux_loss_weight: 辅助损失权重（传统模式）
            z_loss_weight: Z损失权重（传统模式）
            loss_configs: 损失配置列表（灵活模式）
            loss_combiner_name: 损失组合器名称
            use_flexible_loss: 是否使用灵活损失系统
            expert_dropout: 专家dropout
            moe_group: MoE通信组
            all_to_all_dropout: All-to-All dropout
            custom_gate: 自定义门控网络
            custom_expert: 自定义专家网络
            custom_communication: 自定义通信策略
        """
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.intermediate_size = config.get("intermediate_size", 1024)
        self.num_experts = config.get("num_experts", 8)
        self.num_experts_per_tok = config.get("num_experts_per_tok", 2)
        self.num_shared_experts = config.get("num_shared_experts", 0)
        self.expert_parallel_degree = config.get("expert_parallel_degree", 1)
        self.gate_type = config.get("gate_type", "topk")
        self.topk_method = config.get("topk_method", "greedy")
        self.gate_activation = config.get("gate_activation", "softmax")
        self.expert_activation = config.get("expert_activation", "silu")
        self.aux_loss_weight = config.get("aux_loss_weight", 0.01)
        self.z_loss_weight = config.get("z_loss_weight", 0.0)
        self.loss_configs = config.get("loss_configs", None)
        self.loss_combiner_name = config.get("loss_combiner_name", "weighted_sum")
        self.use_flexible_loss = config.get("use_flexible_loss", False)
        self.expert_dropout = config.get("expert_dropout", 0.0)
        self.moe_group = config.get("moe_group", "data")
        self.all_to_all_dropout = config.get("all_to_all_dropout", 0.0)
        self.custom_gate = config.get("custom_gate", None)
        self.custom_expert = config.get("custom_expert", None)
        self.custom_communication = config.get("custom_communication", None)
        self.moe_intermediate_size = config.get("moe_intermediate_size", 768)
        self.drop_tokens = config.get("drop_tokens", True)
        self.config = config

        # 初始化EP并行相关参数
        self._init_expert_parallel()
        # 创建门控网络
        if self.custom_gate is not None:
            self.gate = custom_gate
        elif self.use_flexible_loss:
            # 使用灵活损失系统
            if loss_configs is None:
                loss_configs = [
                    LossConfig("auxiliary", LossType.AUXILIARY, weight=aux_loss_weight),
                    LossConfig("z_loss", LossType.Z_LOSS, weight=z_loss_weight),
                ]
            self.gate = FlexibleMoEGate(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                loss_registry=self.loss_registry,
                num_experts_per_tok=self.num_experts_per_tok,
                gate_type=self.gate_type,
                gate_activation=self.gate_activation,
                loss_configs=self.loss_configs,
                loss_combiner_name=self.loss_combiner_name,
            )
        else:
            # self.gate = StandardMoEGate(
            #     hidden_size=hidden_size,
            #     num_experts=num_experts,
            #     num_experts_per_tok=num_experts_per_tok,
            #     gate_type=gate_type,
            #     gate_activation=gate_activation,
            #     aux_loss_weight=aux_loss_weight,
            #     z_loss_weight=z_loss_weight,
            # )
            self.gate = PretrainedMoEGate(
                config=config,
                num_experts=self.num_experts,
                expert_hidden_size=self.hidden_size,
                top_k=self.num_experts_per_tok,
                topk_method=self.topk_method,
                drop_tokens=self.drop_tokens,
            )
            # self.gate = GeneralLinear.create(config.hidden_size, config.num_experts, has_bias=False, linear_type="default")

        # 创建专家网络
        if self.custom_expert is not None:
            # 如果传入的是实例，直接使用
            if isinstance(self.custom_expert, MoEExpertInterface):
                expert_class = type(cself.ustom_expert)
            else:
                expert_class = self.custom_expert
        else:
            expert_class = Qwen3MoeMLP

        self.experts = nn.LayerList([])
        for i in range(self.num_experts):
            if i // self.num_experts_per_device == self.moe_rank:
                # expert = expert_class(
                #     hidden_size=self.hidden_size,
                #     intermediate_size=self.moe_intermediate_size,
                #     expert_activation=self.expert_activation,
                #     expert_dropout=self.expert_dropout,
                #     config=config,
                # )
                expert = expert_class(
                    config,
                    intermediate_size=self.moe_intermediate_size,
                )
                self.experts.append(expert)
            else:
                # 创建一个空的Layer作为占位符
                empty_expert = nn.Layer()
                self.experts.append(empty_expert)

        # 创建共享专家
        if self.num_shared_experts > 0:
            # self.shared_experts = expert_class(
            #     hidden_size=self.hidden_size,
            #     intermediate_size=self.moe_intermediate_size * self.num_shared_experts,
            #     expert_activation=self.expert_activation,
            #     expert_dropout=self.expert_dropout,
            #     config=config,
            # )
            expert = expert_class(
                config,
                intermediate_size=self.moe_intermediate_size * self.num_shared_experts,
            )
        else:
            self.shared_experts = None

        # 创建通信策略
        if self.custom_communication is not None:
            self.communication = self.custom_communication
        else:
            if os.getenv("USE_DEEPEP", "0"):
                self.communication = DeepEPMoECommunication()
            else:
                self.communication = StandardMoECommunication()

    def _init_expert_parallel(self):
        """
        初始化专家并行相关参数
        """

        def _parse_moe_expert_parallel(self, num_experts: int, expert_parallel_degree: int) -> int:
            """
            解析MoE专家并行参数

            Args:
                num_experts: 专家总数
                expert_parallel_degree: 专家并行度

            Returns:
                moe_num_experts_per_device: 每个设备的专家数
            """
            assert (
                num_experts >= expert_parallel_degree
            ), f"expert num_experts={num_experts} >= moe_world_size={expert_parallel_degree}"
            assert (
                num_experts % expert_parallel_degree == 0
            ), f"expert num_experts={num_experts} % moe_world_size={expert_parallel_degree} == 0"

            moe_num_experts_per_device = num_experts // expert_parallel_degree
            return moe_num_experts_per_device

        try:
            dist.fleet.get_hybrid_communicate_group()
            is_fleet_init = True
        except AttributeError:
            is_fleet_init = False

        if (
            is_fleet_init
            and dist.fleet.get_hybrid_communicate_group().get_data_parallel_world_size() > 1
            and self.moe_group == "data"
        ):
            self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            self.expert_parallel_degree = dist.get_world_size(self.moe_group)
            self.expert_parallel_degree = 1 if self.expert_parallel_degree < 0 else self.expert_parallel_degree
            self.num_experts_per_device = self._parse_moe_expert_parallel(
                self.num_experts, self.expert_parallel_degree
            )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_parallel_degree = 1
            self.num_experts_per_device = self.num_experts

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        MoE Layer前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [batch_size, seq_len, hidden_size]

        Returns:
            output: 输出隐藏状态，形状: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        residuals = hidden_states

        # 门控前向传播
        capacity, topk_weights, topk_indices, priorities, aux_loss, z_loss = self.gate(hidden_states)

        # 重塑输入
        reshaped_input = hidden_states.reshape([-1, d_model])

        # MoE前向传播
        if self.expert_parallel_degree > 1:
            # 使用EP并行
            print("----------------- using _forward_with_ep_parallel")
            output = self._forward_with_ep_parallel(reshaped_input, topk_indices, topk_weights)
        else:
            # 使用传统MoE
            print("----------------- using _forward_traditional_moe")
            output = self._forward_traditional_moe(reshaped_input, topk_indices, topk_weights)

        # 恢复原始形状
        output = output.reshape([batch_size, seq_len, d_model])

        # 添加共享专家输出
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)
            output = output + shared_output

        return output, aux_loss, z_loss

    def _forward_traditional_moe(
        self, hidden_states: paddle.Tensor, topk_indices: paddle.Tensor, topk_weights: paddle.Tensor
    ) -> paddle.Tensor:
        """
        传统MoE前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [batch_size*seq_len, hidden_size]
            topk_indices: TopK专家索引，形状: [seq_len, num_experts_per_tok]
            topk_weights: TopK权重，形状: [seq_len, num_experts_per_tok]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=topk_weights.dtype)

        # 创建专家掩码
        expert_mask = paddle.nn.functional.one_hot(topk_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.transpose([2, 0, 1])  # [num_experts, seq_len, num_experts_per_tok]

        # 遍历每个专家
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = paddle.where(mask)

            if token_indices.numel() > 0:
                # 获取专家权重和输入
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]

                # 计算专家输出
                expert_output = expert(expert_input)

                # 加权输出
                weighted_output = expert_output * expert_weights.unsqueeze(-1)

                # 累加到最终输出
                # 使用scatter代替index_add_
                for i, token_idx in enumerate(token_indices):
                    final_hidden_states[token_idx] += weighted_output[i]

        return final_hidden_states.cast(hidden_states.dtype)

    def _forward_with_ep_parallel(
        self, hidden_states: paddle.Tensor, topk_indices: paddle.Tensor, topk_weights: paddle.Tensor
    ) -> paddle.Tensor:
        """
        EP并行MoE前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [seq_len, hidden_size]
            topk_indices: TopK专家索引，形状: [seq_len, num_experts_per_tok]
            topk_weights: TopK权重，形状: [seq_len, num_experts_per_tok]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """
        # 使用通信策略进行EP并行
        output, aux_loss, z_loss = self.communication.forward(
            hidden_states,
            topk_indices,
            topk_weights,
            self.expert_parallel_degree,
            self.moe_group,
            self.experts,
            self.moe_rank,
            self.num_experts_per_device,
            self.num_experts,
            self.num_experts_per_tok,
        )
        return output

    def get_auxiliary_loss(self) -> paddle.Tensor:
        """
        获取辅助损失

        Returns:
            aux_loss: 辅助损失，标量
        """
        return self.gate.get_auxiliary_loss()

    def get_z_loss(self) -> paddle.Tensor:
        """
        获取Z损失

        Returns:
            z_loss: Z损失，标量
        """
        return self.gate.get_z_loss()

    def get_all_losses(self) -> Dict[str, paddle.Tensor]:
        """获取所有损失（灵活模式）"""
        if hasattr(self.gate, "get_all_losses"):
            return self.gate.get_all_losses()
        else:
            return {"auxiliary": self.get_auxiliary_loss(), "z_loss": self.get_z_loss()}

    def get_total_loss(self) -> paddle.Tensor:
        """获取总损失（灵活模式）"""
        if hasattr(self.gate, "get_total_loss"):
            return self.gate.get_total_loss()
        else:
            return self.get_auxiliary_loss() + self.get_z_loss()

    # 灵活损失管理方法
    def add_loss_function(
        self,
        name: str,
        loss_func: LossFunction,
        weight: float = 0.0,
        loss_type: LossType = LossType.CUSTOM,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ):
        """添加自定义损失函数"""
        if not self.use_flexible_loss:
            logger.warning("当前使用传统损失模式，无法添加自定义损失函数")
            return

        # 注册损失函数
        loss_registry.register_loss(name, loss_func)

        # 添加损失配置
        config = LossConfig(name, loss_type, weight, enabled, params or {})
        if hasattr(self.gate, "add_loss_config"):
            self.gate.add_loss_config(config)
        else:
            logger.warning("当前门控层不支持动态添加损失函数")

    def remove_loss_function(self, name: str):
        """移除损失函数"""
        if not self.use_flexible_loss:
            logger.warning("当前使用传统损失模式，无法移除损失函数")
            return

        if hasattr(self.gate, "remove_loss_config"):
            self.gate.remove_loss_config(name)
        else:
            logger.warning("当前门控层不支持动态移除损失函数")

    def update_loss_weights(self, weights: Dict[str, float]):
        """更新损失权重"""
        if not self.use_flexible_loss:
            logger.warning("当前使用传统损失模式，无法动态更新损失权重")
            return

        if hasattr(self.gate, "update_loss_weights"):
            self.gate.update_loss_weights(weights)
        else:
            logger.warning("当前门控层不支持动态更新损失权重")

    def set_loss_combiner(self, combiner_name: str):
        """设置损失组合器"""
        if not self.use_flexible_loss:
            logger.warning("当前使用传统损失模式，无法设置损失组合器")
            return

        if hasattr(self.gate, "set_loss_combiner"):
            self.gate.set_loss_combiner(combiner_name)
        else:
            logger.warning("当前门控层不支持动态设置损失组合器")

    def get_expert_info(self) -> Dict[str, Any]:
        """
        获取专家信息

        Returns:
            expert_info: 专家信息字典
        """
        return {
            "num_experts": self.num_experts,
            "num_experts_per_device": self.num_experts_per_device,
            "expert_parallel_degree": self.expert_parallel_degree,
            "moe_rank": self.moe_rank,
            "is_parallel_enabled": self.expert_parallel_degree > 1,
            "use_flexible_loss": self.use_flexible_loss,
        }
