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

import os
from typing import Any, Dict, Optional

import paddle
import paddle.distributed as dist
from paddle import nn

from .moe_communication import (
    DeepEPMoECommunication,
    MoECommunicationInterface,
    StandardMoECommunication,
)
from .moe_expert import MoEExpertInterface, StandardMoEExpert
from .moe_gate import MoEGateInterface, PretrainedMoEGate
from .token_dispatcher import MoEFlexTokenDispatcher


class ModularMoELayer(nn.Layer):
    """
    模块化MoE Layer EP并行实现

    设计理念：
    1. 高度模块化：门控、专家、通信完全解耦
    2. 易于扩展：支持自定义门控策略和专家架构
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int = 2,
        num_shared_experts: int = 1,
        expert_parallel_degree: int = 1,
        gate_type: str = "topk",
        topk: int = 2,
        topk_method: str = "greedy",
        gate_activation: str = "softmax",
        expert_activation: str = "silu",
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.0,
        expert_dropout: float = 0.0,
        moe_group: str = "data",
        all_to_all_dropout: float = 0.0,
        custom_gate: Optional[MoEGateInterface] = None,
        custom_expert: Optional[MoEExpertInterface] = None,
        custom_communication: Optional[MoECommunicationInterface] = None,
        # custom_loss: Optional[MoELossInterface] = None,
        **kwargs
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
            aux_loss_weight: 辅助损失权重
            z_loss_weight: Z损失权重
            expert_dropout: 专家dropout
            moe_group: MoE通信组
            all_to_all_dropout: All-to-All dropout
            custom_gate: 自定义门控网络
            custom_expert: 自定义专家网络
            custom_communication: 自定义通信策略
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.expert_parallel_degree = expert_parallel_degree
        self.moe_group = moe_group
        self.all_to_all_dropout = all_to_all_dropout
        self.topk = topk

        # 初始化EP并行相关参数
        self._init_expert_parallel()

        # 创建门控网络
        if custom_gate is not None:
            self.gate = custom_gate
        else:
            # self.gate = StandardMoEGate(
            #     hidden_size=hidden_size,
            #     num_experts=num_experts,
            #     num_experts_per_tok=num_experts_per_tok,
            #     gate_type=gate_type,
            #     topk=topk,
            #     gate_activation=gate_activation,
            #     aux_loss_weight=aux_loss_weight,
            #     z_loss_weight=z_loss_weight,
            # )
            self.gate = PretrainedMoEGate(
                config=None,
                num_experts=num_experts,
                expert_hidden_size=None,
                top_k=self.topk,
                topk_method=topk_method,
                drop_tokens=False,
            )

        # 创建专家网络
        if custom_expert is not None:
            # 如果传入的是实例，直接使用
            if isinstance(custom_expert, MoEExpertInterface):
                expert_class = type(custom_expert)
            else:
                expert_class = custom_expert
        else:
            expert_class = StandardMoEExpert

        self.experts = nn.LayerList([])
        for i in range(self.num_experts):
            if i // self.num_experts_per_device == self.moe_rank:
                expert = expert_class(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    expert_activation=expert_activation,
                    expert_dropout=expert_dropout,
                )
                self.experts.append(expert)
            else:
                # 创建一个空的Layer作为占位符
                empty_expert = nn.Layer()
                self.experts.append(empty_expert)

        # 创建共享专家
        if num_shared_experts > 0:
            self.shared_experts = expert_class(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size * num_shared_experts,
                expert_activation=expert_activation,
                expert_dropout=expert_dropout,
            )
        else:
            self.shared_experts = None

        # 创建通信策略
        if custom_communication is not None:
            self.communication = custom_communication
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

        if self.expert_parallel_degree > 1 and self.moe_group == "data":
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
        topk_indices, topk_weights, aux_loss, z_loss = self.gate(hidden_states)

        # 重塑输入
        reshaped_input = hidden_states.reshape([-1, d_model])

        # MoE前向传播
        if self.expert_parallel_degree > 1:
            # 使用EP并行
            output = self._forward_with_ep_parallel(reshaped_input, topk_indices, topk_weights)
        else:
            # 使用传统MoE
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
            hidden_states: 输入隐藏状态，形状: [seq_len, hidden_size]
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
            self.topk,
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
        }


class MoEFlexTokenLayer(nn.Layer):
    def __init__(self, config, num_experts, expert_class, expert_kwargs, gate, moe_group):

        super().__init__()
        self.config = config
        self.moe_group = moe_group
        self.ep_size = dist.get_world_size(self.moe_group)
        self.moe_router_topk = gate.top_k
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size
        self.moe_rank = dist.get_rank(self.moe_group)
        self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
        self.token_dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.moe_router_topk, self.num_experts, moe_group
        )
        self.expert_parallel_degree = 1 if self.ep_size < 0 else self.ep_size
        self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
            self.num_experts, self.expert_parallel_degree
        )
        self.experts = nn.LayerList([])
        for i in range(self.num_experts):
            if i // self.moe_num_experts_per_device == self.moe_rank:
                self.experts.append(expert_class(**expert_kwargs))
            else:
                self.experts.append(None)
        self.gate = gate

    def expert_forward(self, dispatched_input, tokens_per_expert):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist() if not isinstance(tokens_per_expert, list) else tokens_per_expert
        )
        # print(f"all tokens: {sum(tokens_per_expert)}, detail: {tokens_per_expert}")
        chunks = paddle.split(dispatched_input, num_or_sections=tokens_per_expert, axis=0)
        for i, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            outputs += [expert(chunk)]

        return paddle.concat(outputs, axis=0)

    def forward(self, hidden_states: paddle.Tensor):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        probs, routing_map, l_aux, l_zloss = self.gate(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output = self.expert_forward(dispatched_input, tokens_per_expert)
        output, _ = self.token_dispatcher.token_unpermutation(expert_output, None)
        return output, l_aux, l_zloss

    def _parse_moe_expert_parallel(self, num_experts, expert_parallel_degree):
        assert (
            num_experts >= expert_parallel_degree
        ), f"expert num_experts={num_experts} >= moe_world_size={expert_parallel_degree}"
        assert (
            num_experts % expert_parallel_degree == 0
        ), f"expert num_experts={num_experts} % moe_world_size={expert_parallel_degree} == 0"
        moe_num_experts_per_device = num_experts // expert_parallel_degree
        return moe_num_experts_per_device
