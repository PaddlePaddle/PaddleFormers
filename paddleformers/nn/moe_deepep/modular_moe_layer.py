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
from copy import deepcopy
from typing import Any, Dict, Optional

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import GatherOp, ScatterOp

from ...nn.mlp import MLP
from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.token_dispatcher import MoEFlexTokenDispatcher
from .moe_communication import (
    DeepEPMoECommunication,
    MoECommunicationInterface,
    StandardMoECommunication,
)
from .moe_expert import (
    MoEExpertInterface,
    Qwen2MoeMLP,
    StandardMoEExpert,
    expert_class_mapping,
)
from .moe_gate import FlexibleMoEGate, StandardMoEGate
from .moe_loss import LossCombiner, LossConfig, LossFunction, LossRegistry, LossType
from .moe_loss_instance import get_global_loss_registry

logger = logging.getLogger(__name__)
global_loss_registry = get_global_loss_registry()


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
        moe_intermediate_size: int,
        num_experts: int,
        num_shared_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: int,
        expert_activation: str,
        moe_config: Dict,
        model_type: str,
        pretrained_config: Optional[PretrainedConfig] = None,
    ):
        """
        初始化模块化MoE Layer

        Args:
            hidden_size: 隐藏维度
            moe_intermediate_size: MoE 中间维度
            num_experts: 专家数量
            num_experts_per_tok: 每个token选择的专家数(TopK)
            num_shared_experts: 共享专家数量
            norm_topk_prob: 是否归一化TopK的概率
            expert_activation: 专家使用的激活函数
            moe_config: 其他 MoE 相关配置


        moe_config 内参数：
            moe_group: MoE通信组
            custom_gate: 自定义门控网络
            custom_expert: 自定义专家网络
            custom_communication: 自定义通信策略
            expert_parallel_degree: EP 并行度
            gate_activation: 门控激活函数
            aux_loss_weight: 辅助损失权重（传统模式）
            z_loss_weight: Z损失权重（传统模式）
            train_topk_method: 训练时使用的 TopK 具体方法
            inference_topk_method: 推理时使用的 TopK 具体方法
            drop_tokens: 是否在 Expert 满后抛弃 Token
            use_flexible_loss: 是否使用灵活损失系统
            loss_configs: 损失配置列表（灵活模式）
            loss_combiner_name: 损失组合器名称
            expert_dropout: 专家dropout
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.expert_activation = expert_activation
        self.norm_topk_prob = norm_topk_prob
        self.model_type = model_type

        self.sequence_parallel = pretrained_config.get("sequence_parallel", False)
        self.tensor_parallel_degree = pretrained_config.get("tensor_parallel_degree", 1)
        self.seq_length = pretrained_config.get("seq_length", pretrained_config.get("max_seq_len", 1024))

        try:
            moe_group = fleet.get_hybrid_communicate_group().get_expert_parallel_group()
        except:
            moe_group = None
        self.expert_parallel_degree = dist.get_world_size(moe_group) if moe_group is not None else 1

        self.moe_group = moe_config.get("moe_group", "data")
        self.custom_gate = moe_config.get("custom_gate", None)
        self.custom_expert = moe_config.get("custom_expert", None)
        self.custom_communication = moe_config.get("custom_communication", None)
        self.gate_activation = moe_config.get("gate_activation", "softmax")
        self.aux_loss_weight = moe_config.get("aux_loss_weight", 0.01)
        self.z_loss_weight = moe_config.get("z_loss_weight", 0.0)
        self.topk_method = (
            moe_config.get("train_topk_method", "greedy")
            if self.training
            else moe_config.get("inference_topk_method", "greedy")
        )
        self.drop_tokens = moe_config.get("drop_tokens", True)
        self.use_flexible_loss = moe_config.get("use_flexible_loss", False)
        self.expert_dropout = moe_config.get("expert_dropout", 0.0)
        self.loss_configs = moe_config.get("loss_configs", None)
        self.loss_combiner_name = moe_config.get("loss_combiner_name", "weighted_sum")

        # 初始化EP并行相关参数
        self._init_expert_parallel()
        # 创建门控网络
        if self.custom_gate is not None:
            self.gate = self.custom_gate
        elif self.use_flexible_loss:
            # TODO: 使用灵活损失系统，暂未实现
            if self.loss_configs is None:
                self.loss_configs = [
                    LossConfig("auxiliary", LossType.AUXILIARY, weight=self.aux_loss_weight),
                    LossConfig("z_loss", LossType.Z_LOSS, weight=self.z_loss_weight),
                ]
            self.gate = FlexibleMoEGate(
                num_experts=self.num_experts,
                expert_hidden_size=self.hidden_size,
                drop_tokens=self.drop_tokens,
                topk_method=self.topk_method,
                num_experts_per_tok=self.num_experts_per_tok,
                norm_topk_prob=self.norm_topk_prob,
                moe_config=moe_config,
                loss_registry=global_loss_registry,
                loss_configs=self.loss_configs,
                loss_combiner_name=self.loss_combiner_name,
            )
        else:
            self.gate = StandardMoEGate(
                num_experts=self.num_experts,
                expert_hidden_size=self.hidden_size,
                drop_tokens=self.drop_tokens,
                topk_method=self.topk_method,
                num_experts_per_tok=self.num_experts_per_tok,
                norm_topk_prob=self.norm_topk_prob,
                moe_config=moe_config,
                seq_length=self.seq_length,
            )

        # 创建专家网络
        if self.custom_expert is not None:
            expert_class = expert_class_mapping.get(self.custom_expert, StandardMoEExpert)
        else:
            expert_class = StandardMoEExpert

        routed_expert_pretrained_config = deepcopy(pretrained_config)
        shared_expert_pretrained_config = deepcopy(pretrained_config)
        if self.expert_parallel_degree <= 1 and self.sequence_parallel and self.tensor_parallel_degree > 1:
            routed_expert_pretrained_config.sequence_parallel = False
            shared_expert_pretrained_config.sequence_parallel = False
        elif self.expert_parallel_degree > 1 and self.tensor_parallel_degree >= 1:
            routed_expert_pretrained_config.tensor_parallel_degree = 1

        # self.experts = nn.LayerList(
        #     [expert_class(
        #             hidden_size=self.hidden_size,
        #             intermediate_size=self.moe_intermediate_size,
        #             expert_activation=self.expert_activation,
        #             expert_dropout=self.expert_dropout,
        #             config={},
        #         )
        #     for _ in range(self.num_experts)]
        # )

        self.experts = nn.LayerList(
            [
                MLP(config=routed_expert_pretrained_config, intermediate_size=pretrained_config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        if self.expert_parallel_degree > 1:
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_experts_per_device, self.num_experts_per_tok, self.num_experts, self.moe_group
            )
        else:
            self.token_dispatcher = None

        # 创建共享专家
        if self.num_shared_experts > 0:
            # self.shared_experts = expert_class(
            #     hidden_size=self.hidden_size,
            #     intermediate_size=self.moe_intermediate_size * self.num_shared_experts,
            #     expert_activation=self.expert_activation,
            #     expert_dropout=self.expert_dropout,
            #     config={},
            # )
            self.shared_experts = MLP(
                config=shared_expert_pretrained_config, intermediate_size=pretrained_config.moe_intermediate_size
            )
        else:
            self.shared_experts = None

        # 创建通信策略
        if self.custom_communication is not None:
            self.communication = self.custom_communication
        else:
            if os.getenv("USE_DEEPEP", "0") == "1":
                self.communication = DeepEPMoECommunication()
            else:
                self.communication = StandardMoECommunication()

        # self.is_dummy_moe = False if self.expert_parallel_degree > 1 else True
        # for k in self.experts:
        #     if k is not None:
        #         for p in k.parameters():
        #             p.expert = not self.is_dummy_moe
        #             p.no_sync = not self.is_dummy_moe

        if hasattr(dist, "fleet") and dist.is_initialized() and self.expert_parallel_degree > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            for p in self.experts.parameters():
                setattr(p, "is_moe_param", True)
                setattr(p, "color", {"color": "moe_expert", "group": self.moe_grad_group})
                p.no_sync = not self.is_mp_moe
                p.expert = not self.is_mp_moe
                logger.info(f"expert no-sync={p.no_sync}-{p.name}")
                if self.is_mp_moe or self.is_ep_moe:
                    p.is_distributed = True

    def _init_expert_parallel(self):
        """
        初始化专家并行相关参数
        """

        def _parse_moe_expert_parallel(num_experts: int, expert_parallel_degree: int) -> int:
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
        except AttributeError as e:
            is_fleet_init = False

        if (
            is_fleet_init
            and self.expert_parallel_degree > 1
            # and dist.fleet.get_hybrid_communicate_group().get_data_parallel_world_size() > 1
        ):
            if self.moe_group == "data":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            elif self.moe_group == "expert":
                self.moe_group = dist.fleet.get_hybrid_communicate_group().get_expert_parallel_group()
                self.moe_grad_group = dist.fleet.get_hybrid_communicate_group().get_moe_sharding_parallel_group()
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            new_expert_parallel_degree = dist.get_world_size(self.moe_group)
            assert (
                self.expert_parallel_degree == new_expert_parallel_degree
            ), f"self.expert_parallel_degree={self.expert_parallel_degree} != moe_world_size={new_expert_parallel_degree}"
            self.expert_parallel_degree = 1 if new_expert_parallel_degree < 0 else new_expert_parallel_degree
            self.num_experts_per_device = _parse_moe_expert_parallel(self.num_experts, self.expert_parallel_degree)
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
        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        orig_shape = hidden_states.shape
        residuals = hidden_states
        capacity, topk_weights, topk_indices, gates_masked, mask, priorities, aux_loss, z_loss = self.gate(
            hidden_states
        )
        # 重塑输入

        # MoE前向传播
        if self.expert_parallel_degree > 1:
            output = self._forward_with_ep_parallel(hidden_states, topk_indices, topk_weights, gates_masked, mask)
        else:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                reshaped_input = hidden_states.reshape([-1, d_model])
            else:
                reshaped_input = hidden_states
            output = self._forward_traditional_moe(reshaped_input, topk_indices, topk_weights)

        # 恢复原始形状
        output = output.reshape(orig_shape)

        # 添加共享专家输出
        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)
            output = output + shared_output

        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        # currently no need return aux_loss and z_loss
        return output, aux_loss

    def _forward_traditional_moe(
        self, hidden_states: paddle.Tensor, selected_experts: paddle.Tensor, topk_weights: paddle.Tensor
    ) -> paddle.Tensor:
        """
        传统MoE前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [batch_size*seq_len, hidden_size]
            selected_experts: TopK专家索引，形状: [seq_len, num_experts_per_tok]
            topk_weights: TopK权重，形状: [seq_len, num_experts_per_tok]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        final_hidden_states = paddle.zeros_like(hidden_states, dtype=hidden_states.dtype)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).transpose([2, 1, 0])
        # [num_experts, topk, bs*seq]
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(axis=-1)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            current_hidden_states = expert_layer(current_state) * topk_weights[idx, top_x].unsqueeze(-1)
            final_hidden_states.index_add_(
                index=idx.reshape([-1]), axis=0, value=current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states.cast(hidden_states.dtype)

    def _forward_with_ep_parallel(
        self,
        hidden_states: paddle.Tensor,
        topk_indices: paddle.Tensor,
        topk_weights: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        EP并行MoE前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [seq_len, hidden_size]
            topk_indices: 形状: [seq_len, num_experts_per_token]
            topk_weights: 形状: [seq_len, num_experts_per_token]
            gates_masked: mask 后的 hidden_states，形状: [seq_len, num_experts]
            mask: 由每个 token 选中的 TopK 专家转换成 one-hot encoding，每一行会有 num_experts_per_tok 个 1，其他都是0，形状: [seq_len, num_experts]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """
        # 使用通信策略进行EP并行
        output = self.communication.forward(
            hidden_states,
            topk_indices,
            topk_weights,
            gates_masked,
            mask,
            self.expert_parallel_degree,
            self.moe_group,
            self.experts,
            self.moe_rank,
            self.num_experts_per_device,
            self.num_experts,
            self.num_experts_per_tok,
            self.token_dispatcher,
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
