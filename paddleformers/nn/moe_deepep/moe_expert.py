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

from abc import ABC, abstractmethod

import paddle
from paddle import nn


class MoEExpertInterface(ABC):
    """
    MoE专家网络接口

    定义专家网络的标准接口，支持不同的专家架构
    """

    @abstractmethod
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        专家网络前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [seq_len, hidden_size]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """
        pass


class StandardMoEExpert(nn.Layer, MoEExpertInterface):
    """
    标准MoE专家网络实现

    支持多种专家网络架构的统一实现
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_activation: str = "silu",
        expert_dropout: float = 0.0,
        **kwargs
    ):
        """
        初始化标准MoE专家网络

        Args:
            hidden_size: 隐藏维度
            intermediate_size: 中间维度
            expert_activation: 专家激活函数 ("silu", "gelu", "relu")
            expert_dropout: 专家dropout
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_activation = expert_activation
        self.expert_dropout = expert_dropout

        # 创建MLP层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        # 激活函数
        if expert_activation == "silu":
            self.activation = paddle.nn.functional.silu
        elif expert_activation == "gelu":
            self.activation = paddle.nn.functional.gelu
        elif expert_activation == "relu":
            self.activation = paddle.nn.functional.relu
        else:
            self.activation = paddle.nn.functional.silu

        # Dropout
        if expert_dropout > 0.0:
            self.dropout = nn.Dropout(expert_dropout)
        else:
            self.dropout = None

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        专家网络前向传播

        Args:
            hidden_states: 输入隐藏状态，形状: [seq_len, hidden_size]

        Returns:
            output: 输出隐藏状态，形状: [seq_len, hidden_size]
        """
        # 计算门控和上投影
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)

        # 应用激活函数
        intermediate = self.activation(gate) * up

        # 应用dropout
        if self.dropout is not None:
            intermediate = self.dropout(intermediate)

        # 下投影
        output = self.down_proj(intermediate)

        return output
