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
from typing import Dict

import paddle
from paddle import nn
from paddle.incubate.nn.functional import swiglu as fused_swiglu

from ...nn.mlp import MLP
from ...transformers import Linear, linear_utils
from ...transformers.activations import ACT2FN
from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.llama import fusion_ops
from ...transformers.refined_recompute import (
    RRColumnParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowParallelLinear,
    RRRowSequenceParallelLinear,
)



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
        expert_activation: str,
        expert_dropout: float,
        config: Dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_activation = expert_activation
        self.expert_dropout = expert_dropout

        # 创建MLP层
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias_attr=False)

        # 激活函数
        if self.expert_activation == "silu":
            self.activation = paddle.nn.functional.silu
        elif self.expert_activation == "gelu":
            self.activation = paddle.nn.functional.gelu
        elif self.expert_activation == "relu":
            self.activation = paddle.nn.functional.relu
        else:
            self.activation = paddle.nn.functional.silu

        # Dropout
        if self.expert_dropout > 0.0:
            self.dropout = nn.Dropout(self.expert_dropout)
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


class Qwen2MoeMLP(nn.Layer):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_activation: str,
        expert_dropout: float,
        config: Dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_activation = expert_activation
        self.expert_dropout = expert_dropout

        self.skip_recompute_ops = config.get("skip_recompute_ops", {})
        self.fuse_attention_ffn = config.get("skip_recompute_ops", False)
        self.tensor_parallel_degree = config.get("tensor_parallel_degree", 1)
        self.sequence_parallel = config.get("sequence_parallel", 1)
        self.recompute = config.get("recompute", False)
        self.recompute_use_reentrant = config.get("recompute_use_reentrant", False)

        if self.sequence_parallel:
            ColumnParallelLinear = linear_utils.ColumnSequenceParallelLinear
            RowParallelLinear = linear_utils.RowSequenceParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if self.recompute and not self.recompute_use_reentrant:
                if self.skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnSequenceParallelLinear
                if self.skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowSequenceParallelLinear
        else:
            ColumnParallelLinear = linear_utils.ColumnParallelLinear
            RowParallelLinear = linear_utils.RowParallelLinear

            # NOTE: refined_recompute is only supported when `recompute_use_reentrant=False`
            if self.recompute and not self.recompute_use_reentrant:
                if self.skip_recompute_ops.get("mlp_column_ln", False):
                    ColumnParallelLinear = RRColumnParallelLinear
                if self.skip_recompute_ops.get("mlp_row_ln", False):
                    RowParallelLinear = RRRowParallelLinear

        if self.tensor_parallel_degree > 1:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size * 2,
                    gather_output=False,
                    has_bias=False,
                )
            else:
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    gather_output=False,
                    has_bias=False,
                )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            if self.fuse_attention_ffn:
                self.gate_up_fused_proj = Linear(self.hidden_size, self.intermediate_size * 2, bias_attr=False)
            else:
                self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w1
                self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias_attr=False)  # w3
            self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias_attr=False)  # w2

        # if self.expert_activation == "silu":
        #     self.act_fn = fusion_ops.swiglu
        #     self.fuse_swiglu = True
        # else:
        #     self.act_fn = ACT2FN[self.expert_activation]
        #     self.fuse_swiglu = False

        self.act_fn = ACT2FN[self.expert_activation]
        self.fuse_swiglu = config.get("fuse_swiglu", False)

    def forward(self, x):
        if self.fuse_attention_ffn:
            x = self.gate_up_fused_proj(x)
            if self.fuse_swiglu:
                y = None
            else:
                x, y = x.chunk(2, axis=-1)
        else:
            x, y = self.gate_proj(x), self.up_proj(x)

        # if self.fuse_swiglu:
        #     x = self.act_fn(x, y)
        # else:
        #     x = self.act_fn(x) * y

        if self.fuse_swiglu:
            x = paddle.concat([x, y], axis=-1)
            x = fused_swiglu(x)
        else:
            x = self.act_fn(x) * y

        return self.down_proj(x)

expert_class_mapping = {
    "StandardMoEExpert": StandardMoEExpert,
    "Qwen2MoeMLP": Qwen2MoeMLP
}