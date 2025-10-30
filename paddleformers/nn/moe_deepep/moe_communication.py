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
from typing import Any, List, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.communication.group import Group

from ...transformers.token_dispatcher import MoEFlexTokenDispatcher


class MoECommunicationInterface(ABC):
    """
    MoE通信接口

    定义EP并行通信的标准接口，支持不同的通信策略
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: paddle.Tensor,
        topk_indices: paddle.Tensor,
        topk_weights: paddle.Tensor,
        mask: paddle.Tensor,
        hidden_states_masked: paddle.Tensor,
        expert_parallel_degree: int,
        moe_group: Group,
        experts: nn.LayerList,
        moe_rank: int,
        num_experts_per_device: int,
        num_experts: int,
        topk: int,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        EP并行通信前向传播

        Args:
            hidden_states: 输入隐藏状态
            topk_indices: TopK专家索引
            topk_weights: TopK权重
            expert_parallel_degree: 专家并行度
            moe_group: MoE通信组

        Returns:
            output: 输出隐藏状态
            aux_loss: 辅助损失
            z_loss: Z损失
        """
        pass


class StandardMoECommunication(nn.Layer, MoECommunicationInterface):
    """
    标准MoE通信实现

    基于All-to-All通信的EP并行实现
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        topk_indices: paddle.Tensor,
        topk_weights: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
        priorities: paddle.Tensor,
        expert_parallel_degree: int,
        moe_group: Group,
        experts: nn.LayerList,
        moe_rank: int,
        num_experts_per_device: int,
        num_experts: int,
        topk: int,
        token_dispatcher,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        EP并行通信前向传播

        Args:
            hidden_states: 输入隐藏状态
            topk_indices: TopK专家索引
            topk_weights: TopK权重
            expert_parallel_degree: 专家并行度
            moe_group: MoE通信组

        Returns:
            output: 输出隐藏状态
            aux_loss: 辅助损失
            z_loss: Z损失
        """
        if expert_parallel_degree <= 1:
            # 无需EP并行，直接返回
            return hidden_states

        # 计算每个专家的token数量
        # cnts = paddle.zeros([topk_indices.shape[0], num_experts], dtype=topk_indices.dtype)
        # cnts = cnts.put_along_axis(topk_indices, 1, axis=1)
        # tokens_per_expert = cnts.sum(axis=0)

        # 1. Reshape topk_indices to a single list of all expert assignments
        #    Shape: [T * K]
        # Check topk_indices validity
        if paddle.any(topk_indices < 0):
            raise ValueError("Invalid topk_indices found < 0.")
        if paddle.any(topk_indices >= num_experts):
            raise ValueError("Invalid topk_indices found >= num_experts.")
        if topk_indices.shape != topk_weights.shape:
            raise ValueError("topk_indices shape must match topk_weights shape.")
        if paddle.any(paddle.isnan(topk_indices)):
            raise ValueError("Invalid topk_indices found NaN.")
        if paddle.any(paddle.isinf(topk_indices)):
            raise ValueError("Invalid topk_indices found Inf.")

        flat_expert_indices = paddle.flatten(topk_indices)

        tokens_per_expert = paddle.bincount(x=flat_expert_indices, minlength=num_experts)
        tokens_per_expert = tokens_per_expert.detach()

        # 排序token
        idxs = topk_indices.reshape([topk_indices.shape[0] * topk_indices.shape[1]]).argsort()
        sorted_tokens = hidden_states[idxs // topk_indices.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        # EP并行通信
        # 计算每个EP rank的token数量
        tokens_per_ep_rank = tokens_per_expert.reshape([expert_parallel_degree, -1]).sum(axis=1)
        # 第一次All-to-All：交换token数量信息
        tokens_per_expert_group = _AllToAll.apply([tokens_per_expert.shape[0]], tokens_per_expert, group=moe_group)

        # 计算输出分割大小
        tokens_per_expert_group_sum = tokens_per_expert_group.reshape([expert_parallel_degree, -1])
        output_splits = tokens_per_expert_group_sum.sum(axis=1).cpu().tolist()
        input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        # 第二次All-to-All：交换token数据
        gathered_tokens = _AllToAll.apply(
            [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]],
            sorted_tokens,
            out_split_sizes=output_splits,
            in_split_sizes=input_split_sizes,
            group=moe_group,
        )

        # 计算聚合后的每个专家token数量
        tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
            [expert_parallel_degree, num_experts_per_device]
        ).sum(axis=0)
        # 创建聚合索引
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % num_experts_per_device
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather

        # expert 计算前向
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = experts[i + moe_rank * num_experts_per_device]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = paddle.concat(outputs, axis=0) if len(outputs) > 0 else paddle.to_tensor(0, dtype=sorted_tokens.dtype)

        # 第三次All-to-All：将专家输出分发回原始位置
        new_x = paddle.empty_like(outs)
        new_x[gatherd_idxs] = outs
        # assert paddle.max(paddle.to_tensor(gatherd_idxs)) < new_x.shape[0], "Index out of bounds"

        gathered_tokens = _AllToAll.apply(
            sorted_tokens_shape,
            new_x,
            out_split_sizes=input_split_sizes,
            in_split_sizes=output_splits,
            group=moe_group,
        )
        outs = gathered_tokens

        # 最终聚合
        new_x = paddle.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.reshape(topk_indices.shape + [-1])
            .astype(topk_weights.dtype)
            .multiply_(topk_weights.unsqueeze(-1))
            .sum(axis=1)
            .astype(new_x.dtype)
        )

        return final_out


class DeepEPMoECommunication(nn.Layer, MoECommunicationInterface):
    """
    DeepEP MoE 通信实现

    基于 DeepEP 通信的 EP 并行实现
    """

    def expert_forward(self, dispatched_input, tokens_per_expert, experts, moe_rank, num_experts_per_device):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist() if not isinstance(tokens_per_expert, list) else tokens_per_expert
        )
        chunks = paddle.split(dispatched_input, num_or_sections=tokens_per_expert, axis=0)
        for i, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            current_expert_idx = i + moe_rank * num_experts_per_device
            expert = experts[current_expert_idx]
            outputs += [expert(chunk)]

        return paddle.concat(outputs, axis=0)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        topk_indices: paddle.Tensor,
        topk_weights: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
        priorities: paddle.Tensor,
        expert_parallel_degree: int,
        moe_group: Group,
        experts: nn.LayerList,
        moe_rank: int,
        num_experts_per_device: int,
        num_experts: int,
        topk: int,
        token_dispatcher
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        if expert_parallel_degree <= 1:
            return hidden_states
        (dispatched_input, tokens_per_expert) = token_dispatcher.token_permutation(
            hidden_states,
            gates_masked,
            mask,
        )
        expert_output = self.expert_forward(
            dispatched_input, tokens_per_expert, experts, moe_rank, num_experts_per_device
        )
        output, _ = token_dispatcher.token_unpermutation(expert_output, None)
        return output


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx: Any,
        output_shape: List,
        input: Tensor,
        out_split_sizes: List = None,
        in_split_sizes: List = None,
        group: Group = None,
    ) -> Tensor:  # type: ignore
        """
        All-to-all communication in the group.
        Args:
            ctx (Any): Context object.
            output_shape (List): Output shape.
            input (Tensor): Input tensor.
            out_split_sizes (List): Output split sizes.
            in_split_sizes (List): Input split sizes.
            group (Group): The group object.
        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_shape = input.shape
        ctx.out_split_sizes = out_split_sizes
        ctx.in_split_sizes = in_split_sizes

        # return input
        if dist.get_world_size(group) <= 1:
            return input

        output = paddle.empty(output_shape, dtype=input.dtype)
        task = dist.alltoall_single(
            output,
            input,
            out_split_sizes=out_split_sizes,
            in_split_sizes=in_split_sizes,
            sync_op=False,
            group=group,
        )
        task.wait()

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor]:
        """
        Aggregates gradient information from all input tensors into a single tensor.
        Args:
            ctx (Any): The context object used to store information that needs to be passed.
            *grad_output (Tensor): A list of input tensors whose gradients are to be aggregated.
        Returns:
            Tuple[Tensor]: A tuple containing a tensor that holds the gradients of all input tensors.
        """
        # return grad_output
        return _AllToAll.apply(ctx.input_shape, *grad_output, ctx.in_split_sizes, ctx.out_split_sizes, ctx.group)
