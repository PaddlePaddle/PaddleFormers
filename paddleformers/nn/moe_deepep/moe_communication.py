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


class MoECommunicationInterface(ABC):
    @abstractmethod
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
        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size] or [batch_size, seq_len, hidden_size]
            topk_indices: Indices of selected experts for each token, shape: [num_tokens, num_experts_per_token]
            topk_weights: Weights of selected experts for each token, shape: [num_tokens, num_experts_per_token]
            gates_masked: Masked gates. For each token(row), the selected experts are remainded with their normalized gate values, others are 0. Shape: [num_tokens, num_experts]
            mask: Mask. For each token(row), the selected experts are marked with 1, others are 0. Shape: [num_tokens, num_experts]
            priorities: Token priorities, shape: [num_tokens, num_experts]
            expert_parallel_degree: Expert parallel degree
            moe_group: MoE group
            experts: Experts list
            moe_rank: Current rank id in the MoE group
            num_experts_per_device: Number of experts per device
            num_experts: Total number of experts
            topk: Number of experts per token
            token_dispatcher: Token dispatcher

        Returns:
            output: Output tensor
            aux_loss: Auxiliary loss
            z_loss: Z loss
        """
        pass


class AllToAllMoECommunication(nn.Layer, MoECommunicationInterface):
    """
    All-to-All EP
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

        if expert_parallel_degree <= 1:
            return hidden_states

        # 1. Reshape topk_indices to a single list of all expert assignments
        #    Shape: [T * K]
        flat_expert_indices = paddle.flatten(topk_indices)

        tokens_per_expert = paddle.bincount(x=flat_expert_indices, minlength=num_experts)
        tokens_per_expert = tokens_per_expert.detach()

        idxs = topk_indices.reshape([topk_indices.shape[0] * topk_indices.shape[1]]).argsort()
        sorted_tokens = hidden_states[idxs // topk_indices.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_ep_rank = tokens_per_expert.reshape([expert_parallel_degree, -1]).sum(axis=1)
        tokens_per_expert_group = _AllToAll.apply([tokens_per_expert.shape[0]], tokens_per_expert, group=moe_group)

        tokens_per_expert_group_sum = tokens_per_expert_group.reshape([expert_parallel_degree, -1])
        output_splits = tokens_per_expert_group_sum.sum(axis=1).cpu().tolist()
        input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        gathered_tokens = _AllToAll.apply(
            [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]],
            sorted_tokens,
            out_split_sizes=output_splits,
            in_split_sizes=input_split_sizes,
            group=moe_group,
        )

        tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
            [expert_parallel_degree, num_experts_per_device]
        ).sum(axis=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % num_experts_per_device
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather

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

        new_x = paddle.empty_like(outs)
        new_x[gatherd_idxs] = outs

        gathered_tokens = _AllToAll.apply(
            sorted_tokens_shape,
            new_x,
            out_split_sizes=input_split_sizes,
            in_split_sizes=output_splits,
            group=moe_group,
        )
        outs = gathered_tokens

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
    DeepEP EP
    """

    def expert_forward(self, dispatched_input, tokens_per_expert, experts, moe_rank, num_experts_per_device):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist() if not isinstance(tokens_per_expert, list) else tokens_per_expert
        )
        chunks = paddle.split(dispatched_input, num_or_sections=tokens_per_expert, axis=0)
        for i, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
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
        token_dispatcher,
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
    ) -> Tensor:
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
