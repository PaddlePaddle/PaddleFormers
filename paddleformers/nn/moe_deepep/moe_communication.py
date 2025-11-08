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
from typing import Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.communication.group import Group

from ...transformers.moe_layer import _AllToAll


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
        """
        Forward propagation for EP (Expert Parallelism) communication

        Args:
            hidden_states: Input hidden states. Shape: [batch_size * seq_len, d_model]
            topk_indices: Top-K expert indices. Shape: [batch_size * seq_len, num_experts_per_token]
            topk_weights: Top-K weights. Shape: [batch_size * seq_len, num_experts_per_token]
            gates_masked:
                Masked gate scores, where each row contains only `num_experts_per_token` non-zero elements, and the sum of each row is 1.
                Shape: [batch_size * seq_len, num_experts]
            mask:
                Mask tensor indicating which experts are selected.
                Shape: [batch_size * seq_len, num_experts], where [i, j] is 1 when expert j is selected at sequence index i; otherwise it's zero.
            expert_parallel_degree: Degree of expert parallelism
            moe_group: MoE communication group

        Returns:
            output: Output hidden states
            aux_loss: Auxiliary loss
            z_loss: Z-loss
        """
        if expert_parallel_degree <= 1:
            return hidden_states
        mask = mask.to(paddle.int64)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, d_model = hidden_states.shape
        else:
            seq_len, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model])

        tokens_per_expert = mask.sum(axis=0)  # Shape: [num_experts]
        token_indices, expert_indices = paddle.where(mask == 1)
        combined_key = expert_indices * seq_len + token_indices
        sort_indices = paddle.argsort(combined_key)
        sorted_token_indices = token_indices[sort_indices]
        sorted_tokens = reshaped_input[
            sorted_token_indices
        ]  # Tokens that sorted by expert id. First `tokens_per_expert[0]` tokens belong to expert 0, next `tokens_per_expert[1]` tokens belong to expert 1, etc. Shape: [batch_size * seq_len * num_experts_per_token, d_model]

        tokens_per_expert = tokens_per_expert.detach()
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_ep_rank = tokens_per_expert.reshape([expert_parallel_degree, -1]).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        tokens_per_expert_group = _AllToAll.apply([tokens_per_expert.shape[0]], tokens_per_expert, group=moe_group)

        tokens_per_expert_group_sum = tokens_per_expert_group.reshape([expert_parallel_degree, -1])
        output_splits = tokens_per_expert_group_sum.sum(axis=1).cpu().tolist()
        input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        output_shape = [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]]

        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        gathered_tokens = _AllToAll.apply(
            output_shape,
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

        # Expert Forward
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_post_gather):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = experts[i + moe_rank * num_experts_per_device]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = paddle.concat(outputs, axis=0) if len(outputs) > 0 else paddle.to_tensor(0, dtype=sorted_tokens.dtype)

        # Third All-to-All: Exchange expert outputs back to original rank. `gathered_tokens` are the tokens that originally belong to current rank
        new_x = paddle.empty_like(outs)
        new_x[gatherd_idxs] = outs

        gathered_tokens = _AllToAll.apply(
            sorted_tokens_shape,
            new_x,
            out_split_sizes=input_split_sizes,
            in_split_sizes=output_splits,
            group=moe_group,
        )

        # For every processed token, need to multiply the expert weight.
        num_all_tokens = tokens_per_expert.sum().item()  # i.e. batch_size * seq_len * num_experts_per_token
        boundaries = paddle.cumsum(tokens_per_expert, dim=0)
        token_indices = paddle.arange(num_all_tokens)
        expert_ids = paddle.searchsorted(boundaries, token_indices, right=False)  # shape [num_all_tokens]
        expert_major_weights = gates_masked[sorted_token_indices, expert_ids]  # shape [num_all_tokens]
        weighted_gathered_tokens = gathered_tokens * expert_major_weights.unsqueeze(-1).to(
            gathered_tokens.dtype
        )  # shape [num_all_tokens, d_model]

        final_output_empty = paddle.zeros(reshaped_input.shape, dtype=gathered_tokens.dtype)
        token_indices_for_scatter = sorted_token_indices.unsqueeze(-1).expand(
            -1, d_model
        )  # shape [num_all_tokens, d_model]

        token_indices_for_scatter_single = token_indices_for_scatter[:, 0:1].squeeze()  # shape [num_all_tokens, 1]

        final_output = paddle.index_add(
            final_output_empty, index=token_indices_for_scatter_single, axis=0, value=weighted_gathered_tokens
        )

        return final_output


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
