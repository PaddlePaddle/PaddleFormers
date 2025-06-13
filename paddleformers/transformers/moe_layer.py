# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
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

from typing import Any, List, Tuple

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import Tensor, nn
from paddle.distributed.communication.group import Group

from .moe_gate import PretrainedMoEGate
from .token_dispatcher import MoEFlexTokenDispatcher


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):
    """
    Rearranges the input tensor `x` based on gate results, truncates it according to the specified capacity, and performs padding.

    Args:
        x (Tensor)[Seq, Dim]: The input tensor.
        dispatch_mask (List[Tensor[Seq, 1], Tensor[Seq, 1]]): A list of dispatch masks.
        scatter_index (Union[List[Tensor[Seq,], Tensor[Seq]], Tensor[Seq, 2]]): A list or tensor representing scatter indices.
        num_experts (int): The number of experts.
        capacity (int): The capacity size.

    Returns:
        Tensor [Expert*Capacity, Dim]: The output tensor after dispatching.
    """
    output = None
    orig_dtype = x.dtype
    if isinstance(scatter_index, paddle.Tensor):
        scatter_index = scatter_index.unbind(1)
    for i_scatter_index, i_dispatch_mask in zip(scatter_index, dispatch_mask):
        init_output = paddle.zeros([num_experts * capacity, x.shape[-1]], dtype="float32")
        updates = x * i_dispatch_mask.cast(x.dtype)
        if output is None:
            output = paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        else:
            output = output + paddle.scatter(
                init_output,
                i_scatter_index,
                updates,
                overwrite=False,
            )
        if output.dtype != orig_dtype:
            output = output.cast(orig_dtype)
    return output


def combining(x, combine_weights, scatter_index):
    """
    Performs combination and aggregation operations on the input matrix.

    Args:
        x: Tensor[num_experts * capacity, dim] - The input matrix to be processed, where the last dimension represents the number of features.
        combine_weights: Union[List[Tensor[seq, 1], Tensor[seq, 1]], Tensor[seq, 2, 1]] - A list or tensor containing combination weights for each feature.
        scatter_index: Union[List[Tensor[seq], Tensor[seq]], Tensor[seq, 2]] - A tuple of indices indicating which elements are to be aggregated, where the first element is the row index and the second element is the column index.

    Returns:
        Tensor: The output matrix after combination and aggregation, with a shape of [n, dim * num_features], where n is the number of samples in the input matrix.
    """

    dim = x.shape[-1]
    if isinstance(scatter_index, (list, tuple)):
        scatter_index = paddle.concat([i.unsqueeze([-1]) for i in scatter_index], -1)
    scatter_index = scatter_index.reshape([-1])
    num_k = len(combine_weights) if isinstance(combine_weights, (list, tuple)) else combine_weights.shape[-1]
    x = paddle.gather(x, scatter_index).reshape([-1, num_k, dim])  # [seq,2,dim]
    if isinstance(combine_weights, (list, tuple)):
        combine_weights = paddle.concat(combine_weights, -1).unsqueeze([1])
    return paddle.matmul(combine_weights, x).squeeze(1)  # [seq,1,2] @ [seq,2,dim] -> [seq,1,dim]


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


class MoELayer(nn.Layer):
    def __init__(
        self,
        config,
        moe_num_experts: int,
        expert_class: nn.Layer,
        expert_kwargs: dict,
        gate: PretrainedMoEGate,
        capacity: int = 1.0,
        moe_group: str = "data",
        all_to_all_dropout=0.0,
    ):
        super().__init__()

        self.config = config

        self.moe_num_experts = moe_num_experts
        self.capacity = capacity

        try:
            dist.fleet.get_hybrid_communicate_group()
            is_fleet_init = True
        except AttributeError:
            is_fleet_init = False

        if (
            is_fleet_init
            and dist.fleet.get_hybrid_communicate_group().get_data_parallel_world_size() > 1
            and moe_group == "data"
        ):
            self.moe_group = dist.fleet.get_hybrid_communicate_group().get_data_parallel_group()
            self.moe_rank = dist.get_rank(self.moe_group)
            self.moe_rank = 0 if self.moe_rank < 0 else self.moe_rank
            self.expert_parallel_degree = dist.get_world_size(self.moe_group)
            self.expert_parallel_degree = 1 if self.expert_parallel_degree < 0 else self.expert_parallel_degree
            self.moe_num_experts_per_device = self._parse_moe_expert_parallel(
                self.moe_num_experts, self.expert_parallel_degree
            )
            self.is_dummy_moe = False if self.expert_parallel_degree > 1 else True
        else:
            # when moe_group is dummy, we don't need to use all_to_all
            self.moe_group = None
            self.moe_rank = 0
            self.expert_parallel_degree = 1
            self.moe_num_experts_per_device = self.moe_num_experts
            self.is_dummy_moe = True

        self.all_to_all_dropout = all_to_all_dropout
        self.enable_recompute = False

        self.experts = nn.LayerList([])
        for i in range(self.moe_num_experts):
            if i // self.moe_num_experts_per_device == self.moe_rank:
                self.experts.append(expert_class(**expert_kwargs))
            else:
                self.experts.append(None)

        self.gate = gate
        self.gate.group = self.moe_group
        self._post_init()

    def _parse_moe_expert_parallel(self, moe_num_experts, expert_parallel_degree):
        assert (
            moe_num_experts >= expert_parallel_degree
        ), f"expert moe_num_experts={moe_num_experts} >= moe_world_size={expert_parallel_degree}"
        assert (
            moe_num_experts % expert_parallel_degree == 0
        ), f"expert moe_num_experts={moe_num_experts} % moe_world_size={expert_parallel_degree} == 0"
        moe_num_experts_per_device = moe_num_experts // expert_parallel_degree
        return moe_num_experts_per_device

    def _post_init(self):
        for p in self.gate.parameters():
            p.is_gate = True

        for k in self.experts:
            if k is not None:
                for p in k.parameters():
                    p.expert = not self.is_dummy_moe
                    p.no_sync = not self.is_dummy_moe
                    # logger.info(f"expert param={p.name}, no-sync={p.no_sync}")

    def forward(
        self,
        hidden_state: paddle.Tensor,
    ):
        """MoE Layer forward function
            1. Gate Forward.
            2. Dispatch export.
            3. Experts Forward.

        Args:
            hidden_state: MoE Layer input

        Returns:
            final_out: MoE Layer main output.
            l_aux: MoE auxiliary loss.  l_zloss: MoE z loss."""
        batch_size, seq_len, d_model = hidden_state.shape

        reshaped_input = hidden_state.reshape([-1, d_model])

        # self.l_aux       :
        # topk_weight  : se
        # topk_ids    : sk
        # token_priority    : se
        # self.exp_counts  :
        capacity, topk_weight, topk_ids, token_priority, l_aux, l_zloss = self.gate(hidden_state)

        """MoE expert dispatch from: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py"""
        cnts = paddle.zeros([topk_ids.shape[0], len(self.experts)], dtype=topk_ids.dtype)
        cnts = cnts.put_along_axis(topk_ids, 1, axis=1)

        tokens_per_expert = cnts.sum(axis=0)
        idxs = topk_ids.reshape([topk_ids.shape[0] * topk_ids.shape[1]]).argsort()
        sorted_tokens = reshaped_input[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.detach()
        sorted_tokens_shape = sorted_tokens.shape

        if self.expert_parallel_degree > 1:
            tokens_per_ep_rank = tokens_per_expert.reshape([self.expert_parallel_degree, -1]).sum(axis=1)
            tokens_per_expert_group = _AllToAll.apply(
                [tokens_per_expert.shape[0]], tokens_per_expert, group=self.moe_group
            )
            output_splits = (
                tokens_per_expert_group.reshape([self.expert_parallel_degree, -1]).sum(axis=1).cpu().tolist()
            )
            input_split_sizes = tokens_per_ep_rank.cpu().tolist()
            gathered_tokens = _AllToAll.apply(
                [tokens_per_expert_group.sum(axis=0).cpu().item(), sorted_tokens.shape[1]],
                sorted_tokens,
                out_split_sizes=output_splits,
                in_split_sizes=input_split_sizes,
                group=self.moe_group,
            )

            tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
                [self.expert_parallel_degree, self.moe_num_experts_per_device]
            ).sum(axis=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
                gatherd_idxs[s : s + k] = i % self.moe_num_experts_per_device
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
            expert = self.experts[i + self.moe_rank * self.moe_num_experts_per_device]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = paddle.concat(outputs, axis=0) if len(outputs) > 0 else paddle.to_tensor(0, dtype=sorted_tokens.dtype)
        if self.expert_parallel_degree > 1:
            new_x = paddle.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = _AllToAll.apply(
                sorted_tokens_shape,
                new_x,
                out_split_sizes=input_split_sizes,
                in_split_sizes=output_splits,
                group=self.moe_group,
            )
            outs = gathered_tokens

        new_x = paddle.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.reshape(topk_ids.shape + [-1])
            .astype(topk_weight.dtype)
            .multiply_(topk_weight.unsqueeze(-1))
            .multiply_(token_priority.unsqueeze(-1))
            .sum(axis=1)
            .astype(new_x.dtype)
            .reshape([batch_size, seq_len, -1])
        )

        return final_out, l_aux, l_zloss


class MoEFlexTokenLayer(nn.Layer):
    def __init__(self, config, moe_num_experts, expert_class, expert_kwargs, gate, moe_group):

        super().__init__()
        self.config = config
        self.moe_group = moe_group
        self.ep_size = dist.get_world_size(self.moe_group)
        self.moe_router_topk = gate.top_k
        self.moe_num_experts = moe_num_experts
        self.num_local_experts = moe_num_experts // self.ep_size
        self.token_dispatcher = MoEFlexTokenDispatcher(
            self.num_local_experts, self.moe_router_topk, self.moe_num_experts, moe_group
        )

        self.experts = nn.LayerList([expert_class(**expert_kwargs)] * self.num_local_experts)
        self.router = gate

    def expert_forward(self, dispatched_input, tokens_per_expert):
        outputs = []
        tokens_per_expert = tokens_per_expert.tolist()
        # print(f"all tokens: {sum(tokens_per_expert)}, detail: {tokens_per_expert}")
        chunks = paddle.split(dispatched_input, num_or_sections=tokens_per_expert, axis=0)
        for chunk, expert in zip(chunks, self.experts):
            chunk = chunk.contiguous()
            # assert chunk.shape[0] != 0, "Cannot dispatch empty input"
            outputs += [expert(chunk)]

        return paddle.concat(outputs, axis=0)

    def forward(self, hidden_states: paddle.Tensor):
        _, _, d_model = hidden_states.shape
        # reshaped_input = hidden_states.reshape([-1, d_model])
        probs, routing_map, l_aux, l_zloss = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output = self.expert_forward(dispatched_input, tokens_per_expert)
        output, _ = self.token_dispatcher.token_unpermutation(expert_output, None)
        return output, l_aux, l_zloss
