# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Tuple

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F

from paddleformers.transformers import MoEGateMixin


class PretrainedMoEGate(nn.Layer, MoEGateMixin):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super(PretrainedMoEGate, self).__init__()

        self.config = config

        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size

        # force keep in float32 when using amp
        self._cast_to_low_precision = False

        self.moe_expert_capacity_factor = kwargs.pop("moe_expert_capacity_factor", 0.0)
        self.eval_capacity_factor = kwargs.pop("eval_capacity_factor", 1.0)

        self.group = kwargs.pop("group", None)
        self.global_aux_loss = kwargs.pop("global_aux_loss", False)
        if self.global_aux_loss:
            assert self.group is not None, "group is required when global_aux_loss is True"
            self.rank = dist.get_rank(self.group)

        self.expert_drop = kwargs.pop("expert_drop", False)
        self.noisy_gate_policy = kwargs.pop("noisy_gate_policy", None)
        self.drop_tokens = self.moe_expert_capacity_factor is not None and self.moe_expert_capacity_factor != 0.0
        self.use_rts = kwargs.pop("use_rts", True)
        self.top2_2nd_expert_sampling = kwargs.pop("top2_2nd_expert_sampling", True)

        self.moe_token_drop_policy = kwargs.pop("moe_token_drop_policy", "probs")
        # Qwen2MoE: greedy
        # DeepSeekV2&V3: group_limited_greedy for training, and noaux_tc for inference
        self.topk_method = kwargs.pop("topk_method", "greedy")
        self.top_k = kwargs.pop("top_k", 2)
        self.n_group = kwargs.pop("n_group", 1)  # for group_limited_greedy
        self.topk_group = kwargs.pop("topk_group", 1)  # for group_limited_greedy
        self.norm_topk_prob = kwargs.pop("norm_topk_prob", False)
        self.routed_scaling_factor = kwargs.pop("routed_scaling_factor", 1.0)

        # for flex token moe layer
        self.using_flex_token = kwargs.pop("using_flex_token", False)

    def _priority(self, topk_idx: paddle.Tensor, capacity: int) -> paddle.Tensor:
        """_summary_
            The priority is the cumulative sum of the expert indices.

            This method is used in hunyuan model
        Args:
            topk_idx (paddle.Tensor): [batch_size * seq_len, topk]

        Returns:
            paddle.Tensor: cumsum locations
        """
        _, k = topk_idx.shape
        # Shape: [seq_len * k]
        chosen_expert = topk_idx.reshape([-1])
        # Shape: [seq_len * k, num_experts].
        token_priority = F.one_hot(chosen_expert, self.num_experts).cast(paddle.int32)
        token_priority = paddle.logical_and(token_priority > 0, token_priority.cumsum(axis=0) < capacity)
        # Shape: [seq_len, num_experts].
        token_priority = token_priority.reshape([-1, k, self.num_experts]).sum(axis=1)

        return (token_priority > 0.0).astype("float32")

    def _topk_greedy(self, scores: paddle.Tensor, k: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]
        """
        topk_weight, topk_idx = paddle.topk(scores, k=k, axis=-1, sorted=True)
        return topk_weight, topk_idx

    def _topk_group_limited_greedy(
        self, scores: paddle.Tensor, k: int, n_group: int, topk_group: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts in each group
            n_groups (int): the number of groups for all experts
            topk_group (int): the number of groups selected

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]

        Note: the group size is normal greater than the number of k
        """
        bsz_seq_len, n_experts = scores.shape
        assert n_experts % n_group == 0, "n_experts must be divisible by n_groups"

        group_scores = scores.reshape([0, n_group, -1]).max(axis=-1)  # [n, n_group]
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.ones([], dtype="float32"), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=True)

        return topk_weight, topk_idx

    def _topk_noaux_tc(
        self, scores: paddle.Tensor, k: int, n_group: int, topk_group: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts in each group
            n_groups (int): the number of groups for all experts
            topk_group (int): the number of groups selected

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]

        Note: the group size is normal greater than the number of k
        """
        bsz_seq_len, n_experts = scores.shape
        assert n_experts % n_group == 0, "n_experts must be divisible by n_groups"

        assert self.e_score_correction_bias is not None, "e_score_correction_bias is None"
        scores_for_choice = scores.reshape([bsz_seq_len, -1]) + self.e_score_correction_bias.detach().unsqueeze(0)
        reshape_tmp_rst = scores_for_choice.reshape([bsz_seq_len, self.n_group, -1])
        top_k = min(reshape_tmp_rst.shape[2], 2)
        group_scores = reshape_tmp_rst.topk(top_k, axis=-1)[0].sum(axis=-1)  # fmt:skip [n, n_group]
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.ones([], dtype="float32"), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores_for_choice * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=True)
        topk_weight = scores.take_along_axis(topk_idx, axis=1) if not self.training else topk_weight

        return topk_weight, topk_idx

    def top1gating(
        self,
        logits: paddle.Tensor,
        used_token: paddle.Tensor = None,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements Top1Gating on logits."""
        if self.noisy_gate_policy == "RSample":
            logits += self.gumbel_rsample(logits.shape)

        gates = self.gate_score_func(logits=logits)

        # Create a mask for 1st's expert per token
        # noisy gating
        # Only save the position of the maximum value
        indices1_s = paddle.argmax(logits if self.noisy_gate_policy == "RSample" else gates, axis=1)
        # Convert the position of the maximum value to a one-hot vector [s, e]
        mask1 = self._one_hot_to_float(indices1_s, num_classes=self.num_experts)

        # mask only used tokens
        if used_token is not None:
            mask1 = paddle.einsum(
                "s,se->se", used_token, mask1
            )  # Element-wise multiply used_token with mask1 to obtain a new mask1

        # gating decisions
        exp_counts = paddle.sum(mask1, axis=0)  # Calculate the number of tokens for each expert

        # if we don't want to drop any tokens
        if not self.drop_tokens:
            new_capacity = paddle.max(exp_counts)  # Calculate the number of tokens for each expert
            # Communicate across expert processes to pick the maximum capacity.
            if self.group is not None:
                dist.all_reduce(
                    new_capacity, op=dist.ReduceOp.MAX, group=self.group
                )  # Calculate the maximum value among expert processes
            # Make sure the capacity value does not exceed the number of tokens.
            capacity = int(min(new_capacity, paddle.tensor(mask1.size(0))))
        else:
            capacity = self._capacity(gates, self.moe_expert_capacity_factor)

        l_aux = self._cal_aux_loss(gates, mask1)
        l_zloss = self._cal_z_loss(logits)

        # Random Token Selection
        if self.use_rts:
            mask1_rand = mask1 * self.uniform_sample(mask1)
        else:
            mask1_rand = mask1

        _, top_idx = paddle.topk(mask1_rand, k=capacity, axis=0)  # Select top_capacity tokens

        new_mask1 = mask1 * paddle.zeros_like(mask1).put_along_axis(top_idx, paddle.ones([], dtype="float32"), axis=0)
        mask1 = new_mask1

        # Compute locations in capacity buffer
        locations1 = paddle.cumsum(mask1, axis=0) - 1  # Compute the position of each token in mask1

        # Store the capacity location for each token
        locations1_s = paddle.sum(locations1 * mask1, axis=1).cast(paddle.int64)

        # Normalize gate probabilities
        mask1_float = mask1.cast(paddle.float32)
        gates = gates / gates * mask1_float

        locations1_sc = self._one_hot_to_float(locations1_s, capacity)
        combine_weights = paddle.einsum("se,sc->sec", gates, locations1_sc)
        dispatch_mask = combine_weights.cast(paddle.bool).detach()

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss

    def top2gating(
        self,
        logits: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # everything is in fp32 in this function
        gates = self.gate_score_func(logits=logits)

        # Create a mask for 1st's expert per token.
        indices1_s = paddle.argmax(gates, axis=1)  # [S, 1]
        mask1 = self._one_hot_to_int64(indices1_s, self.num_experts)  # [S, E]

        if self.top2_2nd_expert_sampling:
            # Create a mask for 2nd's expert per token using Gumbel-max trick.
            # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
            logits += self.gumbel_rsample(logits)

        # Replace top-expert with min value
        logits_except1 = logits.masked_fill(mask1.cast(paddle.bool), float("-inf"))  # [S, E]
        indices2_s = paddle.argmax(logits_except1, axis=1)  # [S, 1]
        mask2 = self._one_hot_to_int64(indices2_s, self.num_experts)  # [S, E]

        # Note: mask1 and mask2 can be combined to form a single mask.
        # mask = paddle.cat([mask1, mask2], axis=0)
        # locations = paddle.cumsum(mask, axis=0) - 1
        # locations1, locations2 = locations.split(2, axis=0)
        # Compute locations in capacity buffer.
        locations1 = paddle.cumsum(mask1, axis=0) - 1  # [S, E]
        locations2 = paddle.cumsum(mask2, axis=0) - 1  # [S, E]
        # Update 2nd's location by accounting for locations of 1st.
        locations2 += paddle.sum(mask1, axis=0, keepdim=True)

        l_aux = self._cal_aux_loss(gates, mask1)
        l_zloss = self._cal_z_loss(logits)

        # gating decisions
        exp_counts = paddle.sum(mask1 + mask2, axis=0)
        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(gates, self.moe_expert_capacity_factor)
            # Remove locations outside capacity from mask.
            mask1 *= (locations1 < capacity).cast(paddle.int64)
            mask2 *= (locations2 < capacity).cast(paddle.int64)
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            new_capacity = paddle.max(exp_counts)
            if self.group is not None:
                dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(new_capacity)

        # Store the capacity location for each token.
        locations1_s = paddle.sum(locations1 * mask1, axis=1)
        locations2_s = paddle.sum(locations2 * mask2, axis=1)

        # Normalize gate probabilities
        mask1_float = mask1.cast(paddle.float32)
        mask2_float = mask2.cast(paddle.float32)
        gates1_s = paddle.einsum("se,se->s", gates, mask1_float)
        gates2_s = paddle.einsum("se,se->s", gates, mask2_float)
        denom_s = gates1_s + gates2_s
        # Avoid divide-by-zero
        denom_s = paddle.clip(denom_s, min=paddle.finfo(denom_s.dtype).eps)
        gates1_s /= denom_s
        gates2_s /= denom_s

        # Calculate combine_weights and dispatch_mask
        gates1 = paddle.einsum("s,se->se", gates1_s, mask1_float)
        gates2 = paddle.einsum("s,se->se", gates2_s, mask2_float)
        locations1_sc = self._one_hot_to_float(locations1_s, capacity)
        locations2_sc = self._one_hot_to_float(locations2_s, capacity)
        combine1_sec = paddle.einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = paddle.einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.cast(paddle.bool)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss

    def _cal_seq_aux_loss(self, gates, top_k, topk_idx) -> paddle.Tensor:
        """
        Calculate sequence auxiliary loss.
        Args:
            logits (paddle.Tensor): Model output.
        Returns:
            paddle.Tensor: The value of sequence auxiliary loss.
        """
        if self.config.sequence_parallel:
            # [bs * seq_len, dim]
            # Todo: Temporary measure to be compatible with SP input dimensions:
            # this function affects loss_aux, but the glm4moe model does not actually use this result.
            # Correctness unvalidated; to be verified later.
            max_sequence_length = self.config.max_sequence_length
            local_total_tokens, local_num_experts = gates.shape
            batch_size = local_total_tokens * self.config.tensor_model_parallel_size // max_sequence_length
            seq_len = max_sequence_length
            ce = paddle.zeros([local_total_tokens, local_num_experts])
            ce.put_along_axis_(
                indices=topk_idx, values=paddle.ones_like(topk_idx, dtype=ce.dtype), axis=1, reduce="add"
            )
            ce = ce / (top_k / local_num_experts)
            gates_mean = paddle.mean(gates, axis=tuple(range(len(gates.shape) - 1))).unsqueeze(0)
            aux_loss = (ce * gates_mean).sum(axis=1).mean()
        else:
            # [bs, seq_len, dim]
            batch_size, seq_len, num_experts = gates.shape
            gates = gates / (gates.sum(axis=-1, keepdim=True) + 1e-20)
            _, topk_idx = paddle.topk(gates, top_k, axis=-1)
            ce = paddle.zeros([batch_size, self.num_experts])
            topk_idx = topk_idx.reshape([batch_size, -1])
            ce.put_along_axis_(
                indices=topk_idx, values=paddle.ones([batch_size, seq_len * top_k]), axis=1, reduce="add"
            )
            ce = ce / (seq_len * top_k / self.num_experts)
            aux_loss = (ce * paddle.mean(gates, axis=1)).sum(axis=1).mean()
        return aux_loss

    def topkgating(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements TopKGating on logits."""
        batch_size, seq_len, d_model = gates.shape
        gates_ori = gates
        gates = gates.reshape([-1, d_model])

        l_zloss = self._cal_z_loss(gates)

        # get topk gates
        if self.topk_method == "greedy":
            top_gate, top_idx = self._topk_greedy(gates, k=self.top_k)
        elif self.topk_method == "group_limited_greedy":
            top_gate, top_idx = self._topk_group_limited_greedy(
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
        elif self.topk_method == "noaux_tc":
            top_gate, top_idx = self._topk_noaux_tc(
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
            # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator
        top_gate = top_gate * self.routed_scaling_factor

        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.ones([], dtype="float32"), axis=1)
        if hasattr(self.config, "seq_aux") and self.config.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)

        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(
                gates,
                self.moe_expert_capacity_factor * self.top_k,
            )

            # update mask and locations by capacity
            if self.moe_token_drop_policy == "probs":
                topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
                capacity_probs, capacity_indices = paddle.topk(topk_masked_gates, k=capacity, axis=0, sorted=False)
                token_priority = self._priority(capacity_indices, capacity)

            elif self.moe_token_drop_policy == "position":
                token_priority = self._priority(top_idx, capacity)
            else:
                raise ValueError(f"Invalid moe_token_drop_policy: {self.moe_token_drop_policy}")
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            local_capacity = paddle.max(exp_counts)
            if self.group is not None:
                dist.all_reduce(local_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(local_capacity)
            token_priority = self._priority(top_idx, capacity)

        # normalize gates
        gates_masked = gates * mask
        if self.training:
            gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
            denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
            if self.norm_topk_prob:
                gates_masked = gates_masked / denom_s

        return (
            capacity,
            gates_masked.take_along_axis(top_idx, axis=-1),
            top_idx,
            token_priority.take_along_axis(top_idx, axis=-1),
            l_aux,
            l_zloss,
        )

    def topkgating_nodrop(self, gates: paddle.Tensor):
        """Implements TopKGating on logits."""
        batch_size, seq_len, d_model = gates.shape
        gates_ori = gates
        gates = gates.reshape([-1, d_model])

        l_zloss = self._cal_z_loss(gates)

        # get topk gates
        if self.topk_method == "greedy":
            top_gate, top_idx = self._topk_greedy(gates, k=self.top_k)
        elif self.topk_method == "group_limited_greedy":
            top_gate, top_idx = self._topk_group_limited_greedy(
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
        elif self.topk_method == "noaux_tc":
            top_gate, top_idx = self._topk_noaux_tc(
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )

            # norm gate to sum 1
        # if self.top_k > 1 and self.norm_topk_prob:
        #     denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
        #     top_gate = top_gate / denominator
        # top_gate = top_gate * self.routed_scaling_factor

        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.ones([], dtype="float32"), axis=1)

        gates_masked = gates * mask
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)

        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s

        gates_masked *= self.routed_scaling_factor

        if hasattr(self.config, "seq_aux") and self.config.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)
        # topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
        return gates_masked, mask, exp_counts, l_aux, l_zloss
