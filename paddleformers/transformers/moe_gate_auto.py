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

from typing import Tuple

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils.log import logger
from .auto_utils import einsum


class MoEGateMixin:
    def gate_score_func(self, logits: paddle.Tensor) -> paddle.Tensor:
        # [..., hidden_dim] -> [..., num_experts]
        with paddle.amp.auto_cast(False):
            scoring_func = getattr(self, "scoring_func", None)
            if scoring_func == "softmax":
                scores = F.softmax(logits.cast("float32"), axis=-1)
            elif scoring_func == "sigmoid":
                scores = F.sigmoid(logits.cast("float32"))
            elif scoring_func == "tanh":
                scores = F.tanh(logits.cast("float32"))
            elif scoring_func == "relu":
                scores = F.relu(logits.cast("float32"))
            elif scoring_func == "gelu":
                scores = F.gelu(logits.cast("float32"))
            elif scoring_func == "leaky_relu":
                scores = F.leaky_relu(logits.cast("float32"))
            else:
                logger.warning_once(
                    f"insupportable scoring function for MoE gating: {scoring_func}, use softmax instead"
                )
                scores = F.softmax(logits.cast("float32"), axis=-1)
        return scores

    def gumbel_rsample(self, logits: paddle.Tensor) -> paddle.Tensor:
        gumbel = paddle.distribution.gumbel.Gumbel(0, 1)
        return gumbel.rsample(logits.shape)

    def uniform_sample(self, logits: paddle.Tensor) -> paddle.Tensor:
        uniform = paddle.distribution.uniform.Uniform(0, 1)
        return uniform.sample(logits.shape)

    @paddle.no_grad()
    def _one_hot_to_float(self, x, num_classes):
        if x.dtype not in (paddle.int32, paddle.int64):
            x = paddle.cast(x, paddle.int64)
        return F.one_hot(x, num_classes=num_classes).cast(paddle.get_default_dtype())

    @paddle.no_grad()
    def _one_hot_to_int64(self, x, num_classes):
        if x.dtype not in (paddle.int32, paddle.int64):
            x = paddle.cast(x, paddle.int64)
        return F.one_hot(x, num_classes=num_classes).cast(paddle.int64)

    @paddle.no_grad()
    def _capacity(
        self,
        gates: paddle.Tensor,
        capacity_factor: float,
        max_capacity: int,
        min_capacity: int,
    ) -> paddle.Tensor:
        """Calculate the capacity for each expert based on the gates and capacity factor.

        Args:
            gates (paddle.Tensor): A tensor of shape [num_tokens, num_experts] representing the probability distribution
                over experts for each token.
            capacity_factor (float): A scalar float value representing the capacity factor for each expert.
            min_capacity (int): A scalar integer value representing the minimum capacity for each expert.

        Returns:
            int: A tensor value representing the calculated capacity for each expert.
        """
        assert gates.ndim == 2, f"gates should be 2D, but got {gates.ndim}, {gates.shape}"
        # gates has shape of SE
        num_tokens = gates.shape[0]
        num_experts = gates.shape[1]
        capacity = int((num_tokens // num_experts) * capacity_factor)
        if capacity < min_capacity:
            capacity = min_capacity
        if capacity > max_capacity:
            capacity = max_capacity
        assert capacity > 0, f"requires capacity > 0, capacity_factor: {capacity_factor}, input_shape: {gates.shape}"

        return capacity

    def _cal_aux_loss(self, gates, mask):
        """
        Calculate auxiliary loss

        Args:
            gates (paddle.Tensor): Represents the output probability of each expert. The shape is [batch_size, num_experts]
            mask (paddle.Tensor): Represents whether each sample belongs to a certain expert. The shape is [batch_size, num_experts]

        Returns:
            paddle.Tensor: The value of auxiliary loss.

        """
        # TODO: @DrownFish19 update aux_loss for Qwen2MoE and DeepSeekV2&V3
        me = paddle.mean(gates, axis=0)
        ce = paddle.mean(mask.cast("float32"), axis=0)
        if self.global_aux_loss:
            me_list, ce_list = [], []
            # dist.all_gather(me_list, me, group=self.group)
            # dist.all_gather(ce_list, ce, group=self.group)

            me_list[self.rank] = me
            ce_list[self.rank] = ce
            me = paddle.stack(me_list).mean(0)
            ce = paddle.stack(ce_list).mean(0)
        aux_loss = paddle.sum(me * ce) * float(self.num_experts)
        return aux_loss

    def _cal_seq_aux_loss(self, gates, top_k, topk_idx) -> paddle.Tensor:
        """
        Calculate sequence auxiliary loss.
        Args:
            logits (paddle.Tensor): Model output.
        Returns:
            paddle.Tensor: The value of sequence auxiliary loss.
        """
        batch_size, seq_len, _ = gates.shape
        ce = paddle.zeros([batch_size, self.num_experts])
        topk_idx = topk_idx.reshape([batch_size, -1])
        ce.put_along_axis_(indices=topk_idx, values=paddle.ones([batch_size, seq_len * top_k]), axis=1)
        ce = ce / (seq_len * top_k / self.num_experts)
        aux_loss = (ce * paddle.mean(gates, axis=1)).sum(axis=1).mean()
        return aux_loss

    def _cal_z_loss(self, logits) -> paddle.Tensor:
        """
        Calculate the z loss.

        Args:
            logits (paddle.Tensor): Model output. The shape is [batch_size, num_experts].

        Returns:
            paddle.Tensor: The z loss value.
        """
        l_zloss = logits.exp().sum(1).log().square().mean()
        return l_zloss

    def _cal_orthogonal_loss(self) -> paddle.Tensor:
        """Gate weight orthogonal loss.

        Returns:
            Paddle.Tensor: orthogonal loss
        """
        weight = F.normalize(self.weight, axis=0)
        orthogonal_loss = paddle.mean(paddle.square(paddle.matmul(weight.T, weight) - paddle.eye(self.num_experts)))
        return orthogonal_loss


class PretrainedMoEGate(nn.Layer, MoEGateMixin):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super(PretrainedMoEGate, self).__init__()

        self.config = config

        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.expert_parallel_degree = kwargs.pop("expert_parallel_degree", 1)

        # force keep in float32 when using amp
        self._cast_to_low_precision = False

        self.capacity_factor = kwargs.pop("capacity_factor", 1.0)
        self.eval_capacity_factor = kwargs.pop("eval_capacity_factor", 1.0)
        self.min_capacity = kwargs.pop("min_capacity", 1.0)
        self.max_capacity = kwargs.pop("max_capacity", pow(2, 32))

        self.group = kwargs.pop("group", None)
        self.global_aux_loss = kwargs.pop("global_aux_loss", False)
        if self.global_aux_loss:
            assert self.group is not None, "group is required when global_aux_loss is True"
            self.rank = dist.get_rank(self.group)

        self.expert_drop = kwargs.pop("expert_drop", False)
        self.noisy_gate_policy = kwargs.pop("noisy_gate_policy", None)
        self.drop_tokens = kwargs.pop("drop_tokens", True)
        self.use_rts = kwargs.pop("use_rts", True)
        self.top2_2nd_expert_sampling = kwargs.pop("top2_2nd_expert_sampling", True)

        self.drop_policy = kwargs.pop("drop_policy", "probs")
        # Qwen2MoE: greedy
        # DeepSeekV2&V3: group_limited_greedy for training, and noaux_tc for inference
        self.topk_method = kwargs.pop("topk_method", "greedy")
        self.top_k = kwargs.pop("top_k", 2)
        self.n_group = kwargs.pop("n_group", 1)  # for group_limited_greedy
        self.topk_group = kwargs.pop("topk_group", 1)  # for group_limited_greedy
        self.norm_topk_prob = kwargs.pop("norm_topk_prob", False)
        self.routed_scaling_factor = kwargs.pop("routed_scaling_factor", 1.0)

    def _priority(self, topk_idx: paddle.Tensor, capacity: int) -> paddle.Tensor:
        """_summary_
            The priority is the cumulative sum of the expert indices.

            This method is used in hunyuan model
        Args:
            topk_idx (paddle.Tensor): [batch_size * seq_len, topk]

        Returns:
            paddle.Tensor: cumsum locations
        """
        # Make num_selected_experts the leading axis to ensure that top-1 choices
        # have priority over top-2 choices, which have priority over top-3 choices,
        # etc.
        expert_index = paddle.transpose(topk_idx, [1, 0])  # [topk, B*S]
        # Shape: [num_selected_experts * tokens_per_group]
        expert_index = expert_index.reshape([-1])

        # Create mask out of indices.
        # Shape: [tokens_per_group * num_selected_experts, num_experts].
        expert_mask = F.one_hot(expert_index, self.num_experts).cast(paddle.int32)

        # Experts have a fixed capacity that we cannot exceed. A token's priority
        # within the expert's buffer is given by the masked, cumulative capacity of
        # its target expert.
        # Shape: [tokens_per_group * num_selected_experts, num_experts].
        token_priority = paddle.cumsum(expert_mask, axis=0) * expert_mask - 1
        # Shape: [num_selected_experts, tokens_per_group, num_experts].
        token_priority = token_priority.reshape((self.top_k, -1, self.num_experts))
        # Shape: [tokens_per_group, num_selected_experts, num_experts].
        token_priority = paddle.transpose(token_priority, [1, 0, 2])
        # For each token, across all selected experts, select the only non-negative
        # (unmasked) priority. Now, for group G routing to expert E, token T has
        # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
        # is its targeted expert.
        # Shape: [tokens_per_group, num_experts].
        token_priority = paddle.max(token_priority, axis=1)

        # Token T can only be routed to expert E if its priority is positive and
        # less than the expert capacity. One-hot matrix will ignore indices outside
        # the range [0, expert_capacity).
        # Shape: [tokens_per_group, num_experts, expert_capacity].
        valid_mask = paddle.logical_and(token_priority >= 0, token_priority < capacity)
        token_priority = paddle.masked_fill(token_priority, ~valid_mask, 0)
        dispatch_mask = F.one_hot(token_priority, capacity).cast(paddle.int32)
        valid_mask = valid_mask.unsqueeze(-1).expand(valid_mask.shape + [capacity])
        dispatch_mask = paddle.masked_fill(dispatch_mask, ~valid_mask, 0)

        return dispatch_mask

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
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=False)

        return topk_weight, topk_idx

    def _topk_noaux_tc(
        self, scores: paddle.Tensor, e_score_correction_bias, k: int, n_group: int, topk_group: int
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

        assert e_score_correction_bias is not None, "e_score_correction_bias is None"
        # scores = scores.reshape([bsz_seq_len, -1]) + self.e_score_correction_bias.unsqueeze(0)
        scores = scores.reshape([bsz_seq_len, -1]) + e_score_correction_bias.unsqueeze(0)
        group_scores = scores.reshape([bsz_seq_len, self.n_group, -1]).topk(2, axis=-1)[0].sum(axis=-1)  # [n, n_group]
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=False)
        topk_weight = scores.gather(topk_idx, axis=1) if not self.training else topk_weight

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
        capacity = self._capacity(gates, self.capacity_factor, self.max_capacity, self.min_capacity)

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

        l_aux = self._cal_aux_loss(gates, mask1)
        l_zloss = self._cal_z_loss(logits)

        # Random Token Selection
        if self.use_rts:
            mask1_rand = mask1 * self.uniform_sample(mask1)
        else:
            mask1_rand = mask1

        assert (
            logits.shape[0] >= self.min_capacity
        ), "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

        _, top_idx = paddle.topk(mask1_rand, k=capacity, axis=0)  # Select top_capacity tokens

        new_mask1 = mask1 * paddle.zeros_like(mask1).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=0)
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
        # mask = paddle.concat([mask1, mask2], axis=0)
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
            capacity = self._capacity(gates, self.capacity_factor, self.max_capacity, self.min_capacity)
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
                gates, self.e_score_correction_bias, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
            # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator
        else:
            top_gate = top_gate * self.routed_scaling_factor

        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)
        if self.config.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)

        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(
                gates,
                self.capacity_factor * self.top_k,
                self.max_capacity,
                self.min_capacity,
            )

            # update mask and locations by capacity
            if self.drop_policy == "probs":
                topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
                capacity_probs, capacity_indices = paddle.topk(topk_masked_gates, k=capacity, axis=0, sorted=False)
                token_priority = self._priority(capacity_indices, capacity)

            elif self.drop_policy == "position":
                token_priority = self._priority(top_idx, capacity)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            local_capacity = paddle.max(exp_counts)
            if self.group is not None:
                dist.all_reduce(local_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(local_capacity)

        # normalize gates
        gates_masked = gates * mask
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s

        if not self.drop_tokens:
            locations = paddle.cumsum(mask, axis=0) - 1
            token_priority = self._one_hot_to_float(locations * mask, capacity)

        combine_weights = paddle.einsum("se,sec->sec", gates_masked, token_priority.cast(paddle.get_default_dtype()))
        dispatch_mask = combine_weights.cast(paddle.bool)

        return capacity, combine_weights, dispatch_mask, exp_counts, l_aux, l_zloss

    def topkgating_part1(self, gates, e_score_correction_bias):
        l_zloss = self._cal_z_loss(gates)
        batch_size, seq_len, d_model = gates.shape
        gates_ori = gates
        gates = gates.reshape([-1, d_model])

        # get topk gates
        if self.topk_method == "greedy":
            top_gate, top_idx = self._topk_greedy(gates, k=self.top_k)
        elif self.topk_method == "group_limited_greedy":
            top_gate, top_idx = self._topk_group_limited_greedy(
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
        elif self.topk_method == "noaux_tc":
            top_gate, top_idx = self._topk_noaux_tc(
                gates, e_score_correction_bias, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
            # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator
        else:
            top_gate = top_gate * self.routed_scaling_factor

        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)
        if self.config.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)

        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(
                gates,
                self.capacity_factor * self.top_k,
                self.max_capacity,
                self.min_capacity,
            )

            # update mask and locations by capacity
            if self.drop_policy == "probs":
                topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
                capacity_probs, capacity_indices = paddle.topk(topk_masked_gates, k=capacity, axis=0, sorted=False)
                token_priority = self._priority(capacity_indices, capacity)

            elif self.drop_policy == "position":
                token_priority = self._priority(top_idx, capacity)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        else:
            # Do not drop tokens - set capacity according to current expert assignments
            # local_capacity = paddle.max(exp_counts)
            # capacity = int(local_capacity)
            capacity = None
            token_priority = None

        # keep these tensor for using in topkgating_part2
        self.mask = mask
        self.capacity = capacity
        self.token_priority = token_priority

        return exp_counts, l_aux, l_zloss

    def topkgating_part2(self, gates):

        gates_masked = gates * self.mask
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s

        if not self.drop_tokens:
            locations = paddle.cumsum(self.mask, axis=0) - 1
            self.token_priority = self._one_hot_to_float(locations * self.mask, self.capacity)

        combine_weights = einsum("se,sec->sec", gates_masked, self.token_priority.cast(paddle.get_default_dtype()))
        dispatch_mask = combine_weights.cast(paddle.bool)

        return combine_weights, dispatch_mask
