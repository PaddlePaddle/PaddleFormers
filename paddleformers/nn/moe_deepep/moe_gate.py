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

from typing import Dict, Tuple

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from ...utils.log import logger


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
            dist.all_gather(me_list, me, group=self.group)
            dist.all_gather(ce_list, ce, group=self.group)

            me_list[self.rank] = me
            ce_list[self.rank] = ce
            me = paddle.stack(me_list).mean(0)
            ce = paddle.stack(ce_list).mean(0)
        aux_loss = paddle.sum(me * ce) * float(self.num_experts)
        return aux_loss

    def _cal_seq_aux_loss(self, probs, top_k, routing_map, max_seq_len):
        sub_max_seq_len = max_seq_len
        if hasattr(self, "moe_subbatch_token_num") and self.moe_subbatch_token_num > 0:
            sub_max_seq_len = self.moe_subbatch_token_num * self.tensor_parallel_degree

        # all_probs and routing_map should be computed using the runtime local sequence length on each worker.
        if self.tensor_parallel_degree > 1:
            assert self.sequence_parallel and max_seq_len % self.tensor_parallel_degree == 0
            local_seq_len = sub_max_seq_len // self.tensor_parallel_degree
            # [B*S, E]
            all_probs = AllGatherOp.apply(probs)
            # [B, S, E]
            all_probs = all_probs.reshape([-1, sub_max_seq_len, self.num_experts])
            batch_size = all_probs.shape[0]
            # [B, S, E]
            routing_map = routing_map.reshape([batch_size, local_seq_len, -1])
        else:
            # [B, S, E]
            if len(probs.shape) == 2:
                probs = probs.reshape([1] + probs.shape)
            batch_size, local_seq_len, _ = probs.shape
            all_probs = probs
            routing_map = routing_map.reshape([batch_size, local_seq_len, -1])

        seq_axis = 1
        # Both cost_coeff and seq_aux_loss must be computed with the global sequence length visible to all workers.
        # [B, E]
        cost_coeff = routing_map.sum(axis=seq_axis, dtype="float32") / paddle.to_tensor(
            max_seq_len * top_k / self.num_experts, dtype="float32"
        )
        # [B, E] -> [B] -> []
        seq_aux_loss = (cost_coeff * all_probs.sum(axis=seq_axis) / max_seq_len).sum(axis=1).mean()
        return seq_aux_loss

    def _cal_z_loss(self, logits) -> paddle.Tensor:
        """
        Calculate the z loss.

        Args:
            logits (paddle.Tensor): Model output. The shape is [batch_size, num_experts].

        Returns:
            paddle.Tensor: The z loss value.
        """
        l_zloss = paddle.logsumexp(logits, axis=1).square().mean()
        return l_zloss

    def _cal_orthogonal_loss(self) -> paddle.Tensor:
        """Gate weight orthogonal loss.

        Returns:
            Paddle.Tensor: orthogonal loss
        """
        weight = F.normalize(self.weight, axis=0)
        orthogonal_loss = paddle.mean(paddle.square(paddle.matmul(weight.T, weight) - paddle.eye(self.num_experts)))
        return orthogonal_loss

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
        token_priority = paddle.logical_and(token_priority > 0, token_priority.cumsum(axis=0) <= capacity)
        # Shape: [seq_len, num_experts].
        token_priority = token_priority.reshape([-1, k, self.num_experts]).sum(axis=1)

        return (token_priority > 0.0).astype("float32")

    def _probs_drop_policy(
        self,
        scores: paddle.Tensor,
        capacity: int,
    ) -> paddle.Tensor:
        """
        Implements the Probability-based (Probs) drop policy to enforce expert capacity.

        A token is assigned (mask value 1.0) to an expert if:
        1. It chose that expert (score > 0). (Implicitly handled by input scores).
        2. Its score for that expert is among the top 'capacity' scores for that expert.

        Args:
            scores (paddle.Tensor): [num_tokens, num_total_experts].
                                This should already contain zeros for non-selected
                                experts (i.e., the result of top-K gating).
            capacity (int): The maximum number of tokens any single expert can handle.
                                    (Not strictly used here, but good practice to include).

        Returns:
            paddle.Tensor: [num_tokens, num_total_experts] boolean mask (converted to float).
                        1.0 = Assigned and within capacity. 0.0 = Dropped or unassigned.
        """
        num_tokens, num_experts = scores.shape

        # --- Step 1: Find the 'capacity' best tokens for *each* expert ---

        # Use paddle.topk along dim=0 (the token dimension) to find the indices
        # of the tokens that have the highest scores for each expert (column).
        # Since 'scores' has shape [Tokens, Experts], dim=0 returns the token indices.

        # topk_token_indices has shape [capacity, num_total_experts]
        # It tells us WHICH tokens (row indices) are prioritized by capacity.

        # We use min(num_tokens, capacity) just in case there are fewer tokens than capacity.
        k_to_use = min(num_tokens, capacity)

        # We only care about the indices of the selected tokens
        _, topk_token_indices = paddle.topk(
            scores, k=k_to_use, dim=0, sorted=True  # Sorted=True is usually faster, but we only use the indices.
        )

        # --- Step 2: Create the final assignment mask using scatter ---

        # Initialize the mask to all zeros (tokens are initially dropped/unassigned).
        # We use boolean type for efficient scattering, then convert to float later.
        final_mask = paddle.zeros(num_tokens, num_experts, dtype=paddle.bool)

        # 2a. Create the column indices for the assignment.
        # We need a tensor of shape [k_to_use, num_experts] where each row is [0, 1, 2, ..., num_experts-1].
        col_indices = paddle.arange(num_experts).unsqueeze(0).expand_as(topk_token_indices)

        # 2b. Flatten the row (token) and column (expert) indices for advanced indexing.
        token_indices_flat = topk_token_indices.flatten()
        col_indices_flat = col_indices.flatten()

        # 2c. Use advanced indexing to set the mask positions to True.
        # This sets mask[token_index, expert_index] = True for all prioritized tokens.
        final_mask[token_indices_flat, col_indices_flat] = True

        # --- Step 3: Ensure only originally selected tokens are kept ---

        # Since paddle.topk can pick up tokens with score 0 if num_tokens < capacity,
        # we must ensure that we only keep tokens that had a positive score initially.
        # This step implicitly cleans up any spurious assignments made by topk on zero scores.

        token_priority_mask = final_mask.float() * (scores > 0).float()

        return token_priority_mask

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
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0), axis=-1)  # fmt:skip
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
        group_scores = (
            scores_for_choice.reshape([bsz_seq_len, self.n_group, -1]).topk(2, axis=-1)[0].sum(axis=-1)
        )  # fmt:skip [n, n_group]
        group_idx = paddle.topk(group_scores, k=topk_group, axis=-1, sorted=True)[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0, dtype="float32"), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1).expand([bsz_seq_len, n_group, n_experts // n_group]).reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores_for_choice * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(tmp_scores, k=k, axis=-1, sorted=True)

        # The bias term b is used only to adjust affinity scores for Top-K expert selection (routing); it does not affect gating.
        # The gate applied during dispatch and to weight the FFN output is computed from the original affinity score s_{i,t} (without the bias).
        topk_weight = scores.take_along_axis(topk_idx, axis=1) if not self.training else topk_weight

        return topk_weight, topk_idx


# Modified from PretrainedMoEGate
class StandardMoEGate(nn.Layer, MoEGateMixin):
    def __init__(
        self,
        num_experts: int,
        expert_hidden_size: int,
        drop_tokens: bool,
        topk_method: str,
        num_experts_per_tok: int,
        norm_topk_prob: bool,
        moe_config: Dict,
        seq_length: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        moe_subbatch_token_num: int,
        tensor_parallel_degree: int,
        sequence_parallel: bool,
        transpose_gate_weight: bool,
    ):
        super(StandardMoEGate, self).__init__()

        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.drop_tokens = drop_tokens
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        # force keep in float32 when using amp
        self._cast_to_low_precision = False
        self.seq_length = seq_length
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_subbatch_token_num = moe_subbatch_token_num
        self.tensor_parallel_degree = tensor_parallel_degree
        self.sequence_parallel = sequence_parallel
        self.transpose_gate_weight = transpose_gate_weight

        self.scoring_func = moe_config.get("gate_activation", "softmax")
        self.capacity_factor = moe_config.get("capacity_factor", 1.0)
        self.eval_capacity_factor = moe_config.get("eval_capacity_factor", 1.0)
        self.min_capacity = moe_config.get("min_capacity", 1)
        self.max_capacity = moe_config.get("max_capacity", pow(2, 32))
        self.group = moe_config.get("group", None)
        self.global_aux_loss = moe_config.get("global_aux_loss", False)
        self.use_rts = moe_config.get("use_rts", True)
        self.top2_2nd_expert_sampling = moe_config.get("top2_2nd_expert_sampling", True)
        self.drop_policy = moe_config.get("drop_policy", "probs")
        self.seq_aux = moe_config.get("seq_aux", True)

        if self.global_aux_loss:
            assert self.group is not None, "group is required when global_aux_loss is True"
            self.rank = dist.get_rank(self.group)

        # Accordding to the shape of gate weights in model checkpoint
        if not transpose_gate_weight:
            self.weight = paddle.create_parameter(
                shape=[self.expert_hidden_size, self.num_experts],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Uniform(),
            )
        else:
            self.weight = paddle.create_parameter(
                shape=[self.num_experts, self.expert_hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Uniform(),
            )

        if self.topk_method == "noaux_tc":
            self.register_buffer("e_score_correction_bias", paddle.zeros((self.num_experts,), dtype=paddle.float32))
            self._cast_to_low_precision = False
            self.expert_usage = paddle.zeros(
                shape=[self.num_experts],
                dtype=paddle.int64,
            )  # Used in MoECorrectionBiasAdjustCallback
            self.expert_usage.stop_gradient = True

    def forward(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss = self.topkgating(gates)
        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)
        if self.topk_method == "noaux_tc":
            with paddle.no_grad():
                self.expert_usage += exp_counts
        return capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss

    def topkgating(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements TopKGating on logits."""

        if len(gates.shape) == 3:
            batch_size, seq_len, d_model = gates.shape
            gates = gates.reshape([-1, d_model])
        elif len(gates.shape) == 2:
            batch_size_seq_len, d_model = gates.shape

        with paddle.amp.auto_cast(False):
            gates = gates.cast(self.weight.dtype)
            if not self.transpose_gate_weight:
                logits = F.linear(gates.cast("float32"), self.weight.cast("float32"))
            else:
                logits = F.linear(gates.cast("float32"), self.weight.cast("float32").t())
            gates = self.gate_score_func(logits=logits)
            gates = gates.cast(paddle.float32)

        gates_ori = gates
        if self.scoring_func == "sigmoid":
            gates_ori = gates_ori / (gates_ori.sum(axis=-1, keepdim=True) + 1e-20)

        l_zloss = self._cal_z_loss(gates)

        if self.topk_method == "greedy":
            top_gate, top_idx = self._topk_greedy(gates, k=self.num_experts_per_tok)
        elif self.topk_method == "group_limited_greedy":
            top_gate, top_idx = self._topk_group_limited_greedy(
                gates, k=self.num_experts_per_tok, n_group=self.n_group, topk_group=self.topk_group
            )
        elif self.topk_method == "noaux_tc":
            top_gate, top_idx = self._topk_noaux_tc(
                gates, k=self.num_experts_per_tok, n_group=self.n_group, topk_group=self.topk_group
            )
        else:
            raise NotImplementedError(f"Invalid topk_method: {self.topk_method}")

        # norm gate to sum 1
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator
        top_gate = top_gate * self.routed_scaling_factor

        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1)

        if self.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.num_experts_per_tok, mask, self.seq_length)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)

        if self.drop_tokens:
            # Calculate configured capacity and remove locations outside capacity from mask
            capacity = self._capacity(
                gates,
                self.capacity_factor * self.num_experts_per_tok,
                self.max_capacity,
                self.min_capacity,
            )

            # update mask and locations by capacity
            if self.drop_policy == "probs":
                topk_masked_gates = paddle.zeros_like(gates).put_along_axis(top_idx, top_gate, axis=1)
                token_priority = self._probs_drop_policy(topk_masked_gates, capacity)

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
            token_priority = self._priority(top_idx, capacity)

        # normalize gates
        gates_masked = gates * mask

        # if self.training:
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s
        gates_masked *= self.routed_scaling_factor
        return (
            capacity,  # new capacity
            top_gate,  # weights of selected experts for each token [num_tokens, num_experts_per_token]
            top_idx,  # indices of selected experts for each token [num_tokens, num_experts_per_token]
            gates_masked.to(
                paddle.float32
            ),  # masked gates. for each token, the selected experts are remainded with their original values, others are 0 [num_tokens, num_experts]
            mask,  # mask. for each token, the selected experts are marked with 1s [num_tokens, num_experts]
            token_priority.take_along_axis(top_idx, axis=-1),  # token priority
            l_aux,
            l_zloss,
        )
