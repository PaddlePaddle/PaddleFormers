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

from ...utils.log import logger
from .moe_loss import LossCombiner, LossConfig, LossFunction, LossRegistry, LossType


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
        ce.put_along_axis_(indices=topk_idx, values=paddle.ones([batch_size, seq_len * top_k]), axis=1, reduce="add")
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
        topk_weight = scores.take_along_axis(topk_idx, axis=1) if not self.training else topk_weight

        return topk_weight, topk_idx


class PretrainedMoEGate(nn.Layer, MoEGateMixin):
    def __init__(self, config, num_experts, expert_hidden_size, **kwargs):
        super(PretrainedMoEGate, self).__init__()

        self.config = config
        self.scoring_func = config.get("scoring_func", None)

        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size

        # force keep in float32 when using amp
        self._cast_to_low_precision = False

        self.capacity_factor = config.get("capacity_factor", 1.0)
        self.eval_capacity_factor = config.get("eval_capacity_factor", 1.0)
        self.min_capacity = config.get("min_capacity", 1.0)
        self.max_capacity = config.get("max_capacity", pow(2, 32))

        self.group = config.get("group", None)
        self.global_aux_loss = config.get("global_aux_loss", False)
        if self.global_aux_loss:
            assert self.group is not None, "group is required when global_aux_loss is True"
            self.rank = dist.get_rank(self.group)

        self.noisy_gate_policy = config.get("noisy_gate_policy", None)
        self.drop_tokens = config.get("drop_tokens", False)
        self.use_rts = config.get("use_rts", True)
        self.top2_2nd_expert_sampling = config.get("top2_2nd_expert_sampling", True)

        self.drop_policy = config.get("drop_policy", "probs")
        # Qwen2MoE: greedy
        # DeepSeekV2&V3: group_limited_greedy for training, and noaux_tc for inference
        self.topk_method = config.get("topk_method", "greedy")
        self.top_k = config.get("num_experts_per_tok", 2)
        self.n_group = config.get("n_group", 1)  # for group_limited_greedy
        self.topk_group = config.get("topk_group", 1)  # for group_limited_greedy
        self.norm_topk_prob = config.get("norm_topk_prob", False)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

        # weight of hidden_state -> score
        self.weight = paddle.create_parameter(
            shape=[self.expert_hidden_size, self.num_experts],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Uniform(),
        )

    def forward(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # return self.topkgating(gates)
        return self.topkgating_nodrop(gates)

    def topkgating(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements TopKGating on logits."""
        print("forward input gates: ", gates.shape, gates.dtype)  # [8, 97, 2048] paddle.bfloat16

        batch_size, seq_len, d_model = gates.shape
        gates_ori = gates
        gates = gates.reshape([-1, d_model])

        # 将 hidden_state 转换成 score（每个 token 对每个专家的偏好分数）
        with paddle.amp.auto_cast(False):
            hidden_states = gates.cast(self.weight.dtype)
            logits = F.linear(hidden_states.cast("float32"), self.weight.cast("float32"))
            gates = self.gate_score_func(logits=logits)
        print("forward after logits gates: ", gates.shape, gates.dtype)  # [776, 2048] paddle.bfloat16

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
        print("gates: ", gates.shape, gates.dtype)  # [776, 2048] paddle.bfloat16
        print("top_gate: ", top_gate.shape, top_gate.dtype)  #
        print("top_idx: ", top_idx.shape, top_idx.dtype)  # [776, 8] paddle.int64
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1)

        if hasattr(self.config, "seq_aux") and self.config.seq_aux:
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
            print("----------MOE Gate, no drop tokens: local_capacity = ", local_capacity)
            print("----------SELF.GROUP ", self.group)
            if self.group is not None:
                dist.all_reduce(local_capacity, op=dist.ReduceOp.MAX, group=self.group)
            capacity = int(local_capacity)
            token_priority = self._priority(top_idx, capacity)

        # normalize gates
        # gates_masked is equal to top_gate.
        gates_masked = gates * mask

        # if self.training:
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        print("gates_s: ", gates_s.shape, gates_s.dtype)  # [776, 1] paddle.bfloat16
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s
        gates_masked *= self.routed_scaling_factor
        print("gates_masked after: ", gates_masked.shape, gates_masked.dtype)  # [776, 2048] paddle.bfloat16

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

        # 将 hidden_state 转换成 score（每个 token 对每个专家的偏好分数）
        with paddle.amp.auto_cast(False):
            hidden_states = gates.cast(self.weight.dtype)
            logits = F.linear(hidden_states.cast("float32"), self.weight.cast("float32"))
            gates = self.gate_score_func(logits=logits)
        print("forward after logits gates: ", gates.shape, gates.dtype)  # [776, 2048] paddle.bfloat16

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
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0), axis=1)

        if hasattr(self.config, "seq_aux") and self.config.seq_aux:
            l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        else:
            l_aux = self._cal_aux_loss(gates, mask)

        # The gate applied during dispatch and to weight the FFN output is computed from the original affinity score s_{i,t} (without the bias).
        gates_masked = gates * mask
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s
        gates_masked *= self.routed_scaling_factor

        exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)
        return gates_masked, mask, exp_counts, l_aux, l_zloss


class FlexibleMoEGate(nn.Layer, MoEGateMixin):
    """自定义损失函数的 MoE Gate"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        loss_registry,
        num_experts_per_tok: int = 2,
        gate_type: str = "topk",
        gate_activation: str = "softmax",
        loss_configs: Optional[List[LossConfig]] = None,
        loss_combiner_name: str = "weighted_sum",
        **kwargs
    ):
        super().__init__()
        self.loss_registry = loss_registry

        # 设置损失配置
        self.loss_configs = loss_configs or [
            LossConfig("auxiliary", LossType.AUXILIARY, weight=0.01),
            LossConfig("z_loss", LossType.Z_LOSS, weight=0.0),
        ]

        # 设置损失组合器
        self.loss_combiner = loss_registry.get_combiner(loss_combiner_name)
        if self.loss_combiner is None:
            logger.warning(f"未找到损失组合器: {loss_combiner_name}, 使用默认组合器")
            self.loss_combiner = loss_registry.get_combiner("weighted_sum")

        # 初始化损失存储
        self.current_losses = {}
        self.total_loss = paddle.to_tensor(0.0)

    def _gumbel_softmax(self, logits: paddle.Tensor, temperature: float = 1.0) -> paddle.Tensor:
        """Gumbel-Softmax采样"""
        gumbel_noise = -paddle.log(-paddle.log(paddle.rand_like(logits) + 1e-8) + 1e-8)
        return paddle.nn.functional.softmax((logits + gumbel_noise) / temperature)

    def forward(
        self,
        gates: paddle.Tensor,
    ) -> Tuple[int, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Implements TopKGating on logits."""
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
                gates, k=self.top_k, n_group=self.n_group, topk_group=self.topk_group
            )
            # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
            top_gate = top_gate / denominator
        top_gate = top_gate * self.routed_scaling_factor

        # get topk mask
        mask = paddle.zeros_like(gates).put_along_axis(top_idx, paddle.to_tensor(1.0, dtype="float32"), axis=1)

        # l_zloss = self._cal_z_loss(gates)
        # if hasattr(self.config, "seq_aux") and self.config.seq_aux:
        #     l_aux = self._cal_seq_aux_loss(gates_ori, self.top_k, top_idx)
        # else:
        #     l_aux = self._cal_aux_loss(gates, mask)

        # 计算所有损失函数
        self.current_losses = {}
        config_dict = {config.name: config for config in self.loss_configs}

        for config in self.loss_configs:
            if not config.enabled:
                continue

            loss_func = self.loss_registry.get_loss(config.name)
            if loss_func is None:
                logger.warning(f"未找到损失函数: {config.name}")
                continue

            try:
                loss_value = loss_func(
                    routing_weights=routing_weights,
                    selected_experts=selected_experts,
                    gate_logits=gate_logits,
                    num_experts=self.num_experts,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    **config.params,
                )
                self.current_losses[config.name] = loss_value
            except Exception as e:
                logger.warning(f"计算损失函数 {config.name} 时出错: {e}")
                self.current_losses[config.name] = paddle.to_tensor(0.0)

        # 组合损失
        self.total_loss = self.loss_combiner(self.current_losses, config_dict)

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
            token_priority = self._priority(top_idx, capacity)

        # normalize gates
        # gates_masked is equal to top_gate.
        gates_masked = gates * mask
        # if self.training:
        gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
        denom_s = paddle.clip(gates_s, min=paddle.finfo(gates_masked.dtype).eps)
        if self.norm_topk_prob:
            gates_masked = gates_masked / denom_s
        gates_masked *= self.routed_scaling_factor

        return (
            capacity,
            gates_masked.take_along_axis(top_idx, axis=-1),
            top_idx,
            token_priority.take_along_axis(top_idx, axis=-1),
            self.total_loss,
            paddle.to_tensor(0.0),
        )

    def add_loss_config(self, config: LossConfig):
        """添加损失配置"""
        self.loss_configs.append(config)
        logger.info(f"添加损失配置: {config.name}, 权重: {config.weight}")

    def remove_loss_config(self, name: str):
        """移除损失配置"""
        self.loss_configs = [config for config in self.loss_configs if config.name != name]
        logger.info(f"移除损失配置: {name}")

    def update_loss_weights(self, weights: Dict[str, float]):
        """更新损失权重"""
        for config in self.loss_configs:
            if config.name in weights:
                config.weight = weights[config.name]
        logger.info(f"更新损失权重: {weights}")

    def set_loss_combiner(self, combiner_name: str):
        """设置损失组合器"""
        combiner = self.loss_registry.get_combiner(combiner_name)
        if combiner is not None:
            self.loss_combiner = combiner
            logger.info(f"设置损失组合器: {combiner_name}")
        else:
            logger.warning(f"未找到损失组合器: {combiner_name}")

    def get_auxiliary_loss(self) -> paddle.Tensor:
        """获取辅助损失（兼容性方法）"""
        return self.current_losses.get("auxiliary", paddle.to_tensor(0.0))

    def get_z_loss(self) -> paddle.Tensor:
        """获取Z损失（兼容性方法）"""
        return self.current_losses.get("z_loss", paddle.to_tensor(0.0))

    def get_all_losses(self) -> Dict[str, paddle.Tensor]:
        """获取所有损失"""
        return self.current_losses.copy()

    def get_total_loss(self) -> paddle.Tensor:
        """获取总损失"""
        return self.total_loss
