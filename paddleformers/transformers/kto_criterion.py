# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import os

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import ParallelCrossEntropy

from ..utils import infohub
from .model_outputs import CausalLMOutputWithPast
from .sequence_parallel_utils import (
    AllGatherVarlenOp,
    sequence_parallel_sparse_mask_labels,
)
from .tensor_parallel_utils import (
    fused_head_and_loss_fn,
    parallel_linear,
    parallel_matmul,
)


class KTOCriterion(nn.Layer):
    """KTO Criterion"""

    def __init__(self, config, kto_config=None, ignore_label=0, use_infohub=False):
        super(KTOCriterion, self).__init__()
        self.config = config
        if kto_config is None:
            if getattr(self.config, "kto_config", None) is None:
                raise ValueError("KTO Criterion requires model_config.kto_config.")
            self.kto_config = copy.deepcopy(config.kto_config)
        else:
            self.kto_config = kto_config
        if self.config.tensor_parallel_output and self.config.tensor_parallel_degree > 1:
            self.logprobs = ParallelCrossEntropy()
        else:
            self.logprobs = nn.CrossEntropyLoss(reduction="none")
        self.use_infohub = use_infohub
        self.ignore_label = ignore_label
        # allgather kl in criterion
        self.comm_group = None
        if dist.get_world_size() > 1:
            topo = fleet.get_hybrid_communicate_group()._topo
            parallel_groups = topo.get_comm_list("pipe")
            ranks = []
            for group in parallel_groups:
                ranks.append(group[-1])
            self.comm_group = paddle.distributed.new_group(ranks=ranks)

    def _nested_gather(self, tensors):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        local_rank = -1
        env_local_rank = int(os.environ.get("PADDLE_RANK_IN_NODE", -1))
        if env_local_rank != -1 and env_local_rank != local_rank and paddle.distributed.get_world_size() > 1:
            local_rank = env_local_rank
        if tensors is None:
            return
        if local_rank != -1:
            output_tensors = []
            paddle.distributed.all_gather(
                output_tensors, paddle.tile(tensors, repeat_times=[1, 1]), group=self.comm_group
            )
            tensors = paddle.concat(output_tensors, axis=0)
        return tensors

    def kto_logps(self, logits, response_labels, response_kl_labels, response_indexs):
        """KTO logprobs"""
        use_fused_head_and_loss_fn = getattr(self.config, "use_fused_head_and_loss_fn", False)
        use_sparse_head_and_loss_fn = getattr(self.config, "use_sparse_head_and_loss_fn", False)
        labels = response_labels + response_kl_labels
        if use_fused_head_and_loss_fn:
            hidden_states, weight, bias, transpose_y = logits
        elif use_sparse_head_and_loss_fn:
            hidden_states, weight, bias = logits
        if use_sparse_head_and_loss_fn:
            if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel:
                labels, sparse_tgt_idx = sequence_parallel_sparse_mask_labels(labels, self.ignore_label)

                hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                labels = labels.flatten()
                sparse_tgt_idx = paddle.nonzero(labels != self.ignore_label).flatten()
                labels = paddle.take_along_axis(labels, sparse_tgt_idx, axis=0)

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx.unsqueeze(-1), axis=0)
        if use_fused_head_and_loss_fn:
            per_token_logps = -fused_head_and_loss_fn(
                hidden_states,
                weight,
                bias,
                labels,
                None,
                transpose_y,
                self.config.vocab_size,
                self.config.tensor_parallel_degree,
                self.config.tensor_parallel_output,
                self.config.fused_linear,
                getattr(self.config, "chunk_size", 1024),
                return_token_loss=True,
                ignore_index=self.ignore_label,
            )
        elif use_sparse_head_and_loss_fn:
            if bias is None:
                logits = parallel_matmul(hidden_states, weight, self.config.tensor_parallel_output)
            else:
                logits = parallel_linear(
                    hidden_states,
                    weight,
                    bias,
                    self.config.tensor_parallel_output,
                )
            logits = logits.astype("float32")
            per_token_logps = -self.logprobs(logits, labels)
        else:
            if isinstance(logits, tuple):
                logits = logits[0]
            elif isinstance(logits, CausalLMOutputWithPast):
                logits = logits.logits
            logits = logits.astype("float32")
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
            # bs, seq
            per_token_logps = -self.logprobs(logits, labels.unsqueeze(2)).squeeze(2)

        if len(response_indexs.shape) == 3:
            response_indexs = response_indexs[0]
        if use_sparse_head_and_loss_fn:
            chosen_logps_list = [
                (per_token_logps[response_index[1] : response_index[2]]).sum()
                for response_index in response_indexs
                if response_index[4] == 1
            ]
            rejected_logps_list = [
                (per_token_logps[response_index[1] : response_index[2]]).sum()
                for response_index in response_indexs
                if response_index[4] == 0
            ]
            kl_logps_list = [
                (per_token_logps[response_index[2] : response_index[3]]).sum() for response_index in response_indexs
            ]
        else:
            chosen_logps_list = [
                (per_token_logps[response_index[0]][response_index[1] : response_index[2]]).sum()
                for response_index in response_indexs
                if response_index[4] == 1
            ]
            rejected_logps_list = [
                (per_token_logps[response_index[0]][response_index[1] : response_index[2]]).sum()
                for response_index in response_indexs
                if response_index[4] == 0
            ]
            kl_logps_list = [
                (per_token_logps[response_index[0]][response_index[2] : response_index[3]]).sum()
                for response_index in response_indexs
            ]
        if len(chosen_logps_list) == 0:
            chosen_logps = paddle.zeros([0], dtype="float32")
        else:
            chosen_logps = paddle.stack(chosen_logps_list, axis=0)
        if len(rejected_logps_list) == 0:
            rejected_logps = paddle.zeros([0], dtype="float32")
        else:
            rejected_logps = paddle.stack(rejected_logps_list, axis=0)
        kl_logps = paddle.stack(kl_logps_list, axis=0)
        return chosen_logps, rejected_logps, kl_logps

    def kto_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        policy_kl_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        reference_kl_logps,
    ):
        """KTO Loss"""
        kl = (policy_kl_logps - reference_kl_logps).mean().detach()
        if dist.get_world_size() > 1:
            kl = self._nested_gather(paddle.tile(kl, repeat_times=[1, 1])).mean().clip(min=0)
        if policy_chosen_logps.shape[0] == 0 or reference_chosen_logps.shape[0] == 0:
            chosen_losses = paddle.zeros([0])
        else:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.kto_config.beta * (chosen_logratios - kl))
        if policy_rejected_logps.shape[0] == 0 or reference_rejected_logps.shape[0] == 0:
            rejected_losses = paddle.zeros([0])
        else:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.kto_config.beta * (kl - rejected_logratios))
        losses = paddle.concat(
            (
                self.kto_config.desirable_weight * chosen_losses,
                self.kto_config.undesirable_weight * rejected_losses,
            ),
            0,
        )
        return losses.mean(), kl

    def forward(
        self,
        logits,
        labels,
    ):
        """Forward"""
        (
            response_labels,
            response_kl_labels,
            response_indexs,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        ) = labels
        if reference_chosen_logps is None or reference_rejected_logps is None or reference_kl_logps is None:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                reference_kl_logps,
            ) = self.kto_logps(logits, response_labels, response_kl_labels, response_indexs)
            if self.use_infohub:
                infohub.reference_chosen_logps.append(reference_chosen_logps)
                infohub.reference_rejected_logps.append(reference_rejected_logps)
                infohub.reference_kl_logps.append(reference_kl_logps)
                # pipeline mode requires return loss when self._compute_loss is True
                return paddle.zeros([1])
            else:
                return (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_kl_logps,
                )
        policy_chosen_logps, policy_rejected_logps, policy_kl_logps = self.kto_logps(
            logits, response_labels, response_kl_labels, response_indexs
        )
        loss, kl = self.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
        )
        if self.use_infohub:
            infohub.policy_chosen_logps.append(policy_chosen_logps.detach())
            infohub.policy_rejected_logps.append(policy_rejected_logps.detach())
            infohub.policy_kl_logps.append(policy_kl_logps.detach())
            infohub.kl.append(kl.detach())
            return loss
        else:
            return (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_kl_logps,
                loss,
                kl,
            )
