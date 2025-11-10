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

import os

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from ...transformers.model_outputs import CausalLMOutputWithPast
from ...transformers.sequence_parallel_utils import (
    AllGatherVarlenOp,
    sequence_parallel_sparse_mask_labels,
)
from ...transformers.tensor_parallel_utils import (
    fused_head_and_loss_fn,
    parallel_matmul,
)
from ...utils import infohub
from .loss_utils import subbatch


def kto_preprocess_inputs(self, logits, labels):
    hidden_states, lm_head_weight, lm_head_bias, transpose_y = None, None, None, None

    def unpack_logits(obj):
        if isinstance(obj, tuple):
            if len(obj) == 1:
                return unpack_logits(obj[0])
            elif len(obj) == 4:
                return None, *obj  # unpack logits when using fused head loss
        return obj, None, None, None, None

    logits, hidden_states, lm_head_weight, lm_head_bias, transpose_y = unpack_logits(logits)
    return logits, labels, hidden_states, lm_head_weight, lm_head_bias, transpose_y


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
        paddle.distributed.all_gather(output_tensors, paddle.tile(tensors, repeat_times=[1, 1]), group=self.comm_group)
        tensors = paddle.cat(output_tensors, axis=0)
    return tensors


def kto_logps(
    self,
    logits,
    response_labels,
    response_kl_labels,
    response_indexs,
    hidden_states,
    weight,
    bias,
    transpose_y,
    **kwargs,
):
    """KTO logprobs"""
    labels = response_labels + response_kl_labels

    if self.use_filtered_label_loss:
        if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel and logits is None:
            labels, sparse_tgt_idx = sequence_parallel_sparse_mask_labels(labels, self.ignored_index)

            hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx, axis=0)
            hidden_states = AllGatherVarlenOp.apply(hidden_states)
        else:
            labels = labels.flatten()
            sparse_tgt_idx = paddle.nonzero(labels != self.ignored_index).flatten()
            labels = paddle.take_along_axis(labels, sparse_tgt_idx, axis=0)

            hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
            hidden_states = paddle.take_along_axis(hidden_states, sparse_tgt_idx.unsqueeze(-1), axis=0)
            if logits is not None:
                logits = paddle.gather(logits, sparse_tgt_idx, axis=1)
    else:
        if hidden_states is not None:
            hidden_states = AllGatherOp.apply(hidden_states)

    # bsz,seq_len,hidden_size or seq_len,hidden_size
    seq_len = labels.shape[1] if labels.ndim == 2 else labels.shape[0]
    if self.use_fused_head_and_loss_fn and self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:
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
            self.loss_subbatch_sequence_length,
            return_token_loss=True,
            ignore_index=self.ignored_index,
        )
        per_token_logps = per_token_logps.reshape([1, per_token_logps.shape[-1], 1])

    else:
        if self.use_fused_head_and_loss_fn:
            logits = parallel_matmul(
                hidden_states,
                weight,
                bias,
                transpose_y=transpose_y,
                tensor_parallel_output=self.config.tensor_parallel_output,
            )
        if isinstance(logits, tuple):
            logits = logits[0]
        elif isinstance(logits, CausalLMOutputWithPast):
            logits = logits.logits
        logits = logits.astype("float32")
        if logits.dim() == 2 and labels.dim() == 2:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3 and labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:
            sb_loss_func = subbatch(
                self.loss_func,
                [0, 1],
                [1, 1],
                self.loss_subbatch_sequence_length,
                1,
            )
            per_token_logps = sb_loss_func(logits, labels.unsqueeze(-1))
        else:
            per_token_logps = self.loss_func(logits, labels.unsqueeze(-1))

    if len(response_indexs.shape) == 3:
        response_indexs = response_indexs[0]
    if self.use_filtered_label_loss:
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
        kl = _nested_gather(paddle.tile(kl, repeat_times=[1, 1])).mean().clip(min=0)
    if policy_chosen_logps.shape[0] == 0 or reference_chosen_logps.shape[0] == 0:
        chosen_losses = paddle.zeros([0])
    else:
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        chosen_losses = 1 - F.sigmoid(self.config.kto_config.beta * (chosen_logratios - kl))
    if policy_rejected_logps.shape[0] == 0 or reference_rejected_logps.shape[0] == 0:
        rejected_losses = paddle.zeros([0])
    else:
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        rejected_losses = 1 - F.sigmoid(self.config.kto_config.beta * (kl - rejected_logratios))
    losses = paddle.cat(
        (
            self.config.kto_config.desirable_weight * chosen_losses,
            self.config.kto_config.undesirable_weight * rejected_losses,
        ),
        0,
    )
    return losses.mean(), kl


def kto_loss_forward(
    self: nn.Layer,
    logits,
    labels,
    **kwargs,
):
    # preprocess inputs and label
    logits, labels, hidden_states, lm_head_weight, lm_head_bias, transpose_y = kto_preprocess_inputs(
        self, logits, labels, **kwargs
    )
    (
        response_labels,
        response_kl_labels,
        response_indexs,
        reference_chosen_logps,
        reference_rejected_logps,
        reference_kl_logps,
    ) = labels

    if reference_chosen_logps is None or reference_rejected_logps is None or reference_kl_logps is None:
        (reference_chosen_logps, reference_rejected_logps, reference_kl_logps,) = kto_logps(
            self,
            logits,
            response_labels,
            response_kl_labels,
            response_indexs,
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            **kwargs,
        )
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

    policy_chosen_logps, policy_rejected_logps, policy_kl_logps = kto_logps(
        self,
        logits,
        response_labels,
        response_kl_labels,
        response_indexs,
        hidden_states,
        lm_head_weight,
        lm_head_bias,
        **kwargs,
    )

    loss, kl = kto_loss(
        self,
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
