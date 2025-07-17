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

"""DPO Criterion"""

import paddle
import paddle.nn.functional as F
from paddleformers.transformers import fused_head_and_loss_fn
from paddleformers.trl import DPOCriterion
from paddleformers.utils import infohub

from ..distributed.common_dist_utils import (
    AllGatherVarlenOp,
    GatherOp,
    sequence_parallel_sparse_mask_labels,
)
from ..modeling import parallel_matmul

LOOP_CHUNK_SIZE = 1024


class ErnieDPOCriterion(DPOCriterion):
    """DPO Criterion"""

    def dpo_logps(
        self,
        logits,
        chosen_labels,
        rejected_labels,
        response_indexs,
        average_log_prob=False,
    ):
        """DPO logprobs"""
        labels = chosen_labels + rejected_labels
        if self.config.use_fused_head_and_loss_fn:
            hidden_states, weight, bias, transpose_y = logits
        elif self.config.use_sparse_head_and_loss_fn:
            hidden_states, weight, bias = logits
        if self.config.use_sparse_head_and_loss_fn:
            if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel:
                labels, sparse_tgt_idx = sequence_parallel_sparse_mask_labels(labels, 0)

                hidden_states = paddle.gather(hidden_states, sparse_tgt_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
            else:
                labels = labels.flatten()
                sparse_tgt_idx = paddle.nonzero(labels != 0).flatten()
                labels = paddle.gather(labels, sparse_tgt_idx, axis=0)

                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.gather(hidden_states, sparse_tgt_idx, axis=0)
        elif self.config.use_fused_head_and_loss_fn:
            if self.config.tensor_parallel_degree > 1 and self.config.sequence_parallel:
                hidden_states = GatherOp.apply(hidden_states)
                hidden_states = hidden_states.reshape(
                    [
                        -1,
                        self.config.max_sequence_length,
                        hidden_states.shape[-1],
                    ]
                )

        if self.config.use_fused_head_and_loss_fn:
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
                self.config.fuse_linear,
                LOOP_CHUNK_SIZE,
                return_token_loss=True,
                ignore_index=0,
            )
        elif self.config.use_sparse_head_and_loss_fn:
            logits = parallel_matmul(
                hidden_states,
                weight,
                bias=bias,
                transpose_y=self.config.tie_word_embeddings,
                tensor_parallel_output=self.config.tensor_parallel_output,
                fuse_linear=self.config.fuse_linear,
            )
            logits = logits.astype("float32")
            per_token_logps = -self.logprobs(logits, labels)
        else:
            logits = logits.astype("float32")
            if logits.shape[:-1] != labels.shape:
                raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
            # bs, seq
            per_token_logps = -self.logprobs(logits, labels.unsqueeze(2)).squeeze(2)

        if len(response_indexs.shape) == 3:
            response_indexs = response_indexs[0]

        if self.config.use_sparse_head_and_loss_fn:
            chosen_logps = paddle.stack(
                [
                    (
                        paddle.gather(
                            per_token_logps.reshape([-1]),
                            paddle.arange(response_index[1], response_index[2], dtype=paddle.int32),
                            axis=0,
                        ).sum()
                        if response_index[3] != 0
                        else paddle.to_tensor(100.0)
                    )
                    for response_index in response_indexs
                ],
                axis=0,
            )
            rejected_logps = paddle.stack(
                [
                    (
                        paddle.gather(
                            per_token_logps.reshape([-1]),
                            paddle.arange(response_index[2], response_index[3], dtype=paddle.int32),
                            axis=0,
                        ).sum()
                        if response_index[3] != 0
                        else paddle.to_tensor(100.0)
                    )
                    for response_index in response_indexs
                ],
                axis=0,
            )
        else:
            chosen_logps = paddle.stack(
                [
                    (
                        paddle.gather(
                            paddle.gather(per_token_logps, response_index[0], axis=0),
                            paddle.arange(response_index[1], response_index[2], dtype=paddle.int32),
                            axis=0,
                        ).sum()
                        if response_index[3] != 0
                        else paddle.to_tensor(100.0)
                    )
                    for response_index in response_indexs
                ],
                axis=0,
            )
            rejected_logps = paddle.stack(
                [
                    (
                        paddle.gather(
                            paddle.gather(per_token_logps, response_index[0], axis=0),
                            paddle.arange(response_index[2], response_index[3], dtype=paddle.int32),
                            axis=0,
                        ).sum()
                        if response_index[3] != 0
                        else paddle.to_tensor(100.0)
                    )
                    for response_index in response_indexs
                ],
                axis=0,
            )
        sft_loss = -chosen_logps.sum() / (chosen_labels != 0).sum()
        if average_log_prob:
            chosen_response_length = response_indexs[:, 2] - response_indexs[:, 1]
            rejected_response_length = response_indexs[:, 3] - response_indexs[:, 2]
            chosen_logps /= chosen_response_length.astype("float32")
            rejected_logps /= rejected_response_length.astype("float32")
        elif self.dpo_config.normalize_logps:
            avg_response_length = (response_indexs[:, 3] - response_indexs[:, 1]) / 2
            chosen_response_length = response_indexs[:, 2] - response_indexs[:, 1]
            rejected_response_length = response_indexs[:, 3] - response_indexs[:, 2]
            chosen_logps *= avg_response_length / chosen_response_length.astype("float32")
            rejected_logps *= avg_response_length / rejected_response_length.astype("float32")
        return (
            chosen_logps,
            rejected_logps,
            sft_loss * self.dpo_config.sft_loss_ratio,
        )

    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        score_deltas,
    ):
        """DPO Loss"""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.dpo_config.loss_type == "sigmoid":
            if self.dpo_config.offset_alpha > 0:
                logits = logits - self.dpo_config.offset_alpha / self.dpo_config.beta * paddle.log(score_deltas + 1e-6)
            loss = (
                -F.log_sigmoid(self.dpo_config.beta * logits) * (1 - self.dpo_config.label_smoothing)
                - F.log_sigmoid(-self.dpo_config.beta * logits) * self.dpo_config.label_smoothing
            )
        elif self.dpo_config.loss_type == "hinge":
            loss = F.relu(1 - self.dpo_config.beta * logits)
        elif self.dpo_config.loss_type == "simpo":
            gamma_logratios = self.dpo_config.simpo_gamma / self.dpo_config.beta
            logits -= gamma_logratios
            loss = (
                -F.log_sigmoid(self.dpo_config.beta * logits) * (1 - self.dpo_config.label_smoothing)
                - F.log_sigmoid(-self.dpo_config.beta * logits) * self.dpo_config.label_smoothing
            )
        elif self.dpo_config.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            loss = (logits - 1 / (2 * self.dpo_config.beta)) ** 2
        elif self.dpo_config.loss_type == "dpop":
            loss = -F.log_sigmoid(self.dpo_config.beta * logits)
            positive_reg = reference_chosen_logps - policy_chosen_logps
            loss += self.dpo_config.dpop_lambda * paddle.clip(positive_reg, min=0)
        elif self.dpo_config.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clip(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clip(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is
            # estimated using the rejected (chosen) half.
            loss = paddle.concat(
                (
                    1 - F.sigmoid(self.dpo_config.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.dpo_config.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.dpo_config.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of
            # the trainer class. The version described here is the hard probability version, where P
            # in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            loss = (a - 0.5 / self.dpo_config.beta) ** 2 + (b + 0.5 / self.dpo_config.beta) ** 2
        elif self.dpo_config.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.dpo_config.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.dpo_config.beta
            loss = (
                -F.log_sigmoid(chosen_rewards)
                - 0.5 * F.log_sigmoid(-chosen_rewards)
                - 0.5 * F.log_sigmoid(-rejected_rewards)
            )
        elif self.dpo_config.loss_type == "or":
            # Derived from Eqs. (4) and (7) from https://arxiv.org/abs/2403.07691 by using
            # log identities and exp(log(P(y|x)) = P(y|x)
            log_odds = (policy_chosen_logps - policy_rejected_logps) - (
                paddle.log1p(-paddle.exp(policy_chosen_logps)) - paddle.log1p(-paddle.exp(policy_rejected_logps))
            )
            loss = -F.log_sigmoid(log_odds)
        else:
            raise ValueError(
                f"Unknown loss type: {self.dpo_config.loss_type}. "
                "Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair',"
                "'sppo_hard', 'nca_pair', 'dpop', 'or', 'simpo']"
            )
        return loss.mean() * self.dpo_config.pref_loss_ratio

    def forward(
        self,
        logits,
        labels,
    ):
        """Forward"""
        if self.dpo_config.offset_alpha > 0:
            (
                chosen_labels,
                rejected_labels,
                response_indexs,
                score_deltas,
                reference_chosen_logps,
                reference_rejected_logps,
            ) = labels
        else:
            (
                chosen_labels,
                rejected_labels,
                response_indexs,
                reference_chosen_logps,
                reference_rejected_logps,
            ) = labels
            score_deltas = None

        if self.dpo_config.loss_type in ["ipo", "or", "simpo"]:
            average_log_prob = True
        else:
            average_log_prob = False
        if reference_chosen_logps is None or reference_rejected_logps is None:
            reference_chosen_logps, reference_rejected_logps, sft_loss = self.dpo_logps(
                logits, chosen_labels, rejected_labels, response_indexs, average_log_prob
            )
            if self.use_infohub:
                infohub.reference_chosen_logps.append(reference_chosen_logps)
                infohub.reference_rejected_logps.append(reference_rejected_logps)
                # pipeline mode requires return loss when self._compute_loss is True
                return paddle.zeros([1])
            else:
                return reference_chosen_logps, reference_rejected_logps
        policy_chosen_logps, policy_rejected_logps, sft_loss = self.dpo_logps(
            logits, chosen_labels, rejected_labels, response_indexs, average_log_prob
        )
        dpo_loss = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, score_deltas
        )
        loss = dpo_loss + sft_loss
        if self.use_infohub:
            infohub.policy_chosen_logps.append(policy_chosen_logps.detach())
            infohub.policy_rejected_logps.append(policy_rejected_logps.detach())
            infohub.sft_loss.append(sft_loss.detach())
            infohub.dpo_loss.append(dpo_loss.detach())
            return loss
        else:
            return policy_chosen_logps, policy_rejected_logps, sft_loss, dpo_loss, loss
