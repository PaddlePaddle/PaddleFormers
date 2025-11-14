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
from typing import Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

from ...transformers.sequence_parallel_utils import (
    AllGatherVarlenOp,
    sequence_parallel_sparse_mask_labels,
)
from ...transformers.tensor_parallel_utils import fused_head_and_loss_fn
from .loss_utils import calc_lm_head_logits, subbatch


def sft_preprocess_inputs(self, logits, labels):
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


def sft_postprocess_loss(self, masked_lm_loss, labels, loss_mask, **kwargs):
    if self.use_filtered_label_loss or loss_mask is None:
        loss_mask = labels != self.ignored_index
    loss_mask = loss_mask.reshape([-1]).cast(paddle.float32)
    # 逐位对齐, 全精度聚合
    masked_lm_loss = paddle.sum(masked_lm_loss.cast(paddle.float32).reshape([-1]) * loss_mask)
    loss = masked_lm_loss / loss_mask.sum()
    loss_sum = masked_lm_loss.sum().detach()

    if not self.return_tuple:  # only used in pp
        if self.training:
            return loss
        return loss_sum
    return loss, loss_sum


def loss_impl(self, logits, labels):
    logits = logits.cast("float32")
    loss = self.loss_func(logits, labels)
    return loss


def sft_calculate_loss(self, logits, hidden_states, lm_head_weight, lm_head_bias, labels, loss_mask, transpose_y):
    seq_len = labels.shape[1] if labels.ndim == 2 else labels.shape[0]
    if self.use_fused_head_and_loss_fn and self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:
        masked_lm_loss = fused_head_and_loss_fn(
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            None,
            transpose_y,
            self.config.vocab_size,
            self.config.tensor_parallel_degree,
            self.config.tensor_parallel_output,
            False,
            self.loss_subbatch_sequence_length,
            return_token_loss=True,
            ignore_index=self.ignored_index,
        )
    else:
        if self.use_fused_head_and_loss_fn:
            # go back to non-subbatch fused head loss
            logits = calc_lm_head_logits(
                self.config,
                hidden_states,
                lm_head_weight,
                lm_head_bias,
                training=self.training,
            )
        if self.enable_parallel_cross_entropy:
            assert logits.shape[-1] != self.config.vocab_size, (
                f"enable_parallel_cross_entropy, the vocab_size should be splited:"
                f" {logits.shape[-1]}, {self.config.vocab_size}"
            )
        else:
            assert logits.shape[-1] == self.config.vocab_size, (
                f"disable_parallel_cross_entropy, the vocab_size should not be splited:"
                f" {logits.shape[-1]}, {self.config.vocab_size}"
            )

        if logits.dim() == 2 and labels.dim() == 2:
            logits = logits.unsqueeze(0)
        elif logits.dim() == 3 and labels.dim() == 1:
            labels = labels.unsqueeze(0)

        # logits: bsz seq_len
        # labels: bsz seq_len vocab_size
        if self.use_subbatch and seq_len > self.loss_subbatch_sequence_length:
            sb_loss_func = subbatch(
                loss_impl,
                arg_idx=[1, 2],
                axis=[1, 1],
                bs=self.loss_subbatch_sequence_length,
                out_idx=1,
            )
            masked_lm_loss = sb_loss_func(self, logits, labels.unsqueeze(-1))
        else:
            masked_lm_loss = loss_impl(self, logits, labels.unsqueeze(-1))

    masked_lm_loss = sft_postprocess_loss(self, masked_lm_loss, labels, loss_mask)
    return masked_lm_loss


def sft_loss_forward(
    self: nn.Layer,
    logits: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    labels: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    loss_mask: paddle.Tensor = None,
    **kwargs
):
    logits, labels, hidden_states, lm_head_weight, lm_head_bias, transpose_y = sft_preprocess_inputs(
        self, logits, labels
    )
    if self.use_filtered_label_loss:
        if self.tensor_parallel and self.sequence_parallel and logits is None:
            masked_lm_labels, sparse_label_idx = sequence_parallel_sparse_mask_labels(labels, self.ignored_index)
            sparse_label_idx = sparse_label_idx.reshape([-1, 1])
            if hidden_states is not None:
                hidden_states = paddle.gather(hidden_states, sparse_label_idx, axis=0)
                hidden_states = AllGatherVarlenOp.apply(hidden_states)
        else:
            masked_lm_labels = labels.flatten()
            sparse_label_idx = paddle.nonzero(masked_lm_labels != self.ignored_index).flatten()
            masked_lm_labels = paddle.take_along_axis(masked_lm_labels, sparse_label_idx, axis=0)
            if hidden_states is not None:
                hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
                hidden_states = paddle.take_along_axis(hidden_states, sparse_label_idx.reshape([-1, 1]), axis=0)
            if logits is not None:
                logits = paddle.gather(logits, sparse_label_idx, axis=1)
        labels = masked_lm_labels
    else:
        if self.sequence_parallel:
            if hidden_states is not None:
                hidden_states = AllGatherOp.apply(hidden_states)

    masked_lm_labels = labels
    # bsz,seq_len,hidden_size or seq_len,hidden_size
    if self.config.recompute:
        loss = recompute(
            sft_calculate_loss,
            self,
            logits,
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            loss_mask,
            transpose_y,
        )
    else:
        loss = sft_calculate_loss(
            self,
            logits,
            hidden_states,
            lm_head_weight,
            lm_head_bias,
            labels,
            loss_mask,
            transpose_y,
        )
    return loss


def mtp_sft_loss_forward(
    self: nn.Layer,
    logits: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    labels: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    loss_mask: paddle.Tensor = None,
    router_loss: paddle.Tensor = None,
    mtp_logits: paddle.Tensor = None,
    **kwargs
):
    num_nextn_predict_layers = self.config.get("num_nextn_predict_layers", 0)
    multi_token_pred_lambda = self.config.get("multi_token_pred_lambda", 0.3)
    if num_nextn_predict_layers > 0:
        labels_ori = labels
        labels = labels[:, :-num_nextn_predict_layers]
        if loss_mask is not None:
            loss_mask = loss_mask[:, :-num_nextn_predict_layers]
        seq_length = labels.shape[1]

    sft_loss = sft_loss_forward(self, logits, labels, loss_mask, **kwargs)

    if num_nextn_predict_layers > 0:
        mtp_loss_res = []
        for depth in range(num_nextn_predict_layers):
            logtis_cur_depth = mtp_logits[depth]
            labels_cur_depth = labels_ori[:, (depth + 1) : (depth + 1 + seq_length)]
            res_cur_depth = sft_loss_forward(logtis_cur_depth, labels_cur_depth, loss_mask)
            mtp_loss_res.append(res_cur_depth)

    def add_loss(main_loss, loss):
        return main_loss + loss - loss.detach()

    if self.return_tuple:
        loss, loss_sum = sft_loss
    else:
        loss, loss_sum = sft_loss, None

    if num_nextn_predict_layers > 0:
        loss = add_loss(
            loss,
            multi_token_pred_lambda * sum([x[0] for x in mtp_loss_res]) / len(mtp_loss_res),
        )

    if loss_sum is not None:
        loss_sum = loss_sum + multi_token_pred_lambda * sum([x[1].detach() for x in mtp_loss_res]) / len(mtp_loss_res)

    if router_loss is not None and isinstance(router_loss, paddle.Tensor):
        loss = loss + router_loss - router_loss.detach()

    if self.return_tuple:
        return loss, loss_sum
    else:
        return loss
