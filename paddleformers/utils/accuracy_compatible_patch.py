# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import paddle
import paddle.distributed as dist

_APPLY_DSV4_ACCURACY_COMPATIBLE_PATCH = os.environ.get("APPLY_DSV4_ACCURACY_COMPATIBLE_PATCH", "0") == "1"
_LOAD_FIXED_DATA_CALL_COUNT = 0


def apply_dsv4_accuracy_compatible_patch() -> bool:
    return _APPLY_DSV4_ACCURACY_COMPATIBLE_PATCH


@dataclass
class FixedTrainingData:
    input_ids: list
    labels: list
    position_ids: list
    max_seq_len: int
    batched: bool = False


def load_fixed_training_data(
    training_args,
    mtp_depth: int,
    calc_padding_size: Callable[[int, object], int],
) -> FixedTrainingData | None:
    fixed_tokens_path = os.environ.get("LOAD_FIXED_DATA_PATH")
    if not fixed_tokens_path:
        return None

    global _LOAD_FIXED_DATA_CALL_COUNT

    rank = paddle.distributed.get_rank() if paddle.distributed.is_initialized() else 0
    seq_len = training_args.max_seq_len
    accumulation_steps = max(1, int(getattr(training_args, "gradient_accumulation_steps", 1)))
    call_index = _LOAD_FIXED_DATA_CALL_COUNT
    _LOAD_FIXED_DATA_CALL_COUNT += 1
    fixed_step = call_index // accumulation_steps
    micro_step = call_index % accumulation_steps
    suffix = f"step{fixed_step}_micro{micro_step}_rank{rank}_seq{seq_len}.npy"
    tokens_file = os.path.join(fixed_tokens_path, f"tokens_{suffix}")
    labels_file = os.path.join(fixed_tokens_path, f"labels_{suffix}")
    if not os.path.exists(tokens_file):
        suffix = f"step{fixed_step}_rank{rank}_seq{seq_len}.npy"
        tokens_file = os.path.join(fixed_tokens_path, f"tokens_{suffix}")
        labels_file = os.path.join(fixed_tokens_path, f"labels_{suffix}")

    fixed_input_arr = np.load(tokens_file)
    fixed_label_arr = np.load(labels_file)
    if fixed_input_arr.ndim == 1:
        input_ids = fixed_input_arr.tolist()
        labels = fixed_label_arr.tolist()
        return FixedTrainingData(
            input_ids=input_ids,
            labels=labels,
            position_ids=list(range(len(input_ids))),
            max_seq_len=calc_padding_size(len(input_ids), training_args),
        )

    if fixed_input_arr.ndim != 2:
        raise ValueError("LOAD_FIXED_DATA_PATH expects 1D or 2D token arrays, " f"got shape {fixed_input_arr.shape}")
    if fixed_label_arr.shape != fixed_input_arr.shape:
        raise ValueError(
            f"LOAD_FIXED_DATA_PATH labels shape {fixed_label_arr.shape} "
            f"does not match tokens shape {fixed_input_arr.shape}"
        )

    input_ids = [row.tolist() for row in fixed_input_arr]
    labels = [row.tolist() for row in fixed_label_arr]
    position_ids = [list(range(len(row))) for row in input_ids]
    return FixedTrainingData(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        max_seq_len=calc_padding_size(max(len(row) for row in input_ids), training_args),
        batched=True,
    )


def fixed_data_iter(fixed_data: FixedTrainingData | None, batch: list):
    if fixed_data is not None and fixed_data.batched:
        return [None] * len(fixed_data.input_ids)
    return batch


def fixed_data_sample(fixed_data: FixedTrainingData, index: int):
    if fixed_data.batched:
        return (
            [fixed_data.position_ids[index]],
            [fixed_data.input_ids[index]],
            [fixed_data.labels[index]],
            [fixed_data.position_ids[index]],
        )
    return (
        [fixed_data.position_ids],
        [fixed_data.input_ids],
        [fixed_data.labels],
        [fixed_data.position_ids],
    )


def _get_param_grad(param):
    grad = getattr(param, "main_grad", None)
    if grad is not None:
        return grad
    grad_attr = getattr(param, "grad", None)
    return grad_attr() if callable(grad_attr) else grad_attr


def flush_sequence_first_wgrad(model) -> None:
    attrs = (
        "_dsv4_hc_mapping_seqfirst_wgrad",
        "_dsv4_attn_o_group_seqfirst_wgrad",
    )
    with paddle.no_grad():
        for _, param in model.named_parameters():
            seqfirst_wgrad = None
            matched_attr = None
            for attr in attrs:
                seqfirst_wgrad = getattr(param, attr, None)
                if seqfirst_wgrad is not None:
                    matched_attr = attr
                    break
            if matched_attr is None:
                continue
            grad = _get_param_grad(param)
            if grad is None:
                continue
            grad.set_value(seqfirst_wgrad.cast(grad.dtype))
            setattr(param, matched_attr, None)


def set_loss_acc_steps(acc_steps: int) -> None:
    from paddlefleet.accuracy_compatible_patch import LossScaleBeforeBackward

    LossScaleBeforeBackward.set_acc_steps(acc_steps)


def pop_raw_lm_loss():
    from paddlefleet.accuracy_compatible_patch import LossScaleBeforeBackward

    raw_lm_loss = LossScaleBeforeBackward.pop()

    if raw_lm_loss is not None:
        loss_sum, loss_count = raw_lm_loss
    else:
        loss_sum = paddle.zeros([1], dtype=paddle.float32)
        loss_count = paddle.zeros([1], dtype=paddle.float32)

    if dist.is_initialized():
        dist.all_reduce(loss_sum, dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, dist.ReduceOp.SUM)

    if loss_count.item() <= 0:
        return None

    return float((loss_sum / loss_count).cast("float32").numpy()[0])


def set_pipeline_loss_scale(acc_steps: int) -> None:
    if acc_steps <= 1:
        return
    acc_scale = paddle.to_tensor(1.0 / acc_steps, dtype=paddle.float32)
    from paddlefleet.transformer.dsa_attention import DSAIndexerLossAutoScaler
    from paddlefleet.transformer.multi_token_prediction import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(acc_scale)
    DSAIndexerLossAutoScaler.set_loss_scale(acc_scale)


def has_optimizer_state(struct_name, state_dict_metadata, optimizer_state_names):
    if apply_dsv4_accuracy_compatible_patch():
        return any(struct_name + state_name in state_dict_metadata for state_name in optimizer_state_names)
    if os.getenv("HACK_CONVERT_CKPT", "0").lower() in ["true", "1"]:
        return True
    return any(struct_name + state_name in state_dict_metadata for state_name in optimizer_state_names)
