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

"""
This module provides functions related to training configuration management.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reset_per_device_batch_size(
    global_batch_size, per_device_train_batch_size, dataset_world_size
):
    """
    Adjust `gradient_accumulation_steps` and `per_device_train_batch_size`
    based on `global_batch_size` and `per_device_train_batch_size`.

    The `per_device_train_batch_size` specifies the **maximum**
    batch size for a single forward pass on each device.
    The returned `per_device_train_batch_size` may be smaller than the input value.
    The resulting combination is guaranteed to satisfy the given `global_batch_size`.
    """
    assert (
        global_batch_size % dataset_world_size == 0
    ), f"global_bsz={global_batch_size} not evenly devided by world_size={dataset_world_size}"
    batch_per_device = global_batch_size // dataset_world_size
    if batch_per_device < per_device_train_batch_size:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = batch_per_device
        logger.info(
            f"reset `per_device_train_batch_size` to {per_device_train_batch_size},"
            f"global_batch_size={global_batch_size}, "
            f"dp_worldsize={ dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    else:
        assert (
            batch_per_device % per_device_train_batch_size == 0
        ), f"""global_bsz={global_batch_size} not evenly devided by world_size={dataset_world_size},
        batch_per_device={batch_per_device}"""

        gradient_accumulation_steps = batch_per_device // per_device_train_batch_size
        logger.info(
            f"per_device_train_batch_size={per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    return per_device_train_batch_size, gradient_accumulation_steps


def progressive_accumulate_steps(
    acc_step_begin, acc_step_end, warmup_global_steps, increment, step
):
    """
    Calculate the number of accumulation steps during
    Progressive Batch Size Warmup at global step `step`.

    Args:
        acc_step_begin: Initial accumulation step count.
        acc_step_end: Final accumulation step count.
        warmup_global_steps: Number of global steps for progressive batch-size warmup.
        step: Current global step.

    Returns:
        The accumulation step count at step `step`.
    """
    assert step >= 0, step
    if step >= warmup_global_steps:
        return acc_step_end
    slope = (acc_step_end - acc_step_begin) / warmup_global_steps
    acc_steps = int(slope * step + acc_step_begin)
    acc_steps = int(np.ceil(acc_steps / increment) * increment)
    return acc_steps


def progressive_consumed_examples_per_device(
    acc_step_begin, acc_step_end, warmup_global_steps, micro_batch_size, increment, step
):
    """
    Calculate the number of examples consumed from a per-device perspective during Progressive Batch Size Warmup at global step `step`.

    Args:
        acc_step_begin: Initial accumulation step count.
        acc_step_end: Final accumulation step count.
        warmup_global_steps: Number of global steps for progressive batch-size warmup.
        micro_batch_size: Micro batch size per device.
        dp_world_size: Data parallel world size.
        step: Current global step.

    Returns:
        The number of examples consumed at step `step`.
    """

    if step == 0:
        return 0
    accumulate_steps = 0
    for gstep in range(step):
        accumulate_steps += progressive_accumulate_steps(
            acc_step_begin, acc_step_end, warmup_global_steps, increment, gstep
        )

    return accumulate_steps * micro_batch_size
