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

import logging

logger = logging.getLogger(__name__)


def reset_per_device_batch_size(global_batch_size, per_device_train_batch_size, dataset_world_size):
    assert (
        global_batch_size % dataset_world_size == 0
    ), f"global_bsz={global_batch_size} not evenly divided by world_size={dataset_world_size}"
    batch_per_device = global_batch_size // dataset_world_size
    if batch_per_device < per_device_train_batch_size:
        gradient_accumulation_steps = 1
        per_device_train_batch_size = batch_per_device
        logger.info(
            f"reset `per_device_train_batch_size` to {per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={ dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    else:
        assert (
            batch_per_device % per_device_train_batch_size == 0
        ), f"global_bsz={global_batch_size} not evenly divided by world_size={dataset_world_size}, batch_per_device={batch_per_device}"
        gradient_accumulation_steps = batch_per_device // per_device_train_batch_size
        logger.info(
            f"per_device_train_batch_size={per_device_train_batch_size}, global_batch_size={global_batch_size }, "
            f"dp_worldsize={dataset_world_size}, accumulate_steps={gradient_accumulation_steps} "
        )
    return per_device_train_batch_size, gradient_accumulation_steps
