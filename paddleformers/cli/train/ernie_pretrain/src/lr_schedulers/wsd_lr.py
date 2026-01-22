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

import math

from paddle.optimizer.lr import LambdaDecay


def get_wsd_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    decay_function: str = "half_life",
    last_epoch: int = -1,
    min_lr: float = 0.0,
    num_steady_steps=None,
):
    if num_steady_steps is None:
        num_steady_steps = 0.9 * num_training_steps

    def wsd_scheduler(current_step, base=0.05):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_steady_steps:
            return 1.0
        num_decay_steps = num_training_steps - num_steady_steps
        progress = float(current_step - num_steady_steps) / float(max(1, num_decay_steps))

        if decay_function == "half_life":
            ratio = base**progress
            normalize_ratio = (ratio - base) * (1 / (1 - base))
            return normalize_ratio * (1 - min_lr / learning_rate) + min_lr / learning_rate
        elif decay_function == "1-sqrt":
            ratio = 1 - math.sqrt(progress)
            return ratio * (1 - min_lr / learning_rate) + min_lr / learning_rate
        else:
            raise ValueError(f"Invalid decay function: {decay_function}")

    return LambdaDecay(learning_rate, wsd_scheduler, last_epoch)
