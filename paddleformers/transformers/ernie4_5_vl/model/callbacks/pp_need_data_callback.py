# !/usr/bin/env python3

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
pp_need_data_callback

"""
import logging

import paddle
from paddleformers.trainer.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from paddleformers.trainer.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class PPNeedDataCallback(TrainerCallback):
    """
    Adjust the loss_scale coefficient when `pp_need_data_degree` is enabled.
    """

    def on_optimizer_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """_summary_

        Args:
            args (TrainingArguments): _description_
            state (TrainerState): _description_
            control (TrainerControl): _description_

        Returns:
            _type_: _description_
        """

        if not args.pp_need_data_degree:
            return

        def _enable_delay_scale_loss():
            key = "enable_delay_scale_loss"
            if args.pipeline_parallel_degree > 1:
                return key in args.pipeline_parallel_config.split(" ")
            elif args.tensor_parallel_degree > 1:
                return key in args.tensor_parallel_config.split(" ")
            else:
                return False

        # TODO When performing inference, you need to disable `delay_scale_loss`.
        model = kwargs.pop("model")
        if args.pipeline_parallel_degree > 1 and _enable_delay_scale_loss():
            # paddle.device.synchronize()
            # logger.info(f"scale grad")
            for p in model.parameters():
                with paddle.no_grad():
                    if hasattr(p, "main_grad") and p.main_grad is not None:
                        assert p.grad is None
                        p.main_grad.scale_(1.0 / max(1.0, args.pp_need_data_degree))
                    elif p.grad is not None:
                        p.grad.scale_(1.0 / max(1.0, args.pp_need_data_degree))
