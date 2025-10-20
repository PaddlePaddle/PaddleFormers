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
logging callback
"""
import logging
import time

from paddle import _C_ops
from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class ReshardSaveExitCallback(TrainerCallback):
    """
    After resharding the model, save the model directly and exit the program.
    """

    def __init__(self, trainer):
        "init func"
        self.trainer = trainer
        self.step = 0
        for name in ["adam", "adamw"]:
            for suffix in ["", "_"]:
                op = name + suffix
                if not hasattr(_C_ops, op):
                    continue

                logger.info(f"[zengjinle debug] faking {op} ...")

                def fake_func(*args, **kwargs):
                    logger.info(f"[zengjinle debug] fake_func for {op}")
                    return [None] * 6

                setattr(_C_ops, op, fake_func)

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        """
        Only use at the begin of training.
        """
        self.step += 1
        if self.step < 5:
            return
        new_opt_state_dict = {}
        opt_state_dict = self.trainer.optimizer.state_dict()
        if "master_weights" in opt_state_dict:
            new_opt_state_dict["master_weights"] = opt_state_dict["master_weights"]
            self.trainer.optimizer.set_state_dict(opt_state_dict)

        self.trainer.state.global_step = int(args.resume_from_checkpoint.split("-")[-1])
        self.trainer._save_checkpoint(self.trainer.model, metrics=None)

        logger.info(
            "In ReshardSaveExitCallback, finishing saving reshared model, will exit after 20s..."
        )
        time.sleep(10)
        exit(0)
