# !/usr/bin/env python3

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

""" ClipGradByAdaptiveNormCallback """

# from ernie.utils import logger
import logging
import os

import paddle
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.trainer.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
)

logger = logging.getLogger(__name__)


class ClipGradByAdaptiveNormCallback(TrainerCallback):
    """
    Load and save adaptive norm state hook, hack version
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """
        load adaptive norm state at the beginning of training.
        """
        optimizer = kwargs.get("optimizer", None)
        assert optimizer is not None
        if optimizer._grad_clip is None:
            logger.info("grad_clip is None.")
            return
        elif not hasattr(optimizer._grad_clip, "state_dict"):
            logger.info("grad_clip {optimizer._grad_clip} has not state_dict method.")
            return

        if args.adaptive_norm_force_clear_state:
            logger.info("force clear ClipGradByAdaptiveNorm state dict.")
            return

        resume_from_checkpoint = (
            None if not args.resume_from_checkpoint else args.resume_from_checkpoint
        )
        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is None:
            return

        # if use distributed training
        if args.world_size > 1:
            process_index = args.process_index
            path = os.path.join(
                resume_from_checkpoint, f"adaptivenorm_clip_state_{process_index}.pth"
            )
            if not os.path.isfile(path):
                logger.info(
                    f"Didn't find an adaptivenorm clip state file for process {process_index}, if you are resuming "
                    "a training that wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            path = os.path.join(resume_from_checkpoint, "adaptivenorm_clip_state.pth")
            if not os.path.isfile(path):
                logger.info(
                    "Didn't find an adaptivenorm clip state file, if you are resuming a training that was "
                    "launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return

        logger.info(f"Loading adaptivenorm clip state state to {path}")
        state_dict = paddle.load(path)

        optimizer._grad_clip.set_state_dict(state_dict)
        logger.info("load ClipGradByAdaptiveNorm state dict success.")

    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint save.
        """
        optimizer = kwargs.get("optimizer", None)
        assert optimizer is not None

        if optimizer._grad_clip is None or not hasattr(
            optimizer._grad_clip, "state_dict"
        ):
            return

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        run_dir = args.output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)

        os.makedirs(output_dir, exist_ok=True)

        if args.world_size > 1:
            # use global process_index to save
            process_index = args.process_index
            path = os.path.join(
                output_dir, f"adaptivenorm_clip_state_{process_index}.pth"
            )
        else:
            path = os.path.join(output_dir, "adaptivenorm_clip_state.pth")
        logger.info(f"Saving randompos rng state to {path}")
        paddle.save(optimizer._grad_clip.state_dict(), path)
