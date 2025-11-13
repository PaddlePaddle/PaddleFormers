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
import os

from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class StopperCallback(TrainerCallback):
    """
    Actively stop training based on an external signal.
    """

    def on_substep_end(self, args, state, control, **kwargs):
        """
        Called at the end of a sub-step.

        Args:
            args (dict): A dictionary of parameters used to pass additional required arguments.
            state (State): The current state of the environment.
            control (Control): A control object used to manage the training process.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        if os.path.exists("/root/stop"):
            control.should_training_stop = True
