# !/usr/bin/env python3

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

""" RefinedRecomputeCheckCallback"""

from paddleformers.trainer.trainer_callback import TrainerCallback

from ernie.refined_recompute.utils import global_rr_queue_log


class RefinedRecomputeCheckCallback(TrainerCallback):
    """
    RefinedRecomputeCheckCallback
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """on_train_begin"""
        global_rr_queue_log.check()

    def on_step_end(self, args, state, control, **kwargs):
        """on_step_end"""
        global_rr_queue_log.check()

    def on_step_begin(self, args, state, control, **kwargs):
        """on_step_begin"""
        global_rr_queue_log.check()
