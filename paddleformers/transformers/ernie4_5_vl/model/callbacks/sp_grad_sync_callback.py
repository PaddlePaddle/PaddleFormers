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
Sequence parallel gradient synchronization callback.
"""

from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients_with_group,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    is_sequence_parallel_parameter,
)
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.utils.log import logger


class SPGradSyncCallback(TrainerCallback):
    """
    Callback for synchronizing gradients in sequence parallel training scenarios.

    This callback handles gradient synchronization across model parallel groups
    for sequence parallel parameters during distributed training.

    Attributes:
        _sp_params (List[Parameter]): List of sequence parallel parameters that need gradient synchronization.
    """

    def __init__(self, model):
        """
        Initialize the callback by identifying sequence parallel parameters.

        Args:
            model (nn.Module): The model instance containing parameters to monitor.

        Raises:
            AssertionError: If hybrid communication group is not available (MP not enabled).
        """
        assert hasattr(fleet, "_hcg"), "must use MP when calling this Callback"
        logger.info("using sp callback")
        params = []
        for n, p in model.named_parameters():
            if is_sequence_parallel_parameter(p):
                logger.info(f"register bw hook for:{n}")
                params.append(p)
        logger.info(f"#-sp-sync param:{len(params)}")
        self._sp_params = params

    def on_optimizer_begin(self, args, state, control, **kwargs):
        """
        Perform gradient synchronization before optimizer step.

        This method is called just before the optimizer step to ensure gradients
        are properly synchronized across model parallel groups for sequence parallel parameters.

        Args:
            args (TrainingArguments): Training configuration.
            state (TrainerState): Current training state.
            control (TrainerControl): Training flow control object.
            **kwargs (Any): Additional keyword arguments.
        """
        mp_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
        fused_allreduce_gradients_with_group(self._sp_params, group=mp_group, scale=1.0)  # sum not mean
