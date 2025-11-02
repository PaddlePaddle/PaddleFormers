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

from abc import ABC, abstractmethod
from typing import Dict

import paddle
from paddle import nn
from paddle.incubate.nn.functional import swiglu as fused_swiglu

from ...nn.mlp import MLP
from ...transformers import Linear, linear_utils
from ...transformers.activations import ACT2FN
from ...transformers.configuration_utils import PretrainedConfig
from ...transformers.refined_recompute import (
    RRColumnParallelLinear,
    RRColumnSequenceParallelLinear,
    RRRowParallelLinear,
    RRRowSequenceParallelLinear,
)


class MoEExpertInterface(ABC):

    @abstractmethod
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states: Input hidden states

        Returns:
            output: Output hidden states
        """
        pass

class StandardMLPExpert(MLP):
    def __init__(
        self,
        config: PretrainedConfig,
        moe_intermediate_size: int,
    ):
        super().__init__(config=routed_expert_pretrained_config, intermediate_size=moe_intermediate_size)

