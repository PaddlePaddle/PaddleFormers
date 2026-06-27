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


from copy import deepcopy

import paddle
import paddle.nn.functional as F

from paddleformers.fleet.transformer.mlp import MLP, MLPSublayersSpec
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


class StandardMLPSharedExpert(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        moe_intermediate_size: int,
        is_expert: bool,
        mlp_spec: MLPSublayersSpec,
    ):
        if moe_intermediate_size == config.intermediate_size:
            super().__init__(
                config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        else:
            # Local SequentialMLP can still be used here by overriding the intermediate_size
            # with a deepcopied config.
            sequential_mlp_config = deepcopy(config)
            sequential_mlp_config.intermediate_size = moe_intermediate_size
            super().__init__(
                sequential_mlp_config,
                mlp_spec,
                is_expert=is_expert,
                intermediate_size=moe_intermediate_size,
                # tp_group=pg_collection.expt_tp,
            )
        self.use_shared_expert_gate = config.moe_shared_expert_gate
        if self.use_shared_expert_gate:
            self.gate_weight = paddle.create_parameter(
                shape=[config.hidden_size, 1],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            # Initialize with Normal distribution aligned with Megatron.
            config.init_method(self.gate_weight)
        else:
            self.gate_weight = None

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        output, output_bias = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = F.linear(hidden_states, self.gate_weight)
            gate_score = F.sigmoid(logits)
            output = output * gate_score
        return output, output_bias
