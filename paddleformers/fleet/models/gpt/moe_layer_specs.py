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
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddleformers.fleet.models.backends import BackendSpecProvider
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.transformer.mlp import MLPSublayersSpec
from paddleformers.fleet.transformer.moe.moe_layer import MoELayer, MoESublayers


def get_moe_layer_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: int | None = None,
    moe_expert_fusion: bool | None = False,
) -> LayerSpec:
    """Helper function to get layer spec for MoE"""
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    hidden_act = backend.hidden_act()

    mlp_spec = MLPSublayersSpec(
        up_gate_proj=linear_fc1,
        down_proj=linear_fc2,
        hidden_act=hidden_act,
    )

    moe_layer_spec = LayerSpec(
        layer=MoELayer,
        extra_kwargs={"sublayers": MoESublayers(mlp_spec=mlp_spec)},
    )
    return moe_layer_spec
