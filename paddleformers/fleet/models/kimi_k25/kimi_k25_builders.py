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
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from ..common.empty_layer import EmptyLayer
from .layer_specs import (
    get_kimi_k25_vision_encoder_layers_spec,
    get_kimi_k25_vision_spec,
)


def kimi_k25_vision_builder(config, **kwargs):
    transformer_layer_specs = get_kimi_k25_vision_encoder_layers_spec(
        config=config
    )

    head_empty_layers_spec = []
    for _ in range(config.num_empty_layers_add_in_head):
        head_empty_layers_spec.append(
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        )

    tail_empty_layers_spec = []
    for _ in range(config.num_empty_layers_add_in_tail):
        tail_empty_layers_spec.append(
            LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        )

    res_spec = get_kimi_k25_vision_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layer_specs,
        tail_empty_layer_spec=tail_empty_layers_spec,
    )

    return build_spec_layer(res_spec, **kwargs)
