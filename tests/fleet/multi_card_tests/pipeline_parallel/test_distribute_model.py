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

from dataclasses import dataclass

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LayerSpec,
    PipelineLayer,
    SharedLayerDesc,
)
from paddle.nn import Layer

from paddleformers.fleet.transformer.identity_op import IdentityOp


class SharedLinear(Layer):
    def __init__(self):
        super().__init__()
        self.shared_net = nn.Linear(256, 256)

    @property
    def shared_weight(self):
        return self.shared_net.weight

    def forward(self, hidden_states):
        outputs = self.shared_net(hidden_states)
        return outputs


class ClassifyPipe(Layer):
    def __init__(self):
        super().__init__()
        self.classify_net = nn.Linear(256, 10)

    def forward(self, hidden_states):
        outputs = self.classify_net(hidden_states)
        return outputs


@dataclass
class SimpleNetLayerSpec:
    features: list[LayerSpec] | list[IdentityOp]
    shared: LayerSpec | type = IdentityOp
    classifier: LayerSpec | type = IdentityOp


class SimpleNet(PipelineLayer):
    def __init__(self, sublayers_spec: SimpleNetLayerSpec, **kwargs):
        self.layers = SimpleNet.get_layer_desc_list(sublayers_spec)

        super().__init__(layers=self.layers, **kwargs)

    @staticmethod
    def get_layer_desc_list(spec: SimpleNetLayerSpec):
        def _logits_helper(linear, output):
            return paddle.matmul(output, linear.shared_weight)

        layers = []
        layers.append(
            SharedLayerDesc(
                "shared",
                spec.shared,
                shared_weight_attr="shared_weight",
            )
        )
        for features_spec in spec.features:
            layers.append(LayerDesc(features_spec))
        layers.append(
            SharedLayerDesc(
                "shared",
                spec.shared,
                forward_func=_logits_helper,
                shared_weight_attr="shared_weight",
            )
        )
        layers.append(LayerDesc(spec.classifier))
        return layers


def get_simple_spec(num_classes=10):
    spec = LayerSpec(
        layer=SimpleNet,
        sublayers_spec=SimpleNetLayerSpec(
            shared=LayerSpec(layer=SharedLinear),
            features=[
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.Linear,
                    extra_kwargs={"in_features": 256, "out_features": 256},
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
            ],
            classifier=LayerSpec(
                layer=ClassifyPipe,
            ),
        ),
        extra_kwargs={
            "loss_fn": nn.CrossEntropyLoss(),
        },
    )
    return spec
