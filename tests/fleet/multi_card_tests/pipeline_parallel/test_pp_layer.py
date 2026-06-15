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

import unittest
from dataclasses import dataclass

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    LayerSpec,
    NoPipelineParallel,
    PipelineLayer,
    build_spec_layer,
)
from paddle.nn import Layer

from paddleformers.fleet.transformer.identity_op import IdentityOp


class ReshapeHelp(Layer):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(shape=self.shape)


@dataclass
class AlexNetLayerSpec:
    features: list[LayerSpec] | list[IdentityOp]
    reshape_layer: LayerSpec | type = IdentityOp
    classifier: LayerSpec | type = IdentityOp


class AlexNet(PipelineLayer):
    def __init__(self, sublayers_spec: AlexNetLayerSpec, **kwargs):
        self.layers = AlexNet.get_layer_desc_list(sublayers_spec)

        super().__init__(layers=self.layers, **kwargs)

    @staticmethod
    def get_layer_desc_list(spec: AlexNetLayerSpec):
        layers = []
        for features_spec in spec.features:
            layers.append(LayerDesc(features_spec))
        layers.append(LayerDesc(spec.reshape_layer))
        layers.append(LayerDesc(spec.classifier))
        return layers


def get_alex_spec(num_classes=10):
    spec = LayerSpec(
        layer=AlexNet,
        sublayers_spec=AlexNetLayerSpec(
            features=[
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 11,
                        "stride": 4,
                        "padding": 5,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 5,
                        "padding": 2,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.Conv2D,
                    extra_kwargs={
                        "in_channels": 3,
                        "out_channels": 3,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                ),
                LayerSpec(
                    layer=nn.ReLU,
                ),
                LayerSpec(
                    layer=nn.MaxPool2D,
                    extra_kwargs={"kernel_size": 2, "stride": 2},
                ),
            ],
            reshape_layer=LayerSpec(layer=ReshapeHelp, extra_kwargs={"shape": [-1, 256]}),
            classifier=LayerSpec(
                layer=nn.Linear,
                extra_kwargs={"in_features": 256, "out_features": num_classes},
            ),
        ),
        extra_kwargs={
            "loss_fn": nn.CrossEntropyLoss(),
        },
    )
    return spec


class TestPipeLayerAPI(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
        }
        batch_size = 8
        micro_batch_size = 2
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)
        self.hcg = fleet.get_hybrid_communicate_group()

    def test_pipelayer_desc(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=self.pipeline_parallel_size)
        np.testing.assert_array_equal(len(pipe_model.parameters()), 6)

    def test_pipelayer_desc_single(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        np.testing.assert_array_equal(len(pipe_model.parameters()), 12)
        pipe_model = NoPipelineParallel(pipe_model, self.strategy)
        input = paddle.randn([256, 3, 224, 224])
        label = paddle.randint(0, 10, [147, 1])
        data = [[input, input, input, input], [label, label, label, label]]
        pipe_model.forward_backward_pipeline(data)

    def _create_no_pipeline_model(self):
        """Helper to create a NoPipelineParallel model for eval_batch tests."""
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=1)
        npp = NoPipelineParallel(pipe_model, self.strategy)
        npp._delay_scale_loss = False
        return npp

    def _create_eval_data(self, acc_steps):
        """Create data matching AlexNet architecture.
        Input [256, 3, 224, 224] -> model output [147, 10] -> label [147, 1].
        """
        inputs = [paddle.randn([256, 3, 224, 224]) for _ in range(acc_steps)]
        labels = [paddle.randint(0, 10, [147, 1]) for _ in range(acc_steps)]
        return [inputs, labels]

    def test_eval_batch_compute_loss(self):
        npp = self._create_no_pipeline_model()
        data = self._create_eval_data(npp.accumulate_steps)
        result = npp.eval_batch(data, compute_loss=True)
        self.assertIsInstance(result, paddle.Tensor)

    # TODO(hushenwei2000): enable this test after migrate to paddle pp
    # def test_eval_batch_no_compute_loss(self):
    #     npp = self._create_no_pipeline_model()
    #     data = self._create_eval_data(npp.accumulate_steps)
    #     result = npp.eval_batch(data, compute_loss=False)
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), npp.accumulate_steps)

    def test_eval_batch_invalid_loss_fn_idx(self):
        npp = self._create_no_pipeline_model()
        data = self._create_eval_data(npp.accumulate_steps)
        with self.assertRaises(AssertionError):
            npp.eval_batch(data, compute_loss=True, loss_fn_idx=99)

    def test_eval_batch_compute_loss_delay_scale(self):
        npp = self._create_no_pipeline_model()
        npp._delay_scale_loss = True
        data = self._create_eval_data(npp.accumulate_steps)
        result = npp.eval_batch(data, compute_loss=True)
        self.assertIsInstance(result, paddle.Tensor)

    # TODO(hushenwei2000): enable this test after migrate to paddle pp
    # def test_eval_batch_return_host_tensor(self):
    #     npp = self._create_no_pipeline_model()
    #     data = self._create_eval_data(npp.accumulate_steps)
    #     result = npp.eval_batch(
    #         data, compute_loss=False, return_host_tensor=True
    #     )
    #     self.assertIsInstance(result, list)
    #     self.assertEqual(len(result), npp.accumulate_steps)

    def test_pipelayer_segment_method_list(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(alex_desc, num_stages=self.pipeline_parallel_size, seg_method=[0, 4])
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 4)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 8)

    def test_pipelayer_segment_method_spec(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc,
            num_stages=self.pipeline_parallel_size,
            seg_method="layer:Conv2D|MaxPool2D",
        )
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 4)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 8)

    def test_pipelayer_segment_method_vpp(self):
        alex_desc = get_alex_spec()
        pipe_model = build_spec_layer(
            alex_desc,
            num_stages=self.pipeline_parallel_size,
            seg_method="layer:Conv2D|MaxPool2D",
            num_virtual_pipeline_stages=2,
        )
        stage_id = self.hcg.get_stage_id()
        if stage_id == 0:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 6)
        elif stage_id == 1:
            np.testing.assert_array_equal(len(pipe_model.parameters()), 6)


if __name__ == "__main__":
    unittest.main()
