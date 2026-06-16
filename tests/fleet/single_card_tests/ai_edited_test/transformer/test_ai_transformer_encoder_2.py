# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddle import nn
from paddle.distributed.fleet.meta_parallel import SharedLayerDesc

from paddleformers.fleet.transformer import transformer_encoder


class LightweightEncoder(transformer_encoder.TransformerEncoder):
    def __init__(self, modal=None):
        self.modal = modal
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self.layers = []
        self._stage_id = 0
        self._stage_for_index = 0

    def get_stage_from_index(self, idx):
        del idx
        return self._stage_for_index


class EncoderSpec:
    embedding = None
    head_empty_layers = []
    transformer_layers = []
    tail_empty_layers = []
    layer_norm = None


class DummyLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        del args, kwargs


class DummyEmbedding(DummyLayer):
    pass


class DummyHeadEmpty(DummyLayer):
    pass


class DummyTransformer(DummyLayer):
    pass


class DummyTailEmpty(DummyLayer):
    pass


class DummyLayerNorm(DummyLayer):
    pass


class TestTransformerEncoderDescriptorUtilities(unittest.TestCase):
    def _spec(self):
        spec = EncoderSpec()
        spec.embedding = DummyEmbedding
        spec.head_empty_layers = [DummyHeadEmpty]
        spec.transformer_layers = [DummyTransformer]
        spec.tail_empty_layers = [DummyTailEmpty]
        spec.layer_norm = DummyLayerNorm
        return spec

    def test_get_layer_desc_list_uses_model_prefix(self):
        model = LightweightEncoder(modal=None)

        layers = model.get_layer_desc_list(self._spec())

        self.assertEqual(
            [layer["name_prefix"] for layer in layers],
            [
                "model",
                "model.layers.0",
                "model.layers.1",
                "model.layers.2",
                "model",
            ],
        )

    def test_get_layer_desc_list_uses_modal_prefix(self):
        model = LightweightEncoder(modal="vision")

        layers = model.get_layer_desc_list(self._spec())

        self.assertEqual(
            [layer["name_prefix"] for layer in layers],
            [
                "model.vision",
                "model.vision.layers.0",
                "model.vision.layers.1",
                "model.vision.layers.2",
                "model.vision",
            ],
        )

    def test_sequential_layer_helpers_and_explicit_mapping(self):
        model = LightweightEncoder()
        layers = []
        first = object()
        second = object()
        model.add_sequential_layer(layers, first, "first.prefix")
        model.add_sequential_layer(layers, second, "second.prefix")
        model._sequential_layers = layers
        mapping = {"single": "pipeline"}

        self.assertEqual(model.get_sequential_layers(), [first, second])
        self.assertEqual(
            model.get_sequential_name_prefixes(),
            {"0": "first.prefix", "1": "second.prefix"},
        )
        self.assertIs(model._set_pipeline_name_mapping(mapping), mapping)
        self.assertIs(model._pipeline_name_mapping, mapping)

    def test_get_shardlayer_prefix_success_assertion_and_stage_error(self):
        shared = SharedLayerDesc(
            "embed", DummyEmbedding, shared_weight_attr="embedding_weight"
        )
        model = LightweightEncoder()
        model.layers = [shared]
        model._sequential_layers = [
            {"layer": shared, "name_prefix": "model.embed"}
        ]

        self.assertEqual(
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"]),
            "model.embed",
        )

        with self.assertRaises(AssertionError):
            model.get_shardlayer_prefix(["shared_layers", "missing", "weight"])

        model._stage_for_index = 1
        with self.assertRaises(ValueError):
            model.get_shardlayer_prefix(["shared_layers", "embed", "weight"])


if __name__ == "__main__":
    unittest.main()
