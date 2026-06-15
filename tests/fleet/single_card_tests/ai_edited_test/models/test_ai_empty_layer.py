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

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock


class TestEmptyLayerInit(unittest.TestCase):
    """Test EmptyLayer initialization."""

    def test_basic_init(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        self.assertIsNotNone(layer)


class TestEmptyLayerForward(unittest.TestCase):
    """Test EmptyLayer.forward returns input unchanged."""

    def test_forward_tensor(self):
        import paddle

        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        x = paddle.randn([2, 10, 64])
        result = layer(x)
        self.assertTrue(paddle.equal(x, result).all())

    def test_forward_dict(self):
        import paddle

        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        x = {"hidden_states": paddle.randn([2, 10, 64])}
        result = layer(x)
        self.assertIs(result, x)

    def test_forward_none(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        result = layer(None)
        self.assertIsNone(result)

    def test_forward_int(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        result = layer(42)
        self.assertEqual(result, 42)

    def test_forward_list(self):
        import paddle

        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        x = [paddle.randn([2, 10]), paddle.randn([2, 10])]
        result = layer(x)
        self.assertIs(result, x)


class TestEmptyLayerBuildScheduleNode(unittest.TestCase):
    """Test EmptyLayer.build_schedule_node method."""

    def test_returns_schedule_node(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        node = layer.build_schedule_node()
        self.assertIsNotNone(node)


class TestEmptyLayerInheritance(unittest.TestCase):
    """Test that EmptyLayer properly inherits from FleetLayer."""

    def test_is_fleet_layer(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer
        from paddleformers.fleet.transformer.layer import FleetLayer

        self.assertTrue(issubclass(EmptyLayer, FleetLayer))

    def test_instance_is_fleet_layer(self):
        from paddleformers.fleet.models.common.empty_layer import EmptyLayer
        from paddleformers.fleet.transformer.layer import FleetLayer

        mock_config = MagicMock()
        layer = EmptyLayer(config=mock_config)
        self.assertIsInstance(layer, FleetLayer)


if __name__ == "__main__":
    unittest.main()
