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


class TestVisionLayerInit(unittest.TestCase):
    """Test VisionLayer initialization."""

    def test_basic_init(self):
        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )

        mock_config = MagicMock()
        layer = VisionLayer(config=mock_config)
        self.assertIsNotNone(layer)


class TestVisionLayerInheritance(unittest.TestCase):
    """Test that VisionLayer properly inherits from FleetLayer."""

    def test_is_fleet_layer(self):
        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )
        from paddleformers.fleet.transformer.layer import FleetLayer

        self.assertTrue(issubclass(VisionLayer, FleetLayer))

    def test_instance_is_fleet_layer(self):
        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )
        from paddleformers.fleet.transformer.layer import FleetLayer

        mock_config = MagicMock()
        layer = VisionLayer(config=mock_config)
        self.assertIsInstance(layer, FleetLayer)

    def test_instance_is_paddle_layer(self):
        import paddle

        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )

        mock_config = MagicMock()
        layer = VisionLayer(config=mock_config)
        self.assertIsInstance(layer, paddle.nn.Layer)


class TestVisionLayerConfig(unittest.TestCase):
    """Test VisionLayer stores config correctly."""

    def test_config_stored(self):
        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )

        mock_config = MagicMock()
        layer = VisionLayer(config=mock_config)
        self.assertEqual(layer.config, mock_config)


class TestVisionLayerAsBase(unittest.TestCase):
    """Test VisionLayer can be used as a base class."""

    def test_subclass_init(self):
        import paddle

        from paddleformers.fleet.models.common.vision_layer.vision_layer import (
            VisionLayer,
        )

        class CustomVision(VisionLayer):
            def __init__(self, config):
                super().__init__(config=config)
                self.custom_attr = 42

        mock_config = MagicMock()
        model = CustomVision(config=mock_config)
        self.assertEqual(model.custom_attr, 42)
        self.assertIsInstance(model, VisionLayer)
        self.assertIsInstance(model, paddle.nn.Layer)


if __name__ == "__main__":
    unittest.main()
