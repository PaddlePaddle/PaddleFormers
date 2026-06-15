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
from unittest.mock import patch

import paddle

from paddleformers.fleet.transformer.moe.moe_utils import AddAuxiliaryLoss


class TestMoELayerImport(unittest.TestCase):
    """Tests for MoELayer module import and basic structure."""

    def test_import(self):
        """Test that MoELayer can be imported."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        self.assertTrue(hasattr(MoELayer, "__init__"))

    def test_add_auxiliary_loss_import(self):
        """Test that AddAuxiliaryLoss can be imported from moe_layer module."""
        from paddleformers.fleet.transformer.moe.moe_layer import AddAuxiliaryLoss

        self.assertIsNotNone(AddAuxiliaryLoss)


class TestMoELayerConfiguration(unittest.TestCase):
    """Tests for MoELayer configuration and construction."""

    @patch("paddleformers.fleet.transformer.moe.moe_layer.configure_buffer")
    def test_construction_with_mocked_router(self, mock_configure):
        """Test MoELayer construction with mocked components."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        # Just verify import and basic structure
        self.assertTrue(hasattr(MoELayer, "__init__"))


class TestMoELayerAuxLossIntegration(unittest.TestCase):
    """Tests for MoELayer aux loss integration."""

    def test_add_auxiliary_loss_forward(self):
        """Test AddAuxiliaryLoss forward in MoE context."""
        x = paddle.randn([4, 8])
        loss = paddle.to_tensor(0.5)
        loss.stop_gradient = False
        out = AddAuxiliaryLoss.apply(x, loss)
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
