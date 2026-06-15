# Copyright (c) 2026 PaddleFleet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on applicable law or agreed to in writing, software
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
from unittest.mock import MagicMock, patch

from paddleformers.fleet.models.gpt.gpt_model import GPTModel, GPTSublayersSpec


class TestGPTSublayersSpecDefaults(unittest.TestCase):
    """Tests for GPTSublayersSpec default values."""

    def test_default_embedding_is_none(self):
        """embedding should default to None."""
        spec = GPTSublayersSpec()
        self.assertIsNone(spec.embedding)

    def test_default_transformer_layers_is_none(self):
        """transformer_layers should default to None."""
        spec = GPTSublayersSpec()
        self.assertIsNone(spec.transformer_layers)

    def test_default_lm_head_is_none(self):
        """lm_head should default to None."""
        spec = GPTSublayersSpec()
        self.assertIsNone(spec.lm_head)


class TestGPTModelAddSequentialLayer(unittest.TestCase):
    """Tests for GPTModel.add_sequential_layer."""

    def test_add_sequential_layer_appends_dict(self):
        """add_sequential_layer should append a dict with layer and name_prefix."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            layers = []
            mock_desc = MagicMock()
            model.add_sequential_layer(layers, mock_desc, "test_prefix")
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["layer"], mock_desc)
            self.assertEqual(layers[0]["name_prefix"], "test_prefix")


class TestGPTModelGetSequentialLayers(unittest.TestCase):
    """Tests for GPTModel.get_sequential_layers."""

    def test_get_sequential_layers_extracts_layer_only(self):
        """get_sequential_layers should return only the layer objects."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mock_layer1 = MagicMock()
            mock_layer2 = MagicMock()
            model._sequential_layers = [
                {"layer": mock_layer1, "name_prefix": "a"},
                {"layer": mock_layer2, "name_prefix": "b"},
            ]
            result = model.get_sequential_layers()
            self.assertEqual(result, [mock_layer1, mock_layer2])


class TestGPTModelGetNamePrefixes(unittest.TestCase):
    """Tests for GPTModel.get_sequential_name_prefixes."""

    def test_get_name_prefixes(self):
        """get_sequential_name_prefixes should return index->prefix mapping."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "embed"},
                {"layer": MagicMock(), "name_prefix": "layer.0"},
            ]
            result = model.get_sequential_name_prefixes()
            self.assertEqual(result["0"], "embed")
            self.assertEqual(result["1"], "layer.0")


class TestGPTModelGetHardwareFlops(unittest.TestCase):
    """Tests for GPTModel.get_hardware_flops."""

    def test_get_hardware_flops_returns_expected_value(self):
        """get_hardware_flops should return 989e3."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            result = model.get_hardware_flops()
            self.assertEqual(result, 989e3)


class TestGPTModelFP8Quant(unittest.TestCase):
    """Tests for GPTModel.fp8_quant_weight and use_fp8 method existence."""

    def test_fp8_quant_weight_method_exists(self):
        """fp8_quant_weight should be a method on GPTModel."""
        self.assertTrue(hasattr(GPTModel, "fp8_quant_weight"))
        self.assertTrue(callable(GPTModel.fp8_quant_weight))

    def test_use_fp8_method_exists(self):
        """use_fp8 should be a method on GPTModel."""
        self.assertTrue(hasattr(GPTModel, "use_fp8"))
        self.assertTrue(callable(GPTModel.use_fp8))


if __name__ == "__main__":
    unittest.main()
