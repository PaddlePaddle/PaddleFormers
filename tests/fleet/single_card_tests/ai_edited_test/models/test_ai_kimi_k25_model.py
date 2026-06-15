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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock

from paddleformers.fleet.models.kimi_k25.kimi_k25_model import (
    KimiK25VisionSublayersSpec,
)


class TestKimiK25VisionSublayersSpec(unittest.TestCase):
    """Test KimiK25VisionSublayersSpec dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        spec = KimiK25VisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.final_layernorm)
        self.assertIsNone(spec.sdtpool_merger)
        self.assertIsNone(spec.merger)

    def test_with_values(self):
        """Test setting values."""
        mock_emb = MagicMock()
        mock_merger = MagicMock()
        spec = KimiK25VisionSublayersSpec(
            embedding=mock_emb,
            merger=mock_merger,
        )
        self.assertEqual(spec.embedding, mock_emb)
        self.assertEqual(spec.merger, mock_merger)


class TestKimiK25VisionTransformerLayer(unittest.TestCase):
    """Test KimiK25VisionTransformerLayer class."""

    def _make_config(self, **overrides):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_cpu_initialization": True,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)


class TestKimiK25VisionModel(unittest.TestCase):
    """Test KimiK25VisionModel class."""
