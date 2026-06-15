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

from paddle.distributed.fleet.meta_parallel import ScheduleChunk, ScheduleNode

from paddleformers.fleet.models.gpt.gpt_model import (
    GPTModel,
    GPTSublayersSpec,
    build_overlapped_nodes,
)


class TestBuildOverlappedNodes(unittest.TestCase):
    """Test build_overlapped_nodes function."""

    def _make_schedule_chunk(self, nodes):
        """Create a ScheduleChunk bypassing node validation."""

        chunk = ScheduleChunk.__new__(ScheduleChunk)
        chunk.nodes = nodes
        return chunk

    def test_empty_chunks(self):
        """Test with empty forward and backward chunks."""
        fwd = self._make_schedule_chunk([])
        bwd = self._make_schedule_chunk([])
        result = build_overlapped_nodes(fwd, bwd)
        self.assertEqual(len(result), 5)

    def test_no_overlap_layers(self):
        """Test when there are no overlapping TransformerLayer nodes."""

        fwd = self._make_schedule_chunk([ScheduleNode(lambda x: x)])
        bwd = self._make_schedule_chunk([ScheduleNode(lambda x: x)])
        result = build_overlapped_nodes(fwd, bwd)
        self.assertEqual(len(result), 5)


class TestGPTSublayersSpec(unittest.TestCase):
    """Test GPTSublayersSpec dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        spec = GPTSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.mtp)
        self.assertIsNone(spec.layer_norm)
        self.assertIsNone(spec.lm_head)

    def test_with_values(self):
        """Test setting values."""
        mock_emb = MagicMock()
        mock_lm = MagicMock()
        spec = GPTSublayersSpec(
            embedding=mock_emb,
            lm_head=mock_lm,
        )
        self.assertEqual(spec.embedding, mock_emb)
        self.assertEqual(spec.lm_head, mock_lm)


class TestGPTModelMethods(unittest.TestCase):
    """Test GPTModel methods (unit-level, no GPU)."""

    def test_add_sequential_layer(self):
        """Test add_sequential_layer method."""
        model = GPTModel.__new__(GPTModel)
        layers = []
        mock_desc = MagicMock()
        model.add_sequential_layer(layers, mock_desc, "model")
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["layer"], mock_desc)
        self.assertEqual(layers[0]["name_prefix"], "model")

    def test_get_sequential_layers(self):
        """Test get_sequential_layers method."""
        model = GPTModel.__new__(GPTModel)
        mock_a = MagicMock()
        mock_b = MagicMock()
        model._sequential_layers = [
            {"layer": mock_a, "name_prefix": "model.0"},
            {"layer": mock_b, "name_prefix": "model.1"},
        ]
        result = model.get_sequential_layers()
        self.assertEqual(result, [mock_a, mock_b])

    def test_get_sequential_name_prefixes(self):
        """Test get_sequential_name_prefixes method."""
        model = GPTModel.__new__(GPTModel)
        model._sequential_layers = [
            {"layer": MagicMock(), "name_prefix": "model"},
            {"layer": MagicMock(), "name_prefix": "model.0"},
            {"layer": MagicMock(), "name_prefix": "model.1"},
        ]
        result = model.get_sequential_name_prefixes()
        self.assertEqual(result["0"], "model")
        self.assertEqual(result["1"], "model.0")
        self.assertEqual(result["2"], "model.1")

    def test_get_hardware_flops(self):
        """Test get_hardware_flops returns expected value."""
        model = GPTModel.__new__(GPTModel)
        flops = model.get_hardware_flops()
        self.assertEqual(flops, 989e3)
