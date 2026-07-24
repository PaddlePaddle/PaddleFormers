# Copyright (c) 2026 PaddlePleet Authors. All Rights Reserved.
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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

from paddle.distributed.fleet.meta_parallel import (
    PipelineLayer,
    ScheduleChunk,
    ScheduleNode,
)

from paddleformers.fleet.transformer.transformer_encoder import (
    TransformerEncoder,
    build_overlapped_nodes,
)
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerNode,
)


def _make_schedule_node(name=""):
    """Create a plain ScheduleNode for use in ScheduleChunk."""
    return ScheduleNode(lambda x: x, name=name)


class TestBuildOverlappedNodesNoOverlap(unittest.TestCase):
    """Tests for build_overlapped_nodes when no overlap layers exist."""

    def test_no_overlap_layers_returns_empty_overlap(self):
        """When no TransformerLayerNode in chunks, overlap_node should be empty."""
        node_a = _make_schedule_node("a")
        node_b = _make_schedule_node("b")
        forward_chunk = ScheduleChunk([node_a, node_b])
        backward_chunk = ScheduleChunk([node_a, node_b])
        result = build_overlapped_nodes(forward_chunk, backward_chunk)
        self.assertEqual(len(result), 5)
        forward_pre, backward_pre, overlap, forward_post, backward_post = result
        self.assertEqual(len(overlap.nodes), 0)

    def test_forward_pre_contains_non_overlap_nodes_before_first_overlap(self):
        """Non-overlap nodes before the first overlap node should be in forward_pre."""
        non_overlap = _make_schedule_node("non_overlap")
        overlap_node = TransformerLayerNode.__new__(TransformerLayerNode)
        overlap_node.config = MagicMock()
        forward_chunk = ScheduleChunk([non_overlap, overlap_node])
        backward_chunk = ScheduleChunk([overlap_node])
        result = build_overlapped_nodes(forward_chunk, backward_chunk)
        forward_pre, _, _, _, _ = result
        self.assertEqual(len(forward_pre.nodes), 1)

    def test_backward_pre_contains_non_overlap_nodes_before_first_overlap(self):
        """Non-overlap nodes after the last overlap node in reversed backward should be in backward_pre."""
        overlap_node = TransformerLayerNode.__new__(TransformerLayerNode)
        overlap_node.config = MagicMock()
        non_overlap = _make_schedule_node("non_overlap")
        # reversed order: [overlap_node, non_overlap] -> backward_pre gets non_overlap
        forward_chunk = ScheduleChunk([overlap_node])
        backward_chunk = ScheduleChunk([overlap_node, non_overlap])
        result = build_overlapped_nodes(forward_chunk, backward_chunk)
        _, backward_pre, _, _, _ = result
        self.assertEqual(len(backward_pre.nodes), 1)


class TestTransformerEncoderGetHardwareFlops(unittest.TestCase):
    """Tests for TransformerEncoder.get_hardware_flops."""

    def test_get_hardware_flops_returns_expected_value(self):
        """get_hardware_flops should return 989e3."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            result = encoder.get_hardware_flops()
            self.assertEqual(result, 989e3)


class TestTransformerEncoderAddSequentialLayer(unittest.TestCase):
    """Tests for TransformerEncoder.add_sequential_layer."""

    def test_add_sequential_layer_appends_dict(self):
        """add_sequential_layer should append a dict with layer and name_prefix."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            layers = []
            mock_desc = MagicMock()
            encoder.add_sequential_layer(layers, mock_desc, "test_prefix")
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["layer"], mock_desc)
            self.assertEqual(layers[0]["name_prefix"], "test_prefix")

    def test_add_sequential_layer_default_prefix(self):
        """add_sequential_layer with no prefix should use empty string."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            layers = []
            mock_desc = MagicMock()
            encoder.add_sequential_layer(layers, mock_desc)
            self.assertEqual(layers[0]["name_prefix"], "")

    def test_add_sequential_layer_multiple(self):
        """add_sequential_layer should append multiple entries in order."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            layers = []
            for i in range(5):
                encoder.add_sequential_layer(layers, MagicMock(), f"layer_{i}")
            self.assertEqual(len(layers), 5)
            for i in range(5):
                self.assertEqual(layers[i]["name_prefix"], f"layer_{i}")


class TestTransformerEncoderGetSequentialLayers(unittest.TestCase):
    """Tests for TransformerEncoder.get_sequential_layers."""

    def test_get_sequential_layers_extracts_layer_only(self):
        """get_sequential_layers should return only the layer objects from sequential layers."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            mock_layer1 = MagicMock()
            mock_layer2 = MagicMock()
            encoder._sequential_layers = [
                {"layer": mock_layer1, "name_prefix": "a"},
                {"layer": mock_layer2, "name_prefix": "b"},
            ]
            result = encoder.get_sequential_layers()
            self.assertEqual(result, [mock_layer1, mock_layer2])


class TestTransformerEncoderGetNamePrefixes(unittest.TestCase):
    """Tests for TransformerEncoder.get_sequential_name_prefixes."""

    def test_get_name_prefixes(self):
        """get_sequential_name_prefixes should return index->prefix mapping."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "embed"},
                {"layer": MagicMock(), "name_prefix": "layer.0"},
                {"layer": MagicMock(), "name_prefix": "lm_head"},
            ]
            result = encoder.get_sequential_name_prefixes()
            self.assertEqual(result["0"], "embed")
            self.assertEqual(result["1"], "layer.0")
            self.assertEqual(result["2"], "lm_head")


class TestTransformerEncoderStateDictQwenVL(unittest.TestCase):
    """Tests for TransformerEncoder.state_dict with qwen3_vl model type."""

    def test_state_dict_strips_qwen3_vl_prefix(self):
        """state_dict should strip 'model.language_model.' prefix for qwen3_vl."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.model_type = "qwen3_vl"
            encoder._pipeline_name_mapping = {}
            encoder._pp_to_single_mapping = {"0.weight": "model.weight"}

            mock_super_sd = MagicMock()
            mock_val = MagicMock()
            mock_super_sd.return_value = {
                "model.language_model.0.weight": mock_val
            }
            with patch.object(PipelineLayer, "state_dict", mock_super_sd):
                result = encoder.state_dict()
                self.assertIn("model.weight", result)

    def test_state_dict_no_prefix_for_non_qwen3(self):
        """state_dict should not strip prefix when model_type is not qwen3_vl."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.model_type = "gpt"
            encoder._pipeline_name_mapping = {}
            encoder._pp_to_single_mapping = {}

            mock_super_sd = MagicMock()
            mock_val = MagicMock()
            mock_super_sd.return_value = {"0.weight": mock_val}
            with patch.object(PipelineLayer, "state_dict", mock_super_sd):
                result = encoder.state_dict()
                self.assertIn("0.weight", result)


if __name__ == "__main__":
    unittest.main()
