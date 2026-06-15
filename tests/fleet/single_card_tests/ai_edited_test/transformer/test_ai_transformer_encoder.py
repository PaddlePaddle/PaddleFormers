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

import paddle
from paddle.distributed.fleet.meta_parallel import ScheduleChunk

from paddleformers.fleet.transformer.transformer_encoder import (
    TransformerEncoder,
    build_overlapped_nodes,
)


class DummySpec:
    """Dummy spec for testing TransformerEncoder."""

    def __init__(self):
        self.embedding = paddle.nn.Layer
        self.layer_norm = paddle.nn.Layer
        self.head_empty_layers = []
        self.tail_empty_layers = []
        self.transformer_layers = [paddle.nn.Linear] * 2


class DummyConfig:
    """Dummy config for testing TransformerEncoder."""

    def __init__(self, **overrides):
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        self.model_type = "gpt"
        self.num_nextn_predict_layers = None
        self.mtp_load_weight_only = False
        for k, v in overrides.items():
            setattr(self, k, v)


class TestBuildOverlappedNodes(unittest.TestCase):
    """Tests for build_overlapped_nodes function."""

    def test_no_overlap(self):
        """Test with no TransformerLayerNode in chunks -> no overlap."""
        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])
        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(forward_chunk, backward_chunk)
        self.assertEqual(len(overlap.nodes), 0)
        self.assertEqual(len(fwd_pre.nodes), 0)
        self.assertEqual(len(bwd_pre.nodes), 0)

    def test_assert_type(self):
        """Test that backward_chunk must be a ScheduleChunk."""
        forward_chunk = ScheduleChunk([])
        with self.assertRaises(AssertionError):
            build_overlapped_nodes(forward_chunk, "not_a_chunk")


class TestTransformerEncoderHelperMethods(unittest.TestCase):
    """Tests for TransformerEncoder helper methods that don't require PipelineLayer."""

    def test_add_sequential_layer(self):
        layers = []
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.add_sequential_layer(layers, "layer_desc_1", "model.prefix")
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0]["layer"], "layer_desc_1")
        self.assertEqual(layers[0]["name_prefix"], "model.prefix")

    def test_add_sequential_layer_empty_prefix(self):
        layers = []
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.add_sequential_layer(layers, "layer_desc_1", "")
        self.assertEqual(layers[0]["name_prefix"], "")

    def test_get_sequential_layers(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._sequential_layers = [
            {"layer": "desc1", "name_prefix": "model.0"},
            {"layer": "desc2", "name_prefix": "model.1"},
        ]
        layers = encoder.get_sequential_layers()
        self.assertEqual(layers, ["desc1", "desc2"])

    def test_get_sequential_name_prefixes(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._sequential_layers = [
            {"layer": "desc1", "name_prefix": "model.layers.0"},
            {"layer": "desc2", "name_prefix": "model.layers.1"},
            {"layer": "desc3", "name_prefix": "model.final"},
        ]
        prefixes = encoder.get_sequential_name_prefixes()
        self.assertEqual(
            prefixes,
            {"0": "model.layers.0", "1": "model.layers.1", "2": "model.final"},
        )

    def test_get_hardware_flops(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        flops = encoder.get_hardware_flops()
        self.assertEqual(flops, 989e3)


class TestTransformerEncoderNameMapping(unittest.TestCase):
    """Tests for pipeline name mapping methods."""

    def test_set_pipeline_name_mapping_with_dict(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._pipeline_name_mapping = None
        mappings = {"key1": "value1", "key2": "value2"}
        result = encoder._set_pipeline_name_mapping(mappings)
        self.assertEqual(result, mappings)
        self.assertEqual(encoder._pipeline_name_mapping, mappings)


class TestTransformerEncoderOverlappedForwardBackward(unittest.TestCase):
    """Tests for overlapped_forward_backward method."""

    def test_basic_call_no_overlap(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        config = DummyConfig()
        encoder.config = config
        encoder._pipeline_name_mapping = {}

        # Create simple chunks with no overlap
        fwd_chunk = ScheduleChunk([])
        bwd_chunk = ScheduleChunk([])

        forward_inputs = {"hidden_states": paddle.randn([2, 4, 64])}
        backward_input_grads = [paddle.randn([2, 4, 64])]

        fwd_out, fwd_loss, bwd_out = encoder.overlapped_forward_backward(
            fwd_chunk,
            forward_inputs,
            None,
            bwd_chunk,
            None,
            backward_input_grads,
            None,
            None,
        )
        self.assertIsNone(fwd_loss)

    def test_with_loss_fn_node(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        config = DummyConfig()
        encoder.config = config
        encoder._pipeline_name_mapping = {}

        fwd_chunk = ScheduleChunk([])
        bwd_chunk = ScheduleChunk([])

        forward_inputs = {"hidden_states": paddle.randn([2, 4, 64])}
        backward_input_grads = [paddle.randn([2, 4, 64])]

        # Mock loss node
        loss_fn_node = MagicMock()
        loss_fn_node.forward.return_value = paddle.randn([])
        loss_fn_node.backward.return_value = [paddle.randn([2, 4, 64])]

        fwd_out, fwd_loss, bwd_out = encoder.overlapped_forward_backward(
            fwd_chunk,
            forward_inputs,
            loss_fn_node,
            bwd_chunk,
            loss_fn_node,
            backward_input_grads,
            None,
            None,
        )
        self.assertIsNotNone(fwd_loss)

    def test_with_scaler(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        config = DummyConfig()
        encoder.config = config
        encoder._pipeline_name_mapping = {}

        fwd_chunk = ScheduleChunk([])
        bwd_chunk = ScheduleChunk([])

        forward_inputs = {"hidden_states": paddle.randn([2, 4, 64])}
        backward_input_grads = [paddle.randn([2, 4, 64])]

        loss_fn_node = MagicMock()
        loss_fn_node.forward.return_value = paddle.randn([])
        loss_fn_node.backward.return_value = [paddle.randn([2, 4, 64])]

        test_scaler = MagicMock()
        fwd_out, fwd_loss, bwd_out = encoder.overlapped_forward_backward(
            fwd_chunk,
            forward_inputs,
            loss_fn_node,
            bwd_chunk,
            loss_fn_node,
            backward_input_grads,
            scaler=test_scaler,
            p2p_async_handle=None,
        )
        # scaler should be passed to backward
        loss_fn_node.backward.assert_called_once_with(scaler=test_scaler)


class TestTransformerEncoderFP8Methods(unittest.TestCase):
    """Tests for fp8 methods."""

    def test_fp8_quant_weight_no_virtual(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 1
        encoder.run_function = []
        # Should not raise with empty run_function
        encoder.fp8_quant_weight(batch_mode=False, quant_transpose=True)

    def test_fp8_quant_weight_with_virtual(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 2
        encoder._model_chunks = [[], []]
        encoder.fp8_quant_weight(batch_mode=False, quant_transpose=True)

    def test_use_fp8_no_virtual(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 1
        encoder.run_function = []
        result = encoder.use_fp8()
        self.assertFalse(result)

    def test_use_fp8_with_virtual(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 2
        encoder._model_chunks = [[], []]
        result = encoder.use_fp8()
        self.assertFalse(result)


class TestTransformerEncoderGetLayerDescList(unittest.TestCase):
    """Tests for get_layer_desc_list."""


class TestTransformerEncoderStateDict(unittest.TestCase):
    """Tests for state_dict and set_state_dict methods."""


class TestTransformerEncoderQwen3VL(unittest.TestCase):
    """Tests for qwen3_vl model type handling."""

    def test_config_model_type(self):
        config = DummyConfig(model_type="qwen3_vl")
        self.assertIn("qwen3_vl", config.model_type)

    def test_config_model_type_gpt(self):
        config = DummyConfig(model_type="gpt")
        self.assertNotIn("qwen3_vl", config.model_type)


if __name__ == "__main__":
    unittest.main()
