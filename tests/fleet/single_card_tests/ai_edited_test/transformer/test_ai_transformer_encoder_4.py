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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from unittest.mock import MagicMock, patch

import paddle
from paddle.distributed.fleet.meta_parallel import ScheduleChunk


class TestTransformerEncoderHelperMethods(unittest.TestCase):
    """Tests for TransformerEncoder helper methods."""

    def test_add_sequential_layer(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._sequential_layers = []

            layers = []
            encoder.add_sequential_layer(layers, MagicMock(), "model.layers.0")
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["name_prefix"], "model.layers.0")

    def test_get_sequential_layers(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "model"},
                {"layer": MagicMock(), "name_prefix": "model.layers.0"},
            ]

            layers = encoder.get_sequential_layers()
            self.assertEqual(len(layers), 2)

    def test_get_sequential_name_prefixes(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "model"},
                {"layer": MagicMock(), "name_prefix": "model.layers.0"},
            ]

            prefixes = encoder.get_sequential_name_prefixes()
            self.assertEqual(prefixes["0"], "model")
            self.assertEqual(prefixes["1"], "model.layers.0")

    def test_get_hardware_flops(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config

            flops = encoder.get_hardware_flops()
            self.assertEqual(flops, 989e3)


class TestBuildOverlappedNodesEncoder(unittest.TestCase):
    """Tests for build_overlapped_nodes in transformer_encoder module."""

    def test_build_overlapped_nodes_no_overlap(self):
        """Test build_overlapped_nodes with no TransformerLayerNode instances."""
        from paddleformers.fleet.transformer.transformer_encoder import (
            build_overlapped_nodes,
        )

        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        self.assertEqual(len(overlap.nodes), 0)
        self.assertEqual(len(fwd_pre.nodes), 0)
        self.assertEqual(len(bwd_pre.nodes), 0)

    def test_build_overlapped_nodes_assert_type(self):
        """Test that backward_chunk must be a ScheduleChunk."""
        from paddleformers.fleet.transformer.transformer_encoder import (
            build_overlapped_nodes,
        )

        forward_chunk = ScheduleChunk([])
        backward_chunk = "not_a_chunk"

        with self.assertRaises(AssertionError):
            build_overlapped_nodes(forward_chunk, backward_chunk)


class TestTransformerEncoderOverlappedForwardBackward(unittest.TestCase):
    """Tests for TransformerEncoder.overlapped_forward_backward."""

    def test_overlapped_forward_backward_no_loss(self):
        from paddle.distributed.fleet.meta_parallel import ScheduleChunk

        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config

            forward_chunk = ScheduleChunk([])
            backward_chunk = ScheduleChunk([])

            with patch(
                "paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes"
            ) as mock_build:
                mock_pre = MagicMock()
                mock_pre.forward.return_value = {
                    "input": paddle.randn([2, 8, 64])
                }
                mock_pre.backward.return_value = paddle.randn([2, 8, 64])
                mock_build.return_value = (
                    mock_pre,
                    mock_pre,
                    ScheduleChunk([]),
                    mock_pre,
                    mock_pre,
                )

                forward_inputs = {"input": paddle.randn([2, 8, 64])}
                fwd_out, fwd_loss, bwd_grads = (
                    encoder.overlapped_forward_backward(
                        forward_chunk=forward_chunk,
                        forward_inputs=forward_inputs,
                        forward_loss_fn_node=None,
                        backward_chunk=backward_chunk,
                        backward_loss_fn_node=None,
                        backward_input_grads=paddle.randn([2, 8, 64]),
                        scaler=None,
                        p2p_async_handle=None,
                    )
                )
                self.assertIsNone(fwd_loss)


class TestTransformerEncoderFP8(unittest.TestCase):
    """Tests for TransformerEncoder fp8 methods."""

    def test_fp8_quant_weight_no_virtual_stages(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._num_virtual_pipeline_stages = 1
            encoder.run_function = []
            encoder.fp8_quant_weight()

    def test_use_fp8_no_virtual_stages(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._num_virtual_pipeline_stages = 1
            encoder.run_function = []

            result = encoder.use_fp8()
            self.assertFalse(result)


class TestTransformerEncoderNameMapping(unittest.TestCase):
    """Tests for TransformerEncoder name mapping methods."""

    def test_set_pipeline_name_mapping_with_mappings(self):
        from paddleformers.fleet.transformer.transformer_encoder import (
            TransformerEncoder,
        )

        mock_config = MagicMock()
        mock_config.pipeline_model_parallel_size = 1
        mock_config.virtual_pipeline_model_parallel_size = 1

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = mock_config
            encoder._pipeline_name_mapping = None

            mappings = {"model.layers.0.weight": "0.weight"}
            result = encoder._set_pipeline_name_mapping(mappings=mappings)
            self.assertEqual(result, mappings)
