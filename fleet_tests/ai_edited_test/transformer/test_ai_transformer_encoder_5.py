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

from paddleformers.fleet.transformer.transformer_encoder import (
    TransformerEncoder,
    build_overlapped_nodes,
)


class TestBuildOverlappedNodesWithLayerNodes(unittest.TestCase):
    """Tests for build_overlapped_nodes with TransformerLayerNode instances."""

    def _make_layer_node(self):
        """Create a minimal TransformerLayerNode-like object for testing."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        from paddleformers.fleet.transformer.transformer_layer import (
            TransformerLayerNode,
        )

        # Create a real TransformerLayerNode by bypassing __init__
        node = object.__new__(TransformerLayerNode)
        # Set required ScheduleNode attributes
        ScheduleNode.__init__(node, fwd_func=None, name="test_node")
        node.config = MagicMock()
        node.config.num_nextn_predict_layers = None
        node._is_sparse = False
        node.full_recompute = False
        return node

    def test_build_with_forward_overlap_nodes(self):
        """Test build_overlapped_nodes with TransformerLayerNode in forward chunk."""
        node1 = self._make_layer_node()
        node2 = self._make_layer_node()
        other_node = MagicMock(spec=type("DummyNode", (object,), {}))
        # Make other_node a ScheduleNode so ScheduleChunk accepts it
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        other_node = object.__new__(ScheduleNode)
        ScheduleNode.__init__(other_node, fwd_func=None, name="other")

        forward_chunk = ScheduleChunk([other_node, node1, node2])
        backward_chunk = ScheduleChunk([node2, node1, other_node])

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # other_node goes to forward_pre, both TransformerLayerNodes are overlap
        self.assertEqual(len(fwd_pre.nodes), 1)
        self.assertEqual(len(overlap.nodes), 2)

    def test_build_with_unequal_overlap(self):
        """Test with more forward overlap nodes than backward overlap nodes."""
        fwd_nodes = [self._make_layer_node() for _ in range(4)]
        bwd_nodes = [self._make_layer_node() for _ in range(2)]

        forward_chunk = ScheduleChunk(fwd_nodes)
        backward_chunk = ScheduleChunk(bwd_nodes)

        fwd_pre, bwd_pre, overlap, fwd_post, bwd_post = build_overlapped_nodes(
            forward_chunk, backward_chunk
        )

        # overlap = min(4, 2) = 2, extra 2 go to fwd_post
        self.assertEqual(len(overlap.nodes), 2)
        self.assertEqual(len(fwd_post.nodes), 2)


class TestTransformerEncoderGetLayerDescList(unittest.TestCase):
    """Tests for get_layer_desc_list and get_encoder_layer_desc_list."""

    def _make_encoder(self):
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.pipeline_model_parallel_size = 1
            encoder.config.virtual_pipeline_model_parallel_size = 1
            encoder.modal = None
            encoder._pipeline_name_mapping = None
            encoder._pp_to_single_mapping = None
            encoder._sequential_layers = []
            return encoder

    def test_get_layer_desc_list_no_modal(self):
        """Test get_layer_desc_list without modal prefix."""
        encoder = self._make_encoder()

        class DummySpec:
            def __init__(self):
                self.embedding = paddle.nn.Layer
                self.layer_norm = paddle.nn.Layer
                self.head_empty_layers = []
                self.tail_empty_layers = []
                self.transformer_layers = [paddle.nn.Linear]

        spec = DummySpec()
        layers = encoder.get_layer_desc_list(spec)
        self.assertEqual(
            len(layers), 3
        )  # embedding + 1 transformer + layer_norm
        # Check name_prefixes
        prefixes = [l["name_prefix"] for l in layers]
        self.assertTrue(all(p.startswith("model") for p in prefixes))

    def test_get_layer_desc_list_with_modal(self):
        """Test get_layer_desc_list with modal prefix."""
        encoder = self._make_encoder()
        encoder.modal = "vision"

        class DummySpec:
            def __init__(self):
                self.embedding = paddle.nn.Layer
                self.layer_norm = paddle.nn.Layer
                self.head_empty_layers = []
                self.tail_empty_layers = []
                self.transformer_layers = [paddle.nn.Linear]

        spec = DummySpec()
        layers = encoder.get_layer_desc_list(spec)
        prefixes = [l["name_prefix"] for l in layers]
        self.assertTrue(all(p.startswith("model.vision") for p in prefixes))

    def test_get_encoder_layer_desc_list_with_head_tail(self):
        """Test get_encoder_layer_desc_list with head/tail empty layers."""
        encoder = self._make_encoder()

        class DummySpec:
            def __init__(self):
                self.embedding = paddle.nn.Layer
                self.layer_norm = paddle.nn.Layer
                self.head_empty_layers = [paddle.nn.Layer]
                self.tail_empty_layers = [paddle.nn.Layer]
                self.transformer_layers = [paddle.nn.Linear]

        spec = DummySpec()
        layers = []
        # get_encoder_layer_desc_list modifies layers in-place and returns None
        result = encoder.get_encoder_layer_desc_list(layers, spec, "model")
        self.assertIsNone(result)
        # 1 head + 1 transformer + 1 tail = 3
        self.assertEqual(len(layers), 3)


class TestTransformerEncoderOverlappedForwardBackwardDetailed(
    unittest.TestCase
):
    """Detailed tests for overlapped_forward_backward."""

    def _make_encoder(self):
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            return encoder

    def test_overlapped_with_backward_loss_fn_with_scaler(self):
        """Test overlapped_forward_backward with backward_loss_fn_node and scaler."""
        encoder = self._make_encoder()

        backward_loss_fn_node = MagicMock()
        backward_loss_fn_node.backward.return_value = paddle.randn([2, 4])

        mock_pre = MagicMock()
        mock_pre.forward.return_value = {"input": paddle.randn([2, 4])}
        mock_pre.backward.return_value = paddle.randn([2, 4])

        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])

        with patch(
            "paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes"
        ) as mock_build:
            mock_build.return_value = (
                mock_pre,
                mock_pre,
                ScheduleChunk([]),
                mock_pre,
                mock_pre,
            )

            encoder.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs={"input": paddle.randn([2, 4])},
                forward_loss_fn_node=None,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=backward_loss_fn_node,
                backward_input_grads=paddle.randn([2, 4]),
                scaler=1.0,
                p2p_async_handle=None,
            )
            backward_loss_fn_node.backward.assert_called_once_with(scaler=1.0)

    def test_overlapped_with_backward_loss_fn_no_scaler(self):
        """Test overlapped_forward_backward with backward_loss_fn_node without scaler."""
        encoder = self._make_encoder()

        backward_loss_fn_node = MagicMock()
        backward_loss_fn_node.backward.return_value = paddle.randn([2, 4])

        mock_pre = MagicMock()
        mock_pre.forward.return_value = {"input": paddle.randn([2, 4])}
        mock_pre.backward.return_value = paddle.randn([2, 4])

        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])

        with patch(
            "paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes"
        ) as mock_build:
            mock_build.return_value = (
                mock_pre,
                mock_pre,
                ScheduleChunk([]),
                mock_pre,
                mock_pre,
            )

            encoder.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs={"input": paddle.randn([2, 4])},
                forward_loss_fn_node=None,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=backward_loss_fn_node,
                backward_input_grads=paddle.randn([2, 4]),
                scaler=None,
                p2p_async_handle=None,
            )
            backward_loss_fn_node.backward.assert_called_once_with()

    def test_overlapped_with_forward_loss_fn(self):
        """Test overlapped_forward_backward with forward_loss_fn_node."""
        encoder = self._make_encoder()

        forward_loss_fn_node = MagicMock()
        forward_loss_fn_node.forward.return_value = paddle.randn([])

        mock_pre = MagicMock()
        mock_pre.forward.return_value = {"input": paddle.randn([2, 4])}
        mock_pre.backward.return_value = paddle.randn([2, 4])

        forward_chunk = ScheduleChunk([])
        backward_chunk = ScheduleChunk([])

        with patch(
            "paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes"
        ) as mock_build:
            mock_build.return_value = (
                mock_pre,
                mock_pre,
                ScheduleChunk([]),
                mock_pre,
                mock_pre,
            )

            fwd_out, fwd_loss, bwd_grads = encoder.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs={"input": paddle.randn([2, 4])},
                forward_loss_fn_node=forward_loss_fn_node,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=None,
                backward_input_grads=paddle.randn([2, 4]),
                scaler=None,
                p2p_async_handle=None,
            )
            self.assertIsNotNone(fwd_loss)


class TestTransformerEncoderFP8VirtualStages(unittest.TestCase):
    """Tests for TransformerEncoder fp8 methods with virtual pipeline stages."""

    def test_fp8_quant_weight_with_virtual_stages(self):
        """Test fp8_quant_weight with _num_virtual_pipeline_stages > 1."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder._num_virtual_pipeline_stages = 2
            # Mock _model_chunks with mock layers instead of real nn.Linear
            mock_layer = MagicMock(spec=paddle.nn.Layer)
            encoder._model_chunks = [[mock_layer]]

            # Should iterate _model_chunks without error
            encoder.fp8_quant_weight(batch_mode=False, quant_transpose=True)

    def test_use_fp8_with_virtual_stages(self):
        """Test use_fp8 with _num_virtual_pipeline_stages > 1."""
        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder._num_virtual_pipeline_stages = 2
            # Use mock layers instead of real nn.Linear
            encoder._model_chunks = [[MagicMock(spec=paddle.nn.Layer)]]

            result = encoder.use_fp8()
            self.assertFalse(result)

    def test_use_fp8_with_transformer_layer_in_run_function(self):
        """Test use_fp8 when run_function contains a TransformerLayer-like object."""

        with patch.object(
            TransformerEncoder, "__init__", lambda self, *a, **kw: None
        ):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder._num_virtual_pipeline_stages = 1
            # No TransformerLayer instances - use mock
            encoder.run_function = [MagicMock(spec=paddle.nn.Layer)]
            result = encoder.use_fp8()
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
