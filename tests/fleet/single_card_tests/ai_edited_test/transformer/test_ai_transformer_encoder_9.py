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

from paddleformers.fleet.transformer.transformer_encoder import TransformerEncoder


def _make_encoder(**attrs):
    """Create a TransformerEncoder with mocked __init__ and direct __dict__ setting."""
    with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        object.__setattr__(encoder, "_sub_layers", {})
        object.__setattr__(encoder, "_parameters", {})
        object.__setattr__(encoder, "_buffers", {})
        object.__setattr__(encoder, "_non_persistable_buffers", set())
        for k, v in attrs.items():
            object.__setattr__(encoder, k, v)
        return encoder


class TestTransformerEncoderGetLayerDescListPrefix(unittest.TestCase):
    """Tests for TransformerEncoder.get_layer_desc_list modal prefix logic."""

    def test_modal_prefix_used_when_modal_set(self):
        """When modal is set, name_prefix should start with model.{modal}."""
        encoder = _make_encoder(modal="language_model")
        recorded_prefixes = []
        encoder.add_sequential_layer = lambda ls, d, p: (
            ls.append({"layer": d, "name_prefix": p}),
            recorded_prefixes.append(p),
        )
        encoder.get_encoder_layer_desc_list = MagicMock()
        mock_spec = MagicMock()
        mock_spec.embedding = type("E", (), {})
        mock_spec.layer_norm = type("L", (), {})
        # Avoid LayerDesc calling issubclass on MagicMock
        with patch("paddleformers.fleet.transformer.transformer_encoder.LayerDesc"):
            encoder.get_layer_desc_list(mock_spec)
        self.assertTrue(any("language_model" in p for p in recorded_prefixes))

    def test_model_prefix_used_when_modal_none(self):
        """When modal is None, name_prefix should start with 'model'."""
        encoder = _make_encoder(modal=None)
        recorded_prefixes = []
        encoder.add_sequential_layer = lambda ls, d, p: (
            ls.append({"layer": d, "name_prefix": p}),
            recorded_prefixes.append(p),
        )
        encoder.get_encoder_layer_desc_list = MagicMock()
        mock_spec = MagicMock()
        mock_spec.embedding = type("E", (), {})
        mock_spec.layer_norm = type("L", (), {})
        with patch("paddleformers.fleet.transformer.transformer_encoder.LayerDesc"):
            encoder.get_layer_desc_list(mock_spec)
        self.assertTrue(any(p.startswith("model") for p in recorded_prefixes))


class TestTransformerEncoderGetEncoderLayerDescList(unittest.TestCase):
    """Tests for TransformerEncoder.get_encoder_layer_desc_list."""

    def test_adds_head_empty_layers(self):
        """Should add head_empty_layers to the list."""
        encoder = _make_encoder()
        layers = []
        mock_head = type("H", (), {})
        mock_spec = MagicMock()
        mock_spec.head_empty_layers = [mock_head]
        mock_spec.transformer_layers = []
        mock_spec.tail_empty_layers = []
        encoder.add_sequential_layer = lambda ls, d, p: ls.append({"layer": d, "name_prefix": p})
        with patch("paddleformers.fleet.transformer.transformer_encoder.LayerDesc"):
            encoder.get_encoder_layer_desc_list(layers, mock_spec, "model")
        self.assertEqual(len(layers), 1)

    def test_adds_tail_empty_layers(self):
        """Should add tail_empty_layers after transformer layers."""
        encoder = _make_encoder()
        layers = []
        mock_tail = type("T", (), {})
        mock_spec = MagicMock()
        mock_spec.head_empty_layers = []
        mock_spec.transformer_layers = []
        mock_spec.tail_empty_layers = [mock_tail]
        encoder.add_sequential_layer = lambda ls, d, p: ls.append({"layer": d, "name_prefix": p})
        with patch("paddleformers.fleet.transformer.transformer_encoder.LayerDesc"):
            encoder.get_encoder_layer_desc_list(layers, mock_spec, "model")
        self.assertEqual(len(layers), 1)

    def test_adds_transformer_layers(self):
        """Should add transformer_layers to the list."""
        encoder = _make_encoder()
        layers = []
        mock_tf1 = type("TF1", (), {})
        mock_tf2 = type("TF2", (), {})
        mock_spec = MagicMock()
        mock_spec.head_empty_layers = []
        mock_spec.transformer_layers = [mock_tf1, mock_tf2]
        mock_spec.tail_empty_layers = []
        encoder.add_sequential_layer = lambda ls, d, p: ls.append({"layer": d, "name_prefix": p})
        with patch("paddleformers.fleet.transformer.transformer_encoder.LayerDesc"):
            encoder.get_encoder_layer_desc_list(layers, mock_spec, "model")
        self.assertEqual(len(layers), 2)


class TestTransformerEncoderOverlappedForwardBackwardLogic(unittest.TestCase):
    """Tests for TransformerEncoder.overlapped_forward_backward logic."""

    @patch("paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes")
    def test_forward_loss_none_when_no_loss_fn(self, mock_build):
        """forward_loss should be None when forward_loss_fn_node is None."""
        mock_forward_pre = MagicMock()
        mock_forward_pre.forward.return_value = "fwd_out"
        mock_backward_pre = MagicMock()
        mock_backward_pre.backward.return_value = "bwd_grad"
        mock_overlap = MagicMock()
        mock_overlap.nodes = []
        mock_forward_post = MagicMock()
        mock_forward_post.forward.return_value = "fwd_post_out"
        mock_backward_post = MagicMock()
        mock_backward_post.backward.return_value = "bwd_post_grad"

        mock_build.return_value = (
            mock_forward_pre,
            mock_backward_pre,
            mock_overlap,
            mock_forward_post,
            mock_backward_post,
        )

        encoder = _make_encoder()
        result = encoder.overlapped_forward_backward(
            forward_chunk=MagicMock(),
            forward_inputs="fwd_in",
            forward_loss_fn_node=None,
            backward_chunk=MagicMock(),
            backward_loss_fn_node=None,
            backward_input_grads=None,
            scaler=None,
            p2p_async_handle=None,
        )
        _, forward_loss, _ = result
        self.assertIsNone(forward_loss)

    @patch("paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes")
    def test_forward_loss_from_loss_fn_node(self, mock_build):
        """forward_loss should come from forward_loss_fn_node when provided."""
        mock_forward_pre = MagicMock()
        mock_forward_pre.forward.return_value = "fwd_out"
        mock_backward_pre = MagicMock()
        mock_backward_pre.backward.return_value = "bwd_grad"
        mock_overlap = MagicMock()
        mock_overlap.nodes = []
        mock_forward_post = MagicMock()
        mock_forward_post.forward.return_value = "fwd_post_out"
        mock_backward_post = MagicMock()
        mock_backward_post.backward.return_value = "bwd_post_grad"

        mock_build.return_value = (
            mock_forward_pre,
            mock_backward_pre,
            mock_overlap,
            mock_forward_post,
            mock_backward_post,
        )

        mock_loss_fn = MagicMock()
        mock_loss_fn.forward.return_value = "loss_value"

        encoder = _make_encoder()
        result = encoder.overlapped_forward_backward(
            forward_chunk=MagicMock(),
            forward_inputs="fwd_in",
            forward_loss_fn_node=mock_loss_fn,
            backward_chunk=MagicMock(),
            backward_loss_fn_node=None,
            backward_input_grads=None,
            scaler=None,
            p2p_async_handle=None,
        )
        _, forward_loss, _ = result
        self.assertEqual(forward_loss, "loss_value")


if __name__ == "__main__":
    unittest.main()
