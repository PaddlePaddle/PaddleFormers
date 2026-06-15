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
from unittest.mock import MagicMock, patch

import paddle

from paddleformers.fleet.transformer.transformer_encoder import TransformerEncoder


class TestTransformerEncoderSetPipelineNameMapping(unittest.TestCase):
    """Tests for _set_pipeline_name_mapping with various key formats."""

    def _make_encoder(self):
        with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.pipeline_model_parallel_size = 1
            encoder.config.virtual_pipeline_model_parallel_size = 1
            encoder._pipeline_name_mapping = None
            encoder._pp_to_single_mapping = None
            return encoder

    def test_set_pipeline_name_mapping_with_explicit_mappings(self):
        """Test _set_pipeline_name_mapping when explicit mappings are provided."""
        encoder = self._make_encoder()
        mappings = {"0.weight": "model.layers.0.weight"}
        result = encoder._set_pipeline_name_mapping(mappings=mappings)
        self.assertEqual(result, mappings)
        self.assertEqual(encoder._pipeline_name_mapping, mappings)


class TestTransformerEncoderSetStateDict(unittest.TestCase):
    """Tests for set_state_dict method."""

    def _make_encoder(self):
        with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.pipeline_model_parallel_size = 1
            encoder.config.virtual_pipeline_model_parallel_size = 1
            encoder._pipeline_name_mapping = None
            encoder._pp_to_single_mapping = None
            return encoder

    def test_set_state_dict_with_empty_mapping(self):
        """Test set_state_dict with empty pipeline name mapping."""
        encoder = self._make_encoder()
        encoder._pipeline_name_mapping = {}

        with self.assertRaises(AssertionError):
            # Should assert that mapping is non-empty
            encoder.set_state_dict({})

    def test_set_state_dict_with_mapping(self):
        """Test set_state_dict with non-empty mapping."""
        encoder = self._make_encoder()
        encoder._pipeline_name_mapping = {"model.layers.0.weight": "0.weight"}
        encoder._pp_to_single_mapping = {"0.weight": "model.layers.0.weight"}

        mock_state_dict = {"model.layers.0.weight": paddle.randn([4, 8])}

        with patch.object(type(encoder).__mro__[1], "set_state_dict", return_value=[]):
            result = encoder.set_state_dict(mock_state_dict)
            self.assertIsNotNone(result)


class TestTransformerEncoderCheckSharedModelState(unittest.TestCase):
    """Tests for _check_shared_model_state method."""

    def _make_encoder(self):
        with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            encoder.config.pipeline_model_parallel_size = 1
            encoder.config.virtual_pipeline_model_parallel_size = 1
            # Make sure _pipeline_name_mapping has the keys that
            # _pp_to_single_mapping values point to
            encoder._pipeline_name_mapping = {"model.layers.0.weight": "0.weight"}
            encoder._pp_to_single_mapping = {"0.weight": "model.layers.0.weight"}
            return encoder

    def test_check_shared_model_state(self):
        """Test _check_shared_model_state with basic state dict."""
        encoder = self._make_encoder()

        with patch.object(type(encoder).__mro__[1], "state_dict", return_value={}):
            result = encoder._check_shared_model_state()
            self.assertIsInstance(result, dict)


class TestTransformerEncoderOverlappedWithP2PHandle(unittest.TestCase):
    """Tests for overlapped_forward_backward with p2p_async_handle."""

    def _make_encoder(self):
        with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            return encoder

    def test_overlapped_with_p2p_handle(self):
        """Test overlapped_forward_backward with p2p_async_handle."""
        encoder = self._make_encoder()

        mock_p2p = MagicMock()
        mock_pre = MagicMock()
        mock_pre.forward.return_value = {"input": paddle.randn([2, 4])}
        mock_pre.backward.return_value = paddle.randn([2, 4])

        forward_chunk = MagicMock()
        backward_chunk = MagicMock()

        with patch("paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes") as mock_build:
            mock_build.return_value = (
                mock_pre,
                mock_pre,
                MagicMock(nodes=[]),
                mock_pre,
                mock_pre,
            )

            encoder.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs={"input": paddle.randn([2, 4])},
                forward_loss_fn_node=None,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=None,
                backward_input_grads=paddle.randn([2, 4]),
                scaler=None,
                p2p_async_handle=mock_p2p,
            )
            mock_p2p.forward_handle_wait.assert_called_once()
            mock_p2p.backward_handle_wait.assert_called_once()


class TestTransformerEncoderOverlappedWithOverlapNodes(unittest.TestCase):
    """Tests for overlapped_forward_backward with overlap nodes."""

    def _make_encoder(self):
        with patch.object(TransformerEncoder, "__init__", lambda self, *a, **kw: None):
            encoder = TransformerEncoder.__new__(TransformerEncoder)
            encoder.config = MagicMock()
            return encoder

    def test_overlapped_with_overlap_nodes(self):
        """Test overlapped_forward_backward with overlap nodes present."""
        encoder = self._make_encoder()

        mock_overlap_node = MagicMock()
        mock_overlap_node.forward_backward.return_value = (
            {"input": paddle.randn([2, 4])},
            paddle.randn([2, 4]),
        )
        mock_overlap_chunk = MagicMock()
        mock_overlap_chunk.nodes = [mock_overlap_node]

        mock_pre = MagicMock()
        mock_pre.forward.return_value = {"input": paddle.randn([2, 4])}
        mock_pre.backward.return_value = paddle.randn([2, 4])

        forward_chunk = MagicMock()
        backward_chunk = MagicMock()

        with patch("paddleformers.fleet.transformer.transformer_encoder.build_overlapped_nodes") as mock_build:
            mock_build.return_value = (
                mock_pre,
                mock_pre,
                mock_overlap_chunk,
                mock_pre,
                mock_pre,
            )

            fwd_out, fwd_loss, bwd_grads = encoder.overlapped_forward_backward(
                forward_chunk=forward_chunk,
                forward_inputs={"input": paddle.randn([2, 4])},
                forward_loss_fn_node=None,
                backward_chunk=backward_chunk,
                backward_loss_fn_node=None,
                backward_input_grads=paddle.randn([2, 4]),
                scaler=None,
                p2p_async_handle=None,
            )
            mock_overlap_node.forward_backward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
