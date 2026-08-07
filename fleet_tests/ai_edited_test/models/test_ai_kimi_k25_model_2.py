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
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.models.kimi_k25.kimi_k25_model import (
    KimiK25VisionModel,
    KimiK25VisionSublayersSpec,
    KimiK25VisionTransformerLayer,
)


class _DummyLayer(paddle.nn.Layer):
    """A dummy paddle.nn.Layer subclass for LayerSpec usage."""

    def __init__(self):
        super().__init__()


class TestKimiK25VisionSublayersSpecExtended(unittest.TestCase):
    """Extended tests for KimiK25VisionSublayersSpec dataclass."""

    def test_with_all_fields(self):
        """Test setting all fields."""
        spec = KimiK25VisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[LayerSpec(_DummyLayer)],
            transformer_layers=[LayerSpec(_DummyLayer), LayerSpec(_DummyLayer)],
            tail_empty_layers=[LayerSpec(_DummyLayer)],
            final_layernorm=LayerSpec(_DummyLayer),
            sdtpool_merger=LayerSpec(_DummyLayer),
            merger=LayerSpec(_DummyLayer),
        )
        self.assertIsNotNone(spec.embedding)
        self.assertEqual(len(spec.head_empty_layers), 1)
        self.assertEqual(len(spec.transformer_layers), 2)
        self.assertEqual(len(spec.tail_empty_layers), 1)
        self.assertIsNotNone(spec.final_layernorm)
        self.assertIsNotNone(spec.sdtpool_merger)
        self.assertIsNotNone(spec.merger)


class TestKimiK25VisionTransformerLayerForward(unittest.TestCase):
    """Test KimiK25VisionTransformerLayer forward method."""

    def test_forward_with_full_recompute(self):
        """Test forward with full_recompute=True."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.full_recompute = True
        layer.modal = "vision"

        mock_hidden = paddle.randn([2, 8, 64])
        mock_recompute_result = paddle.randn([2, 8, 64])

        with patch(
            "paddleformers.fleet.models.kimi_k25.kimi_k25_model.recompute",
            return_value=mock_recompute_result,
        ):
            dict_args = {
                "hidden_states": mock_hidden,
                "attention_mask": None,
                "attn_mask_startend_row_indices": None,
                "context": None,
                "context_mask": None,
                "attention_bias": None,
                "packed_seq_params": None,
                "rope_freqs_cis": None,
            }
            result = layer.forward(dict_args)
            self.assertIn("hidden_states", result)

    def test_forward_without_full_recompute(self):
        """Test forward with full_recompute=False."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.full_recompute = False
        layer.modal = "vision"
        layer._forward_impl = MagicMock(return_value=paddle.randn([2, 8, 64]))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
            "grid_thws": None,
            "attention_mask": None,
        }
        result = layer.forward(dict_args)
        self.assertIn("hidden_states", result)

    def test_forward_pops_dynamic_inference_decode_only(self):
        """Test forward pops 'dynamic_inference_decode_only' from dict_args."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.full_recompute = False
        layer.modal = "vision"
        layer._forward_impl = MagicMock(return_value=paddle.randn([2, 8, 64]))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
            "dynamic_inference_decode_only": True,
            "position_ids": None,
            "grid_thws": None,
        }
        result = layer.forward(dict_args)
        self.assertNotIn("dynamic_inference_decode_only", result)
        self.assertNotIn("position_ids", result)

    def test_forward_with_context_3elem(self):
        """Test forward when _forward_impl returns 3-element tuple (context path)."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.full_recompute = False
        layer.modal = "vision"

        hidden = paddle.randn([2, 8, 64])
        context = paddle.randn([2, 8, 64])
        # The forward method checks len(outputs) == 3 for context recovery
        layer._forward_impl = MagicMock(return_value=(hidden, context, None))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
            "grid_thws": None,
        }
        result = layer.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertIn("context", result)


class TestKimiK25VisionTransformerLayerForwardImpl(unittest.TestCase):
    """Test KimiK25VisionTransformerLayer._forward_impl method."""

    def test_forward_impl_2d_hidden_states(self):
        """Test _forward_impl with 2D hidden states (unsqueeze)."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.modal = "vision"
        layer.full_recompute = False
        layer._forward_attention = MagicMock(
            return_value=(paddle.randn([1, 8, 64]), None)
        )
        layer._forward_mlp = MagicMock(return_value=paddle.randn([1, 8, 64]))

        hidden_states = paddle.randn([8, 64])  # 2D
        result = layer._forward_impl(hidden_states=hidden_states)
        self.assertIsNotNone(result)

    def test_forward_impl_3d_hidden_states(self):
        """Test _forward_impl with 3D hidden states."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.modal = "vision"
        layer.full_recompute = False
        layer._forward_attention = MagicMock(
            return_value=(paddle.randn([2, 8, 64]), None)
        )
        layer._forward_mlp = MagicMock(return_value=paddle.randn([2, 8, 64]))

        hidden_states = paddle.randn([2, 8, 64])  # 3D
        result = layer._forward_impl(hidden_states=hidden_states)
        self.assertIsNotNone(result)

    def test_forward_impl_with_context(self):
        """Test _forward_impl returns context when present."""
        layer = KimiK25VisionTransformerLayer.__new__(
            KimiK25VisionTransformerLayer
        )
        layer.modal = "vision"
        layer.full_recompute = False
        context = paddle.randn([2, 8, 64])
        layer._forward_attention = MagicMock(
            return_value=(paddle.randn([2, 8, 64]), context)
        )
        layer._forward_mlp = MagicMock(return_value=paddle.randn([2, 8, 64]))

        hidden_states = paddle.randn([2, 8, 64])
        result = layer._forward_impl(hidden_states=hidden_states)
        # When context is present, result is a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestKimiK25VisionModelGetLayerDescList(unittest.TestCase):
    """Test KimiK25VisionModel.get_layer_desc_list method."""

    def test_with_modal(self):
        """Test get_layer_desc_list with modal set."""
        model = KimiK25VisionModel.__new__(KimiK25VisionModel)
        # Initialize Paddle Layer internals
        model.__dict__.setdefault("_parameters", {})
        model.__dict__.setdefault("_buffers", {})
        model.__dict__.setdefault("_sub_layers", {})
        model.__dict__.setdefault("_loaddict_holder", {})
        model.__dict__.setdefault("_non_persistable_buffers", set())
        model.modal = "vision"
        model._sequential_layers = []

        spec = KimiK25VisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[],
            transformer_layers=[LayerSpec(_DummyLayer)],
            tail_empty_layers=[],
            final_layernorm=LayerSpec(_DummyLayer),
            sdtpool_merger=LayerSpec(_DummyLayer),
            merger=LayerSpec(_DummyLayer),
        )

        with patch.object(model, "get_encoder_layer_desc_list"):
            layers = model.get_layer_desc_list(spec)
            self.assertTrue(len(layers) > 0)

    def test_without_modal(self):
        """Test get_layer_desc_list without modal."""
        model = KimiK25VisionModel.__new__(KimiK25VisionModel)
        # Initialize Paddle Layer internals
        model.__dict__.setdefault("_parameters", {})
        model.__dict__.setdefault("_buffers", {})
        model.__dict__.setdefault("_sub_layers", {})
        model.__dict__.setdefault("_loaddict_holder", {})
        model.__dict__.setdefault("_non_persistable_buffers", set())
        model.modal = None
        model._sequential_layers = []

        spec = KimiK25VisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[],
            transformer_layers=[LayerSpec(_DummyLayer)],
            tail_empty_layers=[],
            final_layernorm=LayerSpec(_DummyLayer),
            sdtpool_merger=LayerSpec(_DummyLayer),
            merger=LayerSpec(_DummyLayer),
        )

        with patch.object(model, "get_encoder_layer_desc_list"):
            layers = model.get_layer_desc_list(spec)
            self.assertTrue(len(layers) > 0)


if __name__ == "__main__":
    unittest.main()
