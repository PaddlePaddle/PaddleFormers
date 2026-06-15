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
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddleformers.fleet.models.qwen3_vl.qwen3_vl_model import (
    Qwen3VLVisionModel,
    Qwen3VLVisionSublayersSpec,
    Qwen3VLVisionTransformerLayer,
    Qwen3VLVsisionTransformerSubLayerSpec,
)


class _DummyLayer(paddle.nn.Layer):
    """A dummy paddle.nn.Layer subclass for LayerSpec usage."""

    def __init__(self):
        super().__init__()


class TestQwen3VLVisionSublayersSpec(unittest.TestCase):
    """Test Qwen3VLVisionSublayersSpec dataclass."""

    def test_default_fields(self):
        """Test default field values."""
        spec = Qwen3VLVisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.merger)

    def test_with_all_fields(self):
        """Test setting all fields."""
        spec = Qwen3VLVisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[],
            transformer_layers=[LayerSpec(_DummyLayer)],
            tail_empty_layers=[],
            merger=LayerSpec(_DummyLayer),
        )
        self.assertIsNotNone(spec.embedding)
        self.assertEqual(len(spec.transformer_layers), 1)
        self.assertIsNotNone(spec.merger)


class TestQwen3VLVsisionTransformerSubLayerSpec(unittest.TestCase):
    """Test Qwen3VLVsisionTransformerSubLayerSpec dataclass."""

    def test_has_deepstack_merger_field(self):
        """Test that deepstack_merger field exists."""
        spec = Qwen3VLVsisionTransformerSubLayerSpec()
        self.assertIsNone(spec.deepstack_merger)

    def test_with_deepstack_merger(self):
        """Test setting deepstack_merger."""
        spec = Qwen3VLVsisionTransformerSubLayerSpec(
            deepstack_merger=LayerSpec(_DummyLayer),
        )
        self.assertIsNotNone(spec.deepstack_merger)


class TestQwen3VLVisionModelGetLayerDescList(unittest.TestCase):
    """Test Qwen3VLVisionModel.get_layer_desc_list method."""

    def test_with_modal(self):
        """Test get_layer_desc_list with modal set."""
        model = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
        model.__dict__.setdefault("_parameters", {})
        model.__dict__.setdefault("_buffers", {})
        model.__dict__.setdefault("_sub_layers", {})
        model.__dict__.setdefault("_loaddict_holder", {})
        model.__dict__.setdefault("_non_persistable_buffers", set())
        model.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        model.modal = "vision"
        model._sequential_layers = []

        spec = Qwen3VLVisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[],
            transformer_layers=[LayerSpec(_DummyLayer)],
            tail_empty_layers=[],
            merger=LayerSpec(_DummyLayer),
        )

        with patch.object(model, "get_encoder_layer_desc_list"):
            layers = model.get_layer_desc_list(spec)
            self.assertTrue(len(layers) > 0)

    def test_without_modal(self):
        """Test get_layer_desc_list without modal."""
        model = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
        model.__dict__.setdefault("_parameters", {})
        model.__dict__.setdefault("_buffers", {})
        model.__dict__.setdefault("_sub_layers", {})
        model.__dict__.setdefault("_loaddict_holder", {})
        model.__dict__.setdefault("_non_persistable_buffers", set())
        model.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        model.modal = None
        model._sequential_layers = []

        spec = Qwen3VLVisionSublayersSpec(
            embedding=LayerSpec(_DummyLayer),
            head_empty_layers=[],
            transformer_layers=[LayerSpec(_DummyLayer)],
            tail_empty_layers=[],
            merger=LayerSpec(_DummyLayer),
        )

        with patch.object(model, "get_encoder_layer_desc_list"):
            layers = model.get_layer_desc_list(spec)
            self.assertTrue(len(layers) > 0)


class TestQwen3VLVisionTransformerLayerForward(unittest.TestCase):
    """Test Qwen3VLVisionTransformerLayer forward paths."""

    def test_forward_pops_dynamic_inference(self):
        """Test forward pops dynamic_inference_decode_only and position_ids."""
        layer = Qwen3VLVisionTransformerLayer.__new__(Qwen3VLVisionTransformerLayer)
        layer.full_recompute = False
        layer.modal = "vision"
        layer.deepstack_merger = None
        layer._forward_impl = MagicMock(return_value=(paddle.randn([2, 8, 64]), None))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
            "dynamic_inference_decode_only": True,
            "position_ids": None,
        }
        result = layer.forward(dict_args)
        self.assertNotIn("dynamic_inference_decode_only", result)
        self.assertNotIn("position_ids", result)

    def test_forward_with_deepstack_feature(self):
        """Test forward with deepstack_merger returns feature."""
        layer = Qwen3VLVisionTransformerLayer.__new__(Qwen3VLVisionTransformerLayer)
        layer.full_recompute = False
        layer.modal = "vision"
        layer.deepstack_merger = MagicMock(return_value=paddle.randn([2, 8, 64]))
        layer._forward_attention = MagicMock(return_value=(paddle.randn([2, 8, 64]), None))
        layer._forward_mlp = MagicMock(return_value=paddle.randn([2, 8, 64]))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
        }
        result = layer.forward(dict_args)
        self.assertIn("deepstack_feature_lists", result)

    def test_forward_without_deepstack_merger(self):
        """Test forward without deepstack_merger."""
        layer = Qwen3VLVisionTransformerLayer.__new__(Qwen3VLVisionTransformerLayer)
        layer.full_recompute = False
        layer.modal = "vision"
        layer.deepstack_merger = None
        layer._forward_impl = MagicMock(return_value=(paddle.randn([2, 8, 64]), None))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
        }
        result = layer.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertIn("deepstack_feature_lists", result)
        # No deepstack features since merger is None
        self.assertEqual(len(result["deepstack_feature_lists"]), 0)

    def test_forward_with_context_3_elem(self):
        """Test forward when _forward_impl returns 3 elements (context path)."""
        layer = Qwen3VLVisionTransformerLayer.__new__(Qwen3VLVisionTransformerLayer)
        layer.full_recompute = False
        layer.modal = "vision"
        layer.deepstack_merger = None

        hidden = paddle.randn([2, 8, 64])
        context = paddle.randn([2, 8, 64])
        layer._forward_impl = MagicMock(return_value=(hidden, context, None))

        dict_args = {
            "hidden_states": paddle.randn([2, 8, 64]),
        }
        result = layer.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertIn("context", result)


if __name__ == "__main__":
    unittest.main()
