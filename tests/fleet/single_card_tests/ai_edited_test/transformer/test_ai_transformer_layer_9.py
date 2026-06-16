# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerNode,
    TransformerLayerSublayersSpec,
    tensors_clone,
)


class TestTransformerLayerNodeSchedule(unittest.TestCase):
    """Tests for TransformerLayerNode schedule functionality."""

    def test_transformer_layer_node_exists(self):
        """TransformerLayerNode should be importable."""
        self.assertTrue(hasattr(TransformerLayerNode, "__init__"))

    def test_transformer_layer_node_has_forward(self):
        """TransformerLayerNode should have a forward method."""
        self.assertTrue(hasattr(TransformerLayerNode, "forward"))


class TestTensorsCloneWithTuple(unittest.TestCase):
    """Tests for tensors_clone with tuple inputs."""

    def test_clone_tuple_of_tensors(self):
        """tensors_clone should handle tuples of tensors."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        result = tensors_clone((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(paddle.allclose(result[0], t1))
        self.assertTrue(paddle.allclose(result[1], t2))

    def test_clone_list_with_none_and_tensor(self):
        """tensors_clone should handle lists with None and tensors."""
        t = paddle.randn([2, 3])
        result = tensors_clone([None, t])
        self.assertIsNone(result[0])
        self.assertTrue(paddle.allclose(result[1], t))


class TestTransformerLayerSublayersSpecCreation(unittest.TestCase):
    """Tests for TransformerLayerSublayersSpec creation with values."""

    def test_spec_with_custom_values(self):
        """TransformerLayerSublayersSpec should accept custom values."""
        mock_attn = MagicMock()
        mock_mlp = MagicMock()
        spec = TransformerLayerSublayersSpec(
            self_attn=mock_attn,
            mlp=mock_mlp,
        )
        self.assertEqual(spec.self_attn, mock_attn)
        self.assertEqual(spec.mlp, mock_mlp)

    def test_spec_has_sharded_state_dict_keys_map(self):
        """TransformerLayerSublayersSpec should have sharded_state_dict_keys_map."""
        spec = TransformerLayerSublayersSpec()
        self.assertIsInstance(spec.sharded_state_dict_keys_map, dict)

    def test_spec_sharded_keys_map_custom(self):
        """TransformerLayerSublayersSpec should accept custom sharded_state_dict_keys_map."""
        spec = TransformerLayerSublayersSpec(
            sharded_state_dict_keys_map={"old": "new"}
        )
        self.assertEqual(spec.sharded_state_dict_keys_map["old"], "new")


class TestTransformerLayerNodeInit(unittest.TestCase):
    """Tests for TransformerLayerNode creation."""

    def test_node_stores_config(self):
        """TransformerLayerNode should store config from the layer."""
        mock_node = MagicMock()
        mock_node.full_recompute = False
        mock_config = MagicMock()
        tln = TransformerLayerNode(
            mock_node, mock_config, name="test", layer_number=3
        )
        self.assertEqual(tln.config, mock_config)
        self.assertEqual(tln.layer_number, 3)


if __name__ == "__main__":
    unittest.main()
