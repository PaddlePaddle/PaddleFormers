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
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle
from paddle.distributed.fleet.meta_parallel import ScheduleNode

from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayerNode,
    TransformerLayerOverlappedScheduleNode,
    tensors_clone,
)


class DenseConfig:
    num_nextn_predict_layers = None
    mtp_load_weight_only = False


class MTPConfig:
    num_nextn_predict_layers = 1
    mtp_load_weight_only = False


class DenseLayer:
    full_recompute = False
    mlp = object()

    def compute_attention(self, *args, **kwargs):
        return None

    def compute_mlp(self, *args, **kwargs):
        return None


class TestTensorsCloneExtra(unittest.TestCase):
    def test_dict_values_are_cloned(self):
        tensor = paddle.ones([2], dtype="float32")
        cloned = tensors_clone({"hidden": tensor})

        self.assertEqual(cloned["hidden"].numpy().tolist(), [1.0, 1.0])
        self.assertIsNot(cloned["hidden"], tensor)

    def test_tuple_preserves_non_tensor_objects(self):
        marker = object()
        tensor = paddle.ones([1], dtype="float32")
        cloned = tensors_clone((tensor, marker))

        self.assertIsNot(cloned[0], tensor)
        self.assertIs(cloned[1], marker)

    def test_unsupported_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            tensors_clone(123)


class TestTransformerLayerNodeExtra(unittest.TestCase):
    def _layer_node(self, config=None):
        return TransformerLayerNode(
            DenseLayer(), config or DenseConfig(), name="dense"
        )

    def test_forward_rejects_mtp_overlap_configuration(self):
        node = self._layer_node(MTPConfig())

        with self.assertRaises(AssertionError):
            node.forward(
                {"hidden_states": paddle.ones([1, 1], dtype="float32")}
            )

    def test_overlapped_node_rejects_non_transformer_layer_nodes(self):
        schedule_node = ScheduleNode(lambda x: x, name="plain")

        with self.assertRaises(AssertionError):
            TransformerLayerOverlappedScheduleNode(schedule_node, schedule_node)

    def test_forward_backward_rejects_split_bw(self):
        node = TransformerLayerOverlappedScheduleNode(
            self._layer_node(), self._layer_node(), name="overlap"
        )

        with self.assertRaises(AssertionError):
            node.forward_backward({}, [], split_bw=True)


if __name__ == "__main__":
    unittest.main()
