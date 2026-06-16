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
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerNode,
    TransformerLayerOverlappedScheduleNode,
    TransformerLayerSublayersSpec,
    tensors_clone,
)


class TestTensorsClone(unittest.TestCase):
    """Tests for tensors_clone utility function."""

    def test_clone_tensor(self):
        """Test cloning a single tensor."""
        x = paddle.randn([4, 8])
        cloned = tensors_clone(x)
        self.assertTrue(paddle.allclose(x, cloned).item())
        # Cloned tensor should have same values but be a different tensor
        self.assertFalse(x is cloned)

    def test_clone_tuple(self):
        """Test cloning a tuple of tensors."""
        x = paddle.randn([4, 8])
        y = paddle.randn([4, 8])
        cloned = tensors_clone((x, y))
        self.assertIsInstance(cloned, tuple)
        self.assertEqual(len(cloned), 2)

    def test_clone_list(self):
        """Test cloning a list of tensors."""
        x = paddle.randn([4, 8])
        y = paddle.randn([4, 8])
        cloned = tensors_clone([x, y])
        self.assertIsInstance(cloned, list)
        self.assertEqual(len(cloned), 2)

    def test_clone_dict(self):
        """Test cloning a dict of tensors."""
        d = {"a": paddle.randn([4, 8]), "b": paddle.randn([4, 8])}
        cloned = tensors_clone(d)
        self.assertIsInstance(cloned, dict)
        self.assertIn("a", cloned)
        self.assertIn("b", cloned)

    def test_clone_mixed_types(self):
        """Test cloning a tuple with mixed types."""
        x = paddle.randn([4, 8])
        val = 42
        cloned = tensors_clone((x, val))
        self.assertIsInstance(cloned, tuple)
        self.assertEqual(cloned[1], 42)

    def test_clone_unsupported_type_raises(self):
        """Test that unsupported type raises ValueError."""
        with self.assertRaises(ValueError):
            tensors_clone("unsupported_string")


class TestTransformerLayerSublayersSpec(unittest.TestCase):
    """Tests for TransformerLayerSublayersSpec dataclass."""

    def test_default_values(self):
        """Test default values of TransformerLayerSublayersSpec."""
        from paddleformers.fleet.transformer.identity_op import (
            IdentityFuncOp,
            IdentityOp,
        )

        spec = TransformerLayerSublayersSpec()
        self.assertEqual(spec.input_layernorm, IdentityOp)
        self.assertEqual(spec.self_attn, IdentityOp)
        self.assertEqual(spec.self_attn_bda, IdentityFuncOp)
        self.assertEqual(spec.pre_cross_attn_layernorm, IdentityOp)
        self.assertEqual(spec.cross_attention, IdentityOp)
        self.assertEqual(spec.cross_attn_bda, IdentityFuncOp)
        self.assertEqual(spec.post_attention_layernorm, IdentityOp)
        self.assertEqual(spec.mlp, IdentityOp)
        self.assertEqual(spec.mlp_bda, IdentityFuncOp)
        self.assertEqual(spec.block_attn_res, IdentityOp)
        self.assertEqual(spec.sharded_state_dict_keys_map, {})

    def test_custom_values(self):
        """Test custom values of TransformerLayerSublayersSpec."""
        spec = TransformerLayerSublayersSpec(
            input_layernorm=MagicMock(),
            self_attn=MagicMock(),
        )
        self.assertIsNotNone(spec.input_layernorm)
        self.assertIsNotNone(spec.self_attn)


class TestTransformerLayerLogMD5(unittest.TestCase):
    """Tests for TransformerLayer._log_md5 static method."""

    def test_log_md5_when_disabled(self):
        """Test _log_md5 does nothing when LOG_LAYER_MD5 is not set."""
        # Should not raise or produce output
        original = TransformerLayer._LOG_LAYER_MD5
        TransformerLayer._LOG_LAYER_MD5 = False
        try:
            TransformerLayer._log_md5(paddle.randn([4, 8]), "test", 1)
        finally:
            TransformerLayer._LOG_LAYER_MD5 = original

    def test_log_md5_skip_mtp_probes(self):
        """Test _log_md5 skips when _skip_mtp_probes is True."""
        original_skip = TransformerLayer._skip_mtp_probes
        original_log = TransformerLayer._LOG_LAYER_MD5
        original_exp = TransformerLayer._gpt_model_use_experimental_version
        TransformerLayer._skip_mtp_probes = True
        TransformerLayer._LOG_LAYER_MD5 = True
        TransformerLayer._gpt_model_use_experimental_version = True
        try:
            # Should not raise
            TransformerLayer._log_md5(paddle.randn([4, 8]), "test", 1)
        finally:
            TransformerLayer._skip_mtp_probes = original_skip
            TransformerLayer._LOG_LAYER_MD5 = original_log
            TransformerLayer._gpt_model_use_experimental_version = original_exp


class TestTransformerLayerNode(unittest.TestCase):
    """Tests for TransformerLayerNode."""

    def _make_node(self, layer_number=0):
        """Create a minimal TransformerLayerNode for testing."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = object.__new__(TransformerLayerNode)
        ScheduleNode.__init__(node, fwd_func=None, name=f"layer_{layer_number}")
        node.config = MagicMock()
        node.config.num_nextn_predict_layers = None
        node.layer_number = layer_number
        node._is_sparse = False
        node.full_recompute = False
        node.attn_node = MagicMock()
        return node

    def test_construction(self):
        """Test TransformerLayerNode construction."""
        node = self._make_node(layer_number=3)
        self.assertEqual(node.layer_number, 3)

    def test_forward(self):
        """Test TransformerLayerNode forward delegates to layer."""
        node = self._make_node(layer_number=0)
        node.layer = MagicMock()
        node.layer.forward.return_value = (paddle.randn([4, 8]), None)
        node.attn_node.forward.return_value = (paddle.randn([4, 8]), None)
        node.mlp_node = MagicMock()
        node.mlp_node.forward.return_value = (paddle.randn([4, 8]), None)
        node.post_process_node = MagicMock()
        node.post_process_node.forward.return_value = (
            paddle.randn([4, 8]),
            None,
        )
        result = node.forward({"hidden_states": paddle.randn([4, 8])})
        self.assertIsNotNone(result)


class TestTransformerLayerOverlappedScheduleNode(unittest.TestCase):
    """Tests for TransformerLayerOverlappedScheduleNode."""

    def _make_layer_node(self):
        """Create a minimal TransformerLayerNode for testing."""
        from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
            ScheduleNode,
        )

        node = object.__new__(TransformerLayerNode)
        ScheduleNode.__init__(node, fwd_func=None, name="layer_0")
        node.config = MagicMock()
        node.config.num_nextn_predict_layers = None
        node.layer_number = 0
        node._is_sparse = False
        node.full_recompute = False
        return node

    def test_construction(self):
        """Test construction with forward and backward nodes."""
        fwd_node = self._make_layer_node()
        bwd_node = self._make_layer_node()
        overlapped = TransformerLayerOverlappedScheduleNode(fwd_node, bwd_node)
        self.assertEqual(overlapped.forward_node, fwd_node)
        self.assertEqual(overlapped.backward_node, bwd_node)
        self.assertEqual(overlapped.config, fwd_node.config)


class TestTransformerLayerClassAttributes(unittest.TestCase):
    """Tests for TransformerLayer class-level attributes."""

    def test_default_experimental_version(self):
        """Test default _gpt_model_use_experimental_version."""
        self.assertFalse(TransformerLayer._gpt_model_use_experimental_version)

    def test_default_log_layer_md5(self):
        """Test default _LOG_LAYER_MD5 is based on environment."""
        expected = os.environ.get("LOG_LAYER_MD5", "0") == "1"
        self.assertEqual(TransformerLayer._LOG_LAYER_MD5, expected)

    def test_default_skip_mtp_probes(self):
        """Test default _skip_mtp_probes."""
        self.assertFalse(TransformerLayer._skip_mtp_probes)


if __name__ == "__main__":
    unittest.main()
