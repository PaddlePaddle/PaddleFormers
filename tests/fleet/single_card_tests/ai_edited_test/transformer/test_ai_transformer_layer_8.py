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
import functools
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

import unittest
from unittest.mock import MagicMock

import paddle

from paddleformers.fleet.gpt_builders import gpt_builder
from paddleformers.fleet.models.gpt import GPTConfig
from paddleformers.fleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
    tensors_clone,
)


def _make_layer():
    config = GPTConfig(
        num_hidden_layers=1,
        hidden_size=8,
        vocab_size=16,
        max_sequence_length=8,
        num_attention_heads=2,
        intermediate_size=16,
        n_routed_experts=2,
        n_shared_experts=0,
        moe_intermediate_size=16,
        moe_layer_freq=1,
        moe_token_dispatcher_type="alltoall",
        moe_expert_fusion=False,
        moe_deep_gemm=False,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        output_layer_init_method=functools.partial(paddle.nn.init.xavier_uniform_, gain=1.0),
        tie_word_embeddings=False,
        use_qk_norm=True,
    )
    model = gpt_builder(config, num_stages=1)
    for layer in model.run_function:
        if isinstance(layer, TransformerLayer):
            return layer
    raise AssertionError("GPTModel did not create a TransformerLayer")


class TestTensorsCloneEdgeCases(unittest.TestCase):
    """Tests for tensors_clone with edge cases."""

    def test_clone_empty_list(self):
        """tensors_clone should handle empty lists."""
        result = tensors_clone([])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_clone_none_in_list(self):
        """tensors_clone should preserve None values in lists."""
        result = tensors_clone([None])
        self.assertIsNone(result[0])

    def test_clone_dict_with_tensors(self):
        """tensors_clone should clone tensors in dicts."""
        t = paddle.randn([2, 3])
        result = tensors_clone({"a": t})
        self.assertTrue(paddle.is_tensor(result["a"]))
        self.assertTrue(paddle.allclose(result["a"], t))

    def test_clone_tuple_of_tensors(self):
        """tensors_clone should handle tuples of tensors."""
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([4, 5])
        result = tensors_clone((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertTrue(paddle.allclose(result[0], t1))
        self.assertTrue(paddle.allclose(result[1], t2))

    def test_clone_list_with_dict(self):
        """tensors_clone should handle lists containing dicts."""
        t = paddle.randn([2, 3])
        result = tensors_clone([{"a": t}])
        self.assertIsInstance(result, list)
        self.assertTrue(paddle.is_tensor(result[0]["a"]))

    def test_clone_raises_on_unsupported_type(self):
        """tensors_clone should raise ValueError on unsupported types."""
        with self.assertRaises(ValueError):
            tensors_clone(42)


class TestTransformerLayerSublayersSpecDefaults(unittest.TestCase):
    """Tests for TransformerLayerSublayersSpec default values."""

    def test_default_self_attn_is_identity(self):
        """self_attn default should be IdentityOp class."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = TransformerLayerSublayersSpec()
        self.assertEqual(spec.self_attn, IdentityOp)

    def test_default_mlp_is_identity(self):
        """mlp default should be IdentityOp class."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = TransformerLayerSublayersSpec()
        self.assertEqual(spec.mlp, IdentityOp)

    def test_default_has_sharded_state_dict_keys_map(self):
        """sharded_state_dict_keys_map should default to empty dict."""
        spec = TransformerLayerSublayersSpec()
        self.assertIsInstance(spec.sharded_state_dict_keys_map, dict)


class TestTransformerLayerFP8Quant(unittest.TestCase):
    """Tests for TransformerLayer.fp8_quant_weight with MoE."""

    def test_fp8_quant_weight_with_moe_layer(self):
        """fp8_quant_weight should call mlp.fp8_quant_weight when mlp is MoELayer."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = _make_layer()
        layer.mlp = MagicMock(spec=MoELayer)
        layer.fp8_quant_weight(batch_mode=True, quant_transpose=False)
        layer.mlp.fp8_quant_weight.assert_called_once_with(batch_mode=True, quant_transpose=False)


class TestTransformerLayerUseFP8(unittest.TestCase):
    """Tests for TransformerLayer.use_fp8."""

    def test_use_fp8_with_moe_layer(self):
        """use_fp8 should delegate to mlp.use_fp8 when mlp is MoELayer."""
        from paddleformers.fleet.transformer.moe.moe_layer import MoELayer

        layer = _make_layer()
        layer.mlp = MagicMock(spec=MoELayer)
        layer.mlp.use_fp8.return_value = True
        self.assertTrue(layer.use_fp8())


class TestTransformerLayerBuildScheduleNode(unittest.TestCase):
    """Tests for TransformerLayer.build_schedule_node."""

    def test_build_schedule_node_exists(self):
        """build_schedule_node should be a method on TransformerLayer."""
        self.assertTrue(hasattr(TransformerLayer, "build_schedule_node"))
        self.assertTrue(callable(TransformerLayer.build_schedule_node))


if __name__ == "__main__":
    unittest.main()
