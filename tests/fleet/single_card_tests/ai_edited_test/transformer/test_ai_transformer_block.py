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

from paddleformers.fleet.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSublayersSpec,
    _get_block_sublayers_spec,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "head_dim": 16,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "cpu_offloading": False,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestTransformerBlockSublayersSpec(unittest.TestCase):
    """Tests for TransformerBlockSublayersSpec dataclass."""

    def test_default_values(self):
        spec = TransformerBlockSublayersSpec()
        self.assertIsNone(spec.layer_specs)
        self.assertIsNone(spec.layer_norm)

    def test_custom_values(self):
        layer_specs = [MagicMock(), MagicMock()]
        layer_norm = MagicMock()
        spec = TransformerBlockSublayersSpec(layer_specs=layer_specs, layer_norm=layer_norm)
        self.assertEqual(spec.layer_specs, layer_specs)
        self.assertEqual(spec.layer_norm, layer_norm)


class TestGetBlockSublayersSpec(unittest.TestCase):
    """Tests for _get_block_sublayers_spec function."""

    def test_with_transformer_block_sublayers_spec(self):
        """When spec is already a TransformerBlockSublayersSpec, return it."""
        spec = TransformerBlockSublayersSpec()
        config = _make_config()
        result = _get_block_sublayers_spec(config, spec)
        self.assertIs(result, spec)

    def test_with_layer_spec_transformer_layer(self):
        """When spec is a LayerSpec for TransformerLayer, construct sublayers_spec."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        from paddleformers.fleet.transformer.transformer_layer import TransformerLayer

        config = _make_config(num_hidden_layers=3)
        layer_spec = LayerSpec(TransformerLayer)
        result = _get_block_sublayers_spec(config, layer_spec)

        self.assertIsInstance(result, TransformerBlockSublayersSpec)
        self.assertEqual(len(result.layer_specs), 3)

    def test_with_layer_spec_transformer_block(self):
        """When spec is a LayerSpec for TransformerBlock, return its sublayers_spec."""
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        inner_spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock()],
            layer_norm=MagicMock(),
        )

        layer_spec = LayerSpec(TransformerBlock, sublayers_spec=inner_spec)
        config = _make_config()
        result = _get_block_sublayers_spec(config, layer_spec)
        self.assertIs(result, inner_spec)

    def test_with_unsupported_spec_raises(self):
        """When spec type is unsupported, raise Exception."""
        config = _make_config()
        with self.assertRaises(Exception):  # noqa: B017
            _get_block_sublayers_spec(config, "unsupported_type")


class TestTransformerBlockConstruction(unittest.TestCase):
    """Tests for TransformerBlock __init__ and _build_layers."""

    @patch("paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups")
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_no_post_layer_norm(self, mock_build, mock_pg):
        """Test TransformerBlock with post_layer_norm=False."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock(spec=paddle.nn.Layer)

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock(), MagicMock()],
            layer_norm=MagicMock(),
        )

        block = TransformerBlock(
            config=config,
            spec=spec,
            post_layer_norm=False,
        )
        self.assertIsNone(block.norm)

    @patch("paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups")
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_no_post_process(self, mock_build, mock_pg):
        """Test TransformerBlock with post_process=False."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock(spec=paddle.nn.Layer)

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock(), MagicMock()],
            layer_norm=MagicMock(),
        )

        block = TransformerBlock(
            config=config,
            spec=spec,
            post_process=False,
        )
        self.assertIsNone(block.norm)

    @patch("paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups")
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_with_post_process_and_layer_norm(self, mock_build, mock_pg):
        """Test TransformerBlock with post_process=True and post_layer_norm=True."""
        mock_pg.return_value = MagicMock()
        mock_norm = MagicMock(spec=paddle.nn.Layer)
        mock_build.return_value = mock_norm

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock(), MagicMock()],
            layer_norm=MagicMock(),
        )

        block = TransformerBlock(
            config=config,
            spec=spec,
            post_layer_norm=True,
            post_process=True,
        )
        self.assertIsNotNone(block.norm)

    def test_vp_stage_raises(self):
        """Test that vp_stage raises assertion error."""
        config = _make_config()
        spec = TransformerBlockSublayersSpec()

        with self.assertRaises(AssertionError):
            TransformerBlock(config=config, spec=spec, vp_stage=1)


class TestTransformerBlockMethods(unittest.TestCase):
    """Tests for TransformerBlock methods."""

    def _make_block(self):
        with (
            patch(
                "paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups"
            ) as mock_pg,
            patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer") as mock_build,
        ):
            mock_pg.return_value = MagicMock()
            # build_spec_layer must return paddle.nn.Layer instances for LayerList
            mock_build.return_value = MagicMock(spec=paddle.nn.Layer)

            config = _make_config()
            spec = TransformerBlockSublayersSpec(
                layer_specs=[MagicMock(), MagicMock()],
                layer_norm=None,
            )
            block = TransformerBlock(config=config, spec=spec)
            return block

    def test_get_layer(self):
        block = self._make_block()
        layer = block._get_layer(0)
        self.assertIsNotNone(layer)

    def test_set_input_tensor(self):
        block = self._make_block()
        tensor = paddle.randn([2, 4, 64])
        block.set_input_tensor(tensor)
        self.assertEqual(block.input_tensor.shape, [2, 4, 64])

    def test_num_layers_per_pipeline_rank(self):
        block = self._make_block()
        self.assertEqual(block.num_layers_per_pipeline_rank, 2)


if __name__ == "__main__":
    unittest.main()
