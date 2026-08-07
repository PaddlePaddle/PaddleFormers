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

from paddleformers.fleet.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSublayersSpec,
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


class TestTransformerBlockForward(unittest.TestCase):
    """Tests for TransformerBlock forward method."""

    @patch(
        "paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups"
    )
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_forward_with_post_layer_norm_true(self, mock_build, mock_pg):
        """Test TransformerBlock construction with post_layer_norm sets norm."""
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

    @patch(
        "paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups"
    )
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_set_input_tensor_stores_tensor(self, mock_build, mock_pg):
        """Test set_input_tensor stores the input tensor."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock(spec=paddle.nn.Layer)

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock(), MagicMock()],
            layer_norm=None,
        )
        block = TransformerBlock(config=config, spec=spec, pre_process=False)

        tensor = paddle.randn([2, 4, 64])
        block.set_input_tensor(tensor)
        self.assertEqual(block.input_tensor.shape, [2, 4, 64])

    @patch(
        "paddleformers.fleet.transformer.transformer_block.ProcessGroupCollection.use_mpu_process_groups"
    )
    @patch("paddleformers.fleet.transformer.transformer_block.build_spec_layer")
    def test_num_layers_matches_spec(self, mock_build, mock_pg):
        """Test num_layers_per_pipeline_rank matches spec."""
        mock_pg.return_value = MagicMock()
        mock_build.return_value = MagicMock(spec=paddle.nn.Layer)

        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[MagicMock(), MagicMock(), MagicMock()],
            layer_norm=None,
        )
        block = TransformerBlock(config=config, spec=spec)
        self.assertEqual(block.num_layers_per_pipeline_rank, 3)


class TestTransformerBlockCPUOffloadingAssert(unittest.TestCase):
    """Tests for TransformerBlock CPU offloading assertion."""

    def test_cpu_offloading_true_raises(self):
        """Test that cpu_offloading=True raises assertion."""
        config = _make_config(cpu_offloading=True)
        spec = TransformerBlockSublayersSpec()

        with self.assertRaises(AssertionError):
            TransformerBlock(config=config, spec=spec)


if __name__ == "__main__":
    unittest.main()
