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

from paddleformers.fleet.transformer.block_attn_res import (
    BlockAttnRes,
    BlockAttnResSublayersSpec,
)
from paddleformers.fleet.transformer.paddle_norm import WrappedPaddleNorm
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestBlockAttnResSublayersSpec(unittest.TestCase):
    """Tests for BlockAttnResSublayersSpec dataclass."""

    def test_default_norm(self):
        """Test default norm is IdentityOp."""
        from paddleformers.fleet.transformer.identity_op import IdentityOp

        spec = BlockAttnResSublayersSpec()
        self.assertEqual(spec.norm, IdentityOp)

    def test_custom_norm(self):
        """Test custom norm."""
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)
        self.assertEqual(spec.norm, WrappedPaddleNorm)


class TestBlockAttnResConstruction(unittest.TestCase):
    """Tests for BlockAttnRes construction."""

    @patch("paddleformers.fleet.transformer.block_attn_res.build_spec_layer")
    def test_construction_basic(self, mock_build):
        """Test basic construction of BlockAttnRes."""
        mock_build.return_value = MagicMock()
        config = _make_config()
        spec = BlockAttnResSublayersSpec()

        block = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertEqual(block.hidden_size, 64)
        self.assertIsNotNone(block.proj_weight)
        self.assertEqual(block.proj_weight.shape, [64])

    @patch("paddleformers.fleet.transformer.block_attn_res.build_spec_layer")
    def test_proj_weight_initialized_to_zero(self, mock_build):
        """Test proj_weight is initialized to zero."""
        mock_build.return_value = MagicMock()
        config = _make_config()
        spec = BlockAttnResSublayersSpec()

        block = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertTrue(
            paddle.allclose(block.proj_weight, paddle.zeros([64])).item()
        )


class TestBlockAttnResForward(unittest.TestCase):
    """Tests for BlockAttnRes forward."""

    @patch("paddleformers.fleet.transformer.block_attn_res.build_spec_layer")
    def test_forward_with_single_block(self, mock_build):
        """Test forward with a single completed block."""
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([1, 2, 4, 64])
        mock_build.return_value = mock_norm

        config = _make_config()
        spec = BlockAttnResSublayersSpec()
        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = [paddle.randn([2, 4, 64])]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [2, 4, 64])

    @patch("paddleformers.fleet.transformer.block_attn_res.build_spec_layer")
    def test_forward_with_multiple_blocks(self, mock_build):
        """Test forward with multiple completed blocks."""
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([3, 2, 4, 64])
        mock_build.return_value = mock_norm

        config = _make_config()
        spec = BlockAttnResSublayersSpec()
        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = [paddle.randn([2, 4, 64]), paddle.randn([2, 4, 64])]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [2, 4, 64])

    @patch("paddleformers.fleet.transformer.block_attn_res.build_spec_layer")
    def test_forward_with_no_completed_blocks(self, mock_build):
        """Test forward with no completed blocks (only partial)."""
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([1, 2, 4, 64])
        mock_build.return_value = mock_norm

        config = _make_config()
        spec = BlockAttnResSublayersSpec()
        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = []

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [2, 4, 64])


class TestBlockAttnResWithRealNorm(unittest.TestCase):
    """Tests for BlockAttnRes with real normalization layer."""

    def test_forward_with_rmsnorm(self):
        """Test forward with real RMSNorm."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = [paddle.randn([2, 4, 64])]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_forward_output_dtype_matches_input(self):
        """Test that output dtype matches partial_block dtype."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64]).cast("float32")
        blocks = [paddle.randn([2, 4, 64]).cast("float32")]

        output = block(partial_block, blocks)
        self.assertEqual(output.dtype, partial_block.dtype)


if __name__ == "__main__":
    unittest.main()
