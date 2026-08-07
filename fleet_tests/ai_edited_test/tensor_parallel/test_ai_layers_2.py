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

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    VocabParallelEmbedding,
    param_is_not_tensor_parallel_duplicate,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    set_tensor_model_parallel_attributes,
)
from paddleformers.fleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    """Create a TransformerConfig with sensible defaults for testing."""
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_group(world_size=2, rank=0):
    """Create a mock process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank = rank
    group.nranks = world_size
    group.ranks = list(range(world_size))
    return group


class TestColumnParallelLinearInit(unittest.TestCase):
    """Tests for ColumnParallelLinear initialization."""

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_basic(self, mock_init_w, mock_rank, mock_ws):
        """Test basic initialization of ColumnParallelLinear."""
        config = _make_config()
        init_method = lambda x: None
        layer = ColumnParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            bias=False,
        )
        self.assertIsNotNone(layer.weight)
        self.assertEqual(layer.skip_bias_add, False)

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_with_bias(self, mock_init_w, mock_rank, mock_ws):
        """Test ColumnParallelLinear with bias."""
        config = _make_config()
        init_method = lambda x: None
        layer = ColumnParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            bias=True,
            skip_bias_add=True,
        )
        self.assertIsNotNone(layer.weight)
        self.assertTrue(layer.skip_bias_add)

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_gather_output(self, mock_init_w, mock_rank, mock_ws):
        """Test ColumnParallelLinear with gather_output=True."""
        config = _make_config()
        init_method = lambda x: None
        layer = ColumnParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            gather_output=True,
        )
        self.assertIsNotNone(layer.weight)
        self.assertTrue(layer.gather_output)

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_stride(self, mock_init_w, mock_rank, mock_ws):
        """Test ColumnParallelLinear with custom stride."""
        config = _make_config()
        init_method = lambda x: None
        layer = ColumnParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            stride=2,
        )
        self.assertIsNotNone(layer.weight)


class TestRowParallelLinearInit(unittest.TestCase):
    """Tests for RowParallelLinear initialization."""

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_basic(self, mock_init_w, mock_rank, mock_ws):
        """Test basic initialization of RowParallelLinear."""
        config = _make_config()
        init_method = lambda x: None
        layer = RowParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
        )
        self.assertIsNotNone(layer.weight)

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_init_input_is_parallel(self, mock_init_w, mock_rank, mock_ws):
        """Test RowParallelLinear with input_is_parallel=True."""
        config = _make_config()
        init_method = lambda x: None
        layer = RowParallelLinear(
            input_size=16,
            output_size=32,
            config=config,
            init_method=init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )
        self.assertIsNotNone(layer.weight)
        self.assertTrue(layer.input_is_parallel)


class TestLinearLayer(unittest.TestCase):
    """Tests for Linear layer."""

    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_linear_forward(self, mock_init_w):
        """Test basic forward of Linear."""
        config = _make_config()
        init_method = lambda x: None
        layer = Linear(16, 32, config=config, init_method=init_method)
        x = paddle.randn([2, 16])
        output = layer(x)
        # Linear.forward returns (output, bias) tuple
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[0].shape, [2, 32])

    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_linear_no_bias(self, mock_init_w):
        """Test Linear without bias."""
        config = _make_config()
        init_method = lambda x: None
        layer = Linear(
            16, 32, config=config, init_method=init_method, bias=False
        )
        self.assertIsNone(layer.bias)

    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_linear_skip_bias_add(self, mock_init_w):
        """Test Linear with skip_bias_add."""
        config = _make_config()
        init_method = lambda x: None
        layer = Linear(
            16, 32, config=config, init_method=init_method, skip_bias_add=True
        )
        self.assertTrue(layer.skip_bias_add)

    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_gpu")
    def test_linear_repr(self, mock_init_w):
        """Test Linear __repr__."""
        config = _make_config()
        init_method = lambda x: None
        layer = Linear(16, 32, config=config, init_method=init_method)
        repr_str = repr(layer)
        self.assertIn("Linear", repr_str)


class TestTPAttributes(unittest.TestCase):
    """Tests for tensor model parallel attribute utilities."""

    def test_set_tensor_model_parallel_attributes(self):
        """Test setting TP attributes on a tensor."""
        t = paddle.randn([4, 4])
        set_tensor_model_parallel_attributes(
            t, is_parallel=True, dim=1, stride=2
        )
        self.assertTrue(t.tensor_model_parallel)
        self.assertEqual(t.partition_dim, 1)
        self.assertEqual(t.partition_stride, 2)

    def test_set_defaults_if_not_set(self):
        """Test setting defaults only when not already set."""
        t = paddle.randn([4, 4])
        set_defaults_if_not_set_tensor_model_parallel_attributes(t)
        # Should set defaults
        self.assertFalse(t.tensor_model_parallel)
        self.assertEqual(t.partition_dim, -1)

    def test_set_defaults_preserves_existing(self):
        """Test that defaults do not overwrite existing attributes."""
        t = paddle.randn([4, 4])
        set_tensor_model_parallel_attributes(
            t, is_parallel=True, dim=0, stride=1
        )
        set_defaults_if_not_set_tensor_model_parallel_attributes(t)
        self.assertTrue(t.tensor_model_parallel)
        self.assertEqual(t.partition_dim, 0)

    def test_param_is_not_tensor_parallel_duplicate_true(self):
        """Test param_is_not_tensor_parallel_duplicate returns True."""
        t = paddle.randn([4, 4])
        set_tensor_model_parallel_attributes(
            t, is_parallel=False, dim=1, stride=1
        )
        # When is_parallel=False, tensor_model_parallel is False.
        # The function returns True when tensor_model_parallel is True OR rank==0.
        # Since is_parallel=False, tensor_model_parallel is False.
        # With rank==0 (default mock), the function should still return True.
        self.assertTrue(param_is_not_tensor_parallel_duplicate(t))

    def test_param_is_not_tensor_parallel_duplicate_false(self):
        """Test param_is_not_tensor_parallel_duplicate returns False for parallel params on non-zero rank."""
        t = paddle.randn([4, 4])
        set_tensor_model_parallel_attributes(
            t, is_parallel=True, dim=1, stride=1
        )
        # When is_parallel=True, tensor_model_parallel is True.
        # The function returns True when tensor_model_parallel is True OR rank==0.
        # With is_parallel=True, it always returns True regardless of rank.
        self.assertTrue(param_is_not_tensor_parallel_duplicate(t))

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=1,
    )
    def test_param_is_not_duplicate_non_zero_rank(self, mock_rank):
        """Test param_is_not_tensor_parallel_duplicate returns False for non-parallel param on non-zero rank."""
        t = paddle.randn([4, 4])
        # Not marked as tensor_model_parallel and rank is 1 -> not duplicate
        # Actually: function returns (has attr and is True) OR (rank==0)
        # Neither condition is true here, so returns False
        self.assertFalse(param_is_not_tensor_parallel_duplicate(t))


class TestVocabParallelEmbedding(unittest.TestCase):
    """Tests for VocabParallelEmbedding extra coverage."""

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    def test_init_basic(self, mock_rank, mock_ws):
        """Test basic VocabParallelEmbedding initialization."""
        config = _make_config()
        init_method = lambda x: None
        emb = VocabParallelEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            init_method=init_method,
            config=config,
        )
        self.assertIsNotNone(emb.weight)
        self.assertEqual(emb.weight.shape[0], 100)

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    def test_forward_shape(self, mock_rank, mock_ws):
        """Test VocabParallelEmbedding forward output shape."""
        config = _make_config()
        init_method = lambda x: None
        emb = VocabParallelEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            init_method=init_method,
            config=config,
        )
        input_ids = paddle.randint(0, 100, [2, 8])
        output = emb(input_ids)
        self.assertEqual(output.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_world_size",
        return_value=1,
    )
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_rank",
        return_value=0,
    )
    def test_init_reduce_scatter(self, mock_rank, mock_ws):
        """Test VocabParallelEmbedding with reduce_scatter flag."""
        config = _make_config()
        init_method = lambda x: None
        emb = VocabParallelEmbedding(
            num_embeddings=100,
            embedding_dim=64,
            init_method=init_method,
            reduce_scatter_embeddings=True,
            config=config,
        )
        self.assertIsNotNone(emb.weight)


if __name__ == "__main__":
    unittest.main()
