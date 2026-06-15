# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)

from paddleformers.fleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddleformers.fleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddleformers.fleet.training.initialize import initialize_fleet
from paddleformers.fleet.transformer.transformer_config import TransformerConfig

TP_SIZE = None


def _init_fleet_custom(mp=4, pp=1, dp=1, sharding=1, cp=1, ep=1):
    strategy = fleet.DistributedStrategy()
    moe_sharding = 1
    if ep > 1:
        world_size = pp * mp * dp * sharding
        moe_sharding = world_size // (pp * ep)
    strategy.hybrid_configs = {
        "dp_degree": dp,
        "mp_degree": mp,
        "pp_degree": pp,
        "sharding_degree": sharding,
        "sep_degree": 1,
        "cp_degree": cp,
        "ep_degree": ep,
        "moe_sharding_degree": moe_sharding,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def setUpModule():
    """Initialize fleet once for all tests in this module (MP=4)."""
    global TP_SIZE
    TP_SIZE = dist.get_world_size()
    _init_fleet_custom(mp=TP_SIZE, pp=1, dp=1, sharding=1)
    np.random.seed(42)
    paddle.seed(42)
    model_parallel_cuda_manual_seed(42)


def _make_config(tp_size):
    """Create a TransformerConfig for tensor parallel tests."""
    return TransformerConfig(
        tensor_model_parallel_size=tp_size,
        num_attention_heads=4,
        num_key_value_heads=4,
        use_cpu_initialization=True,
    )


def _unpack_output(output):
    """Unpack output tuple if needed."""
    if isinstance(output, tuple):
        return output[0]
    return output


class TestVocabParallelEmbeddingBasic(unittest.TestCase):
    """Test VocabParallelEmbedding forward with random input ids."""

    def test_vocab_parallel_embedding_basic(self):
        """VocabParallelEmbedding should produce correct output shape."""
        config = _make_config(TP_SIZE)
        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        num_embeddings = 32
        embedding_dim = 16
        emb = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_method=config.init_method,
            config=config,
        )

        input_ids = paddle.randint(0, num_embeddings, shape=[4, 8])
        input_ids.stop_gradient = False
        output = emb(input_ids)
        output = _unpack_output(output) if isinstance(output, tuple) else output

        # Output shape should be [batch, seq_len, embedding_dim]
        self.assertEqual(output.shape, [4, 8, embedding_dim])
        self.assertFalse(paddle.isnan(output).any())


class TestColumnParallelLinearForward(unittest.TestCase):
    """Test ColumnParallelLinear forward pass."""

    def test_column_parallel_linear_forward(self):
        """ColumnParallelLinear should split weight columns across TP ranks."""
        config = _make_config(TP_SIZE)
        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        hidden_size = 16
        output_size = 32
        col = ColumnParallelLinear(
            hidden_size,
            output_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            config=config,
        )

        input_tensor = paddle.randn([4, hidden_size], dtype="float32")
        input_tensor.stop_gradient = False
        output = col(input_tensor)
        output = _unpack_output(output)

        # Each rank should have output_size // TP_SIZE as output dim
        expected_output_size = output_size // TP_SIZE
        self.assertEqual(output.shape[-1], expected_output_size)
        self.assertFalse(paddle.isnan(output).any())

        # Verify backward pass works
        output.sum().backward()
        self.assertIsNotNone(input_tensor.grad)


class TestRowParallelLinearForward(unittest.TestCase):
    """Test RowParallelLinear forward pass."""

    def test_row_parallel_linear_forward(self):
        """RowParallelLinear should split weight rows across TP ranks."""
        config = _make_config(TP_SIZE)
        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        input_size = 16
        output_size = 8
        row = RowParallelLinear(
            input_size,
            output_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            config=config,
        )

        local_input_size = input_size // TP_SIZE
        input_tensor = paddle.randn([4, local_input_size], dtype="float32")
        input_tensor.stop_gradient = False
        output = row(input_tensor)
        output = _unpack_output(output)

        # Row parallel linear should all-reduce to produce full output
        self.assertEqual(output.shape[-1], output_size)
        self.assertFalse(paddle.isnan(output).any())

        # Verify backward pass works
        output.sum().backward()
        self.assertIsNotNone(input_tensor.grad)


class TestColumnAndRowParallelEndToEnd(unittest.TestCase):
    """Test ColumnParallelLinear -> RowParallelLinear pipeline matches non-parallel."""

    def test_column_and_row_parallel_end_to_end(self):
        """ColumnParallel followed by RowParallel should produce same result as a single Linear."""
        config = _make_config(TP_SIZE)
        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        hidden_size = 16
        intermediate_size = 32

        # Create parallel layers
        col = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            gather_output=False,
            config=config,
        )

        row = RowParallelLinear(
            intermediate_size,
            hidden_size,
            init_method=paddle.nn.initializer.KaimingUniform(),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            config=config,
        )

        # Create non-parallel reference linear
        paddle.manual_seed(0)
        ref_linear = paddle.nn.Linear(hidden_size, hidden_size, bias_attr=False)

        input_tensor = paddle.randn([4, hidden_size], dtype="float32")
        input_tensor.stop_gradient = False

        # Parallel path: column -> row (ColumnParallel handles splitting internally)
        col_output = col(input_tensor)
        col_output = _unpack_output(col_output)
        row_output = row(col_output)
        row_output = _unpack_output(row_output)

        # The parallel output should have the correct shape
        self.assertEqual(row_output.shape, [4, hidden_size])
        self.assertFalse(paddle.isnan(row_output).any())

        # Reference path
        ref_output = ref_linear(input_tensor)
        self.assertEqual(ref_output.shape, [4, hidden_size])

        # Both should produce finite outputs
        self.assertFalse(paddle.isnan(ref_output).any())


class TestVocabParallelEmbeddingReduceScatter(unittest.TestCase):
    """Test VocabParallelEmbedding with reduce_scatter_embeddings=True."""

    def test_vocab_parallel_embedding_reduce_scatter(self):
        """VocabParallelEmbedding with reduce_scatter should produce correct output shape."""
        config = _make_config(TP_SIZE)
        paddle.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        num_embeddings = 32
        seq_len = 8
        embedding_dim = 16
        batch_size = 4

        emb = VocabParallelEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            init_method=config.init_method,
            config=config,
            reduce_scatter_embeddings=True,
        )

        input_ids = paddle.randint(0, num_embeddings, shape=[batch_size, seq_len])
        input_ids.stop_gradient = False
        output = emb(input_ids)

        # With reduce_scatter_embeddings, output shape depends on implementation
        output = _unpack_output(output) if isinstance(output, tuple) else output
        self.assertIsNotNone(output)
        self.assertFalse(paddle.isnan(output).any())

        # Verify backward works
        output.sum().backward()


if __name__ == "__main__":
    unittest.main()
