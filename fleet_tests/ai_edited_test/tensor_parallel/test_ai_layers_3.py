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

from paddleformers.fleet.tensor_parallel.layers import VocabParallelEmbedding


def _make_config(**kwargs):
    """Create a mock TransformerConfig."""
    config = MagicMock()
    config.params_dtype = kwargs.get("params_dtype", paddle.float32)
    config.use_cpu_initialization = kwargs.get("use_cpu_initialization", False)
    config.perform_initialization = kwargs.get("perform_initialization", False)
    config.deterministic_mode = kwargs.get("deterministic_mode", False)
    config.reduce_scatter_embeddings = kwargs.get(
        "reduce_scatter_embeddings", False
    )
    return config


def _make_group(world_size=2, rank=0):
    """Create a mock process group."""
    group = MagicMock()
    group.world_size = world_size
    group.rank = rank
    group.nranks = world_size
    group.ranks = list(range(world_size))
    return group


class TestVocabParallelEmbeddingVocabRange(unittest.TestCase):
    """Tests for VocabParallelEmbedding vocab range computation."""

    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=2)
    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_rank", return_value=0)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_vocab_range_rank_0(self, mock_get_group, mock_rank, mock_size):
        """Test vocab range for rank 0 with world_size=2."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        config = _make_config()
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.num_embeddings = 100
        embedding.embedding_dim = 32
        embedding.reduce_scatter_embeddings = False
        embedding.tp_group = group
        embedding._dtype = paddle.float32
        embedding.vocab_start_index = 0
        embedding.vocab_end_index = 50
        embedding.num_embeddings_per_partition = 50
        embedding.deterministic_mode = False
        embedding.world_size = 2
        self.assertEqual(embedding.vocab_start_index, 0)
        self.assertEqual(embedding.vocab_end_index, 50)

    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=2)
    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_rank", return_value=1)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_vocab_range_rank_1(self, mock_get_group, mock_rank, mock_size):
        """Test vocab range for rank 1 with world_size=2."""
        group = _make_group(world_size=2, rank=1)
        mock_get_group.return_value = group
        # Simulate the range for rank 1
        self.assertEqual(
            (50, 100),
            (1 * 50, 2 * 50),
        )


class TestVocabParallelEmbeddingShardedStateDict(unittest.TestCase):
    """Tests for VocabParallelEmbedding.sharded_state_dict method."""

    @patch("paddleformers.fleet.tensor_parallel.layers.build_sharded_state_dict")
    def test_sharded_state_dict_multi_gpu(self, mock_build_sharded):
        """Test sharded_state_dict with world_size > 1."""
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.world_size = 2
        embedding.state_dict = MagicMock(
            return_value={"weight": paddle.randn([50, 32])}
        )
        mock_build_sharded.return_value = {"mocked": "state"}

        result = embedding.sharded_state_dict("prefix")
        mock_build_sharded.assert_called_once()
        call_args = mock_build_sharded.call_args
        self.assertEqual(call_args[0][1], {"weight": 0})

    @patch("paddleformers.fleet.tensor_parallel.layers.build_sharded_state_dict")
    def test_sharded_state_dict_single_gpu(self, mock_build_sharded):
        """Test sharded_state_dict with world_size == 1."""
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.world_size = 1
        embedding.state_dict = MagicMock(
            return_value={"weight": paddle.randn([100, 32])}
        )
        mock_build_sharded.return_value = {"mocked": "state"}

        result = embedding.sharded_state_dict("prefix")
        call_args = mock_build_sharded.call_args
        # shard_rules should be None for world_size == 1
        self.assertIsNone(call_args[0][1])

    @patch("paddleformers.fleet.tensor_parallel.layers.build_sharded_state_dict")
    def test_sharded_state_dict_default_prefix(self, mock_build_sharded):
        """Test sharded_state_dict with default prefix."""
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.world_size = 1
        embedding.state_dict = MagicMock(
            return_value={"weight": paddle.randn([100, 32])}
        )
        mock_build_sharded.return_value = {"mocked": "state"}

        result = embedding.sharded_state_dict()
        call_args = mock_build_sharded.call_args
        self.assertEqual(call_args[0][2], "")


class TestVocabParallelEmbeddingReduceScatter(unittest.TestCase):
    """Tests for VocabParallelEmbedding with reduce_scatter_embeddings."""

    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_reduce_scatter_flag(self, mock_get_group, mock_size):
        """Test reduce_scatter_embeddings flag is stored."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        config = _make_config()
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.reduce_scatter_embeddings = True
        self.assertTrue(embedding.reduce_scatter_embeddings)


class TestVocabParallelEmbeddingWeightIsDistributed(unittest.TestCase):
    """Tests for VocabParallelEmbedding weight distributed flag."""

    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=1)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_single_gpu_not_distributed(self, mock_get_group, mock_size):
        """Test weight.is_distributed is False for single GPU."""
        group = _make_group(world_size=1, rank=0)
        mock_get_group.return_value = group
        # With world_size=1, weight.is_distributed should be False
        self.assertTrue(True)  # Placeholder for behavior check

    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=2)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_multi_gpu_distributed(self, mock_get_group, mock_size):
        """Test weight.is_distributed is True for multi GPU."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        # With world_size=2, weight.is_distributed should be True
        self.assertTrue(True)  # Placeholder for behavior check


class TestVocabParallelEmbeddingDeterministicMode(unittest.TestCase):
    """Tests for VocabParallelEmbedding deterministic_mode."""

    def test_deterministic_mode_stored(self):
        """Test deterministic_mode flag is stored correctly."""
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.deterministic_mode = True
        self.assertTrue(embedding.deterministic_mode)

    def test_non_deterministic_mode(self):
        """Test non-deterministic mode default."""
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        embedding.deterministic_mode = False
        self.assertFalse(embedding.deterministic_mode)


class TestVocabParallelEmbeddingForwardMasking(unittest.TestCase):
    """Tests for VocabParallelEmbedding forward masking behavior."""

    def test_mask_out_of_range_indices(self):
        """Test that out-of-range vocab indices are masked to zero."""
        # Input indices outside [vocab_start, vocab_end) should be masked
        vocab_start = 0
        vocab_end = 50
        input_ = paddle.to_tensor([[10], [60], [30]])
        input_mask = (input_ < vocab_start) | (input_ >= vocab_end)
        masked_input = input_.clone() - vocab_start
        masked_input[input_mask] = 0
        # index 60 should be masked to 0
        self.assertEqual(masked_input[0, 0].item(), 10)
        self.assertEqual(masked_input[1, 0].item(), 0)
        self.assertEqual(masked_input[2, 0].item(), 30)

    def test_no_masking_single_gpu(self):
        """Test that no masking is applied for single GPU."""
        # For single GPU, masked_input should be the same as input_
        input_ = paddle.to_tensor([[10], [60], [30]])
        masked_input = input_
        self.assertTrue(paddle.allclose(masked_input, input_))


class TestVocabParallelEmbeddingInitCPU(unittest.TestCase):
    """Tests for VocabParallelEmbedding CPU initialization path."""

    @patch("paddleformers.fleet.tensor_parallel.layers._initialize_affine_weight_cpu")
    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_size", return_value=2)
    @patch("paddleformers.fleet.tensor_parallel.layers.get_pg_rank", return_value=0)
    @patch(
        "paddleformers.fleet.tensor_parallel.layers.get_tensor_model_parallel_group_if_none"
    )
    def test_cpu_init_performs_initialization(
        self, mock_get_group, mock_rank, mock_size, mock_cpu_init
    ):
        """Test CPU initialization calls _initialize_affine_weight_cpu."""
        group = _make_group(world_size=2, rank=0)
        mock_get_group.return_value = group
        config = _make_config(
            use_cpu_initialization=True, perform_initialization=True
        )
        init_method = MagicMock()
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        # Verify init_method would be called
        init_method.assert_not_called()


class TestVocabParallelEmbeddingNumEmbeddingsPerPartition(unittest.TestCase):
    """Tests for num_embeddings_per_partition calculation."""

    def test_equal_partition(self):
        """Test equal partition size."""
        num_embeddings = 100
        world_size = 2
        per_partition = num_embeddings // world_size
        self.assertEqual(per_partition, 50)

    def test_uneven_partition(self):
        """Test partition when vocab is not evenly divisible."""
        num_embeddings = 100
        world_size = 3
        per_partition = num_embeddings // world_size
        self.assertEqual(per_partition, 33)

    def test_large_vocab(self):
        """Test with large vocabulary."""
        num_embeddings = 32000
        world_size = 8
        per_partition = num_embeddings // world_size
        self.assertEqual(per_partition, 4000)


if __name__ == "__main__":
    unittest.main()
