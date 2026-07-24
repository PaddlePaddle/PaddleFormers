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


class TestLanguageModelEmbeddingInit(unittest.TestCase):
    """Test LanguageModelEmbedding initialization."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_position_embedding(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_tp.VocabParallelEmbedding.return_value = MagicMock()

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="learned_absolute",
        )
        self.assertTrue(emb.add_position_embedding)
        self.assertTrue(hasattr(emb, "position_embeddings"))

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_rope(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_tp.VocabParallelEmbedding.return_value = MagicMock()

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )
        self.assertFalse(emb.add_position_embedding)
        self.assertFalse(hasattr(emb, "position_embeddings"))

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_tokentypes(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_tp.VocabParallelEmbedding.return_value = MagicMock()

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="learned_absolute",
            num_tokentypes=2,
        )
        self.assertIsNotNone(emb.tokentype_embeddings)
        self.assertEqual(emb.num_tokentypes, 2)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_without_tokentypes(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_tp.VocabParallelEmbedding.return_value = MagicMock()

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="learned_absolute",
        )
        self.assertIsNone(emb.tokentype_embeddings)


class TestLanguageModelEmbeddingEmbeddingWeight(unittest.TestCase):
    """Test LanguageModelEmbedding.embedding_weight property."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_returns_embed_tokens_weight(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_embed = MagicMock()
        mock_embed.weight = MagicMock()
        mock_tp.VocabParallelEmbedding.return_value = mock_embed

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
        )
        self.assertEqual(emb.embedding_weight, mock_embed.weight)


class TestLanguageModelEmbeddingReduceScatter(unittest.TestCase):
    """Test reduce_scatter_embeddings logic."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_reduce_scatter_with_rope_and_sp(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = True
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_embed = MagicMock()
        mock_tp.VocabParallelEmbedding.return_value = mock_embed
        mock_tp.get_tensor_model_parallel_group_if_none.return_value = (
            MagicMock()
        )

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="rope",
        )
        # With rope, no position embedding -> reduce_scatter should be True
        self.assertTrue(emb.reduce_scatter_embeddings)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_no_reduce_scatter_with_position_embedding(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.hidden_dropout_prob = 0.1
        mock_config.sequence_parallel = True
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_embed = MagicMock()
        mock_tp.VocabParallelEmbedding.return_value = mock_embed
        mock_tp.get_tensor_model_parallel_group_if_none.return_value = (
            MagicMock()
        )

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=32000,
            max_sequence_length=2048,
            position_embedding_type="learned_absolute",
        )
        # With learned_absolute position embedding, reduce_scatter should be False
        self.assertFalse(emb.reduce_scatter_embeddings)


class TestLanguageModelEmbeddingForward(unittest.TestCase):
    """Test LanguageModelEmbedding forward method."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_basic(self, mock_tp):
        import paddle

        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.hidden_dropout_prob = 0.0
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_embed = MagicMock()
        mock_embed.return_value = paddle.randn([2, 10, 64])
        mock_tp.VocabParallelEmbedding.return_value = mock_embed

        emb = LanguageModelEmbedding(
            config=mock_config,
            vocab_size=100,
            max_sequence_length=64,
            position_embedding_type="rope",
        )
        input_ids = paddle.randint(0, 100, [2, 10])
        position_ids = paddle.arange(10).unsqueeze(0).expand([2, -1])
        result = emb(input_ids, position_ids)
        self.assertIsNotNone(result)


class TestLanguageModelEmbeddingZeroParameters(unittest.TestCase):
    """Test LanguageModelEmbedding.zero_parameters method."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_zero_parameters_with_position_and_tokentype(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.hidden_dropout_prob = 0.0
        mock_config.sequence_parallel = False
        mock_config.embedding_init_method = MagicMock()
        mock_config.perform_initialization = False
        mock_config.fp32_residual_connection = False
        mock_config.clone_scatter_output_in_embedding = False

        mock_embed = MagicMock()
        mock_embed_weight = MagicMock()
        mock_embed_weight.data = MagicMock()
        mock_embed_weight.shared = False
        mock_embed.weight = mock_embed_weight
        mock_tp.VocabParallelEmbedding.return_value = mock_embed

        # Mock paddle.nn.Embedding to return mock embeddings for position/tokentype
        mock_pos_emb = MagicMock()
        mock_pos_emb.weight = MagicMock()
        mock_pos_emb.weight.data = MagicMock()
        mock_pos_emb.weight.shared = False
        mock_tt_emb = MagicMock()
        mock_tt_emb.weight = MagicMock()
        mock_tt_emb.weight.data = MagicMock()
        mock_tt_emb.weight.shared = False

        with patch(
            "paddle.nn.Embedding", side_effect=[mock_pos_emb, mock_tt_emb]
        ):
            emb = LanguageModelEmbedding(
                config=mock_config,
                vocab_size=100,
                max_sequence_length=64,
                position_embedding_type="learned_absolute",
                num_tokentypes=2,
            )

        emb.zero_parameters()
        mock_embed_weight.data.fill_.assert_called_with(0)
        mock_pos_emb.weight.data.fill_.assert_called_with(0)
        mock_tt_emb.weight.data.fill_.assert_called_with(0)


class TestLanguageModelEmbeddingSequenceParallelAssert(unittest.TestCase):
    """Test that sequence_parallel requires scatter_to_sequence_parallel."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_sp_without_scatter_raises(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.sequence_parallel = True

        mock_tp.VocabParallelEmbedding.return_value = MagicMock()

        with self.assertRaises(AssertionError):
            LanguageModelEmbedding(
                config=mock_config,
                vocab_size=100,
                max_sequence_length=64,
                scatter_to_sequence_parallel=False,
            )


if __name__ == "__main__":
    unittest.main()
