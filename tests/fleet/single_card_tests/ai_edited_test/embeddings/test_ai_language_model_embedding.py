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


def _make_config(**overrides):
    """Helper to create a TransformerConfig with sensible defaults."""
    from paddleformers.fleet.transformer.transformer_config import (
        TransformerConfig,
    )

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "hidden_dropout_prob": 0.0,
        "perform_initialization": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestLanguageModelEmbeddingInit(unittest.TestCase):
    """Tests for LanguageModelEmbedding initialization."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.get_tensor_model_parallel_group_if_none",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_learned_absolute(self, mock_tp, mock_get_group):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="learned_absolute",
        )
        self.assertTrue(emb.add_position_embedding)
        self.assertFalse(emb.reduce_scatter_embeddings)
        self.assertEqual(emb.vocab_size, 1000)
        self.assertEqual(emb.max_sequence_length, 128)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.get_tensor_model_parallel_group_if_none",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_rope(self, mock_tp, mock_get_group):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="rope",
        )
        self.assertFalse(emb.add_position_embedding)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.get_tensor_model_parallel_group_if_none",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_with_tokentypes(self, mock_tp, mock_get_group):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            num_tokentypes=2,
        )
        self.assertIsNotNone(emb.tokentype_embeddings)
        self.assertEqual(emb.num_tokentypes, 2)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.get_tensor_model_parallel_group_if_none",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_init_no_tokentypes(self, mock_tp, mock_get_group):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        )
        self.assertIsNone(emb.tokentype_embeddings)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.get_tensor_model_parallel_group_if_none",
        return_value=None,
    )
    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_sequence_parallel_requires_scatter(self, mock_tp, mock_get_group):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        mock_tp.VocabParallelEmbedding.return_value = MagicMock()
        with self.assertRaises(AssertionError):
            LanguageModelEmbedding(
                config=config,
                vocab_size=1000,
                max_sequence_length=128,
                scatter_to_sequence_parallel=False,
            )


class TestLanguageModelEmbeddingForward(unittest.TestCase):
    """Tests for LanguageModelEmbedding forward pass."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_with_position_embedding(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(hidden_dropout_prob=0.0)
        # Create a real VocabParallelEmbedding mock that returns real embeddings
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="learned_absolute",
        )
        position_ids = paddle.randint(0, 128, [2, 8])
        # Replace the position embedding weight to get deterministic output
        emb.position_embeddings.weight.set_value(
            paddle.randn([128, 64], dtype="float32")
        )

        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_without_position_embedding(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(hidden_dropout_prob=0.0)
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="rope",
        )
        position_ids = paddle.randint(0, 128, [2, 8])

        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_with_tokentype_ids(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(hidden_dropout_prob=0.0)
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            num_tokentypes=2,
        )
        position_ids = paddle.randint(0, 128, [2, 8])
        tokentype_ids = paddle.zeros([2, 8], dtype="int64")

        result = emb(input_ids, position_ids, tokentype_ids=tokentype_ids)
        self.assertEqual(result.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_no_tokentype_ids_with_none_tokentype(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(hidden_dropout_prob=0.0)
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        )
        position_ids = paddle.randint(0, 128, [2, 8])

        # Pass tokentype_ids=None, should work fine when num_tokentypes=0
        result = emb(input_ids, position_ids, tokentype_ids=None)
        self.assertEqual(result.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_forward_with_fp32_residual(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(
            hidden_dropout_prob=0.0,
            fp32_residual_connection=True,
        )
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="rope",
        )
        position_ids = paddle.randint(0, 128, [2, 8])

        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [2, 8, 64])
        self.assertEqual(result.dtype, paddle.float32)


class TestLanguageModelEmbeddingSequenceParallel(unittest.TestCase):
    """Tests for LanguageModelEmbedding with sequence_parallel=True."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_sequence_parallel_scatter_and_clone(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            hidden_dropout_prob=0.0,
            clone_scatter_output_in_embedding=True,
        )
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        # Mock scatter_to_sequence_parallel_region to return identity
        mock_tp.scatter_to_sequence_parallel_region = MagicMock(
            side_effect=lambda x, **kw: x
        )

        # Mock get_cuda_rng_tracker
        mock_tracker = MagicMock()
        mock_tracker.fork.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="rope",
        )
        position_ids = paddle.randint(0, 128, [2, 8])

        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [2, 8, 64])

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_sequence_parallel_reduce_scatter_enabled(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            hidden_dropout_prob=0.0,
        )
        mock_vocab_emb = MagicMock()
        input_ids = paddle.randint(0, 100, [2, 8])
        mock_embeddings = paddle.randn([2, 8, 64], dtype="float32")
        mock_vocab_emb.return_value = mock_embeddings
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        mock_tracker = MagicMock()
        mock_tracker.fork.return_value = MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
        mock_tp.get_cuda_rng_tracker.return_value = mock_tracker

        # When reduce_scatter_embeddings is True, scatter is not called
        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            position_embedding_type="rope",
            num_tokentypes=0,
        )
        # reduce_scatter_embeddings should be True when:
        # not add_position_embedding and num_tokentypes <= 0 and sequence_parallel and scatter_to_sequence_parallel
        self.assertTrue(emb.reduce_scatter_embeddings)

        position_ids = paddle.randint(0, 128, [2, 8])
        result = emb(input_ids, position_ids)
        self.assertEqual(result.shape, [2, 8, 64])


class TestLanguageModelEmbeddingProperties(unittest.TestCase):
    """Tests for properties and utility methods."""

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_embedding_weight_property(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_vocab_emb = MagicMock()
        mock_weight = paddle.randn([1000, 64], dtype="float32")
        mock_vocab_emb.weight = mock_weight
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        )
        self.assertIs(emb.embedding_weight, mock_weight)

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_zero_parameters(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_vocab_emb = MagicMock()
        mock_weight = paddle.randn([1000, 64], dtype="float32")
        mock_vocab_emb.weight = mock_weight
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
            num_tokentypes=2,
        )
        # Initialize position embeddings weight
        emb.position_embeddings.weight.set_value(
            paddle.randn([128, 64], dtype="float32")
        )
        emb.tokentype_embeddings.weight.set_value(
            paddle.randn([2, 64], dtype="float32")
        )

        emb.zero_parameters()

        # Verify all weights are zero
        self.assertTrue(
            paddle.allclose(
                emb.embed_tokens.weight.data, paddle.zeros_like(mock_weight)
            )
        )
        self.assertTrue(
            paddle.allclose(
                emb.position_embeddings.weight.data, paddle.zeros([128, 64])
            )
        )

    @patch(
        "paddleformers.fleet.models.common.embeddings.language_model_embedding.tensor_parallel"
    )
    def test_zero_parameters_no_tokentypes(self, mock_tp):
        from paddleformers.fleet.models.common.embeddings.language_model_embedding import (
            LanguageModelEmbedding,
        )

        config = _make_config()
        mock_vocab_emb = MagicMock()
        mock_weight = paddle.randn([1000, 64], dtype="float32")
        mock_vocab_emb.weight = mock_weight
        mock_tp.VocabParallelEmbedding.return_value = mock_vocab_emb

        emb = LanguageModelEmbedding(
            config=config,
            vocab_size=1000,
            max_sequence_length=128,
        )
        emb.position_embeddings.weight.set_value(
            paddle.randn([128, 64], dtype="float32")
        )

        emb.zero_parameters()
        self.assertTrue(
            paddle.allclose(
                emb.embed_tokens.weight.data, paddle.zeros_like(mock_weight)
            )
        )


class TestLanguageModelEmbeddingWithTokentypeAndSequenceParallel(
    unittest.TestCase
):
    """Test tokentype embedding with sequence parallel."""
