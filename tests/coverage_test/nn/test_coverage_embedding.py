# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest
from unittest.mock import MagicMock

import paddle
import paddle.nn as nn


class TestEmbedding(unittest.TestCase):
    """Tests for paddleformers.nn.embedding.Embedding factory."""

    def _make_config(self, **overrides):
        config = MagicMock()
        config.vocab_size = overrides.get("vocab_size", 100)
        config.hidden_size = overrides.get("hidden_size", 64)
        config.tensor_model_parallel_size = overrides.get("tensor_model_parallel_size", 1)
        config.sequence_parallel = False
        # Make get behave like a real dict-based config
        config_data = {"vocab_size": config.vocab_size, "hidden_size": config.hidden_size}
        config_data.update(overrides)
        config.get = lambda key, default=None: config_data.get(key, default)
        return config

    def test_embedding_create_default_type(self):
        """With tp_size=1, embedding_type should default to 'default'."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config()
        emb = Embedding.create(config)
        self.assertIsInstance(emb, nn.Embedding)

    def test_embedding_create_custom_dims(self):
        """Custom num_embeddings and embedding_dim should be used."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config()
        emb = Embedding.create(config, num_embeddings=200, embedding_dim=128)
        self.assertEqual(emb.weight.shape, [200, 128])

    def test_get_embedding_type_single_gpu(self):
        """get_embedding_type should return 'default' for tp_size=1."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(tensor_model_parallel_size=1)
        self.assertEqual(Embedding.get_embedding_type(config), "default")

    def test_get_embedding_type_multi_gpu(self):
        """get_embedding_type should return 'vocab_parallel' for tp_size>1."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(tensor_model_parallel_size=4)
        self.assertEqual(Embedding.get_embedding_type(config), "vocab_parallel")

    def test_process_kwargs_default(self):
        """process_kwargs for 'default' type should remove mp_group."""
        from paddleformers.nn.embedding import Embedding

        kwargs = {"mp_group": "fake_group", "padding_idx": 0}
        result = Embedding.process_kwargs("default", **kwargs)
        self.assertNotIn("mp_group", result)
        # padding_idx should remain for default type
        self.assertIn("padding_idx", result)

    def test_process_kwargs_vocab_parallel(self):
        """process_kwargs for 'vocab_parallel' type should remove padding_idx and sparse."""
        from paddleformers.nn.embedding import Embedding

        kwargs = {"padding_idx": 0, "sparse": True, "other": 42}
        result = Embedding.process_kwargs("vocab_parallel", **kwargs)
        self.assertNotIn("padding_idx", result)
        self.assertNotIn("sparse", result)
        self.assertEqual(result["other"], 42)

    def test_process_kwargs_vocab_parallel_missing_keys(self):
        """process_kwargs should handle missing keys gracefully via pop(..., None)."""
        from paddleformers.nn.embedding import Embedding

        kwargs = {"other": 42}
        result = Embedding.process_kwargs("vocab_parallel", **kwargs)
        self.assertEqual(result["other"], 42)

    def test_embedding_forward_shape(self):
        """Embedding forward should produce correct output shape."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(vocab_size=50, hidden_size=32)
        emb = Embedding.create(config)
        x = paddle.randint(0, 50, shape=[2, 8])
        out = emb(x)
        self.assertEqual(out.shape, [2, 8, 32])

    def test_embedding_create_missing_num_embeddings(self):
        """Should raise ValueError when num_embeddings and config.vocab_size are both missing."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(vocab_size=None)
        with self.assertRaises((ValueError, TypeError)):
            Embedding.create(config, num_embeddings=None)

    def test_embedding_create_missing_embedding_dim(self):
        """Should raise ValueError when embedding_dim and config.hidden_size are both missing."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(hidden_size=None)
        with self.assertRaises((ValueError, TypeError)):
            Embedding.create(config, embedding_dim=None)

    def test_embedding_create_vocab_parallel_type(self):
        """With tp_size>1, should use vocab_parallel embedding class from _global_mapping."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(tensor_model_parallel_size=2)
        # Patch _global_mapping to avoid distributed initialization issues
        mock_emb_cls = MagicMock(return_value=nn.Embedding(100, 64))
        original_mapping = Embedding._global_mapping.copy()
        try:
            Embedding._global_mapping["vocab_parallel"] = mock_emb_cls
            Embedding.create(config)
            mock_emb_cls.assert_called_once()
        finally:
            Embedding._global_mapping = original_mapping

    def test_embedding_create_explicit_type(self):
        """Explicit embedding_type should override config detection."""
        from paddleformers.nn.embedding import Embedding

        config = self._make_config(tensor_model_parallel_size=1)
        # Patch _global_mapping to avoid distributed initialization issues
        mock_emb_cls = MagicMock(return_value=nn.Embedding(100, 64))
        original_mapping = Embedding._global_mapping.copy()
        try:
            Embedding._global_mapping["vocab_parallel"] = mock_emb_cls
            Embedding.create(config, embedding_type="vocab_parallel")
            mock_emb_cls.assert_called_once()
        finally:
            Embedding._global_mapping = original_mapping
