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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
)


import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle

from paddleformers.fleet.models.qwen3_5.qwen3_5_model import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormPipe,
    Qwen3_5VisionSublayersSpec,
)


class TestQwen3_5RMSNorm(unittest.TestCase):
    """Test Qwen3_5RMSNorm class."""

    def _make_config(self, **overrides):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_cpu_initialization": True,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_constructor_basic(self):
        """Test basic constructor."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, hidden_size=64, eps=1e-5)
        self.assertEqual(norm.normalized_shape, 64)
        self.assertEqual(norm.variance_epsilon, 1e-5)
        # Weight should be initialized to 0
        np.testing.assert_allclose(norm.weight.numpy(), paddle.zeros([64]).numpy())

    def test_constructor_normalized_shape(self):
        """Test constructor with normalized_shape kwarg."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, normalized_shape=128, norm_eps=1e-6)
        self.assertEqual(norm.normalized_shape, 128)
        self.assertEqual(norm.variance_epsilon, 1e-6)

    def test_constructor_defaults_from_config(self):
        """Test constructor uses config defaults when not specified."""
        config = self._make_config(rms_norm_eps=1e-8)
        norm = Qwen3_5RMSNorm(config)
        self.assertEqual(norm.normalized_shape, 64)
        self.assertEqual(norm.variance_epsilon, 1e-8)

    def test_forward_shape(self):
        """Test forward produces correct shape."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, hidden_size=64)
        x = paddle.randn([4, 8, 64])
        output = norm(x)
        self.assertEqual(output.shape, [4, 8, 64])

    def test_forward_dtypes_preserved(self):
        """Test forward preserves input dtype."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, hidden_size=64)
        x = paddle.randn([4, 8, 64]).astype("float16")
        output = norm(x)
        self.assertEqual(output.dtype, paddle.float16)

    def test_forward_values_nonzero(self):
        """Test forward does not produce all zeros."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, hidden_size=64)
        x = paddle.randn([2, 4, 64])
        output = norm(x)
        self.assertFalse(paddle.all(output == 0))

    def test_enable_sequence_parallel(self):
        """Test enable_sequence_parallel method."""
        config = self._make_config()
        norm = Qwen3_5RMSNorm(config, hidden_size=64, input_is_parallel=True)
        # Should not raise
        norm.enable_sequence_parallel()


class TestQwen3_5RMSNormPipe(unittest.TestCase):
    """Test Qwen3_5RMSNormPipe class."""

    def _make_config(self, **overrides):
        from paddleformers.fleet.transformer.transformer_config import TransformerConfig

        defaults = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_cpu_initialization": True,
        }
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_forward_basic(self):
        """Test basic forward with dict input."""
        config = self._make_config()
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=64)
        x = paddle.randn([4, 8, 64])
        result = pipe({"hidden_states": x})
        self.assertEqual(result["hidden_states"].shape, [4, 8, 64])

    def test_forward_preserves_other_keys(self):
        """Test forward preserves other keys in dict."""
        config = self._make_config()
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=64)
        x = paddle.randn([4, 8, 64])
        result = pipe({"hidden_states": x, "attention_mask": paddle.ones([4, 1])})
        self.assertIn("attention_mask", result)

    def test_build_schedule_node(self):
        """Test build_schedule_node returns ScheduleNode."""
        config = self._make_config()
        pipe = Qwen3_5RMSNormPipe(config, hidden_size=64)
        node = pipe.build_schedule_node()
        self.assertIsNotNone(node)


class TestQwen3_5VisionSublayersSpec(unittest.TestCase):
    """Test Qwen3_5VisionSublayersSpec dataclass."""

    def test_default_values(self):
        """Test default values are None."""
        spec = Qwen3_5VisionSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.merger)

    def test_with_values(self):
        """Test setting values."""
        mock_emb = MagicMock()
        mock_merger = MagicMock()
        spec = Qwen3_5VisionSublayersSpec(embedding=mock_emb, merger=mock_merger)
        self.assertEqual(spec.embedding, mock_emb)
        self.assertEqual(spec.merger, mock_merger)


class TestQwen3_5VisionModel(unittest.TestCase):
    """Test Qwen3_5VisionModel class."""
