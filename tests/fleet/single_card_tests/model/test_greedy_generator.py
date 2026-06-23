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

"""
Unit tests for generation module components.
This test file imports only the necessary components without full paddlefleet.
"""

import os
import sys

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import unittest

import paddle

from paddleformers.fleet.generation.greedy_generator import DynamicKVCache


class TestDynamicKVCache(unittest.TestCase):
    """Test cases for DynamicKVCache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = DynamicKVCache(num_layers=4)

        self.assertEqual(len(cache.k), 4)
        self.assertEqual(len(cache.v), 4)

        for i in range(4):
            self.assertIsNone(cache.k[i])
            self.assertIsNone(cache.v[i])

    def test_basic_update(self):
        """Test basic KV cache update."""
        cache = DynamicKVCache(num_layers=2)

        # First update
        k1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        v1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        returned_k, returned_v = cache.update(k1, v1, 0)

        self.assertIsNotNone(returned_k)
        self.assertIsNotNone(returned_v)

        # Should be the same as input (first update)
        self.assertTrue(
            paddle.allclose(returned_k.cast("float32"), k1.cast("float32"))
        )
        self.assertTrue(
            paddle.allclose(returned_v.cast("float32"), v1.cast("float32"))
        )

    def test_second_update_concat(self):
        """Test that second update concatenates."""
        cache = DynamicKVCache(num_layers=2)

        # First update
        k1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        v1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        cache.update(k1, v1, 0)

        # Second update (different length)
        k2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        returned_k, returned_v = cache.update(k2, v2, 0)

        # Should be concatenated
        self.assertEqual(returned_k.shape[1], 6)  # 4 + 2
        self.assertEqual(returned_v.shape[1], 6)

    def test_get_seq_len(self):
        """Test get_seq_len method.

        Note: get_seq_len has a fallback that returns the first non-empty layer's
        sequence length when the requested layer is empty. This is by design since
        all layers should have the same sequence length during inference.
        """
        cache = DynamicKVCache(num_layers=2)

        self.assertEqual(cache.get_seq_len(0), 0)
        self.assertEqual(cache.get_seq_len(1), 0)

        # Update layer 0
        k = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k, v, 0)

        # Layer 0 has 5 tokens
        self.assertEqual(cache.get_seq_len(0), 5)
        # Layer 1 is empty, but fallback returns layer 0's length (by design)
        self.assertEqual(cache.get_seq_len(1), 5)

    def test_reset(self):
        """Test reset functionality."""
        cache = DynamicKVCache(num_layers=3)

        # Update a layer
        k = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        v = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k, v, 0)

        # Reset
        cache.reset()

        # All layers should be None
        for i in range(3):
            self.assertIsNone(cache.k[i])
            self.assertIsNone(cache.v[i])

    def test_multiple_layers(self):
        """Test that different layers have independent caches.

        Note: get_seq_len has a fallback that returns the first non-empty layer's
        sequence length when the requested layer is empty.
        """
        cache = DynamicKVCache(num_layers=4)

        # Update layer 0
        k0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        v0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k0, v0, 0)

        # Update layer 2
        k2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k2, v2, 2)

        # Check layer-specific cache lengths
        self.assertEqual(cache.get_seq_len(0), 3)  # Layer 0 has 3 tokens
        self.assertEqual(cache.get_seq_len(2), 5)  # Layer 2 has 5 tokens
        # Layers 1 and 3 are empty, fallback returns first non-empty (layer 0 = 3)
        self.assertEqual(cache.get_seq_len(1), 3)
        self.assertEqual(cache.get_seq_len(3), 3)


class TestGreedyGeneratorEosStop(unittest.TestCase):
    """Test eos_token_id handling in GreedyGenerator.generate (mocked model)."""

    def _make_generator(self, token_sequence):
        """Create a GreedyGenerator with a fake model that yields given tokens."""
        from unittest.mock import MagicMock

        from paddleformers.fleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        # token_sequence: list of int, tokens the model will output in order
        self._call_idx = 0
        seq = token_sequence

        def fake_forward(inputs):
            # Return logits where argmax gives the desired token
            vocab_size = 100
            logits = paddle.zeros([1, 1, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[0, 0, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_eos_int_stops_early(self):
        """eos_token_id as int should stop generation."""
        # Sequence: 5, 5, 3(eos), 5, 5 — should stop at step 3
        gen = self._make_generator([5, 5, 3, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=10, eos_token_id=3)
        generated = out[0, 2:].tolist()
        # Should contain 5, 5, 3 then stop
        self.assertEqual(generated, [5, 5, 3])

    def test_eos_list_single_token_stops(self):
        """eos_token_id as list of single-token lists (e.g. [[3],[7]]) should stop."""
        gen = self._make_generator([5, 5, 7, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=10, eos_token_id=[[3], [7]]
        )
        generated = out[0, 2:].tolist()
        self.assertEqual(generated, [5, 5, 7])

    def test_eos_list_multi_token_not_early_stop(self):
        """Multi-token stop sequences in list should not trigger early stop."""
        # [[10, 20]] is a multi-token stop — should NOT stop generation
        gen = self._make_generator([10, 5, 5, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=5, eos_token_id=[[10, 20]])
        generated = out[0, 2:].tolist()
        # All 5 tokens generated (no early stop)
        self.assertEqual(len(generated), 5)

    def test_eos_none_generates_max(self):
        """No eos_token_id should generate max_new_tokens."""
        gen = self._make_generator([5] * 10)
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=5, eos_token_id=None)
        generated = out[0, 2:].tolist()
        self.assertEqual(len(generated), 5)


if __name__ == "__main__":
    print("Running greedy generator unit tests...")
    unittest.main(verbosity=2)
