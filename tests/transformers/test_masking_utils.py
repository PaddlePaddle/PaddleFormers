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

import unittest

import paddle

from paddleformers.generation.utils import GenerationMixin
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.transformers.masking_utils import (
    create_causal_mask_and_row_indices,
    create_sliding_window_causal_mask_and_row_indices,
)


class DummyModel(GenerationMixin):
    """
    A dummy model class that inherits `_prepare_decoder_attention_mask`
    directly from `GenerationMixin` to test the exact implementation used in production.
    """

    def __init__(self):
        super().__init__()
        pass


class MaskingUtilsTest(unittest.TestCase):
    """
    Comprehensive test suite for masking_utils.py and the integrated _prepare_decoder_attention_mask logic.
    """

    def setUp(self):
        self.batch_size = 2
        self.seq_length = 5
        self.hidden_dim = 16
        self.dtype = "float32"
        self.inputs_embeds = paddle.randn((self.batch_size, self.seq_length, self.hidden_dim), dtype=self.dtype)
        self.config = PretrainedConfig()
        self.config._attn_implementation = "eager"

        self.dummy_model = DummyModel()
        # Direct reference to the method under test
        self.prepare_fn = self.dummy_model._prepare_decoder_attention_mask

    def _is_visible(self, val):
        """Helper to check if a position is visible (0.0)."""
        return val == 0.0

    def _is_masked(self, val):
        """Helper to check if a position is masked (large negative value)."""
        # Using -100.0 as a safe threshold for float32 min values
        return val < -100.0

    def test_no_mask_causal_behavior(self):
        """
        Test 1: Default behavior when no attention_mask is provided.
        Expectation: Standard causal masking (lower triangular).
        """
        full_mask, row_indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        self.assertIsNone(row_indices)
        self.assertEqual(full_mask.shape, [self.batch_size, 1, self.seq_length, self.seq_length])

        # Self-attention (Diagonal) -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 0, 0]))
        # Past (Lower Triangle) -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 4, 0]))
        # Future (Upper Triangle) -> Masked
        self.assertTrue(self._is_masked(full_mask[0, 0, 0, 4]))

    def test_padding_mask_right(self):
        """
        Test 2: Standard right-side padding mask.
        Mask: [1, 1, 1, 0, 0] (Last 2 tokens are padding).
        """
        padding_mask = paddle.zeros((self.batch_size, self.seq_length), dtype="int64")
        padding_mask[:, :3] = 1

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=padding_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Valid token attending to Valid token -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 2, 0]))
        # Valid token attending to Padding token -> Masked
        self.assertTrue(self._is_masked(full_mask[0, 0, 2, 3]))

    def test_padding_mask_left(self):
        """
        Test 3: Left-side padding mask (common in batch generation).
        Mask: [0, 0, 1, 1, 1] (First 2 tokens are padding).
        """
        padding_mask = paddle.zeros((self.batch_size, self.seq_length), dtype="int64")
        padding_mask[:, 2:] = 1

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=padding_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # 3rd token (Index 2, Valid) attending to 1st token (Index 0, Padding) -> Masked
        self.assertTrue(self._is_masked(full_mask[0, 0, 2, 0]))

        # 3rd token attending to itself -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 2, 2]))

    def test_sliding_window_no_input_mask(self):
        """
        Test 4: Sliding window mask generation without an input attention_mask.
        Expectation: Tokens outside the window lookback range should be masked.
        """
        self.config.sliding_window = 2

        full_mask, _ = create_sliding_window_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=5,
            cache_length=0,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Window=2. Position 4 should see [4, 3].
        self.assertTrue(self._is_visible(full_mask[0, 0, 4, 4]))
        self.assertTrue(self._is_visible(full_mask[0, 0, 4, 3]))

        # Position 4 attending to 2 (Distance=2, Window=2)
        # Note: If window size is W, visible range is [i-W+1, i]. Distances >= W are masked.
        self.assertTrue(self._is_masked(full_mask[0, 0, 4, 2]))

    def test_sliding_window_during_generation(self):
        """
        Test 5: Sliding window behavior during single-step generation (seq_len=1).
        Expectation: Even when generating a single token, it should not attend to
        historical tokens outside the sliding window.
        """
        self.config.sliding_window = 2
        cache_len = 10
        seq_len = 1

        inputs = paddle.randn((self.batch_size, seq_len, self.hidden_dim))

        # Current token index in total sequence is 10 (cache_len).
        # Window=2. Visible indices: 10, 9.
        # Masked indices: 8, 7, ... 0.

        full_mask, _ = create_sliding_window_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=inputs,
            batch_size=self.batch_size,
            seq_length=seq_len,
            cache_length=cache_len,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Index 10 (Self) -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 0, 10]))
        # Index 9 (Distance 1) -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 0, 9]))

        # Index 8 (Distance 2) -> Should be Masked
        self.assertTrue(
            self._is_masked(full_mask[0, 0, 0, 8]),
            msg="Sliding window constraint failed: Token at distance >= window_size should be masked during generation.",
        )

    def test_sliding_window_large_size(self):
        """
        Test 6: Sliding window size larger than sequence length.
        Expectation: Should behave exactly like a standard causal mask.
        """
        self.config.sliding_window = 100  # Larger than seq_len=5

        full_mask, _ = create_sliding_window_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=5,
            cache_length=0,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Distance 4 (Index 4 -> 0) should be Visible (since 4 < 100)
        self.assertTrue(self._is_visible(full_mask[0, 0, 4, 0]))

    def test_or_mask_function_integration(self):
        """
        Test 7: Integration of `or_mask_function`.
        Expectation: Areas masked by causal/padding logic should become visible
        if `or_mask_function` returns True for those positions.
        """

        def force_visible_fn(b, h, q, k):
            # Make (0, 4) visible (normally masked by causal mask)
            mask = paddle.zeros((self.batch_size, 1, self.seq_length, self.seq_length), dtype="bool")
            mask[:, :, 0, 4] = True
            return mask

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            prepare_decoder_attention_mask=self.prepare_fn,
            or_mask_function=force_visible_fn,
        )

        # Standard causal check: (0, 1) Masked
        self.assertTrue(self._is_masked(full_mask[0, 0, 0, 1]))
        # Override check: (0, 4) Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, 0, 4]))

    def test_4d_mask_preservation(self):
        """
        Test 8: Passing a 4D Mask (e.g., boolean-like floats).
        Expectation: The function should respect the input values
        (mapped to 0.0 or min_val based on boolean truthiness).
        """
        # Input: 1.0 (True) -> Keep, 0.0 (False) -> Mask
        custom_4d = paddle.ones((self.batch_size, 1, self.seq_length, self.seq_length), dtype=self.dtype)
        custom_4d[:, :, 0, 4] = 0.0

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=custom_4d,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        self.assertTrue(self._is_visible(full_mask[0, 0, 0, 0]))
        self.assertTrue(self._is_masked(full_mask[0, 0, 0, 4]))

    def test_precomputed_indices_priority(self):
        """
        Test 9: Priority of `attn_mask_startend_row_indices`.
        Expectation: If start/end indices are provided, `causal_mask` should be None,
        and indices should be returned/processed.
        """
        indices = paddle.zeros((self.batch_size, 1, self.seq_length, 2), dtype="int64")

        # Case A: create_causal_mask_and_row_indices
        mask, out_indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attn_mask_startend_row_indices=indices,
            prepare_decoder_attention_mask=self.prepare_fn,
        )
        self.assertIsNone(mask)
        self.assertIsNotNone(out_indices)
        self.assertTrue(paddle.equal_all(out_indices, indices))

        # Case B: create_sliding_window...
        self.config.sliding_window = 2
        mask_sw, out_indices_sw = create_sliding_window_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attn_mask_startend_row_indices=indices,
            prepare_decoder_attention_mask=self.prepare_fn,
        )
        self.assertIsNone(mask_sw)
        # Indices should be modified by sliding window logic
        self.assertIsNotNone(out_indices_sw)

    def test_batch_independence(self):
        """
        Test 10: Independence of masks across the batch dimension.
        Expectation: Padding in one batch sample should not affect others.
        """
        padding_mask = paddle.ones((self.batch_size, self.seq_length), dtype="int64")
        # Mask the last token of the second sample only
        padding_mask[1, -1] = 0

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=padding_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Batch 0: Last token attending to itself -> Visible
        self.assertTrue(self._is_visible(full_mask[0, 0, -1, -1]))
        # Batch 1: Last token attending to itself -> Masked
        self.assertTrue(self._is_masked(full_mask[1, 0, -1, -1]))

    def test_combination_sliding_padding_cache(self):
        """
        Test 11: Complex combination of Sliding Window + Padding + Generation Cache.
        Expectation: Both sliding window constraints AND padding masks must be satisfied.
        """
        self.config.sliding_window = 2
        cache_len = 4
        seq_len = 1

        # Input mask: [1, 1, 1, 0, 1]. Position 3 is padding.
        # Total positions: 0, 1, 2, 3(Pad), 4(Current).
        padding_mask = paddle.to_tensor([[1, 1, 1, 0, 1]], dtype="int64")

        full_mask, _ = create_sliding_window_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=paddle.randn((1, seq_len, 16)),
            batch_size=1,
            seq_length=seq_len,
            cache_length=cache_len,
            attention_mask=padding_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        # Target: Current token (Index 4).

        # 1. Check Window Constraint (Dist 2 -> Index 2)
        # Index 2 is Valid in Padding Mask, but Out of Window (Window=2 covers 4,3).
        # Result: Masked.
        self.assertTrue(
            self._is_masked(full_mask[0, 0, 0, 2]),
            msg="Failed combining constraints: Sliding window should mask index 2.",
        )

        # 2. Check Padding Constraint (Dist 1 -> Index 3)
        # Index 3 is Inside Window, but Invalid in Padding Mask.
        # Result: Masked.
        self.assertTrue(
            self._is_masked(full_mask[0, 0, 0, 3]),
            msg="Failed combining constraints: Padding mask should mask index 3.",
        )

    def test_full_mask_optimization_returns_none(self):
        """
        Test the optimization: If attention_mask is all ones (no padding) and
        no or_mask_function is present, it should return None to verify
        backend optimization (avoiding unnecessary mask creation).
        """
        self.config._attn_implementation = "sdpa"
        # Case 1: All Ones Mask (No Padding)
        all_ones_mask = paddle.ones((self.batch_size, self.seq_length), dtype="int64")

        full_mask, indices = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=all_ones_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )

        self.assertIsNone(full_mask, "Should return None for all-ones mask (optimization)")
        self.assertIsNone(indices)

    def test_full_mask_optimization_bypassed_conditions(self):
        """
        Test that the optimization is SKIPPED (returns Tensor) when:
        1. or_mask_function is provided (even if mask is full).
        2. attention_mask contains zeros (padding).
        """
        all_ones_mask = paddle.ones((self.batch_size, self.seq_length), dtype="int64")

        # Case 2: All Ones Mask + or_mask_function
        def dummy_or_fn(b, h, q, k):
            return paddle.zeros((self.batch_size, 1, self.seq_length, self.seq_length), dtype="bool")

        full_mask, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=all_ones_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
            or_mask_function=dummy_or_fn,
        )
        self.assertIsNotNone(full_mask, "Should NOT optimize to None if or_mask_function is present")

        # Case 3: Mixed Mask (Has Padding)
        mixed_mask = all_ones_mask.clone()
        mixed_mask[0, -1] = 0

        full_mask_padded, _ = create_causal_mask_and_row_indices(
            config=self.config,
            inputs_embeds=self.inputs_embeds,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            cache_length=0,
            attention_mask=mixed_mask,
            prepare_decoder_attention_mask=self.prepare_fn,
        )
        self.assertIsNotNone(full_mask_padded, "Should NOT optimize to None if mask contains padding (0s)")


if __name__ == "__main__":
    unittest.main()
