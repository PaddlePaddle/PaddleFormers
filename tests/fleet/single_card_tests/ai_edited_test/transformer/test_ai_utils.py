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
from types import SimpleNamespace

import paddle

from paddleformers.fleet.transformer.utils import (
    attention_mask_func,
    get_default_causal_mask,
    get_sliding_window_causal_mask,
    get_sliding_window_left_size,
    is_layer_window_attention,
    startend_row_indices_add_sliding_window,
)


class TestGetDefaultCausalMask(unittest.TestCase):
    """Tests for get_default_causal_mask."""

    def test_basic_shape(self):
        mask = get_default_causal_mask(4)
        self.assertEqual(mask.shape, [4, 4])

    def test_upper_triangular(self):
        mask = get_default_causal_mask(4)
        # upper triangle (excluding diagonal) should be True
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 0].item())
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(mask[0, 3].item())

    def test_single_token(self):
        mask = get_default_causal_mask(1)
        self.assertEqual(mask.shape, [1, 1])
        self.assertFalse(mask[0, 0].item())

    def test_lru_cache_same_result(self):
        mask1 = get_default_causal_mask(4)
        mask2 = get_default_causal_mask(4)
        self.assertTrue(paddle.allclose(mask1, mask2))

    def test_different_sizes(self):
        mask_small = get_default_causal_mask(2)
        mask_large = get_default_causal_mask(8)
        self.assertEqual(mask_small.shape, [2, 2])
        self.assertEqual(mask_large.shape, [8, 8])


class TestGetSlidingWindowCausalMask(unittest.TestCase):
    """Tests for get_sliding_window_causal_mask."""

    def test_basic_shape(self):
        mask = get_sliding_window_causal_mask(4, 4, (3, 3))
        self.assertEqual(mask.shape, [4, 4])

    def test_output_is_bool(self):
        mask = get_sliding_window_causal_mask(4, 4, (3, 3))
        self.assertEqual(mask.dtype, paddle.bool)

    def test_square_mask(self):
        mask = get_sliding_window_causal_mask(4, 4, (3, 3))
        # diagonal should be False (not masked)
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[1, 1].item())

    def test_lru_cache(self):
        mask1 = get_sliding_window_causal_mask(4, 4, (3, 3))
        mask2 = get_sliding_window_causal_mask(4, 4, (3, 3))
        self.assertTrue(
            paddle.allclose(mask1.cast(mask1.dtype), mask2.cast(mask2.dtype))
        )

    def test_non_square_mask(self):
        mask = get_sliding_window_causal_mask(2, 8, (3, 3))
        self.assertEqual(mask.shape, [2, 8])

    def test_int_equivalent_to_tuple_left_zero(self):
        """The int W branch should produce a mask equivalent to the tuple (W, 0)
        two-sided branch (HF causal one-sided semantics)."""
        mask_int = get_sliding_window_causal_mask(128, 128, 32)
        mask_tuple = get_sliding_window_causal_mask(128, 128, (32, 0))
        self.assertTrue(paddle.equal_all(mask_int, mask_tuple).item())

    def test_int_branch_shape(self):
        mask = get_sliding_window_causal_mask(4, 4, 3)
        self.assertEqual(mask.shape, [4, 4])
        self.assertEqual(mask.dtype, paddle.bool)

    def test_int_negative_is_infinite_causal_mask(self):
        """int `-1` (infinite window) should degrade to a plain causal mask."""
        mask = get_sliding_window_causal_mask(4, 4, -1)
        causal = get_default_causal_mask(4)
        self.assertTrue(paddle.equal_all(mask, causal).item())

    def test_int_negative_not_all_masked(self):
        """Regression: `-1` must not mask out every position."""
        mask = get_sliding_window_causal_mask(4, 4, -1)
        # diagonal (self-attention) must stay visible (not masked)
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[3, 3].item())

    def test_tuple_negative_left_is_causal(self):
        """A negative left in tuple form also means infinite past window."""
        mask = get_sliding_window_causal_mask(4, 4, (-1, 0))
        causal = get_default_causal_mask(4)
        self.assertTrue(paddle.equal_all(mask, causal).item())


class TestGetSlidingWindowLeftSize(unittest.TestCase):
    """Tests for get_sliding_window_left_size (shared window-size helper)."""

    def test_int(self):
        self.assertEqual(get_sliding_window_left_size(512), 512)

    def test_tuple(self):
        self.assertEqual(get_sliding_window_left_size((128, 0)), 128)
        self.assertEqual(get_sliding_window_left_size((256, 64)), 256)

    def test_negative_int(self):
        self.assertEqual(get_sliding_window_left_size(-1), -1)

    def test_negative_tuple(self):
        self.assertEqual(get_sliding_window_left_size((-1, 0)), -1)


class TestStartendRowIndicesAddSlidingWindow(unittest.TestCase):
    """Smoke tests for startend_row_indices_add_sliding_window int/tuple branches."""

    def _make_indices(self, bsz=1, heads=1, seq=8, num_vec=1):
        return paddle.full(
            [bsz, heads, seq, num_vec], fill_value=seq, dtype=paddle.int32
        )

    def test_int_and_tuple_equivalent(self):
        indices_int = self._make_indices()
        indices_tuple = self._make_indices()
        out_int = startend_row_indices_add_sliding_window(
            indices_int, 4, 0.0, 2
        )
        out_tuple = startend_row_indices_add_sliding_window(
            indices_tuple, (4, 0), 0.0, 2
        )
        self.assertTrue(paddle.equal_all(out_int, out_tuple).item())

    def test_int_none_passthrough(self):
        indices = self._make_indices()
        out = startend_row_indices_add_sliding_window(indices, None, 0.0, 2)
        self.assertTrue(paddle.equal_all(out, indices).item())

    def test_int_negative_passthrough(self):
        """`-1` (infinite window) must not truncate the row indices."""
        indices = self._make_indices()
        out = startend_row_indices_add_sliding_window(indices, -1, 0.0, 2)
        self.assertTrue(paddle.equal_all(out, indices).item())


class TestGreedyGeneratorWindowSize(unittest.TestCase):
    """Regression: GreedyGenerator must accept int sliding_window without
    hitting `TypeError: 'int' object is not subscriptable`."""

    def _make_model(self, sliding_window, window_attn_skip_freq=None):
        cfg = SimpleNamespace(
            num_hidden_layers=4,
            sliding_window=sliding_window,
            window_attn_skip_freq=window_attn_skip_freq,
            head_wise_swa_ratio=0.0,
            sequence_parallel=False,
            apply_rope_fusion=False,
            recompute_granularity=None,
        )
        return SimpleNamespace(config=cfg)

    def test_int_sliding_window_inits_cache(self):
        from paddleformers.fleet.generation.greedy_generator import GreedyGenerator

        gen = GreedyGenerator(self._make_model(512))
        self.assertEqual(gen.window_size, 512)
        self.assertEqual(gen.cache.window_size, 512)
        self.assertTrue(all(gen.cache.swa_layers))

    def test_tuple_sliding_window_inits_cache(self):
        from paddleformers.fleet.generation.greedy_generator import GreedyGenerator

        gen = GreedyGenerator(self._make_model((512, 0)))
        self.assertEqual(gen.window_size, 512)

    def test_infinite_window_disables_truncation(self):
        from paddleformers.fleet.generation.greedy_generator import GreedyGenerator

        gen = GreedyGenerator(self._make_model(-1))
        self.assertIsNone(gen.window_size)

    def test_no_sliding_window(self):
        from paddleformers.fleet.generation.greedy_generator import GreedyGenerator

        gen = GreedyGenerator(self._make_model(None))
        self.assertIsNone(gen.window_size)


class TestAttentionMaskFunc(unittest.TestCase):
    """Tests for attention_mask_func."""

    def test_masked_fill(self):
        scores = paddle.ones([2, 2], dtype=paddle.float32)
        mask = paddle.zeros([2, 2], dtype=paddle.bool)
        mask[0, 1] = True
        result = attention_mask_func(scores, mask)
        self.assertAlmostEqual(result[0, 1].item(), -10000.0)
        self.assertAlmostEqual(result[0, 0].item(), 1.0)

    def test_all_masked(self):
        scores = paddle.ones([3, 3], dtype=paddle.float32)
        mask = paddle.ones([3, 3], dtype=paddle.bool)
        result = attention_mask_func(scores, mask)
        self.assertTrue(paddle.all(result == -10000.0).item())

    def test_no_mask(self):
        scores = paddle.ones([2, 2], dtype=paddle.float32)
        mask = paddle.zeros([2, 2], dtype=paddle.bool)
        result = attention_mask_func(scores, mask)
        self.assertTrue(paddle.allclose(result, scores))

    def test_in_place_modification(self):
        scores = paddle.ones([2, 2], dtype=paddle.float32)
        mask = paddle.zeros([2, 2], dtype=paddle.bool)
        mask[0, 0] = True
        result = attention_mask_func(scores, mask)
        # The function modifies in place
        self.assertIs(result, scores)


class TestIsLayerWindowAttention(unittest.TestCase):
    """Tests for is_layer_window_attention."""

    def test_no_sliding_window(self):
        result = is_layer_window_attention(None, None, 1)
        self.assertFalse(result)

    def test_sliding_window_no_skip_freq(self):
        result = is_layer_window_attention((3, 3), None, 1)
        self.assertTrue(result)

    def test_sliding_window_int_skip_freq_not_multiple(self):
        result = is_layer_window_attention((3, 3), 2, 1)
        self.assertTrue(result)

    def test_sliding_window_int_skip_freq_is_multiple(self):
        result = is_layer_window_attention((3, 3), 2, 2)
        self.assertFalse(result)

    def test_sliding_window_list_skip_freq_true(self):
        result = is_layer_window_attention((3, 3), [1, 0, 1], 0)
        self.assertTrue(result)

    def test_sliding_window_list_skip_freq_false(self):
        result = is_layer_window_attention((3, 3), [1, 0, 1], 1)
        self.assertFalse(result)

    def test_sliding_window_list_all_true(self):
        result = is_layer_window_attention((3, 3), [True, True, True], 2)
        self.assertTrue(result)

    def test_sliding_window_list_all_false(self):
        result = is_layer_window_attention((3, 3), [False, False], 1)
        self.assertFalse(result)

    def test_invalid_skip_freq_type(self):
        with self.assertRaises(ValueError):
            is_layer_window_attention((3, 3), {"invalid": 1}, 1)

    def test_empty_sliding_window(self):
        result = is_layer_window_attention(None, 2, 1)
        self.assertFalse(result)

    def test_sliding_window_list_boundary(self):
        result = is_layer_window_attention((3, 3), [1, 1, 1, 1, 0, 0], 5)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
