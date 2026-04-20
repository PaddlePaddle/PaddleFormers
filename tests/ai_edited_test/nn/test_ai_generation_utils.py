# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import unittest

import paddle


class TestGetUnfinishedFlag(unittest.TestCase):
    """Tests for paddleformers.generation.utils.get_unfinished_flag."""

    def test_single_eos_token_not_finished(self):
        """When last token is not eos, unfinished_flag should remain True."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 5]])
        unfinished = paddle.to_tensor([[True]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertTrue(result.item())

    def test_single_eos_token_finished(self):
        """When last token is eos, unfinished_flag should become False."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 0]])
        unfinished = paddle.to_tensor([[True]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertFalse(result.item())

    def test_already_finished_stays_finished(self):
        """When unfinished_flag is already False, it should remain False."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 5]])
        unfinished = paddle.to_tensor([[False]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertFalse(result.item())

    def test_list_eos_tokens(self):
        """List of eos token ids should stop when any matches."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 5]])
        unfinished = paddle.to_tensor([[True]])
        # eos_token_id=[0, 5] -- 5 matches last token
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=[0, 5])
        self.assertFalse(result.item())

    def test_list_eos_tokens_none_match(self):
        """List of eos token ids: none match should stay unfinished."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 9]])
        unfinished = paddle.to_tensor([[True]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=[0, 5])
        self.assertTrue(result.item())

    def test_nested_list_eos_tokens(self):
        """Nested list of eos token ids: should stop when any sublist matches."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 5]])
        unfinished = paddle.to_tensor([[True]])
        # [[0], [5]] -- second sublist matches
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=[[0], [5]])
        self.assertFalse(result.item())

    def test_nested_list_eos_none_match(self):
        """Nested list of eos token ids: no match should stay unfinished."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 3, 9]])
        unfinished = paddle.to_tensor([[True]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=[[0], [5]])
        self.assertTrue(result.item())

    def test_batch_input(self):
        """Batch input with mixed finished/unfinished states."""
        from paddleformers.generation.utils import get_unfinished_flag

        input_ids = paddle.to_tensor([[1, 2, 0], [1, 2, 3]])
        unfinished = paddle.to_tensor([[True], [True]])
        result = get_unfinished_flag(input_ids, unfinished, eos_token_id=0)
        self.assertFalse(result[0].item())
        self.assertTrue(result[1].item())


class TestBeamHypotheses(unittest.TestCase):
    """Tests for paddleformers.generation.utils.BeamHypotheses."""

    def test_init(self):
        """BeamHypotheses should initialize with empty beams."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
        self.assertEqual(len(hyps), 0)
        self.assertEqual(hyps.num_beams, 3)

    def test_add_within_capacity(self):
        """Adding beams within capacity should store them."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
        hyp1 = paddle.to_tensor([1, 2, 3])
        hyps.add(hyp1, sum_logprobs=-1.0)
        self.assertEqual(len(hyps), 1)

    def test_add_up_to_capacity(self):
        """Adding up to num_beams should keep all beams."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
        for i in range(3):
            hyp = paddle.to_tensor([1, 2, 3 + i])
            hyps.add(hyp, sum_logprobs=-(i + 1) * 0.5)
        self.assertEqual(len(hyps), 3)

    def test_add_evicts_worst(self):
        """When exceeding capacity, the worst beam should be evicted."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=False)
        hyps.add(paddle.to_tensor([1, 2, 3]), sum_logprobs=-2.0)
        hyps.add(paddle.to_tensor([1, 2, 4]), sum_logprobs=-1.0)
        # worst_score is the lower of the two
        self.assertAlmostEqual(
            hyps.worst_score,
            min(
                -2.0 / (((3 - 0 + 5) / 6) ** 1.0),
                -1.0 / (((3 - 0 + 5) / 6) ** 1.0),
            ),
        )

    def test_add_better_than_worst(self):
        """A beam better than worst should replace it."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=False)
        hyps.add(paddle.to_tensor([1, 2, 3]), sum_logprobs=-5.0)
        hyps.add(paddle.to_tensor([1, 2, 4]), sum_logprobs=-4.0)
        old_worst = hyps.worst_score
        # Add a much better beam
        hyps.add(paddle.to_tensor([1, 2, 5]), sum_logprobs=-0.1)
        self.assertEqual(len(hyps), 2)
        self.assertGreater(hyps.worst_score, old_worst)

    def test_is_done_not_enough_beams(self):
        """is_done should return False when not enough beams."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
        self.assertFalse(hyps.is_done(best_sum_logprobs=0.0, cur_len=5))

    def test_is_done_early_stopping(self):
        """is_done should return True with early_stopping when beams are full."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=True)
        hyps.add(paddle.to_tensor([1, 2, 3]), sum_logprobs=-1.0)
        hyps.add(paddle.to_tensor([1, 2, 4]), sum_logprobs=-2.0)
        self.assertTrue(hyps.is_done(best_sum_logprobs=0.0, cur_len=5))

    def test_is_done_no_early_stopping_better_available(self):
        """Without early_stopping, should return False if a better score is possible."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=False)
        hyps.add(paddle.to_tensor([1, 2, 3]), sum_logprobs=-1.0)
        hyps.add(paddle.to_tensor([1, 2, 4]), sum_logprobs=-2.0)
        # A much better score should make is_done return False
        self.assertFalse(hyps.is_done(best_sum_logprobs=100.0, cur_len=5))

    def test_add_with_origin_len(self):
        """add with origin_len should use it in length penalty calculation."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=3, length_penalty=1.0, early_stopping=False)
        hyp = paddle.to_tensor([1, 2, 3, 4, 5, 6, 7, 8])
        hyps.add(hyp, sum_logprobs=-1.0, origin_len=5)
        self.assertEqual(len(hyps), 1)

    def test_is_done_with_origin_len(self):
        """is_done with origin_len should use it in score comparison."""
        from paddleformers.generation.utils import BeamHypotheses

        hyps = BeamHypotheses(num_beams=2, length_penalty=1.0, early_stopping=False)
        hyps.add(paddle.to_tensor([1, 2, 3, 4, 5]), sum_logprobs=-1.0, origin_len=2)
        hyps.add(paddle.to_tensor([1, 2, 3, 4, 5]), sum_logprobs=-2.0, origin_len=2)
        self.assertFalse(hyps.is_done(best_sum_logprobs=100.0, cur_len=5, origin_len=2))


class TestMakeSlidingWindowMask(unittest.TestCase):
    """Tests for paddleformers.generation.utils._make_sliding_window_mask."""

    def test_basic_shape(self):
        """Output shape should be [bsz, 1, tgt_seq_len, src_seq_len]."""
        from paddleformers.generation.utils import _make_sliding_window_mask

        mask = _make_sliding_window_mask((2, 8), past_key_values_length=0, window_size=5)
        self.assertEqual(mask.shape, [2, 1, 8, 8])

    def test_with_past_key_values(self):
        """With past_key_values_length, src_seq_len should be tgt + past."""
        from paddleformers.generation.utils import _make_sliding_window_mask

        mask = _make_sliding_window_mask((2, 4), past_key_values_length=10, window_size=5)
        self.assertEqual(mask.shape, [2, 1, 4, 14])

    def test_causal_property(self):
        """Each position should only attend to itself and earlier positions within window."""
        from paddleformers.generation.utils import _make_sliding_window_mask

        mask = _make_sliding_window_mask((1, 5), past_key_values_length=0, window_size=3)
        # Position 0 should only see position 0
        self.assertTrue(mask[0, 0, 0, 0].item())
        self.assertFalse(mask[0, 0, 0, 1].item())
        # Position 2 should see positions 0, 1, 2 (window_size=3)
        self.assertTrue(mask[0, 0, 2, 0].item())
        self.assertTrue(mask[0, 0, 2, 1].item())
        self.assertTrue(mask[0, 0, 2, 2].item())
        self.assertFalse(mask[0, 0, 2, 3].item())
        # Position 4 should see positions 2, 3, 4 (window of 3)
        self.assertFalse(mask[0, 0, 4, 1].item())
        self.assertTrue(mask[0, 0, 4, 2].item())
        self.assertTrue(mask[0, 0, 4, 3].item())
        self.assertTrue(mask[0, 0, 4, 4].item())

    def test_window_size_larger_than_seq(self):
        """When window_size >= seq_length, all causal positions should be visible."""
        from paddleformers.generation.utils import _make_sliding_window_mask

        mask = _make_sliding_window_mask((1, 3), past_key_values_length=0, window_size=10)
        # All positions should see all previous and current
        for i in range(3):
            for j in range(i + 1):
                self.assertTrue(mask[0, 0, i, j].item(), f"pos {i} should see pos {j}")


class TestBeamSearchScorer(unittest.TestCase):
    """Tests for paddleformers.generation.utils.BeamSearchScorer."""

    def test_init_validation_num_beams(self):
        """BeamSearchScorer should raise ValueError for num_beams <= 1."""
        from paddleformers.generation.utils import BeamSearchScorer

        with self.assertRaises(ValueError):
            BeamSearchScorer(batch_size=1, max_length=10, num_beams=1)

    def test_init_validation_beam_groups(self):
        """BeamSearchScorer should raise ValueError for invalid num_beam_groups."""
        from paddleformers.generation.utils import BeamSearchScorer

        with self.assertRaises(ValueError):
            BeamSearchScorer(batch_size=1, max_length=10, num_beams=4, num_beam_groups=3)

    def test_init_valid(self):
        """BeamSearchScorer should initialize correctly with valid params."""
        from paddleformers.generation.utils import BeamSearchScorer

        scorer = BeamSearchScorer(batch_size=2, max_length=20, num_beams=4)
        self.assertEqual(scorer.num_beams, 4)
        self.assertEqual(scorer.group_size, 4)
        self.assertFalse(scorer.is_done.item())

    def test_init_with_beam_groups(self):
        """BeamSearchScorer with multiple beam groups."""
        from paddleformers.generation.utils import BeamSearchScorer

        scorer = BeamSearchScorer(batch_size=2, max_length=20, num_beams=4, num_beam_groups=2)
        self.assertEqual(scorer.group_size, 2)
