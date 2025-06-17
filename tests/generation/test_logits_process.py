# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import paddle
from paddle import nn


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return paddle.to_tensor(data=values).reshape(shape)


from paddlenlp.generation.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    TemperatureLogitsWarper,
    TopKProcess,
    TopPProcess,
)


class LogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = paddle.ones((batch_size, length)) / length
        return scores

    def test_min_length_dist_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        min_dist_processor = MinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)

        # check that min length is applied at length 5
        input_ids = ids_tensor((batch_size, 5), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertListEqual(scores_before_min_length[:, eos_token_id].tolist(), 4 * [paddle.finfo(scores.dtype).min])

        # check that min length is not applied anymore at length 15
        input_ids = ids_tensor((batch_size, 15), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_min_length = min_dist_processor(input_ids, scores)
        self.assertFalse((scores_before_min_length == paddle.finfo(scores.dtype).min).any())

    def test_temperature_dist_warper(self):
        input_ids = None
        length = 20

        scores = self._get_uniform_logits(batch_size=2, length=length)

        # tweak scores to not be uniform anymore
        scores[1, 5] = (1 / length) + 0.1  # peak, 1st batch
        scores[1, 10] = (1 / length) - 0.4  # valley, 1st batch

        # compute softmax
        probs = nn.functional.softmax(scores, axis=-1)

        temp_dist_warper_sharper = TemperatureLogitsWarper(temperature=0.5)
        temp_dist_warper_smoother = TemperatureLogitsWarper(temperature=1.3)

        warped_prob_sharp = nn.functional.softmax(temp_dist_warper_sharper(input_ids, scores.clone()), axis=-1)
        warped_prob_smooth = nn.functional.softmax(temp_dist_warper_smoother(input_ids, scores.clone()), axis=-1)

        # uniform distribution stays uniform
        self.assertTrue(paddle.allclose(probs[0, :], warped_prob_sharp[0, :], atol=1e-3))
        self.assertTrue(paddle.allclose(probs[0, :], warped_prob_smooth[0, :], atol=1e-3))

        # sharp peaks get higher, valleys get lower
        self.assertLess(probs[1, :].max(), warped_prob_sharp[1, :].max())
        self.assertGreater(probs[1, :].min(), warped_prob_sharp[1, :].min())

        # smooth peaks get lower, valleys get higher
        self.assertGreater(probs[1, :].max(), warped_prob_smooth[1, :].max())
        self.assertLess(probs[1, :].min(), warped_prob_smooth[1, :].min())

    def test_repetition_penalty_dist_process(self):
        input_ids = paddle.to_tensor([[0, 1], [5, 0]])
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        # give values special values
        scores[0, 0] = -(1 / vocab_size)
        scores[1, 5] = 4 / vocab_size

        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)

        scores = rep_penalty_proc(input_ids, scores.clone())

        # check that values were correctly changed
        self.assertAlmostEqual(scores[0, 0].item(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(scores[0, 1].item(), (1 / vocab_size) / 2)

        self.assertAlmostEqual(scores[1, 0].item(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(scores[1, 5].item(), (4 / vocab_size) / 2)

    def test_top_k_dist_warper(self):
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = paddle.arange(vocab_size).unsqueeze(0).tile((batch_size, 1))
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size
        ramp_logits = ramp_logits.astype("float32")
        scores = TopKProcess(ramp_logits, 3, 1)

        # check that correct tokens are filtered
        self.assertListEqual((scores[0] == 0.0).tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual((scores[1] == 0.0).tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        scores = TopKProcess(logits, top_k=1, min_tokens_to_keep=3)
        # uniform dist is not changed
        self.assertListEqual((scores == 0.0).sum(axis=-1).tolist(), [0, 0])

        ramp_logits = paddle.arange(length).unsqueeze(0).tile((batch_size, 1))
        ramp_logits = ramp_logits.astype("float32")
        scores = TopKProcess(ramp_logits, top_k=1, min_tokens_to_keep=3)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual((scores == 0.0).sum(axis=-1).tolist(), [2, 2])

    def test_top_p_dist_warper(self):
        vocab_size = 10
        batch_size = 2

        # create distribution and take log (inverse to Softmax as taken in TopPProcess)
        # dist = paddle.log(paddle.to_tensor([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]]))
        dist = paddle.to_tensor([[0.3, 0.1, 0.1, 0.5], [0.15, 0.3, 0.3, 0.25]])

        # filtered_dist = paddle.exp(TopPProcess(dist, 0.80001, 1))
        filtered_dist = TopPProcess(dist, 0.79999, 1)

        EXPECTED_FILTERED_DIST = paddle.to_tensor([[0.3, 0.0, 0.0, 0.5], [0.0, 0.3, 0.3, 0.25]])
        self.assertTrue(paddle.allclose(filtered_dist, EXPECTED_FILTERED_DIST, atol=1e-3))

        # check edge cases with negative and extreme logits
        ramp_logits = paddle.arange(vocab_size).unsqueeze(0).tile((batch_size, 1)) - (vocab_size // 2)
        ramp_logits = ramp_logits.astype("float32")
        # make ramp_logits more extreme
        ramp_logits[1] = ramp_logits[1] * 10.0
        sft_ramp_logits = paddle.nn.functional.softmax(ramp_logits, axis=-1)
        # make sure at least 2 tokens are kept
        filtered_dist = TopPProcess(sft_ramp_logits, 0.9, min_tokens_to_keep=2)

        # first batch should keep three tokens, second batch would keep only 1, but due to `min_tokens_to_keep=2` keeps 2.
        self.assertListEqual((filtered_dist != 0.0).sum(axis=-1).tolist(), [3, 2])

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2

        input_ids = paddle.to_tensor([[1, 1, 2, 1], [0, 1, 0, 1]])
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = NoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = NoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores.clone())
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores.clone())

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch

        self.assertListEqual(
            (filtered_scores_2_gram == paddle.finfo(scores.dtype).min).tolist(),
            [[False, True, True], [True, False, False]],
        )

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(
            (filtered_scores_3_gram == paddle.finfo(scores.dtype).min).tolist(),
            [[False, False, False], [True, False, False]],
        )

    def test_processor_list(self):
        batch_size = 4
        sequence_length = 10
        vocab_size = 15
        eos_token_id = 0

        # dummy input_ids and scores
        input_ids = ids_tensor((batch_size, sequence_length), vocab_size)
        input_ids_comp = input_ids.clone()

        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_comp = scores.clone()

        # instantiate all dist processors
        min_dist_proc = MinLengthLogitsProcessor(min_length=10, eos_token_id=eos_token_id)
        temp_dist_warp = TemperatureLogitsWarper(temperature=0.5)
        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)
        no_repeat_proc = NoRepeatNGramLogitsProcessor(2)

        # no processor list
        scores = min_dist_proc(input_ids, scores)
        scores = temp_dist_warp(input_ids, scores)
        scores = rep_penalty_proc(input_ids, scores)
        scores = no_repeat_proc(input_ids, scores)

        # with processor list
        processor = LogitsProcessorList(
            [
                min_dist_proc,
                temp_dist_warp,
                rep_penalty_proc,
                no_repeat_proc,
            ]
        )
        scores_comp = processor(input_ids, scores_comp)

        # scores should be equal
        self.assertTrue(paddle.allclose(scores, scores_comp, atol=1e-3))

        # input_ids should never be changed
        self.assertListEqual(input_ids.tolist(), input_ids_comp.tolist())

    def test_hamming_diversity(self):
        vocab_size = 4
        num_beams = 2
        num_beam_groups = 2

        scores = self._get_uniform_logits(num_beams, vocab_size)
        # batch_idx = 0 -> index batch_idx * num_beam_groups -> idx = 0 * 2 = 0 -> penalises tokens 1
        # batch_idx = 1 -> index batch_idx * num_beam_groups -> idx = 1 * 2 = 2 -> penalises tokens 1
        current_tokens = paddle.to_tensor([0, 3, 1, 2])

        diversity_logits_processor = HammingDiversityLogitsProcessor(
            diversity_rate=1.0, num_beams=num_beams, num_beam_groups=num_beam_groups
        )

        processed_scores = diversity_logits_processor(None, scores, current_tokens, 1)

        self.assertTrue(
            paddle.allclose(processed_scores[0], paddle.to_tensor([-0.7500, 0.2500, 0.2500, 0.2500]), atol=1e-3)
        )
        self.assertTrue(
            paddle.allclose(processed_scores[1], paddle.to_tensor([0.2500, -0.7500, 0.2500, 0.2500]), atol=1e-3)
        )

    def test_forced_bos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        bos_token_id = 0

        logits_processor = ForcedBOSTokenLogitsProcessor(forced_bos_token_id=bos_token_id)

        # check that all scores are -inf except the bos_token_id score
        input_ids = ids_tensor((batch_size, 1), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertTrue((scores[:, bos_token_id + 1 :] == paddle.finfo(scores.dtype).min).all())
        self.assertListEqual(scores[:, bos_token_id].tolist(), 4 * [0])  # score for bos_token_id should be zero

        # check that bos_token_id is not forced if current length is greater than 1
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertFalse((scores == paddle.finfo(scores.dtype).min).any())

    def test_forced_eos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0
        max_length = 5

        logits_processor = ForcedEOSTokenLogitsProcessor(max_length=max_length, forced_eos_token_id=eos_token_id)

        # check that all scores are -inf except the eos_token_id when max_length-1 is reached
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertTrue((scores[:, eos_token_id + 1 :] == paddle.finfo(scores.dtype).min).all())
        self.assertListEqual(scores[:, eos_token_id].tolist(), 4 * [0])  # score for eos_token_id should be zero

        # check that eos_token_id is not forced if max_length-1 is not reached
        input_ids = ids_tensor((batch_size, 3), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores = logits_processor(input_ids, scores)
        self.assertFalse((scores == paddle.finfo(scores.dtype).min).any())

    def test_bias_dist_processor(self):
        vocab_size = 5
        batch_size = 2

        input_ids = paddle.to_tensor([[0, 1, 3, 1], [0, 1, 0, 1]])
        positive_bias = {(1,): 100.0, (4,): 100.0}
        negative_bias = {(1, 0): -100.0, (0, 1, 2): -100.0, (1, 3, 1, 3): -100.0}
        # biases the same termination twice, to ensure we can handle overlapping terminations (it won't have an effect
        # on the test cases, though)
        negative_bias.update({(1, 3, 1, 3, 1, 3): -100.0})
        sequence_bias = {**positive_bias, **negative_bias}

        # scores = 0 to facilitate checks
        scores = paddle.zeros((batch_size, vocab_size))

        bias_dist_proc = SequenceBiasLogitsProcessor(sequence_bias=sequence_bias)
        filtered_scores = bias_dist_proc(input_ids, scores.clone())

        # batch 1: positive bias: tokens (1, 4); negative bias: tokens (0, 3); neutral: tokens (2)
        # batch 2: positive bias: tokens (1, 4); negative bias: tokens (0, 2); neutral: tokens (3)
        self.assertListEqual(
            filtered_scores.tolist(), [[-100.0, 100.0, 0.0, -100.0, 100.0], [-100.0, 100.0, -100.0, 0.0, 100.0]]
        )

    def test_no_bad_words_dist_processor(self):
        vocab_size = 5
        batch_size = 2
        eos_token_id = 4

        input_ids = paddle.to_tensor([[0, 1, 3, 1], [0, 1, 0, 1]])
        bad_word_tokens = [[1], [4], [1, 0], [0, 1, 2], [1, 3, 1, 3]]
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=bad_word_tokens, eos_token_id=eos_token_id)

        filtered_scores = no_bad_words_dist_proc(input_ids, scores.clone())

        # batch 1: 1st, 2nd, and 4th (0, 1, 3) token are forbidden
        # batch 2: 1st, 2nd, and 3rd (0, 1, 2) token are forbidden
        # Note that 5th element cannot be forbidden as it is EOS token
        self.assertListEqual(
            paddle.isinf(filtered_scores).tolist(),
            [[True, True, False, True, False], [True, True, True, False, False]],
        )

        # check edge case
        no_bad_words_dist_proc = NoBadWordsLogitsProcessor(bad_words_ids=[[4]], eos_token_id=eos_token_id)
        filtered_scores = no_bad_words_dist_proc(input_ids, scores.clone())
        self.assertTrue(paddle.allclose(scores, filtered_scores, atol=1e-3).numpy())

    def test_prefix_constrained_logits_processor(self):
        vocab_size = 5
        batch_size = 2

        input_ids = paddle.to_tensor([[0, 1, 3, 1], [0, 1, 0, 1]])
        scores = self._get_uniform_logits(batch_size, vocab_size)

        def prefix_allowed_tokens_fn(batch_id, inputs_ids):
            return [[0, 1], [2, 3]][batch_id]

        prefix_constrained_logits_proc = PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, 1)

        filtered_scores = prefix_constrained_logits_proc(input_ids, scores.clone())

        # batch 1: 1st, 2nd (0, 1) token are allowed
        # batch 2: 3rd, 4th (2, 3) token are allowed
        self.assertListEqual(
            (filtered_scores == paddle.finfo(filtered_scores.dtype).min).tolist(),
            [[False, False, True, True, True], [True, True, False, False, True]],
        )
