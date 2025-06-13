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
from __future__ import annotations

import inspect
from abc import ABC
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import paddle
from paddle.nn.layer.layers import in_declarative_mode


class LogitsProcessor(ABC):
    """
    Abstract base class for all logit processors that can be applied during
    generation.
    """

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. " "Only classes inheriting this class can be called."
        )


class LogitsProcessorList:
    """use ordered dict to store processors"""

    def __init__(self, processors: List[LogitsProcessor] = None) -> None:
        self._processors = OrderedDict()
        processors = processors or []
        for processor in processors:
            self.append(processor)

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor, **kwargs):
        for processor in self._processors.values():
            processor_args = inspect.signature(processor.__call__).parameters
            if len(processor_args) > 2:
                assert all(
                    arg in kwargs for arg in list(processor_args.keys())[2:]
                ), f"The parameters don't match for {processor.__class__}"
                logits = processor(input_ids, logits, **kwargs)
            else:
                logits = processor(input_ids, logits)
        return logits

    def append(self, processor: LogitsProcessor):
        self._processors[len(self._processors)] = processor


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (int): The minimum length of generation sequence.
        eos_token_id (int): The id of the `end-of-sequence` token.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if min_length < 0 and not in_declarative_mode():
            raise ValueError("`min_length` should be a positive integer, but get {}".format(min_length))

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError("`eos_token_id` should be a positive integer, but get {}".format(eos_token_id))

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            logits[:, self.eos_token_id] = paddle.finfo(logits.dtype).min
        return logits


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    Enforcing an exponential penalty on repeated sequences.

    Args:
        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not (penalty > 0) and not in_declarative_mode():
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: paddle.Tensor, logits: paddle.Tensor):
        score = paddle.index_sample(logits, input_ids)
        score = paddle.where(score < 0, score * self.penalty, score / self.penalty)
        input_ids = input_ids + paddle.arange(logits.shape[0], dtype="int64").unsqueeze(-1) * logits.shape[-1]
        outputs = paddle.scatter(logits.flatten(), input_ids.flatten(), score.flatten()).reshape(logits.shape)
        return outputs


def _get_ngrams(ngram_size: int, prev_input_ids: paddle.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`paddle.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.

    Args:
        banned_ngrams (`dict`):
            A dictionary containing previously generated n-grams for each hypothesis.
        prev_input_ids (`paddle.Tensor`):
            Generated token ids for the current hypothesis.
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        cur_len (`int`):
            The current length of the token sequences for which the n-grams are being checked.

    Returns:
        List of tokens that are banned.
    """
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(ngram_size: int, prev_input_ids: paddle.Tensor, num_hypos: int, cur_len: int):
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)

    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            if len(banned_tokens) == 0:
                continue
            scores[i, banned_tokens] = paddle.finfo(scores.dtype).min

        return scores


class HammingDiversityLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces diverse beam search. Note that this logits
    processor is only effective for `group_beam_search`. See
    `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.

    Args:
        diversity_rate (float): This value is subtracted from a beam's score if
            it generates a token same as any beam from other group at a particular
            time.
        num_beams (int): Number of beams used for group beam search.
        num_beam_groups (int): Number of groups to divide `num_beams` into in order
            to ensure diversity among different groups of beams.
    """

    def __init__(self, diversity_rate: float, num_beams: int, num_beam_groups: int):
        if not isinstance(diversity_rate, float) or (not diversity_rate > 0.0):
            raise ValueError("`diversity_rate` should be a float strictly larger than 0.")
        self._diversity_rate = diversity_rate
        if not isinstance(num_beams, int) or num_beams < 2:
            raise ValueError("`num_beams` should be an integer strictly larger than 1.")
        self._num_beams = num_beams
        if not isinstance(num_beam_groups, int) or num_beam_groups < 2:
            raise ValueError("`num_beam_groups` should be an integer strictly larger than 1.")
        self._num_sub_beams = num_beams // num_beam_groups

    def __call__(
        self, input_ids: paddle.Tensor, scores: paddle.Tensor, current_tokens: paddle.Tensor, beam_group_idx: int
    ):
        batch_size = current_tokens.shape[0] // self._num_beams
        group_start_idx = beam_group_idx * self._num_sub_beams
        group_end_idx = min(group_start_idx + self._num_sub_beams, self._num_beams)
        group_size = group_end_idx - group_start_idx
        vocab_size = scores.shape[-1]

        if group_start_idx == 0:
            return scores

        for batch_idx in range(batch_size):
            previous_group_tokens = current_tokens[
                batch_idx * self._num_beams : batch_idx * self._num_beams + group_start_idx
            ]
            token_frequency = paddle.bincount(previous_group_tokens, minlength=vocab_size)
            scores[batch_idx * group_size : (batch_idx + 1) * group_size] -= self._diversity_rate * token_frequency

        return scores


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the first generated token to be the selected `forced_bos_token`.

    Args:
        forced_bos_token_id (:obj:`int`):
            The id of the token to be generated as the first token.
    """

    def __init__(self, forced_bos_token_id: int):
        self.forced_bos_token_id = forced_bos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        cur_len = input_ids.shape[-1]
        if cur_len == 1:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_bos_token_id] = 0
        return scores


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    """
    This `LogitsProcessor` enforces the last generated token to be the selected `forced_eos_token`.

    Args:
        max_length (int): The maximum length of the sequence to be generated.
        forced_eos_token_id (int): The id of the token to be generated as the last token.
    """

    def __init__(self, max_length: int, forced_eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        self.forced_eos_token_id = forced_eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len == self.max_length - 1:
            scores[:] = paddle.finfo(scores.dtype).min
            scores[:, self.forced_eos_token_id] = 0
        return scores


def TopKProcess(probs: paddle.Tensor, top_k: int, min_tokens_to_keep: int):
    top_k = paddle.minimum(
        paddle.maximum(paddle.to_tensor(top_k), paddle.to_tensor(min_tokens_to_keep)),
        paddle.to_tensor(probs.shape[-1]),
    )
    # Remove all tokens with a probability less than the last token of the top-k
    # cast to float16 to support generation & d2s
    if probs.dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)
        topk_probs, _ = paddle.topk(probs, k=top_k)
        topk_probs = paddle.cast(topk_probs, paddle.bfloat16)
    else:
        topk_probs, _ = paddle.topk(probs, k=top_k)

    probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
    return probs


def TopPProcess(probs: paddle.Tensor, top_p: float, min_tokens_to_keep: int):
    if probs.dtype == paddle.bfloat16:
        probs = paddle.cast(probs, paddle.float32)

        sorted_indices = paddle.argsort(probs, descending=True)
        sorted_probs = paddle.sort(probs, descending=True)

        sorted_probs = paddle.cast(sorted_probs, paddle.bfloat16)

    else:
        sorted_indices = paddle.argsort(probs, descending=True)
        sorted_probs = paddle.sort(probs, descending=True)

    cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

    # Remove tokens with cumulative probs above the top_p, But keep at
    # least min_tokens_to_keep tokens
    sorted_indices_to_remove = cumulative_probs > top_p
    if min_tokens_to_keep > 1:
        # Set 'min_tokens_to_keep - 1' because the first token is kept
        sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
    # Keep the first token
    sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    # Scatter sorted tensors to original indexing
    sorted_indices = sorted_indices + paddle.arange(probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1]
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
    )
    condition = paddle.cast(condition, "bool").reshape(probs.shape)
    probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
    return probs


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TemperatureLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for temperature (exponential scaling output probability distribution).
    Args:
        temperature (`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor):
        scores = scores / self.temperature
        return scores


class SequenceBiasLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that applies an additive bias on sequences. The bias is applied to the last token of a sequence
    when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than
    one token, consider using beam methods (to gracefully work around partially completed sequences that have a
    negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

    <Tip>

    In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when
    initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The
    `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers' prefixing behaviours
    come from `pre tokenizers`.

    </Tip>

    Args:
        sequence_bias (`Dict[Tuple[int], float]`):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias
            will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be
            completed (in the token selection step after this processor is applied).

    Examples:

    ```python
    >>> from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2-en")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2-en")
    >>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

    >>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
    >>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Trump Jr

    >>> # Now let's control generation through a bias. Please note that the tokenizer is initialized differently!
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2-en")


    >>> def get_tokens_as_tuple(word):
    ...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


    >>> # If we add a negative bias without beam search, it may become "stuck" in a prefix without good continuations
    >>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald J. Donald,

    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Rumsfeld,

    >>> # We can also add a positive bias to nudge the model towards specific tokens or continuations
    >>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
    >>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
    >>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
    The full name of Donald is Donald Duck.
    ```
    """

    def __init__(self, sequence_bias: Dict[Tuple[int], float]):
        self.sequence_bias = sequence_bias
        self._validate_arguments()

        # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
        # is inferred in the first usage, which inhibits initializing here)
        self.length_1_bias = None
        self.prepared_bias_variables = False

    def __call__(self, input_ids, scores):
        # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        # 2 - prepares an empty bias to add
        bias = paddle.zeros_like(scores)

        # 3 - include the bias from length = 1
        if self.length_1_bias is not None:
            bias += self.length_1_bias

        # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = (
                paddle.equal(
                    input_ids[:, -prefix_length:],
                    paddle.to_tensor(sequence_ids[:-1], dtype=input_ids.dtype),
                )
                .astype(paddle.int64)
                .prod(axis=1)
            )
            bias[:, last_token] += paddle.where(
                matching_rows == 1,
                paddle.to_tensor(sequence_bias),
                paddle.to_tensor(0.0),
            )

        # 5 - apply the bias to the scores
        scores = scores + bias
        return scores

    def _prepare_bias_variables(self, scores):
        vocabulary_size = scores.shape[-1]

        # Check biased tokens out of bounds
        invalid_biases = []
        for sequence_ids in self.sequence_bias:
            for token_id in sequence_ids:
                if token_id >= vocabulary_size:
                    invalid_biases.append(token_id)
        if len(invalid_biases) > 0:
            raise ValueError(
                f"The model vocabulary size is {vocabulary_size}, but the following tokens were being biased: "
                f"{invalid_biases}"
            )

        # Precompute the bias tensors to be applied. Sequences of length 1 are kept separately, as they can be applied
        # with simpler logic.
        self.length_1_bias = paddle.zeros((vocabulary_size,))
        for sequence_ids, bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:
                self.length_1_bias[sequence_ids[-1]] = bias

        self.prepared_bias_variables = True

    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) or len(sequence_bias) == 0:
            raise ValueError(f"`sequence_bias` has to be a non-empty dictionary, but is {sequence_bias}.")
        if any(not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias.keys()):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias.keys()
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )
        if any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")


class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):
    """
    [`LogitsProcessor`] that enforces that specified sequences will never be selected.

    <Tip>

    In order to get the token ids of the words that should not appear in the generated text, make sure to set
    `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words,
    add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers,
    as fast tokenizers' prefixing behaviours come from `pre tokenizers`. Read more
    [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

    </Tip>

    Args:
        bad_words_ids (`List[List[int]]`):
            List of list of token ids that are not allowed to be generated.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

    Examples:

    ```python
    >>> from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("gpt2-en")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2-en")
    >>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

    >>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a mess.

    >>> # Now let's take the bad words out. Please note that the tokenizer is initialized differently
    >>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2-en", add_prefix_space=True)


    >>> def get_tokens_as_list(word_list):
    ...     "Converts a sequence of words into a list of tokens"
    ...     tokens_list = []
    ...     for word in word_list:
    ...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
    ...         tokens_list.append(tokenized_word)
    ...     return tokens_list


    >>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```

    >>> from paddleformers.transformers.generation import NoBadWordsLogitsProcessor, LogitsProcessorList
    >>> logits_processors = LogitsProcessorList([NoBadWordsLogitsProcessor([[5,6]], eos_token_id=tokenizer.eos_token_id)])
    >>> output_ids = model.generate(
    ...     inputs["input_ids"], max_new_tokens=5, logits_processors=logits_processors, pad_token_id=tokenizer.eos_token_id
    ... )
    >>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    In a word, the cake is a bit of a surprise.
    ```
    """

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Union[int, List[int]]):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        # Filter EOS token from bad_words_ids
        if eos_token_id is None:
            eos_token_id = []
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        bad_words_ids = list(
            filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id), bad_words_ids)
        )

        # Forbidding a sequence is equivalent to setting its bias to -inf
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, paddle.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        mask = paddle.full_like(scores, paddle.finfo(scores.dtype).min)
        for batch_id, beam_sent in enumerate(input_ids.reshape([-1, self._num_beams, input_ids.shape[-1]])):
            for beam_id, sent in enumerate(beam_sent):
                mask[batch_id * self._num_beams + beam_id, self._prefix_allowed_tokens_fn(batch_id, sent)] = 0

        return scores + mask
