# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import copy
import inspect
from typing import Optional, Union

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor
from paddle.common_ops_import import convert_dtype
from paddle.utils import map_structure

from ..transformers.model_outputs import ModelOutput
from ..transformers.utils import get_scale_by_dtype
from ..utils.log import logger
from .configuration_utils import DEFAULT_MAX_NEW_TOKENS, GenerationConfig
from .logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TopKProcess,
    TopPProcess,
)
from .stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from .streamers import BaseStreamer

__all__ = [
    "GenerationMixin",
    "BeamSearchScorer",
    "BeamHypotheses",
    "LogitsProcessorList",
    "LogitsProcessor",
    "MinLengthLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "TopKProcess",
    "TopPProcess",
    "get_unfinished_flag",
]


def get_unfinished_flag(
    input_ids: Tensor, unfinished_flag: Tensor, eos_token_id: Union[int, list[int], list[list[int]]]
) -> Tensor:
    """get unfinished flag for generation step

    Args:
        input_ids (Tensor): the input_ids
        eos_token_id (Union[int, list[int], list[list[int]]]): the end os sentence flag, which can be:
            * single token id, eg: 10
            * multiple token ids to stop generation, eg: [10, 10]
            * some more tokens to stop generations, eg: [[10], [20, 20], [30, 30, 30]]

    Returns:
        Tensor: the unfinished flag tensor
    """
    if isinstance(eos_token_id, int):
        unfinished_flag = paddle.logical_and(unfinished_flag, input_ids[:, -1:] != eos_token_id)
    else:
        batch_unfinish_flag = None
        for batch_eos_token_id in eos_token_id:
            if batch_unfinish_flag is None:
                batch_unfinish_flag = ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
            else:
                batch_unfinish_flag = paddle.logical_or(
                    batch_unfinish_flag, ~get_unfinished_flag(input_ids, unfinished_flag, batch_eos_token_id)
                )

        unfinished_flag = ~batch_unfinish_flag
    return unfinished_flag


class BeamHypotheses:
    def __init__(self, num_beams, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = get_scale_by_dtype()

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, origin_len=0):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (((hyp.shape[-1] - origin_len + 5) / 6) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len, origin_len=0):
        """
        If there are enough hypotheses and that none of the hypotheses being
        generated can become better than the worst one in the heap, then we
        are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / ((cur_len - origin_len + 5) / 6) ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


class BeamSearchScorer(object):
    """
    implementing standard beam search decoding.
    """

    def __init__(
        self,
        batch_size,
        max_length,
        num_beams,
        length_penalty=1.0,
        do_early_stopping=False,
        num_beam_hyps_to_keep=1,
        num_beam_groups=1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams, length_penalty=self.length_penalty, early_stopping=self.do_early_stopping
            )
            for _ in range(batch_size)
        ]
        self._done = paddle.to_tensor([0 for _ in range(batch_size)], dtype="int64")

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                "`num_beams` has to be an integer strictly greater than 1, but "
                "received {}. For `num_beams` == 1, one should make use of "
                "`greedy_search` instead.".format(num_beams)
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                "`num_beam_groups` has to be an integer smaller or equal than "
                "`num_beams` and `num_beams` has to be divisible by "
                "`num_beam_groups`, but received num_beam_groups={}, num_beams="
                "{}.".format(num_beam_groups, num_beams)
            )

    @property
    def is_done(self):
        return paddle.min(self._done) == 1

    def process(
        self, input_ids, next_scores, next_tokens, next_indices, origin_len=0, pad_token_id=None, eos_token_id=None
    ):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        next_beam_scores = paddle.zeros([batch_size, self.group_size], dtype=next_scores.dtype)
        next_beam_tokens = paddle.zeros([batch_size, self.group_size], dtype=next_tokens.dtype)
        next_beam_indices = paddle.zeros([batch_size, self.group_size], dtype=next_indices.dtype)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # If beam_token does not belong to top num_beams tokens,
                    # it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(input_ids[batch_beam_idx.item()].clone(), next_score.item(), origin_len)

                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token.item()
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx.item()
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    "At most {} tokens in `next_tokens[batch_idx]` can be equal "
                    "to `eos_token_id: {}`. Make sure `next_tokens[batch_idx]` "
                    "are corrected.".format(self.group_size, eos_token_id)
                )

            # Check if we are done so that we can save a pad step if all(done)
            if beam_hyp.is_done(next_scores[batch_idx].max().item(), cur_len, origin_len):
                self._done[batch_idx] = 1

        return {
            "next_beam_scores": next_beam_scores.reshape([-1]),
            "next_beam_tokens": next_beam_tokens.reshape([-1]),
            "next_beam_indices": next_beam_indices.reshape([-1]),
        }

    def finalize(
        self,
        input_ids,
        final_beam_scores,
        final_beam_tokens,
        final_beam_indices,
        origin_len=0,
        pad_token_id=None,
        eos_token_id=None,
    ):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx] == 1:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, origin_len=origin_len)

        # select the best hypotheses
        sent_lengths = paddle.zeros([batch_size * self.num_beam_hyps_to_keep], dtype=input_ids.dtype)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_score, best_hyp = sorted_hyps.pop()
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append([best_hyp, best_score])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, sent_max_len], dtype=input_ids.dtype)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded[:, :] = pad_token_id
        decoded_score = paddle.zeros([batch_size * self.num_beam_hyps_to_keep, 1])

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, (hypo, score) in enumerate(best):
            decoded[i, : sent_lengths[i].item()] = hypo.cpu().numpy()
            decoded_score[i] = score
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i].item()] = eos_token_id
        return decoded, decoded_score


class GenerationMixin(object):
    r"""
    This class implements the interface for generation task.

    It's used as the base class of `paddleformers.transformers.PretrainedModel
    """
    # enable `to_static` method for CausalLM Model
    enable_to_static_method = False

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * get_scale_by_dtype(
                return_positive=False
            )
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    @staticmethod
    def prepare_seq_len_for_generation(input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            seq_len = paddle.sum(input_ids != pad_token_id, axis=1).unsqueeze(-1)
        else:
            seq_len = paddle.full((input_ids.shape[0], 1), input_ids.shape[1], dtype="int64")
        return seq_len

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        logits_processors=None,
    ):
        processors = LogitsProcessorList()

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        if logits_processors is not None:
            custom_processors = LogitsProcessorList()
            custom_processors_type = [type(lp) for lp in logits_processors]

            for processor in processors:
                if type(processor) not in custom_processors_type:
                    custom_processors.append(processor)
            custom_processors.extend(logits_processors)

            return custom_processors
        else:
            return processors

    @staticmethod
    def expand_inputs_for_generation(input_ids, expand_size, attention_mask=None, **model_kwargs):

        index = paddle.tile(paddle.arange(input_ids.shape[0], dtype="int64").unsqueeze(-1), [1, expand_size]).reshape(
            [-1]
        )

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(attention_mask, index)

        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(token_type_ids, index)

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs and model_kwargs["seq_len"] is not None:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if "encoder_output" in model_kwargs and model_kwargs["encoder_output"] is not None:
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(encoder_output, index)

        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, model_kwargs

    @staticmethod
    def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["cache"] = outputs[1]
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs:
            model_kwargs["cache"] = outputs.past_key_values
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat([position_ids, position_ids[..., -1:] + 1], axis=-1)

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                cur_device = paddle.get_device()
                if cur_device.split(":")[0] == "npu":
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="constant")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=0)(attention_mask)
                else:
                    attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                    attention_mask = nn.Pad2D([0, 1, 0, 0], value=get_scale_by_dtype(return_positive=False))(
                        attention_mask
                    )

                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
            model_kwargs["attention_mask"] = attention_mask

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        return model_kwargs

    @staticmethod
    def update_scores_for_generation(scores, next_scores, length, unfinished_flag):
        # update scores

        unfinished_scores = (scores * paddle.to_tensor(length, dtype=scores.dtype) + next_scores) / (
            paddle.to_tensor(length, dtype=scores.dtype) + 1
        )
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs):
        if "encoder_output" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_") or argument.startswith("cross_attn") or argument == "use_cache"
                )
            }
            # Use inputs_embeds as the priority if inputs_embeds exists
            if "inputs_embeds" in encoder_kwargs:
                model_kwargs["encoder_output"] = encoder(**encoder_kwargs)
            else:
                model_kwargs["encoder_output"] = encoder(input_ids=input_ids, **encoder_kwargs)
        return model_kwargs

    def prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else bos_token_id

        decoder_input_ids = paddle.ones([input_ids.shape[0], 1], dtype="int64") * decoder_start_token_id

        return decoder_input_ids

    def get_decoder_start_token_id(self, decoder_start_token_id=None, bos_token_id=None):
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif self.config.decoder_start_token_id is not None:
            return self.config.decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        elif self.config.bos_token_id is not None:
            return self.config.bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Implement in subclasses for custom behavior to prepare inputs in the
        # generate method.

        return {"input_ids": input_ids}

    def adjust_logits_during_generation(self, logits):
        # Implement in subclasses for custom behavior to adjust the logits in
        # the generate method.

        return logits

    def prepare_fast_entry(self, kwargs):
        return False

    def _convert_to_fast(self, kwargs):
        # try general convert
        pass

    def _build_fast(self, kwargs):
        self._fast_entry = False
        if kwargs["num_beam_groups"] != 1:
            # not support for group_beam_search yet in the fast version
            raise AttributeError("'num_beam_groups != 1' is not supported yet in the fast version")
        if paddle.get_default_dtype() == "float16" and kwargs["use_fp16_decoding"] is False:
            logger.info(
                "Since the default dtype is float16, float16 would be used " "though 'use_fp16_decoding=False'."
            )
            kwargs["use_fp16_decoding"] = True
        self.prepare_fast_entry(kwargs)

    def set_pad_token_id(self, pad_token_id, eos_token_id):
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to `eos_token_id`:{} for " "open-end generation.".format(eos_token_id)
            )
            if isinstance(eos_token_id, list):
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id
        return pad_token_id

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor = None,
        generation_config: GenerationConfig = None,
        stopping_criteria: StoppingCriteria = None,
        streamer: BaseStreamer = None,
        synced_gpus: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        The interface for generation task. This method can generate sequences
        by using decoding strategy. Currently, there are three decoding
        strategies supported: "greedy_search", "sampling" and "beam_search".

        Args:
            input_ids (Tensor, optional): The input sequence ids for the
                generation. It is a Tensor with shape [batch_size, sequence_length].
                The data type should be int32 or int64. Default to None, which
                we will initialize it as a Tensor with shape [1, 1], filled
                with the value `bos_token_id`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            streamer (`~streamer.BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            kwargs (dict): It can be used to specify additional kwargs
                passed to the model.

        Returns:
            tuple[Tensor]: It is a tuple contains two elements: ids and scores.
            Each element is a Tensor.

            With the fields:

            - ids (Tensor):
                The ids of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, sequence_length]. The data
                type is same as the input `input_ids`.
            - scores (Tensor):
                The scores of the generated sequences. It is a Tensor with shape
                [batch_size * num_return_sequences, 1]. The data type is float32
                or float64, which is the same as the parameters in the model.

        Example:
            .. code-block::

                import paddle
                from paddleformers.transformers import (
                    UnifiedTransformerLMHeadModel,
                    UnifiedTransformerTokenizer
                )

                paddle.seed(2)

                # Initialize the model and tokenizer
                model_name_or_path = 'unified_transformer-12L-cn-luge'
                model = UnifiedTransformerLMHeadModel.from_pretrained(model_name_or_path)
                tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name_or_path)

                # Prepare the model inputs.
                history = "早上好，今天空气质量不错。"
                inputs = tokenizer.dialogue_encode(history, task_type='chitchat',
                    add_start_token_as_response=True, return_tensors=True)

            .. code-block::

                # Generate the sequence by using "greedy_search" strategy
                ids, scores = model.generate(
                    **inputs,
                    decode_strategy="greedy_search")
                print(ids.shape, scores.shape)
                # [1, 3] [1, 1]
                sequence_ids = ids.cpu().numpy().tolist()[0]
                sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                response = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                print(response)
                # 是的

            .. code-block::

                # Generate 2 sequences by using "sampling" strategy (top_k=5)
                generation_config = GenerationConfig(
                    decode_strategy="sampling",
                    top_k=5,
                    num_return_sequences=2
                )
                ids, scores = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    )
                print(ids.shape, scores.shape)
                # [2, 7] [2, 1]
                response = []
                for sequence_ids in ids.cpu().numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['天气好,心情也好', '你也是']

            .. code-block::

                # Generate 2 sequences by using "beam_search" strategy (num_beams=5)
                generation_config = GenerationConfig(
                    decode_strategy="beam_search",
                    num_beams=5,
                    num_return_sequences=2
                )
                ids, scores = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    )
                print(ids.shape, scores.shape)
                # [2, 3] [2, 1]
                response = []
                for sequence_ids in ids.cpu().numpy().tolist():
                    sequence_ids = sequence_ids[:sequence_ids.index(tokenizer.sep_token_id)]
                    text = tokenizer.convert_ids_to_string(sequence_ids, keep_space=False)
                    response.append(text)
                print(response)
                # ['是的', '嗯嗯']
        """
        if generation_config is None:
            if self.generation_config is None or self.generation_config._from_model_config:
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    logger.warning(
                        "model.generation_config is in conflict with model.config, " "model.config is used."
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        # without update model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        assert generation_config.decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            generation_config.decode_strategy
        )

        if getattr(self, "deprecated_warnings", None) is None:
            self.deprecated_warnings = {}

        use_fast = False
        if "use_faster" in model_kwargs:
            raise ValueError("`use_faster` is deprecated now.")

        if "use_fast" in model_kwargs:
            raise ValueError("`use_fast` is deprecated now.")

        bos_token_id = (
            generation_config.bos_token_id if generation_config.bos_token_id is not None else self.config.bos_token_id
        )
        eos_token_id = (
            generation_config.eos_token_id if generation_config.eos_token_id is not None else self.config.eos_token_id
        )
        pad_token_id = (
            generation_config.pad_token_id if generation_config.pad_token_id is not None else self.config.pad_token_id
        )
        forced_bos_token_id = (
            generation_config.forced_bos_token_id
            if generation_config.forced_bos_token_id is not None
            else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            generation_config.forced_eos_token_id
            if generation_config.forced_eos_token_id is not None
            else self.config.forced_eos_token_id
        )
        decoder_start_token_id = (
            generation_config.decoder_start_token_id
            if generation_config.decoder_start_token_id is not None
            else self.config.decoder_start_token_id
        )
        no_repeat_ngram_size = (
            generation_config.no_repeat_ngram_size
            if generation_config.no_repeat_ngram_size is not None
            else self.config.no_repeat_ngram_size
        )

        if getattr(self, "_fast_entry", None) is not False and use_fast:
            fg_args = locals()
            fg_args.pop("self")
            fg_args.pop("__class__", None)
            model_kwargs = fg_args.pop("model_kwargs")
            fg_args.update(model_kwargs)
            try:
                if getattr(self, "_fast_entry", None) is None:
                    self._build_fast(fg_args)
                if self._fast_entry:
                    output = self._fast_entry(**fg_args)
                    if isinstance(output, tuple):
                        output_ids, dummy_srore = output
                    else:
                        output_ids = output
                        # make result and fast result oneconsistent
                        dummy_srore = None
                    if generation_config.decode_strategy == "beam_search":
                        output_ids = output_ids.transpose([1, 2, 0])
                        output_ids = output_ids[:, : generation_config.num_return_sequences, :].reshape(
                            [-1, output_ids.shape[-1]]
                        )
                        if dummy_srore is not None:
                            dummy_srore = dummy_srore[:, : generation_config.num_return_sequences].flatten()
                    else:
                        output_ids = output_ids.transpose([1, 0])
                    return output_ids, dummy_srore

            except Exception as e:
                fg_args["model_kwargs"] = model_kwargs
                # TODO
                # Prevent self._convert_to_fast to throw Exception
                self._convert_to_fast(fg_args)
                logger.warning(e)
                logger.warning("FastGeneration is not available, " "and the original version would be used instead.")

        # input_ids in model_kwargs is supported
        if "input_ids" in model_kwargs:
            _input_ids = model_kwargs.pop("input_ids")
            if input_ids is None:
                input_ids = _input_ids

        # params check
        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)
        elif "inputs_embeds" in model_kwargs:
            # Add input embeds support
            input_ids = self.prepare_input_ids_for_generation(
                bos_token_id, encoder_output=model_kwargs["inputs_embeds"]
            )

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )
        self.is_encoder_decoder = self.config.is_encoder_decoder

        if self.is_encoder_decoder:
            model_kwargs = self.prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)
            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self.prepare_decoder_input_ids_for_generation(
                    input_ids, decoder_start_token_id, bos_token_id
                )
        # streamer
        if streamer is not None:
            # streamer couldn't support beam_search strategy
            if generation_config.decode_strategy == "beam_search" or generation_config.num_beams > 1:
                raise ValueError(
                    "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
                )

        pad_token_id = self.set_pad_token_id(pad_token_id, eos_token_id)

        if generation_config.max_length != 0 and generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS:
            logger.warning("`max_length` will be deprecated in future releases, use `max_new_tokens` instead.")
            generation_config.max_new_tokens = generation_config.max_length

        if generation_config.min_length != 0 and generation_config.min_new_tokens == 0:
            logger.warning("`min_length` will be deprecated in future releases, use `min_new_tokens` instead.")
            generation_config.min_new_tokens = generation_config.min_length

        max_length = generation_config.max_new_tokens
        min_length = generation_config.min_new_tokens

        input_len = input_ids.shape[-1]
        min_len = input_len + min_length
        max_len = input_len + max_length

        logits_processors = self.get_logits_processor(
            min_length=min_len if min_length > 0 else None,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=generation_config.num_beams,
            num_beam_groups=generation_config.num_beam_groups,
            diversity_rate=generation_config.diversity_rate,
            repetition_penalty=generation_config.repetition_penalty,
            no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
            logits_processors=model_kwargs["logits_processors"]
            if "logits_processors" in model_kwargs
            and isinstance(model_kwargs["logits_processors"], LogitsProcessorList)
            else None,
        )
        if "logits_processors" in model_kwargs:
            model_kwargs.pop("logits_processors")

        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.decode_strategy == "greedy_search":
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "`num_return_sequences` has to be 1, but is {} "
                    "when doing greedy search.".format(generation_config.num_return_sequences)
                )
            return self.greedy_search(
                input_ids,
                logits_processors,
                max_len,
                pad_token_id,
                eos_token_id,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
                fast_ptq_sampling=generation_config.fast_ptq_sampling,
                trunc_input=generation_config.trunc_input,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_config.decode_strategy == "sampling":
            if generation_config.num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=generation_config.num_return_sequences, **model_kwargs
                )

            return self.sample(
                input_ids,
                logits_processors,
                max_len,
                pad_token_id,
                eos_token_id,
                generation_config.top_k,
                generation_config.top_p,
                generation_config.temperature,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
                fast_ptq_sampling=generation_config.fast_ptq_sampling,
                trunc_input=generation_config.trunc_input,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_config.decode_strategy == "beam_search":
            batch_size = input_ids.shape[0]
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError(
                    "`num_return_sequences` has to be smaller or equal to "
                    "`num_beams`. But received `num_return_sequences` is {}, "
                    "`num_beams` is {}".format(generation_config.num_return_sequences, generation_config.num_beams)
                )
            if generation_config.num_beams <= 1:
                raise ValueError(
                    "`num_beams` has to be bigger than 1. But received "
                    "`num_beams` is {}. If `num_beams` is 1, `decode_strategy` "
                    "should be 'greedy_search'".format(generation_config.num_beams)
                )
            if generation_config.num_beam_groups > 1:
                diverse_beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=generation_config.num_beams,
                    length_penalty=generation_config.length_penalty,
                    do_early_stopping=generation_config.early_stopping,
                    num_beam_hyps_to_keep=generation_config.num_return_sequences,
                    num_beam_groups=generation_config.num_beam_groups,
                )

                # interleave with `num_beams`
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=generation_config.num_beams, **model_kwargs
                )

                return self.group_beam_search(
                    input_ids,
                    diverse_beam_scorer,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    stopping_criteria=stopping_criteria,
                    fast_ptq_sampling=generation_config.fast_ptq_sampling,
                    trunc_input=generation_config.trunc_input,
                    synced_gpus=synced_gpus,
                    **model_kwargs,
                )
            else:
                beam_scorer = BeamSearchScorer(
                    batch_size=batch_size,
                    max_length=max_len,
                    num_beams=generation_config.num_beams,
                    length_penalty=generation_config.length_penalty,
                    do_early_stopping=generation_config.early_stopping,
                    num_beam_hyps_to_keep=generation_config.num_return_sequences,
                )

                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=generation_config.num_beams, **model_kwargs
                )

                return self.beam_search(
                    input_ids,
                    beam_scorer,
                    logits_processors,
                    max_len,
                    generation_config.diversity_rate,
                    pad_token_id,
                    eos_token_id,
                    stopping_criteria=stopping_criteria,
                    fast_ptq_sampling=generation_config.fast_ptq_sampling,
                    trunc_input=generation_config.trunc_input,
                    synced_gpus=synced_gpus,
                    **model_kwargs,
                )

    def greedy_search(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        stopping_criteria=None,
        streamer=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if synced_gpus and generate_end:
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            probs = logits_processors(input_ids, next_token_logits)
            # greedy
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            cur_len += 1

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())

            if stopping_criteria(input_ids, scores):
                generate_end = True

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            # Stop when there is a </s> in all sentences
            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

        if streamer is not None:
            streamer.end()

        return input_ids[:, origin_len:] if trunc_input else input_ids, scores

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        stopping_criteria=None,
        streamer=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # NOTE: to decrease ref-count and clear outdate cache in-time
            model_kwargs["cache"] = None
            model_kwargs["past_key_values"] = None
            outputs = self(**model_inputs)
            if synced_gpus and generate_end:
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            origin_probs = paddle.log(origin_probs)
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                probs = TopPProcess(probs, top_p, min_tokens_to_keep)
            if paddle.device.is_compiled_with_custom_device("gcu"):
                probs = paddle.cast(probs, "float32")
            if paddle.device.is_compiled_with_xpu():
                probs = paddle.cast(probs, "float32")

            # multinomial already support fp16 and bf16 currently, fix issue: https://github.com/PaddlePaddle/Paddle/issues/51852
            next_tokens = paddle.multinomial(probs)

            if self.config.tensor_parallel_degree > 1:
                # Maybe no need to broadcast if seed is set correctly.
                from paddle.distributed import fleet

                try:
                    hcg = fleet.get_hybrid_communicate_group()
                    group = hcg.get_model_parallel_group()
                    src = hcg.get_model_parallel_group_src_rank()
                except:
                    group, src = None, 0
                paddle.distributed.broadcast(next_tokens, src=src, group=group)
            # config does not include pipeline_parallel_degree, and pipeline parallel
            # uses trainer.model_wrapped to run in both train and predict mode
            # which has pp_group as a attribute
            # TODO(guosheng): only let the last stage of pipeline to do softmax
            # and sampling, and then broadcast to avoid broadcast logits.
            if getattr(self, "pp_group", None) is not None:
                paddle.distributed.broadcast(
                    next_tokens, src=self.pp_group.ranks[0], group=self.pp_group  # use rank 0 for same seed to check
                )

            next_scores = paddle.index_sample(origin_probs, next_tokens)
            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())

            if stopping_criteria(input_ids, scores):
                generate_end = True

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            # Stop when there is a </s> in all sentences
            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

        if streamer is not None:
            streamer.end()

        return input_ids[:, origin_len:] if trunc_input else input_ids, scores

    def _get_model_inputs_spec(self, dtype: str):
        spec = {
            "input_ids": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
            "attention_mask": paddle.static.InputSpec(shape=[None, None], dtype="int64"),
        }
        if "position_ids" in inspect.getfullargspec(self.forward).args:
            spec["position_ids"] = paddle.static.InputSpec(shape=[None, None], dtype="int64")
        return spec

    def to_static(self, path: str, config: dict):
        """export generation model to static

        Args:
            path (str): path of saved inference model
            config (dict): configuration for generation
                bos_token_id (int): token id of begin-of-sentence
                eos_token_id (int): token id of end-of-sentence
                pad_token_id (int): token id of pad token
                use_top_p (bool): whether use top_p decoding strategy
        """

        use_top_p = config.get("use_top_p", True)

        top_k_spec = paddle.static.InputSpec(shape=[1], dtype="int64") if not use_top_p else 0

        top_p_spec = paddle.static.InputSpec(shape=[1], dtype="float32") if use_top_p else 1.0
        temperature = paddle.static.InputSpec(shape=[1], dtype="float32") if use_top_p else 1.0
        dtype = config.get("dtype", None)

        logits_processors = config.get("logits_processors", None)
        model_inputs_spec = self._get_model_inputs_spec(dtype)

        input_spec = [
            model_inputs_spec["input_ids"],  # input_ids
            model_inputs_spec["attention_mask"],  # attention_mask
            model_inputs_spec.get("position_ids", None),  # attention_mask
            logits_processors,
            paddle.static.InputSpec(shape=[1], dtype="int64"),  # max_length
            self.generation_config.pad_token_id or config.get("pad_token_id", None),
            self.generation_config.eos_token_id or config.get("eos_token_id", None),
            top_k_spec,  # top_k
            top_p_spec,  # top_p
            temperature,  # temperature
            1,
        ]

        model = paddle.jit.to_static(self.sample_d2s, input_spec=input_spec)

        paddle.jit.save(model, path)

    def sample_d2s(
        self,
        input_ids,
        attention_mask,
        position_ids,
        logits_processors,
        max_new_tokens,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
    ):

        pad_token_id = self.set_pad_token_id(pad_token_id, eos_token_id)
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        if paddle.is_tensor(top_k) and not paddle.is_tensor(top_p):
            use_top_p = False
        elif not paddle.is_tensor(top_k) and paddle.is_tensor(top_p):
            use_top_p = True

        # top_k and top_p are the const value
        elif isinstance(top_p, float) or isinstance(top_k, int):
            use_top_p = True
        else:
            if top_p is None and top_k is None:
                raise ValueError("top_k and top_p should not be None")
            raise ValueError(
                "you should not specify InputSpec for top_k and top_p parameters, one of InputSpec is expected"
            )

        batch_size, cur_len = input_ids.shape
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = input_ids.shape[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")

        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        # use_cache is immutable, we split it off other mutable kwargs.
        immutable = {"use_cache": True}
        model_kwargs = {"attention_mask": attention_mask, "position_ids": position_ids}

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            assert "use_cache" in model_inputs
            del model_inputs["use_cache"]
            return self(**model_inputs, **immutable)

        def _post_process_(
            outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs, pad_token_id
        ):
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)

            logits = logits_processors(input_ids, logits)
            probs = F.softmax(logits)

            # sample
            origin_probs = F.log_softmax(logits)
            # compute next_tokens
            if use_top_p:
                logits = logits / temperature
                top_ps_tensor = paddle.full(shape=[probs.shape[0], 1], fill_value=top_p, dtype=probs.dtype)
                _, next_tokens = paddle.tensor.top_p_sampling(probs, top_ps_tensor)
            else:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
                if top_k == 1:
                    next_tokens = paddle.unsqueeze_(paddle.argmax(probs, axis=-1), -1)
                else:
                    next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)
            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            return input_ids, scores, unfinished_flag, model_kwargs

        outputs = _forward_(**model_kwargs)
        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len_gpu, origin_len_gpu, scores, unfinished_flag, model_kwargs, pad_token_id
        )

        cur_len += 1
        cur_len_gpu += 1

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        model_kwargs["attention_mask"] = paddle.reshape(attn_mask, attn_mask.shape)
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None
        max_new_tokens = paddle.full([1], max_new_tokens + cur_len - 1, dtype="int64")

        while cur_len < max_new_tokens and paddle.any(unfinished_flag):
            input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                input_ids,
                cur_len_gpu,
                origin_len_gpu,
                scores,
                unfinished_flag,
                model_kwargs,
                pad_token_id,
            )
            cur_len += 1
            cur_len_gpu += 1

        return input_ids[:, origin_len:], scores

    def reorder_cache(self, cache, beam_idx):
        cache = map_structure(lambda x: paddle.index_select(x, beam_idx), cache)
        return cache

    def beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processors,
        max_length,
        diversity_rate,
        pad_token_id,
        eos_token_id,
        stopping_criteria=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.zeros((batch_size, num_beams), dtype=paddle.get_default_dtype())

        beam_scores[:, 1:] = get_scale_by_dtype(return_positive=False)
        beam_scores = paddle.reshape(beam_scores, [-1])

        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)
            if synced_gpus and generate_end:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = self.adjust_logits_during_generation(logits)
            # beam search
            # [batch_size * num_beams, vocab_size]
            next_scores = F.softmax(logits)
            next_scores = paddle.log(next_scores)
            next_scores = logits_processors(input_ids, next_scores)
            next_scores = next_scores + beam_scores.unsqueeze(-1)

            vocab_size = next_scores.shape[-1]
            if diversity_rate == 0.0:
                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, num_beams * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

            else:
                next_scores, next_tokens = paddle.topk(next_scores, 2 * num_beams, axis=1)

                sibling_score = paddle.arange(1, 2 * num_beams + 1, dtype="int64").unsqueeze(0) * diversity_rate

                diversed_score = next_scores - sibling_score

                next_scores = next_scores.reshape([batch_size, 2 * num_beams * num_beams])
                next_tokens = next_tokens.reshape([batch_size, 2 * num_beams * num_beams])

                diversed_score = diversed_score.reshape([batch_size, 2 * num_beams * num_beams])
                diversed_score, diversed_tokens = paddle.topk(diversed_score, 2 * num_beams, axis=1)

                # TODO
                # Use gather_nd() to select origan token and score
                next_scores = paddle.stack(
                    [paddle.index_select(next_scores[i], diversed_tokens[i]) for i in range(next_scores.shape[0])]
                )
                next_tokens = paddle.stack(
                    [paddle.index_select(next_tokens[i], diversed_tokens[i]) for i in range(next_tokens.shape[0])]
                )

                next_indices = diversed_tokens // (2 * num_beams)

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_scores,
                next_tokens,
                next_indices,
                origin_len=origin_len,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            # beam_idx may contain element -1 and cause error
            # PR: https://github.com/PaddlePaddle/Paddle/issues/57366
            beam_idx = paddle.maximum(beam_idx, paddle.full_like(beam_idx, 0))

            cur_len += 1
            input_ids = paddle.concat(
                [paddle.index_select(input_ids, beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
            )

            if beam_scorer.is_done or stopping_criteria(input_ids, beam_scores):
                if not synced_gpus:
                    break
                else:
                    generate_end = True

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )
            if "cache" in model_kwargs:
                # reorder the cache
                model_kwargs["cache"] = self.reorder_cache(model_kwargs["cache"], beam_idx)
            if "past_key_values" in model_kwargs:
                # reorder the cache
                model_kwargs["past_key_values"] = self.reorder_cache(model_kwargs["past_key_values"], beam_idx)
            if fast_ptq_sampling:
                break

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:] if trunc_input else input_ids, scores

    def group_beam_search(
        self,
        input_ids,
        beam_scorer,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        stopping_criteria=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()

        # max_length will be convert to MaxLengthCriteria
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            # logger.warning(
            #    "`max_length` is deprecated in this function, use"
            #    " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead."
            # )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups

        batch_beam_size, cur_len = input_ids.shape
        origin_len = cur_len

        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {}, but received {}.".format(
            num_beams * batch_size, batch_beam_size
        )

        beam_scores = paddle.full((batch_size, num_beams), get_scale_by_dtype(return_positive=False), dtype="float32")
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = paddle.reshape(beam_scores, [-1])

        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break
            # predicted tokens in cur_len step
            current_tokens = paddle.zeros(shape=[batch_size * num_beams], dtype=input_ids.dtype)

            # indices which will form the beams in the next time step
            reordering_indices = paddle.zeros(shape=[batch_size * num_beams], dtype="int64")
            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)
            if synced_gpus and generate_end:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )

                group_input_ids = input_ids[batch_group_indices]

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif isinstance(outputs, ModelOutput):
                    logits = outputs.logits
                else:
                    logits = outputs

                logits = logits[:, -1, :]
                logits = paddle.index_select(logits, paddle.to_tensor(batch_group_indices))
                logits = self.adjust_logits_during_generation(logits)

                next_scores = F.softmax(logits)
                next_scores = paddle.log(next_scores)
                vocab_size = next_scores.shape[-1]

                next_scores = logits_processors(
                    group_input_ids, next_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )

                next_scores = next_scores + beam_scores[batch_group_indices].unsqueeze(-1)

                # reshape for beam search
                next_scores = next_scores.reshape([batch_size, group_size * vocab_size])

                next_scores, next_tokens = paddle.topk(next_scores, 2 * group_size, axis=1)

                next_indices = next_tokens // vocab_size
                next_tokens = next_tokens % vocab_size

                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_scores,
                    next_tokens,
                    next_indices,
                    origin_len=origin_len,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]
                # beam_idx may contain element -1 and cause error
                # PR: https://github.com/PaddlePaddle/Paddle/issues/57366
                beam_idx = paddle.maximum(beam_idx, paddle.full_like(beam_idx, 0))

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = paddle.concat(
                    [paddle.index_select(group_input_ids, index=beam_idx), beam_next_tokens.unsqueeze(-1)], axis=-1
                )
                current_tokens[batch_group_indices] = beam_next_tokens

                reordering_indices[batch_group_indices] = (
                    num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
                )

            input_ids = paddle.concat([input_ids, current_tokens.unsqueeze(-1)], axis=-1)

            cur_len += 1

            if beam_scorer.is_done or stopping_criteria(input_ids, beam_scores):
                if not synced_gpus:
                    break
                else:
                    generate_end = True

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            if "cache" in model_kwargs:
                # reorder the cache
                model_kwargs["cache"] = self.reorder_cache(model_kwargs["cache"], reordering_indices)
            if "past_key_values" in model_kwargs:
                # reorder the cache
                model_kwargs["past_key_values"] = self.reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            if fast_ptq_sampling:
                break

        pred_ids, scores = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            origin_len=origin_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return pred_ids[:, origin_len:] if trunc_input else input_ids, scores
