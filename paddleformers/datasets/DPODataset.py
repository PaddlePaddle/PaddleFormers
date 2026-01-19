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
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from paddle.io import IterableDataset

from paddleformers.datasets.data_utils import (
    get_worker_sliced_iterator,
    postprocess_fc_sequence,
    print_debug_info,
)
from paddleformers.datasets.reader.mix_datasets import create_dataset_instance
from paddleformers.datasets.reader.multi_source_datasets import MultiSourceDataset
from paddleformers.utils.env import NONE_CHAT_TEMPLATE
from paddleformers.utils.log import logger

LOGGER_COUNT = 0


@dataclass
class Sequence:
    """Sequence."""

    token_ids: Optional[List[int]]
    position_ids: Optional[List[int]]
    attention_mask: Optional[List[List[int]]]
    attn_mask_startend_row_indices: Optional[List[int]]
    chosen_labels: List[int]
    rejected_labels: List[int]
    response_index: List[int]
    score_delta: float


class DPODataSet(IterableDataset):
    def __init__(self, **dataset_config):

        # parameter init
        self.tokenizer = dataset_config.get("tokenizer", None)
        self.processor = dataset_config.get("processor", None)
        self.max_seq_len = dataset_config.get("max_seq_len", 8192)
        self.mask_out_eos_token = dataset_config.get("mask_out_eos_token", True)
        self.template = dataset_config.get("template_instance", None)
        self.use_attn_mask_startend_row_indices = dataset_config.get("use_attn_mask_startend_row_indices", True)
        self.template_backend = dataset_config.get("template_backend", "jinja")
        self.split_multi_turn = dataset_config.get("split_multi_turn", False)
        self.is_valid = dataset_config.get("is_valid", False)
        self.packing = dataset_config.get("packing", False)
        self.greedy_intokens = dataset_config.get("greedy_intokens", True)
        self.buffer_size = dataset_config.get("buffer_size", 500)
        self.efficient_eos = True if not self.template else getattr(self.template, "efficient_eos", True)
        self.use_filtered_label_loss = dataset_config.get("use_filtered_label_loss", False)

        # data loader + multisource dataset mix
        if self.is_valid:
            dataset_config["random_shuffle"] = False
            dataset_config["greedy_intokens"] = False
            multi_source_dataset = MultiSourceDataset(**dataset_config)
            self.mix_datasets = create_dataset_instance(
                "concat",
                multi_source_dataset,
                **dataset_config,
            )
        else:
            multi_source_dataset = MultiSourceDataset(**dataset_config)
            self.mix_datasets = create_dataset_instance(
                dataset_config["mix_strategy"],
                multi_source_dataset,
                **dataset_config,
            )

    def __len__(self):
        return len(self.mix_datasets)

    def __iter_func(self):

        # prepare epoch data
        batch_sequence, cur_len = [], 0
        dataset_iterator = get_worker_sliced_iterator(self.mix_datasets)

        if not self.packing:
            for _ in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                try:
                    sequence = self._postprocess_sequence(example)
                except Exception as e:
                    print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                    continue
                if sequence is None:
                    continue

                batch_sequence, cur_len = [sequence], len(sequence.token_ids)
                yield batch_sequence

            if len(batch_sequence) > 0:
                yield batch_sequence
        else:
            if not self.greedy_intokens:
                # base
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    try:
                        sequence = self._postprocess_sequence(example)
                    except Exception as e:
                        print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                        continue
                    if sequence is None:
                        continue
                    if cur_len + len(sequence.token_ids) <= self.max_seq_len:
                        batch_sequence.append(sequence)
                        cur_len += len(sequence.token_ids)
                    else:
                        yield batch_sequence
                        batch_sequence, cur_len = [sequence], len(sequence.token_ids)

                if len(batch_sequence) > 0:
                    yield batch_sequence
            else:
                sequence_buffer = []
                buffer_size = self.buffer_size
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    try:
                        sequence = self._postprocess_sequence(example)
                    except Exception as e:
                        print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                        continue
                    if sequence is None:
                        continue
                    sequence_buffer.append(sequence)

                    if len(sequence_buffer) == buffer_size:
                        sequence_pack = self._generate_greedy_packs(sequence_buffer)
                        for pack in sequence_pack:
                            yield pack
                        sequence_buffer = []
                if len(sequence_buffer) > 0:
                    sequence_pack = self._generate_greedy_packs(sequence_buffer)
                    for pack in sequence_pack:
                        yield pack

    def __iter__(self):
        """
        Rewrite the __iter__ method to implement dataset iteration.
        Each iteration returns a Sequence-type element.
        """
        if self.is_valid:
            yield from self.__iter_func()
        else:
            while True:
                yield from self.__iter_func()

    def _generate_greedy_packs(self, sequences):
        """Generate packed sequences using greedy strategy.

        Args:
            examples: List of examples to pack.
            actual_example_num_list: List of example counts.

        Returns:
            list: List of packed sequences.
        """

        left_len_list = np.array([])
        sequence_pack = []
        for sequence in sequences:
            sequence_len = len(sequence.token_ids)
            if len(left_len_list) > 0:
                max_left_len_index = left_len_list.argmax()

            if len(left_len_list) == 0 or left_len_list[max_left_len_index] < sequence_len:
                sequence_pack.append([sequence])
                left_len_list = np.append(left_len_list, np.array([self.max_seq_len - sequence_len]))
            else:
                sequence_pack[max_left_len_index].append(sequence)
                left_len_list[max_left_len_index] -= sequence_len
        return sequence_pack

    def _preprocess_dpo_example(self, example):

        chosen_m, rejected_m = deepcopy(example["messages"]), deepcopy(example["messages"])
        if self.template_backend == "jinja":
            # The Jinja backend will concatenate the "system" separately and place it at the beginning.
            session_start_index = (
                len(example["messages"])
                if example["messages"][0]["role"] != "system"
                else len(example["messages"]) - 1
            )
        else:
            # Custom backends will concatenate the "system" message and the first "user" message together.
            session_start_index = len(example["messages"])
        chosen_m.extend(example["chosen_response"])
        rejected_m.extend(example["rejected_response"])

        example["chosen"] = {"messages": chosen_m}
        example["rejected"] = {"messages": rejected_m}
        example["session_start_index"] = session_start_index
        example["score_delta"] = 1.0

        return example

    def __postprocess_before_concat(self, example):
        """Process multi-turn conversation data into tokenized sequences with dynamic truncation."""
        prompt_token_ids = []

        cur_len = 0

        system = example.get("system", None)
        tools = example.get("tools", None)
        images = example.get("images", [])
        videos = example.get("videos", [])
        audios = example.get("audios", [])

        if self.template_backend == "jinja":
            if not self.tokenizer.chat_template:
                self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
            if self.split_multi_turn:
                chosen_encoded_messages = postprocess_fc_sequence(self.tokenizer, example["chosen"])
                rejected_encoded_messages = postprocess_fc_sequence(self.tokenizer, example["rejected"])
            else:
                chosen_encoded_messages = self.tokenizer.encode_chat_inputs(example["chosen"])
                rejected_encoded_messages = self.tokenizer.encode_chat_inputs(example["rejected"])
        else:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                images, videos, audios, [len(images)], [len(videos)], [len(audios)], None, self.processor
            )
            chosen_messages = self.template.mm_plugin.process_messages(
                example["chosen"]["messages"], images, videos, audios, mm_inputs, self.processor
            )
            rejected_messages = self.template.mm_plugin.process_messages(
                example["rejected"]["messages"], images, videos, audios, mm_inputs, self.processor
            )
            chosen_encoded_messages = self.template.encode_multiturn(self.tokenizer, chosen_messages, system, tools)
            rejected_encoded_messages = self.template.encode_multiturn(
                self.tokenizer, rejected_messages, system, tools
            )

        if self.template_backend == "custom":
            suffix_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
        else:
            suffix_ids = [self.tokenizer.eos_token_id]
        if self.efficient_eos:
            if len(chosen_encoded_messages) > 0 and len(chosen_encoded_messages[-1]) > 1:
                chosen_encoded_messages[-1][1].extend(suffix_ids)
            if len(rejected_encoded_messages) > 0 and len(rejected_encoded_messages[-1]) > 1:
                rejected_encoded_messages[-1][1].extend(suffix_ids)

        # chosen/rejected response
        response_token_ids_list = []
        response_label_ids_list = []
        response_len_list = []
        split_index = example["session_start_index"] // 2
        for responses in [
            chosen_encoded_messages[split_index:],
            rejected_encoded_messages[split_index:],
        ]:
            responses_token_ids = []
            responses_label_ids = []
            response_len = 0
            for i, response in enumerate(responses):
                q, a = response
                label_ids, res = [], []

                if i != 0:
                    # prompt
                    label_ids += [0] * (len(q) - 1)
                    res += q

                # response
                if self.mask_out_eos_token:
                    label_ids += a[:-1] + [0, 0]
                    response_len += len(a) - 1
                    res += a
                else:
                    label_ids += a + [0]
                    response_len += len(a)
                    res += a
                responses_token_ids += res
                responses_label_ids += label_ids
            response_token_ids_list.append(responses_token_ids)
            response_label_ids_list.append(responses_label_ids)
            response_len_list.append(response_len)

        cur_len += sum(map(len, response_token_ids_list))

        # create at least one turn
        turn_index = split_index
        while turn_index >= 0:
            if turn_index == split_index:
                cur_turn_token = chosen_encoded_messages[turn_index][0]
            else:
                cur_turn_token = chosen_encoded_messages[turn_index][0] + chosen_encoded_messages[turn_index][1]

            if cur_len + len(cur_turn_token) > self.max_seq_len:
                break

            prompt_token_ids = cur_turn_token + prompt_token_ids
            cur_len += len(cur_turn_token)
            turn_index -= 1

        # at least one turn
        if turn_index == split_index:
            sub_src = example["chosen"]["messages"][0]["content"].strip()[:5]
            global LOGGER_COUNT
            LOGGER_COUNT += 1
            if LOGGER_COUNT <= 5:
                logger.warning(
                    f"[SKIP] max_seq_len({self.max_seq_len}) is insufficient to include "
                    f"even one turn, example_output:'{{'src':[{sub_src}, ……]}}'"
                )
            return (None,) * 5

        if cur_len > self.max_seq_len:
            logger.warning(f"[SKIP] Example is too long: {example}")
            return (None,) * 5

        return (
            prompt_token_ids,
            response_token_ids_list,
            response_label_ids_list,
            response_len_list,
            cur_len,
        )

    def _postprocess_sequence(self, example):
        example = self._preprocess_dpo_example(example)
        # sequence: system + knowledge_tokens + prompt + chosen + reject
        (
            prompt_token_ids,
            response_token_ids_list,
            response_label_ids_list,
            response_len_list,
            cur_len,
        ) = self.__postprocess_before_concat(example)

        # The sequnece is too long, just return None
        if prompt_token_ids is None:
            return None
        # 1.concat all tokens
        # 1.1 input_ids
        input_ids = prompt_token_ids + response_token_ids_list[0] + response_token_ids_list[1]
        if cur_len != len(input_ids):
            logger.warning(f"[SKIP] code bug: {example}")
            return None

        # 1.2. position_ids
        prompt_len = len(prompt_token_ids)
        chosen_len = len(response_token_ids_list[0])
        rejected_len = len(response_token_ids_list[1])
        position_ids = (
            list(range(prompt_len))  # prompt
            + list(range(prompt_len, prompt_len + chosen_len))  # chosen
            + list(range(prompt_len, prompt_len + rejected_len))  # rejected
        )

        # 1.3 labels
        chosen_labels = [0] * (prompt_len - 1) + response_label_ids_list[0] + [0] * len(response_token_ids_list[1])
        rejected_labels = [0] * (prompt_len - 1) + [0] * len(response_token_ids_list[0]) + response_label_ids_list[1]

        # 1.4 response index
        if self.use_filtered_label_loss:
            response_index = [0, response_len_list[0], sum(response_len_list)]
        else:
            response_index = [
                prompt_len,
                prompt_len + len(response_token_ids_list[0]),
                prompt_len + sum(response_len_list),  # end
            ]

        # 1.5 attention mask
        if self.use_attn_mask_startend_row_indices:
            attn_mask_startend_row_indices = (
                [cur_len] * (prompt_len) + [prompt_len + chosen_len] * chosen_len + [cur_len] * rejected_len
            )
            attention_mask = None
        else:
            attention_mask = np.tri(cur_len, cur_len, dtype=bool)
            attention_mask[
                (prompt_len + chosen_len) :,
                prompt_len : (prompt_len + chosen_len),
            ] = False
            attn_mask_startend_row_indices = None

        # print
        enable_dataset_debug = os.getenv("FLAGS_enable_dataset_debug", "false").lower() in ("true", "1", "t")
        if enable_dataset_debug:
            logger.info("\n" + "=" * 50)
            logger.info("[dataset debug] Debug mode enabled")
            if hasattr(self, "tokenizer"):
                print("========================================")
                print_debug_info(self.tokenizer, input_ids, "input")
                print("========================================\n")

                filtered_labels = [x for x in chosen_labels if x != 0]  # remove -100
                print("========================================")
                print_debug_info(self.tokenizer, filtered_labels, "chosen_labels")
                print("========================================\n")

                filtered_labels = [x for x in rejected_labels if x != 0]  # remove -100
                print("========================================")
                print_debug_info(self.tokenizer, filtered_labels, "rejected_labels")
                print("========================================\n")
            else:
                logger.info("[dataset debug] Tokenizer not available")
            logger.info("=" * 50 + "\n")

        # 2. return sequence
        return Sequence(
            token_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            response_index=response_index,
            score_delta=example["score_delta"],
        )
