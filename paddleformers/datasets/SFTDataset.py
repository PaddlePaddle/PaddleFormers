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
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from paddle.io import IterableDataset

from paddleformers.datasets.data_utils import (
    get_worker_sliced_iterator,
    postprocess_fc_sequence,
    print_debug_info,
)
from paddleformers.datasets.reader.mix_datasets import create_dataset_instance
from paddleformers.datasets.reader.multi_source_datasets import MultiSourceDataset
from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.utils.env import NONE_CHAT_TEMPLATE
from paddleformers.utils.log import logger


@dataclass
class Sequence:
    """Encapsulated sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    num_examples: int
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    mm_inputs: Dict = field(default_factory=dict)


class SFTDataSet(IterableDataset):
    def __init__(self, **dataset_config):

        # parameter init
        self.tokenizer = dataset_config.get("tokenizer", None)
        self.processor = dataset_config.get("processor", None)
        self.max_seq_len = dataset_config.get("max_seq_len", 8192)
        self.template = dataset_config.get("template_instance", None)
        self.template_backend = dataset_config.get("template_backend", "jinja")
        self.use_template = dataset_config.get("use_template", True)
        self.efficient_eos = True if not self.template else getattr(self.template, "efficient_eos", True)
        self.auto_add_bos = True if not self.template else getattr(self.template, "auto_add_bos", False)
        self.split_multi_turn = dataset_config.get("split_multi_turn", False)
        self.encode_one_turn = dataset_config.get("encode_one_turn", True)
        self.is_pretraining = dataset_config.get("is_pretraining", False)
        self.truncate_packing = dataset_config.get("truncate_packing", True)
        self.is_valid = dataset_config.get("is_valid", False)
        if self.truncate_packing and not self.is_pretraining:
            logger.warning_once("Truncate packing is only valid in pretraining data flow")
        self.packing = dataset_config.get("packing", False)
        self.greedy_intokens = dataset_config.get("greedy_intokens", True)
        if self.is_pretraining and self.packing and self.truncate_packing:
            logger.info("[dataflow] pretrain dataflow using truncate packing.")

        # special token
        self.begin_token = getattr(self.tokenizer.special_tokens_map, "cls_token", "<|begin_of_sentence|>")
        if isinstance(self.tokenizer, PretrainedTokenizer):
            self.begin_token_id = self.tokenizer._convert_token_to_id([self.begin_token])[0]
        else:
            self.begin_token_id = self.tokenizer.convert_tokens_to_ids([self.begin_token])[0]

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
                reverse=True,
            )

        self.estimate = False
        # The number of valid samples and skipped samples in estimation
        self.unused_samples = 0
        self.used_samples = 0
        # If used_estimate_samples exceeds max_estimate_samples,stop estimating.
        self.used_estimate_samples = 0
        self.max_estimate_samples = 0
        # set max estimate samples
        if not self.is_valid:
            self.max_estimate_samples = len(self.mix_datasets)

        self.last_printed_percent = 0

    def __len__(self):
        return len(self.mix_datasets)

    def __iter_func(self):

        # prepare epoch data
        batch_sequence, cur_len = [], 0
        dataset_iterator = get_worker_sliced_iterator(self.mix_datasets)
        actual_example_num = 1

        # pre-training:
        # 1. tokenize all the samples in the sampling pool,
        # 2. combine them into one large sample
        # 3. truncate it into multiple new samples based on the max_seq_len.
        if self.is_pretraining and self.packing and self.truncate_packing:
            take_lengths = []
            buffer = []
            for _ in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                try:
                    tokens = self._encode_pretraining_example(example, actual_example_num)
                except Exception as e:
                    print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                    if self.estimate:
                        self.unused_samples += actual_example_num
                    continue
                if tokens is None:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                    continue
                if self.estimate:
                    self.used_samples += actual_example_num

                idx = 0
                tokens_len = len(tokens)

                while idx < tokens_len:
                    remaining = self.max_seq_len + 1 - len(buffer)
                    take = min(remaining, tokens_len - idx)
                    take_lengths.append(take)
                    buffer.extend(tokens[idx : idx + take])
                    idx += take
                    if len(buffer) == self.max_seq_len + 1:
                        # label shift
                        res_tokens = buffer[:-1]
                        res_labels = buffer[1:]
                        take_lengths[-1] -= 1
                        position_ids = [list(range(item)) for item in take_lengths]
                        sequence = Sequence(
                            token_ids=res_tokens,
                            position_ids=position_ids,
                            labels=res_labels,
                            num_examples=actual_example_num,
                        )
                        batch_sequence = [sequence]
                        yield batch_sequence
                        buffer = []
                        take_lengths = []

                if self.estimate:
                    self.used_estimate_samples += actual_example_num
                    self.print_max_steps_estimate_progress()
                    if self.used_estimate_samples >= self.max_estimate_samples:
                        if buffer:
                            # label shift
                            res_tokens = buffer[:-1]
                            res_labels = buffer[1:]
                            take_lengths[-1] -= 1
                            position_ids = [list(range(item)) for item in take_lengths]
                            sequence = Sequence(
                                token_ids=res_tokens,
                                position_ids=position_ids,
                                labels=res_labels,
                                num_examples=actual_example_num,
                            )
                            batch_sequence = [sequence]
                            yield batch_sequence
                        self.used_estimate_samples = 0
                        # Set flag to False and yield empty list to signal the end of estimation
                        self.estimate = False
                        yield []

            if buffer:
                # label shift
                res_tokens = buffer[:-1]
                res_labels = buffer[1:]
                take_lengths[-1] -= 1
                position_ids = [list(range(item)) for item in take_lengths]
                sequence = Sequence(
                    token_ids=res_tokens,
                    position_ids=position_ids,
                    labels=res_labels,
                    num_examples=actual_example_num,
                )
                batch_sequence = [sequence]
                yield batch_sequence
        else:
            if not self.packing:
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    try:
                        if self.is_pretraining:
                            sequence = self._postprocess_pretraining_sequence(example, actual_example_num)
                        else:
                            sequence = self._postprocess_sequence(example, actual_example_num)
                    except Exception as e:
                        print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                        if self.estimate:
                            self.unused_samples += actual_example_num
                        continue
                    # unused_samples and used_samples are used to calculate skip_samples and actual_train_samples
                    if sequence is None:
                        if self.estimate:
                            self.unused_samples += actual_example_num
                        continue
                    if self.estimate:
                        self.used_samples += actual_example_num
                    batch_sequence, cur_len = [sequence], len(sequence.token_ids)
                    yield batch_sequence

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        self.print_max_steps_estimate_progress()
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            self.used_estimate_samples = 0
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []
                if len(batch_sequence) > 0:
                    yield batch_sequence
            else:
                if not self.greedy_intokens:
                    # base
                    for _ in range(len(self.mix_datasets)):
                        example = next(dataset_iterator)
                        try:
                            if self.is_pretraining:
                                sequence = self._postprocess_pretraining_sequence(example, actual_example_num)
                            else:
                                sequence = self._postprocess_sequence(example, actual_example_num)
                        except Exception as e:
                            print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                            if self.estimate:
                                self.unused_samples += actual_example_num
                            continue
                        if sequence is None:
                            if self.estimate:
                                self.unused_samples += actual_example_num
                            continue
                        if self.estimate:
                            self.used_samples += actual_example_num
                        if cur_len + len(sequence.token_ids) <= self.max_seq_len:
                            batch_sequence.append(sequence)
                            cur_len += len(sequence.token_ids)
                        else:
                            yield batch_sequence
                            batch_sequence, cur_len = [sequence], len(sequence.token_ids)

                        if self.estimate:
                            self.used_estimate_samples += actual_example_num
                            self.print_max_steps_estimate_progress()
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Yield left batch sequence before estimation ends
                                if len(batch_sequence) > 0:
                                    yield batch_sequence
                                self.used_estimate_samples = 0
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []
                    if len(batch_sequence) > 0:
                        yield batch_sequence
                else:
                    # Pseudo multiple rounds + group greedy intokens.
                    buffer_size = 500
                    examples = []
                    actual_example_num_list = []
                    i = 0
                    for _ in range(len(self.mix_datasets)):
                        example = next(dataset_iterator)
                        if i < buffer_size:
                            examples.append(example)
                            actual_example_num_list.append(actual_example_num)
                            i += 1
                        else:
                            # Running greedy strategy in examples.
                            generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                            for pack in generate_packs:
                                if len(pack) > 0:
                                    yield pack
                            examples = [example]
                            i = 1

                        if self.estimate:
                            self.used_estimate_samples += actual_example_num
                            self.print_max_steps_estimate_progress()
                            # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Yield left packs before estimation ends
                                if len(examples) > 0:
                                    generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                                    for pack in generate_packs:
                                        if len(pack) > 0:
                                            yield pack
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []

                    if len(examples) > 0:
                        generate_packs = self._generate_greedy_packs(examples, actual_example_num_list)
                        for pack in generate_packs:
                            if len(pack) > 0:
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

    def _postprocess_pretraining_sequence(self, example, actual_example_num):
        tokens = self._encode_pretraining_example(example, actual_example_num)
        if len(tokens) > self.max_seq_len + 1:
            # Truncate the sequence to the maximum length
            tokens = tokens[: self.max_seq_len + 1]
        res_tokens = tokens[:-1]
        res_labels = tokens[1:]
        pos_ids = list(range(len(res_tokens)))
        sequence = Sequence(
            token_ids=res_tokens,
            position_ids=pos_ids,
            labels=res_labels,
            num_examples=actual_example_num,
        )
        return sequence

    def _encode_pretraining_example(self, example, actual_example_num):
        # tokens
        content = example["messages"][0]["content"]
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
        # Add an EOS token at the end of each sample
        tokens = tokens + [self.tokenizer.eos_token_id]
        return tokens

    def _postprocess_sequence(self, example, actual_example_num):
        """Process code completion examples into token sequences.

        Args:
            example: The input example containing code components.
            actual_example_num (int): Number of examples used.

        Returns:
            Sequence: Processed sequence or None if invalid.
        """
        system = example.get("system", None)
        tools = example.get("tools", None)
        images = example.get("images", [])
        videos = example.get("videos", [])
        audios = example.get("audios", [])
        objects = example.get("objects", {})
        mm_inputs = None

        if self.use_template:
            if self.template_backend == "jinja":
                if not self.tokenizer.chat_template:
                    self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
                if self.split_multi_turn:
                    encoded_pairs = postprocess_fc_sequence(self.tokenizer, example)
                else:
                    encoded_pairs = self.tokenizer.encode_chat_inputs(example, encode_one_turn=self.encode_one_turn)
            else:
                messages = self.template.grounding_plugin.process_messages(
                    example["messages"],
                    objects,
                )
                mm_inputs = self.template.mm_plugin.get_mm_inputs(
                    images, videos, audios, [len(images)], [len(videos)], [len(audios)], None, self.processor
                )
                messages = self.template.mm_plugin.process_messages(
                    messages, images, videos, audios, mm_inputs, self.processor
                )
                encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        else:
            encoded_pairs = self.tokenizer.encode_chat_inputs_with_no_template(
                example, encode_one_turn=self.encode_one_turn
            )

        num_reserved_tokens_for_each_dialog = 1
        num_reserved_tokens_for_each_turn = 8

        cur_len = num_reserved_tokens_for_each_dialog

        turn_index = len(encoded_pairs) - 1

        tokens = []
        labels = []
        while turn_index >= 0:
            tokens_src, tokens_target = encoded_pairs[turn_index]
            if len(tokens_target) == 0:
                logger.warning(f"[SKIP] The length of encoded assistant tokens is 0: {example}")
                return None
            if len(tokens_src) + len(tokens_target) > (
                self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn
            ):
                if len(images) != 0 or len(videos) != 0 or len(audios) != 0:
                    # If there is multimodal data, do not truncate it; just discard it directly.
                    sub_src = example["messages"][0]["content"].strip()[:50]
                    logger.warning(f"[SKIP] This data is too long: {sub_src}...")
                    return None
                # If the source (src) exceeds length limit, discard this round of conversation data
                # If the target (tgt) exceeds length limit, truncate it
                if len(tokens_src) > self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn:
                    break
                else:
                    reverse_len = self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn - len(tokens_src)
                    tokens_target = tokens_target[:reverse_len]

            labels_src = [-100] * len(tokens_src)

            # Perform additional processing on chat sep.
            # If eos is valid, replace it with eos for learning;
            # otherwise, replace it with -100 and do not learn
            if not self.use_template or self.template_backend == "jinja":
                labels_target = tokens_target
            else:
                sep_token_len = len(self.tokenizer.tokenize(self.template.chat_sep))
                if turn_index != (len(encoded_pairs) - 1):
                    labels_target = tokens_target[: len(tokens_target) - sep_token_len] + [-100] * sep_token_len
                else:
                    labels_target = tokens_target

            if not example["label"][turn_index]:
                labels_target = [-100] * len(labels_target)
            tokens = tokens_src + tokens_target + tokens
            labels = labels_src + labels_target + labels

            assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

            cur_len = len(tokens)

            turn_index -= 1

        # Not even one turn can be added, so need to do warning and skip this example
        if len(tokens) <= num_reserved_tokens_for_each_dialog + num_reserved_tokens_for_each_turn:
            try:
                # For print log
                sub_src = example["messages"][0]["content"].strip()[:50]
                sub_tgt = example["messages"][-1]["content"].strip()[-50:]
                if len(tokens) > 0:
                    logger.warning(f"This data is too short: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
                else:
                    logger.warning(f"This data is too long: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
            except Exception:
                logger.warning("[SKIP] wrong example")
            return None

        if self.use_template:
            # add dynamic eos
            if self.template_backend == "custom":
                suffix_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
            else:
                suffix_ids = [self.tokenizer.eos_token_id]
            self._add_dynamic_eos(tokens, labels, suffix_ids)
            # Maybe left truncated, so need to add begin_token
            if self.auto_add_bos and self.begin_token_id:
                if tokens[0] != self.begin_token_id:
                    tokens = [self.begin_token_id] + tokens
                    labels = [-100] + labels
                    if len(tokens) > self.max_seq_len:
                        raise RuntimeError(f"token_ids is too long: {len(tokens)}")
            # Add EOS token at the end
            if self.efficient_eos:
                tokens = tokens + suffix_ids
                labels = labels + suffix_ids
                if len(tokens) > self.max_seq_len:
                    raise RuntimeError(f"token_ids is too long: {len(tokens)}")
            # label shift
            labels = labels[1:] + [-100]
        else:
            # label shift
            labels = labels[1:] + [-100]
            if len(tokens) > self.max_seq_len:
                raise RuntimeError(f"token_ids is too long: {len(tokens)}")

        pos_ids = list(range(len(tokens)))

        if all(x == -100 for x in labels):
            logger.warning(f"[SKIP] all labels set to 0: {example}")
            return None

        assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

        enable_dataset_debug = os.getenv("FLAGS_enable_dataset_debug", "false").lower() in ("true", "1", "t")
        if enable_dataset_debug:
            logger.info("\n" + "=" * 50)
            logger.info("[dataset debug] Debug mode enabled")
            if hasattr(self, "tokenizer"):
                print("========================================")
                print("tokens: ", [tokens])
                print_debug_info(self.tokenizer, tokens, "input")
                print("========================================\n")

                filtered_labels = [x for x in labels if x != -100]  # remove -100
                print("========================================")
                print("labels: ", [labels])
                print_debug_info(self.tokenizer, filtered_labels, "labels")
                print("========================================\n")
            else:
                logger.info("[dataset debug] Tokenizer not available")
            logger.info("=" * 50 + "\n")

        return Sequence(
            token_ids=tokens,
            position_ids=pos_ids,
            labels=labels,
            num_examples=actual_example_num,
            images=images,
            videos=videos,
            audios=audios,
            mm_inputs=mm_inputs,
        )

    def _generate_greedy_packs(self, examples, actual_example_num_list):
        """Generate packed sequences using greedy strategy.

        Args:
            examples: List of examples to pack.
            actual_example_num_list: List of example counts.

        Returns:
            list: List of packed sequences.
        """

        left_len = np.zeros([len(examples)]) - 1
        left_len[0] = self.max_seq_len  # At the beginning, only the first pack is valid.
        generate_packs = [[] for i in range(len(examples))]
        index = 0
        left_index = 0

        while index < len(examples):
            try:
                if self.is_pretraining:
                    sequence = self._postprocess_pretraining_sequence(examples[index], actual_example_num_list[index])
                else:
                    sequence = self._postprocess_sequence(examples[index], actual_example_num_list[index])
            except Exception as e:
                print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                if self.estimate:
                    self.unused_samples += actual_example_num_list[index]
                index += 1
                continue
            if sequence is None:
                if self.estimate:
                    self.unused_samples += actual_example_num_list[index]
                index += 1
                continue

            max_left_index = left_len.argmax()
            # Put the current sequence into the largest left space valid pack.
            if len(sequence.token_ids) <= left_len[max_left_index]:
                generate_packs[max_left_index].append(sequence)
                left_len[max_left_index] -= len(sequence.token_ids)
                if self.estimate:
                    self.used_samples += actual_example_num_list[index]
                index += 1
            else:
                left_index += 1
                left_len[left_index] = self.max_seq_len

        return generate_packs

    def print_max_steps_estimate_progress(self):
        current_percent = (self.used_estimate_samples / self.max_estimate_samples) * 100
        # Print progress at every 5% interval.
        if int(current_percent) // 5 > self.last_printed_percent // 5:
            print(f"[Estimate Max Steps Progress]: {current_percent:.0f}%")
            self.last_printed_percent = current_percent

    @staticmethod
    def _add_dynamic_eos(input_ids, labels, suffix_tokens_id):
        # Adapted from:
        # https://github.com/modelscope/ms-swift
        # Original author: modelscope
        # License: Apache-2.0
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels) + 1):
            if labels[i - 1] >= 0 and i < len(labels) and labels[i] == -100:
                start = i
            elif start > 0 and labels[i - 1] == -100 and (i == len(labels) or labels[i] >= 0):
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len and input_ids[start : start + suffix_len] == suffix_tokens_id:
                    labels[start : start + suffix_len] = suffix_tokens_id
