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

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Literal, Optional

import numpy as np
from paddle.io import Dataset, IterableDataset

from paddleformers.datasets.data_utils import (
    calculate_matched_group,
    generate_greedy_packs_from_sequences,
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
class TextSequence:
    """Encapsulated text sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    num_examples: int


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


class BaseSFTDataset:
    def __init__(self, **dataset_config):

        # parameter init
        self.tokenizer = dataset_config.get("tokenizer", None)
        self.processor = dataset_config.get("processor", None)
        self.dataset_num_proc = dataset_config.get("dataset_num_proc", 1)
        logger.info(f"self.dataset_num_proc: {self.dataset_num_proc}")
        self.dataloader_num_workers = dataset_config.get("dataloader_num_workers", 0)
        self.max_seq_len = dataset_config.get("max_seq_len", 8192)
        self.template = dataset_config.get("template_instance", None)
        self.template_backend = dataset_config.get("template_backend", "jinja")
        self.use_template = dataset_config.get("use_template", True)
        self.efficient_eos = True if not self.template else getattr(self.template, "efficient_eos", True)
        self.auto_add_bos = True if not self.template else getattr(self.template, "auto_add_bos", False)
        self.split_multi_turn = dataset_config.get("split_multi_turn", False)
        self.encode_one_turn = dataset_config.get("encode_one_turn", True)
        self.is_pretraining = dataset_config.get("is_pretraining", False)
        self.truncation_strategy = dataset_config.get("truncation_strategy", "delete")
        self.truncate_packing = dataset_config.get("truncate_packing", True)
        self.is_valid = dataset_config.get("is_valid", False)
        self.packing = dataset_config.get("packing", False)
        self.greedy_intokens = dataset_config.get("greedy_intokens", True)
        self.dtype = dataset_config.get("dtype", None)
        self.binpacking = dataset_config.get("binpacking", False)
        self.packing_interval = dataset_config.get("packing_interval", 1000)

        # check
        if not self.dataset_num_proc:
            self.dataset_num_proc = 1
        if self.dataset_num_proc > 1 and self.dataloader_num_workers > 0:
            raise ValueError("dataset_num_proc and dataloader_num_workers can not be set simultaneously now.")
        if self.truncate_packing and not self.is_pretraining:
            logger.warning_once("Truncate packing is only valid in pretraining data flow")
        if self.is_pretraining and self.packing and self.truncate_packing:
            logger.info("[dataflow] pretrain dataflow using truncate packing.")
        assert self.truncation_strategy in [
            "oral",
            "delete",
            "right",
            "left",
        ], f"truncation_strategy must be in [oral, delete, right, left], but got {self.truncation_strategy}"
        logger.info(f"[dataflow] truncation_strategy: {self.truncation_strategy}")

        # special token
        self.begin_token = getattr(self.tokenizer.special_tokens_map, "cls_token", "<|begin_of_sentence|>")
        if isinstance(self.tokenizer, PretrainedTokenizer):
            self.begin_token_id = self.tokenizer._convert_token_to_id([self.begin_token])[0]
        else:
            self.begin_token_id = self.tokenizer.convert_tokens_to_ids([self.begin_token])[0]
        self.sep_token_len = 0
        if self.use_template and self.template_backend != "jinja":
            self.sep_token_len = len(self.tokenizer.tokenize(self.template.chat_sep))

        # The number of reserved tokens for each dialog
        self.num_reserved_tokens_for_each_dialog = 0
        if self.use_template:
            # add dynamic eos
            suffix_ids = (
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
                if self.template_backend == "custom"
                else [self.tokenizer.eos_token_id]
            )
            self.num_reserved_tokens_for_each_dialog += len(suffix_ids)

            # bos token
            self.num_reserved_tokens_for_each_dialog += 1
        logger.info(f"self.num_reserved_tokens_for_each_dialog: {self.num_reserved_tokens_for_each_dialog}")

        # media placeholder token
        self.placeholder_tokens = []
        if self.template and self.template.mm_plugin:
            for tok in [
                self.template.mm_plugin.image_token,
                self.template.mm_plugin.video_token,
                self.template.mm_plugin.audio_token,
            ]:
                if tok:
                    self.placeholder_tokens.append(tok)
        for i, token in enumerate(self.placeholder_tokens):
            if isinstance(token, str):
                if isinstance(self.tokenizer, PretrainedTokenizer):
                    self.placeholder_tokens[i] = self.tokenizer._convert_token_to_id(token)
                else:
                    self.placeholder_tokens[i] = self.tokenizer.convert_tokens_to_ids(token)

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

        # max_steps estimate
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
        self._estimate_start_time = None

        # flags
        self.enable_dataset_debug = os.getenv("FLAGS_enable_dataset_debug", "false").lower() in ("true", "1", "t")
        self.mem_debug = os.getenv("FLAGS_enable_mem_debug", "false").lower() in ("true", "1", "t")

        # The flag indicating whether all examples have been iterated
        self.iter_all_examples = False

        # multiprocessing initialization
        if self.is_pretraining and self.packing and self.truncate_packing:
            self._current_processor_func = self._tokenize_pretraining
        else:
            if self.is_pretraining:
                self._current_processor_func = self._process_pretraining_sequence
            else:
                self._current_processor_func = self._process_sft_sequence

        if self.dataset_num_proc > 1:
            self.prefetch_size = self.dataset_num_proc * 2
            self._in_queue = mp.Queue(maxsize=self.prefetch_size)
            self._out_queue = mp.Queue(maxsize=self.prefetch_size)
            self.workers = []
            for _ in range(self.dataset_num_proc):
                worker = mp.Process(target=self._worker_loop, daemon=True)
                worker.start()
                self.workers.append(worker)

    def __len__(self):
        return len(self.mix_datasets)

    def _worker_loop(self):
        """Worker process main loop."""
        while True:
            try:
                i, example, actual_example_num = self._in_queue.get()
                result = None
                try:
                    result = self._current_processor_func(example, actual_example_num)
                except Exception as e:
                    # result remains None, will be counted as unused_samples in _get_processed_data_iterator
                    print(f"Warning: Error processing example in worker, skipping. Error: {str(e)}")
                self._out_queue.put((i, result))
            except Exception:
                break

    def _get_processed_data_iterator(
        self, dataset_iterator, actual_example_num, processor_func, yield_with_index=False
    ):
        """Get an iterator that yields processed data, using multiprocessing if enabled.

        Args:
            dataset_iterator: Raw data iterator.
            actual_example_num: Number of examples used.
            processor_func: Function to process each example.
            yield_with_index: whether yield (raw_idx, result) tuples or yield result

        Yields:
            Processed results in order (skips None results).
        """

        def _rss_mb():
            try:
                with open("/proc/self/status") as _f:
                    for _line in _f:
                        if _line.startswith("VmRSS:"):
                            return int(_line.split()[1]) / 1024
            except Exception:
                pass
            return -1

        _log_interval = 200
        _yield_cnt = 0

        if self.dataset_num_proc > 1:
            # Multiprocessing mode
            if self.mem_debug:
                print(f"[MemDebug] workers started, RSS={_rss_mb():.0f} MB, " f"num_proc={self.dataset_num_proc}")
            try:
                pending = 0
                send_idx = 0
                recv_idx = 0
                result_buffer = {}  # Buffer for out-of-order results
                total_samples = len(self.mix_datasets)

                # Pre-fill the queue
                for _ in range(self.prefetch_size):
                    if send_idx >= total_samples:
                        break
                    example = next(dataset_iterator)
                    self._in_queue.put((send_idx, example, actual_example_num))
                    send_idx += 1
                    pending += 1

                if self.mem_debug:
                    print(
                        f"[MemDebug] pre-fill done, RSS={_rss_mb():.0f} MB, "
                        f"pending={pending}, in_q~{self._in_queue.qsize()}, "
                        f"out_q~{self._out_queue.qsize()}"
                    )

                # Process data in streaming fashion, maintaining order
                while pending > 0:
                    idx, result = self._out_queue.get()
                    pending -= 1

                    while send_idx < total_samples and pending < self.prefetch_size:
                        example = next(dataset_iterator)
                        self._in_queue.put((send_idx, example, actual_example_num))
                        send_idx += 1
                        pending += 1

                    # Store result in buffer
                    result_buffer[idx] = result

                    # Yield results in order, skip None
                    while recv_idx in result_buffer:
                        res = result_buffer.pop(recv_idx)
                        current_idx = recv_idx
                        recv_idx += 1
                        if res is not None:
                            _yield_cnt += 1
                            if self.mem_debug and _yield_cnt % _log_interval == 0:
                                print(
                                    f"[MemDebug] yielded={_yield_cnt}, RSS={_rss_mb():.0f} MB | "
                                    f"pending={pending}, result_buf={len(result_buffer)}, "
                                    f"in_q~{self._in_queue.qsize()}, out_q~{self._out_queue.qsize()}"
                                )
                            yield (current_idx, res) if yield_with_index else res
                        else:
                            if self.estimate:
                                self.used_estimate_samples += actual_example_num
                                self.unused_samples += actual_example_num
            finally:
                if self.mem_debug:
                    print(f"[MemDebug] iteration finished, RSS={_rss_mb():.0f} MB, " f"workers kept alive for reuse")
        else:
            # Single process mode
            for raw_idx in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                try:
                    result = processor_func(example, actual_example_num)
                except Exception as e:
                    print(f"Warning: Error processing example, skipping. Error: {str(e)}")
                    result = None
                if result is not None:
                    _yield_cnt += 1
                    if self.mem_debug and _yield_cnt % _log_interval == 0:
                        print(f"[MemDebug][single] yielded={_yield_cnt}, RSS={_rss_mb():.0f} MB")
                    yield (raw_idx, result) if yield_with_index else result
                else:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                        self.used_estimate_samples += actual_example_num

    def _generate_sequences(self):

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
            data_iter = self._get_processed_data_iterator(
                dataset_iterator, actual_example_num, self._current_processor_func
            )
            for tokens in data_iter:
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
            self.iter_all_examples = True
        else:
            if not self.packing:
                logger.info("Not using packing mode for data iteration.")
                # No packing mode
                data_iter = self._get_processed_data_iterator(
                    dataset_iterator, actual_example_num, self._current_processor_func
                )
                for sequence in data_iter:
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
                self.iter_all_examples = True
            else:
                if self.binpacking:
                    logger.info("Using binpacking mode for data iteration.")
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._current_processor_func
                    )
                    accumulated_data = []

                    while True:
                        batch_data, num_samples = self._binpacking_process_batch(data_iter, self.packing_interval)
                        finished = num_samples != self.packing_interval

                        accumulated_data += batch_data

                        sequences, accumulated_data = calculate_matched_group(
                            accumulated_data, self.max_seq_len, is_finished=finished
                        )

                        for row in sequences:
                            yield [r[0] for r in row]

                        if self.estimate:
                            self.used_estimate_samples += num_samples
                            self.print_max_steps_estimate_progress()
                            # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []

                        if finished:
                            self.iter_all_examples = True
                            break
                elif not self.greedy_intokens:
                    logger.info("Using base packing mode for data iteration.")
                    # base packing mode
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._current_processor_func
                    )
                    for sequence in data_iter:
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
                    self.iter_all_examples = True
                else:
                    logger.info("Using greedy packing mode for data iteration.")
                    # Pseudo multiple rounds + group greedy intokens.
                    buffer_size = self.packing_interval
                    sequences_buffer = []
                    data_iter = self._get_processed_data_iterator(
                        dataset_iterator, actual_example_num, self._current_processor_func
                    )
                    for sequence in data_iter:
                        if self.estimate:
                            self.used_samples += actual_example_num

                        sequences_buffer.append(sequence)

                        if len(sequences_buffer) >= buffer_size:
                            # Running greedy strategy in sequences_buffer.
                            generate_packs = generate_greedy_packs_from_sequences(self.max_seq_len, sequences_buffer)
                            for pack in generate_packs:
                                if len(pack) > 0:
                                    yield pack
                            sequences_buffer = []

                        if self.estimate:
                            self.used_estimate_samples += actual_example_num
                            self.print_max_steps_estimate_progress()
                            # Stop estimation if the number of samples used in estimation is larger than max_estimate_samples
                            if self.used_estimate_samples >= self.max_estimate_samples:
                                # Yield left packs before estimation ends
                                if len(sequences_buffer) > 0:
                                    generate_packs = generate_greedy_packs_from_sequences(
                                        self.max_seq_len, sequences_buffer
                                    )
                                    for pack in generate_packs:
                                        if len(pack) > 0:
                                            yield pack
                                # Set flag to False and yield empty list to signal the end of estimation
                                self.estimate = False
                                yield []

                    if len(sequences_buffer) > 0:
                        generate_packs = generate_greedy_packs_from_sequences(self.max_seq_len, sequences_buffer)
                        for pack in generate_packs:
                            if len(pack) > 0:
                                yield pack

                    self.iter_all_examples = True

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

    def _tokenize_pretraining(self, example, actual_example_num):
        """Process a pretraining example into tokens."""
        content = example["messages"][0]["content"]
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
        # Add an EOS token at the end of each sample
        tokens = tokens + [self.tokenizer.eos_token_id]
        return tokens

    def _process_pretraining_sequence(self, example, actual_example_num):

        messages = example.get("messages", [])
        images = example.get("images", [])
        videos = example.get("videos", [])
        audios = example.get("audios", [])

        if len(images) == 0 and len(videos) == 0 and len(audios) == 0:
            tokens = self._tokenize_pretraining(example, actual_example_num)
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
        else:
            mm_inputs = self.template.mm_plugin.get_mm_inputs(
                images,
                videos,
                audios,
                self.processor,
                imglens=[len(images)],
                vidlens=[len(videos)],
                audlens=[len(audios)],
                batch_ids=None,
                messages=messages,
            )

            messages = self.template.mm_plugin.process_messages(
                messages, images, videos, audios, mm_inputs, self.processor
            )
            example["messages"] = messages
            tokens = self._tokenize_pretraining(example, actual_example_num)
            if len(tokens) > self.max_seq_len + 1:
                # Truncate the sequence to the maximum length
                tokens = tokens[: self.max_seq_len + 1]

            labels = self.template.mm_plugin.process_tokens(tokens, self.processor)

            # label shift
            labels = labels[1:] + [-100]

            pos_ids = list(range(len(tokens)))  # only pure text, mm_position_ids will be reconstructed in collate.py

            if all(x == -100 for x in labels):
                logger.warning(f"[SKIP] all labels set to 0: {example}")
                return None

            assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

            if self.enable_dataset_debug:
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

    def _process_sft_sequence(self, example, actual_example_num):
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
                    images,
                    videos,
                    audios,
                    self.processor,
                    imglens=[len(images)],
                    vidlens=[len(videos)],
                    audlens=[len(audios)],
                    batch_ids=None,
                    messages=messages,
                    dtype=self.dtype,
                )
                messages = self.template.mm_plugin.process_messages(
                    messages, images, videos, audios, mm_inputs, self.processor
                )
                encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        else:
            encoded_pairs = self.tokenizer.encode_chat_inputs_with_no_template(
                example, encode_one_turn=self.encode_one_turn
            )

        cur_len = self.num_reserved_tokens_for_each_dialog
        tokens_chunks = []
        labels_chunks = []
        accumulated_tokens_len = 0

        for turn_index in range(len(encoded_pairs) - 1, -1, -1):
            tokens_src, tokens_target = encoded_pairs[turn_index]
            if len(tokens_target) == 0:
                logger.warning(f"[SKIP] The length of encoded assistant tokens is 0: {example}")
                return None

            if self.truncation_strategy == "oral":
                remaining_len = self.max_seq_len - cur_len
                if len(tokens_src) + len(tokens_target) > remaining_len:
                    if images or videos or audios:
                        # If there is multimodal data, do not truncate it; just discard it directly.
                        sub_src = example["messages"][0]["content"].strip()[:50]
                        logger.warning(f"[SKIP] This data is too long: {sub_src}...")
                        return None
                    # If the source (src) exceeds length limit, discard this round of conversation data
                    # If the target (tgt) exceeds length limit, truncate it
                    if len(tokens_src) > remaining_len:
                        break
                    else:
                        tokens_target = tokens_target[: remaining_len - len(tokens_src)]

            labels_src = [-100] * len(tokens_src)

            # Perform additional processing on chat sep.
            # If eos is valid, replace it with eos for learning;
            # otherwise, replace it with -100 and do not learn
            if not self.use_template or self.template_backend == "jinja":
                labels_target = tokens_target
            else:
                if turn_index != (len(encoded_pairs) - 1):
                    labels_target = (
                        tokens_target[: len(tokens_target) - self.sep_token_len] + [-100] * self.sep_token_len
                    )
                else:
                    labels_target = tokens_target

            if not example["label"][turn_index]:
                labels_target = [-100] * len(labels_target)

            tokens_chunks.append(tokens_src + tokens_target)
            labels_chunks.append(labels_src + labels_target)

            accumulated_tokens_len += len(tokens_src) + len(tokens_target)
            cur_len = accumulated_tokens_len

        tokens_chunks.reverse()
        labels_chunks.reverse()
        tokens = list(chain.from_iterable(tokens_chunks))
        labels = list(chain.from_iterable(labels_chunks))
        del tokens_chunks, labels_chunks

        # Not even one turn can be added, so need to do warning and skip this example
        if len(tokens) <= self.num_reserved_tokens_for_each_dialog:
            try:
                # For print log
                sub_src = example["messages"][0]["content"].strip()[:50]
                sub_tgt = example["messages"][-1]["content"].strip()[-50:]
                msg = "too short" if len(tokens) > 0 else "too long"
                logger.warning(f"This data is {msg}: '{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
            except Exception:
                logger.warning("[SKIP] wrong example")
            return None

        if self.use_template:
            # add dynamic eos
            suffix_ids = (
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.template.suffix[-1]))
                if self.template_backend == "custom"
                else [self.tokenizer.eos_token_id]
            )
            self._add_dynamic_eos(tokens, labels, suffix_ids)

            # Maybe left truncated, so need to add begin_token
            if self.auto_add_bos and self.begin_token_id and tokens[0] != self.begin_token_id:
                tokens = [self.begin_token_id] + tokens
                labels = [-100] + labels

            # Add EOS token at the end
            if self.efficient_eos:
                tokens.extend(suffix_ids)
                labels.extend(suffix_ids)

        # data truncate
        if self.truncation_strategy != "oral":
            tokens, labels = self._encode_truncated(tokens, labels)
            if not tokens:
                sub_src = example["messages"][0]["content"].strip()[:50]
                logger.warning(f"[SKIP] data is deleted by truncation strategy: {sub_src}...")
                return None
        else:
            if len(tokens) > self.max_seq_len:
                raise RuntimeError(f"token_ids is too long: {len(tokens)}")

        # label shift
        labels = labels[1:] + [-100]

        pos_ids = list(range(len(tokens)))

        if all(x == -100 for x in labels):
            logger.warning(f"[SKIP] all labels set to -100: {example}")
            return None

        assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

        if self.enable_dataset_debug:
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

    @staticmethod
    def _get_length(input_ids, labels):
        # input_ids might be a tensor.
        lengths = [0]
        if input_ids is not None:
            lengths.append(len(input_ids))
        if labels is not None:
            lengths.append(len(labels))
        length = max(lengths)
        return length

    def _truncate(
        self,
        input_ids: List[int],
        labels: Optional[List[int]],
        truncation_strategy: Literal["left", "right"],
    ):
        max_len = self.max_seq_len
        placeholder_set = set(self.placeholder_tokens)

        is_placeholder = [tok in placeholder_set for tok in input_ids]
        placeholder_idx = [i for i, v in enumerate(is_placeholder) if v]

        if len(placeholder_idx) >= max_len:
            keep_idx = placeholder_idx[:max_len]
        else:
            remain = max_len - len(placeholder_idx)
            non_placeholder_idx = [i for i, v in enumerate(is_placeholder) if not v]

            if truncation_strategy == "left":
                extra_idx = non_placeholder_idx[-remain:]
            else:
                extra_idx = non_placeholder_idx[:remain]

            keep_idx = sorted(placeholder_idx + extra_idx)

        input_ids = [input_ids[i] for i in keep_idx]
        labels = [labels[i] for i in keep_idx]

        return input_ids, labels

    def _encode_truncated(self, input_ids, labels):
        length = self._get_length(input_ids, labels)
        if self.max_seq_len is not None and length > self.max_seq_len:
            if self.truncation_strategy == "delete":
                return None, None
            if self.truncation_strategy in {"right", "left"}:
                input_ids, labels = self._truncate(input_ids, labels, truncation_strategy=self.truncation_strategy)
        return input_ids, labels

    def print_max_steps_estimate_progress(self):
        current_percent = (self.used_estimate_samples / self.max_estimate_samples) * 100
        if self._estimate_start_time is None:
            self._estimate_start_time = time.time()
        # Print progress at every 5% interval.
        if int(current_percent) // 5 > self.last_printed_percent // 5:
            elapsed = time.time() - self._estimate_start_time
            print(f"[Estimate Max Steps Progress]: {current_percent:.0f}% (elapsed: {elapsed:.1f}s)")
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

    def _binpacking_process_batch(self, iterator, batch_size, index_only=False):
        batch = []
        count = 0
        for _ in range(batch_size):
            try:
                item = next(iterator)
                if self.estimate:
                    self.used_samples += 1
                if index_only:
                    raw_idx, encoded = item
                    if encoded:
                        batch.append((raw_idx, len(encoded.token_ids)))
                else:
                    encoded = item
                    if encoded:
                        batch.append((encoded, len(encoded.token_ids)))
                count += 1
            except StopIteration:
                break
        return batch, count


class IteratorSFTDataset(BaseSFTDataset, IterableDataset):
    def __init__(self, **dataset_config):
        super().__init__(**dataset_config)

    def __iter__(self):
        if self.is_valid:
            yield from self._generate_sequences()
        else:
            while True:
                yield from self._generate_sequences()


class MapSFTDataset(BaseSFTDataset, Dataset):
    def __init__(self, **dataset_config):
        super().__init__(**dataset_config)

        self.packed_idx_cache_dir = dataset_config.get("packed_idx_cache_dir", None)

        self.raw_data = list(self.mix_datasets)
        logger.info(f"[MapSFTDataset] Total samples: {len(self.raw_data)}")

        self.n_try_fetch = min(10, len(self.raw_data))
        self.random_state = np.random.RandomState(None)
        self.traceback_limit = 10
        self._traceback_counter = 0
        self._idx = 0
        self._idx_list = self.random_state.permutation(len(self.raw_data)).tolist()
        self.packed_idx = None

        if self.packing and self.binpacking:
            self.packed_idx = self._build_or_load_packed_idx()

        elif self.packing and (not self.binpacking):
            raise ValueError("[MapSFTDataset] packing only support binpacking")

    def _build_or_load_packed_idx(self):

        if self.packed_idx_cache_dir is not None:
            split = "eval" if self.is_valid else "train"
            self.cache_path = os.path.join(self.packed_idx_cache_dir, f"{split}_packed_idx.npz")
            packed_idx = self._load_packed_idx_cache(self.cache_path)
            if packed_idx is not None:
                return packed_idx

        packed_idx = self._compute_packed_idx()

        if self.packed_idx_cache_dir is not None:
            self._save_packed_idx_cache(packed_idx, self.cache_path)

        return packed_idx

    def _compute_packed_idx(self):
        """
        Mirrors the binpacking loop in _generate_sequences
        but uses index_only=True:
        (Sequence, token_length) -> (raw_idx, token_length)
        """
        logger.info("[MapSFTDataset] Computing token lengths for bin packing...")
        actual_example_num = 1

        dataset_iterator = iter(self.raw_data)
        data_iter = self._get_processed_data_iterator(
            dataset_iterator, actual_example_num, self._current_processor_func, yield_with_index=True
        )

        accumulated_data = []
        packed_idx = []

        while True:
            batch_data, num_samples = self._binpacking_process_batch(data_iter, self.packing_interval, index_only=True)
            finished = num_samples != self.packing_interval

            accumulated_data += batch_data

            groups, accumulated_data = calculate_matched_group(
                accumulated_data, self.max_seq_len, is_finished=finished
            )

            for row in groups:
                packed_idx.append([item[0] for item in row])

            if finished:
                break

        logger.info(
            f"[MapSFTDataset] {sum(len(g) for g in packed_idx)} valid samples -> " f"{len(packed_idx)} packed groups."
        )
        return packed_idx

    def _load_packed_idx_cache(self, cache_path):
        """from an .npz cache file."""
        if not os.path.isfile(cache_path):
            logger.info(f"[MapSFTDataset] No packed_idx cache found at {cache_path}")
            return None

        try:
            data = np.load(cache_path, allow_pickle=False)

            group_offsets = data["group_offsets"]
            flat_indices = data["flat_indices"]

            packed_idx = []
            for i in range(len(group_offsets) - 1):
                start = group_offsets[i]
                end = group_offsets[i + 1]
                packed_idx.append(flat_indices[start:end].tolist())

            logger.info(f"[MapSFTDataset] Loaded packed_idx cache from {cache_path}: " f"{len(packed_idx)} groups.")
            return packed_idx

        except Exception as e:
            logger.warning(f"[MapSFTDataset] Failed to load packed_idx cache from {cache_path}: {e}. " "Recomputing.")
            return None

    def _save_packed_idx_cache(self, packed_idx, cache_path):
        """to an .npz cache file."""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            offsets = [0]
            flat = []
            for group in packed_idx:
                flat.extend(group)
                offsets.append(len(flat))

            np.savez(
                cache_path,
                group_offsets=np.array(offsets, dtype=np.int64),
                flat_indices=np.array(flat, dtype=np.int64),
            )

            logger.info(f"[MapSFTDataset] Saved packed_idx cache to {cache_path}: " f"{len(packed_idx)} groups")
        except Exception as e:
            logger.warning(
                f"[MapSFTDataset] Failed to save packed_idx cache to {cache_path}: {e}. " "Continuing without cache."
            )

    def __len__(self):
        if self.packed_idx is not None:
            return len(self.packed_idx)
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.packed_idx is not None:
            return self._getitem_packed(idx)
        return self._getitem_single(idx)

    def _getitem_packed(self, idx):
        """Get a packed group of sequences by bin packing index."""
        group_indices = self.packed_idx[idx]
        actual_example_num = 1
        sequences = []

        for raw_idx in group_indices:
            example = self.raw_data[raw_idx]
            try:
                sequence = self._current_processor_func(example, actual_example_num)

                if sequence is not None:
                    sequences.append(sequence)
                else:
                    logger.warning(
                        f"[MapSFTDataset] Sample {raw_idx} in packed group {idx} "
                        "returned None during __getitem__. Skipping within group."
                    )
            except Exception:
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback

                    logger.info(traceback.format_exc())
                    logger.warning(
                        f"[MapSFTDataset] Error processing sample {raw_idx} in packed group {idx}. "
                        "Skipping within group."
                    )
                    self._traceback_counter += 1

        return sequences

    def _getitem_single(self, idx):
        """same as original __getitem__"""
        actual_example_num = 1

        for i in range(self.n_try_fetch):
            if i == 0:
                current_idx = idx
            else:
                current_idx = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.raw_data)

            example = self.raw_data[current_idx]
            try:
                sequence = self._current_processor_func(example, actual_example_num)

                if sequence is not None:
                    return [sequence]

                # sequence is None, try next
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    logger.warning(
                        f"[MapSFTDataset] Example at index {current_idx} returned None, "
                        "another piece of data will be randomly selected."
                    )
                    self._traceback_counter += 1

            except Exception:
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback

                    logger.info(traceback.format_exc())
                    logger.warning(
                        "[MapSFTDataset] There are errors in data processing, "
                        "another piece of data will be randomly selected."
                    )
                    self._traceback_counter += 1

        raise ValueError(
            f"[MapSFTDataset] Failed to retrieve valid data after {self.n_try_fetch} attempts. "
            "You can avoid this issue by checking your data quality."
        )
