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


import copy
import os
from dataclasses import dataclass
from typing import List

import numpy as np
from paddle.io import IterableDataset

from paddleformers.transformers.tokenizer_utils import PretrainedTokenizer
from paddleformers.utils.env import NONE_CHAT_TEMPLATE

from ..utils.log import logger
from .base import MultiSourceDataset
from .data_utils import Example, pad_batch_data, postprocess_fc_sequence
from .mix_datasets import create_dataset_instance

LOGGER_COUNT = 0


@dataclass
class Sequence:
    """Encapsulated sequence class."""

    token_ids: List[int]
    position_ids: List[int]
    labels: List[int]
    loss_mask: List[int]
    num_examples: int


def create_dataset(**dataset_config):
    """Create SFT dataset based on configuration parameters.

    Args:
        dataset_config (dict): Configuration dictionary containing parameters like:

    Returns:
        SequenceDataset: Configured sequence dataset for SFT tasks
    """
    task_dataset_path = [path for path in str(dataset_config["task_group"]).replace(" ", "").split(",") if path != ""]
    task_dataset_prob = [
        float(prob) for prob in str(dataset_config["task_group_prob"]).replace(" ", "").split(",") if prob != ""
    ]
    sub_dataset_type = [
        type_ for type_ in str(dataset_config["sub_dataset_type"]).replace(" ", "").split(",") if type_ != ""
    ]

    if not (len(task_dataset_path) == len(task_dataset_prob) == len(sub_dataset_type)):
        raise ValueError("The len of dataset path, prob, type are inconsistent, please check the configuration.")

    if len(task_dataset_path) == 0:
        raise ValueError("The len of dataset path is zero, please check the configuration.")

    example_dataset = MultiSourceDataset(
        task_dataset_path=task_dataset_path,
        task_dataset_prob=task_dataset_prob,
        sub_dataset_type=sub_dataset_type,
        process_fn=process_example,
        process_fn_pt=process_pretraining_example,
        process_fn_fc=process_fc,
        is_pretraining=dataset_config["is_pretraining"],
    )
    sequence_dataset = SequenceDataset(
        dataset=example_dataset,
        tokenizer=dataset_config["tokenizer"],
        max_seq_len=dataset_config["max_seq_len"],
        num_samples_each_epoch=dataset_config["num_samples_each_epoch"],
        is_valid=dataset_config.get("is_valid", False),
        random_seed=dataset_config["random_seed"],
        random_shuffle=dataset_config["random_shuffle"],
        greedy_intokens=dataset_config["greedy_intokens"],
        packing=dataset_config["packing"],
        mix_strategy=dataset_config["mix_strategy"],
        encode_one_turn=dataset_config["encode_one_turn"],
        use_template=dataset_config["use_template"],
        is_pretraining=dataset_config["is_pretraining"],
    )
    return sequence_dataset


def create_indexed_dataset(data_file_prefix):
    """Create indexed dataset from raw data files.

    Args:
        data_file_prefix (str): Path prefix for raw data files

    Returns:
        IndexedDataset: Preprocessed dataset with memory-efficient indexing
    """
    from paddleformers.data.indexed_dataset import (
        make_sft_dataset as make_sft_indexed_dataset,
    )

    indexed_dataset = make_sft_indexed_dataset(
        path=data_file_prefix,
        dataclass=Sequence,
    )
    return indexed_dataset


def collate_fn(batch: List[List[Sequence]], tokenizer, training_args, model_args, max_seq_len: int):
    """Convert batch of sequences into training tensors.

    Args:
        batch (List[List[Sequence]]): Batch of input sequences
        tokenizer: Tokenizer for text conversion
        model_args: Model configuration parameters
        max_seq_len (int): Maximum sequence length for padding

    Returns:
        dict: Dictionary containing:
            - input_ids: Padded token IDs
            - labels: Shifted labels for prediction
            - loss_mask: Mask for computing loss
    """
    input_keys = ["input_ids", "labels", "loss_mask"]
    if training_args.num_nextn_predict_layers > 0:
        input_keys.append("nbatch_pack_offset")
    if model_args.use_attn_mask_startend_row_indices:
        input_keys.append("attn_mask_startend_row_indices")
    else:
        input_keys.append("attention_mask")
    return_list = []
    if max_seq_len is None:
        max_seq_len = max(len(item.token_ids) for sequence in batch for item in sequence)
    for batch_sequence in batch:
        original_token_ids = [seq.token_ids for seq in batch_sequence]
        token_ids = [sum(original_token_ids, [])]
        loss_mask = [sum([seq.loss_mask for seq in batch_sequence], [])]
        labels = [sum([seq.labels for seq in batch_sequence], [])]
        # padding
        padded_token_ids = pad_batch_data(token_ids, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        padded_labels = pad_batch_data(labels, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        padded_loss_mask = pad_batch_data(loss_mask, pad_idx=0, max_seq_len=max_seq_len)
        padded_labels = np.where(padded_loss_mask == 1, padded_labels, -100)
        return_list.append(
            [
                padded_token_ids,
                padded_labels,
                padded_loss_mask,
            ]
        )

        if training_args.num_nextn_predict_layers > 0:
            # each sequence end index
            batch_sequence_len = [len(sequence) for sequence in original_token_ids]
            nbatch_pack_offset = [0] * sum(batch_sequence_len)
            prefix_sum = 0
            for sequence_len in batch_sequence_len[:-1]:
                prefix_sum += sequence_len
                nbatch_pack_offset[prefix_sum - 1] = 1
            padded_nbatch_pack_offset = pad_batch_data([nbatch_pack_offset], pad_idx=0, max_seq_len=max_seq_len)
            return_list[-1].append(padded_nbatch_pack_offset)

        if not model_args.stage.lower() == "pt":
            if model_args.use_attn_mask_startend_row_indices:
                return_list[-1].append(gen_attn_mask_startend_row_indices(original_token_ids, max_seq_len))
            else:
                return_list[-1].append(gen_self_attn_mask(original_token_ids, max_seq_len))

    return_list = [np.concatenate(tensor_list) for tensor_list in zip(*return_list)]
    input_dict = dict(zip(input_keys, return_list))
    return input_dict


def process_fc(data, input_file):
    multi_turns_messages = data["messages"]
    tools_list = data["tools"] if "tools" in data else None
    label = data["label"] if "label" in data else None

    system = ""
    is_system = False
    if "system" in multi_turns_messages[0]["role"]:
        system = multi_turns_messages[0]["content"]
        is_system = True

    # be default, all assistant output should be learned, labels are all 1
    if label is None:
        label = []
        for index, turn in enumerate(multi_turns_messages):
            if "assistant" in turn["role"]:
                label.append(1)

    assistant_index = 0
    for index, turn in enumerate(multi_turns_messages):
        if "assistant" in turn["role"] and label[assistant_index]:
            message = copy.deepcopy(multi_turns_messages[: index + 1])
            ex = Example(
                request={"messages": message, "tools": tools_list},
                system=system,
                label=[1],
                is_system=is_system,
                source=input_file,
                is_function_call=True,
            )
            yield ex
            assistant_index += 1


def process_example(data, input_file):
    """Convert raw data example into training example.

    Args:
        data (dict): Raw example data with:
        input_file (str): Source file path

    Returns:
        Example: Processed example for sequence generation
    """
    # We have the code completion dataset, which has the following fields
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]

    if len(data["src"]) == 0 or len(data["tgt"]) == 0:
        raise ValueError("Ignore example with empty src or empty tgt.")

    for item in data["src"] + data["tgt"]:
        if len(item.strip()) == 0:
            raise ValueError("Ignore example with empty string in str / tgt field.")

    if "label" not in data:
        data["label"] = [1] * len(data["src"])

    if not (len(data["src"]) == len(data["tgt"]) == len(data["label"])):
        raise ValueError(
            f"The length of src & tgt & label must be equal, but get len(data['src']) : {len(data['src'])}, ' len(data['tgt']) : {len(data['tgt'])}, ' len(data['label']) : {len(data['label'])}"
        )

    if "is_system" not in data:
        # If is_system is 1, it indicates that the sample includes system settings
        # and no other sample should be concatenated before it.
        data["is_system"] = 0

    if data["is_system"] == 1:
        data["system"] = data["src"][0]
        data["src"] = data["src"][1:]
        data["tgt"] = data["tgt"][1:]
        data["label"] = data["label"][1:]

    # update "system"
    if "system" in data:
        if not isinstance(data["system"], str):
            raise ValueError("System field must be a string.")
        data["is_system"] = 1

    # convert to OpenAI format
    data["messages"] = []
    if "system" in data:
        data["messages"].append({"role": "system", "content": data["system"]})
    for q, a in zip(data["src"], data["tgt"]):
        data["messages"].append({"role": "user", "content": q.strip()})
        data["messages"].append({"role": "assistant", "content": a.strip()})

    return Example(
        request={"messages": data["messages"]},
        system=data["system"] if data["is_system"] else "",
        label=data["label"],
        is_system=data["is_system"],
        source=input_file,
    )


def process_pretraining_example(data, input_file):
    """Convert raw data example into training example.
    Args:
        data (dict): Raw example data with:
        input_file (str): Source file path
    Returns:
        Example: Processed example for sequence generation
    """
    # We have the code completion dataset, which has the following fields
    if isinstance(data["text"], str):
        data["text"] = [data["text"]]

    if len(data["text"]) == 0:
        raise ValueError("Ignore example with empty src or empty tgt.")

    for item in data["text"]:
        if len(item.strip()) == 0:
            raise ValueError("Ignore example with empty string in str / tgt field.")

    if "label" not in data:
        data["label"] = [1] * len(data["text"])

    # convert to OpenAI format
    data["messages"] = []
    for a in data["text"]:
        data["messages"].append({"role": "assistant", "content": a.strip()})

    return Example(
        request={"messages": data["messages"]},
        system="",
        label=data["label"],
        is_system=0,
        source=input_file,
    )


class SequenceDataset(IterableDataset):
    """Dataset for creating sequences from multi-source examples.

    This is a stateful dataset that handles sequence generation and packing.
    """

    def __init__(
        self,
        dataset: MultiSourceDataset,
        tokenizer,
        max_seq_len: int = 4096,
        num_samples_each_epoch: int = 100000,
        is_valid: bool = False,
        random_seed: int = 11,
        random_shuffle: bool = True,
        greedy_intokens: bool = False,
        packing: bool = False,
        mix_strategy: str = "random",
        encode_one_turn: bool = True,
        use_template: bool = True,
        is_pretraining: bool = False,
    ):
        """Initialize SequenceDataset.

        Args:
            dataset (MultiSourceDataset): The multi-source example dataset.
            tokenizer: Tokenizer for text processing.
            max_seq_len (int): Maximum sequence length.
            num_samples_each_epoch (int): Target samples per epoch.
            is_valid (bool): Flag for validation mode.
            random_seed (int): Seed for random number generation.
            random_shuffle (bool): Enable random shuffling.
            greedy_intokens (bool): Use greedy in-token packing strategy.
        """

        self.example_dataset = dataset
        self.tokenizer = tokenizer
        self.start_token = tokenizer.bos_token  # "<s>"
        self.end_token = tokenizer.eos_token  # "</s>"
        self.break_token = tokenizer.sep_token  # "<sep>"
        self.break_turn_token = tokenizer.cls_token  # "<cls>"
        self.sys_start_token = getattr(tokenizer, "sys_start_token", None)
        self.sys_end_token = getattr(tokenizer, "sys_end_token", None)
        self.max_seq_len = max_seq_len
        self.is_valid = is_valid
        self.random_seed = random_seed
        self.random_shuffle = random_shuffle
        self.greedy_intokens = greedy_intokens
        self.packing = packing
        self.mix_strategy = mix_strategy
        self.encode_one_turn = encode_one_turn
        self.use_template = use_template
        self.num_samples_each_epoch = num_samples_each_epoch
        self.reverse = True
        self.is_pretraining = is_pretraining

        # For new data concatenation mode
        self.begin_of_query = self.tokenizer.tokenize("User: ")
        self.begin_of_response = self.tokenizer.tokenize("\nAssistant: ")
        self.end_of_response = getattr(self.tokenizer.special_tokens_map, "sep_token", "<|end_of_sentence|>")
        self.begin_token = getattr(self.tokenizer.special_tokens_map, "cls_token", "<|begin_of_sentence|>")
        self.newline_token = self.tokenizer.tokenize("\n")  # Same effect as sys_end_token
        if isinstance(self.tokenizer, PretrainedTokenizer):
            self.end_of_response_id = self.tokenizer._convert_token_to_id([self.end_of_response])[0]
            self.begin_token_id = self.tokenizer._convert_token_to_id([self.begin_token])[0]
        else:
            self.end_of_response_id = self.tokenizer.convert_tokens_to_ids([self.end_of_response])[0]
            self.begin_token_id = self.tokenizer.convert_tokens_to_ids([self.begin_token])[0]

        datasets_list = [task["dataset"] for task in self.example_dataset._task_group]
        datasets_prob = [task["prob"] for task in self.example_dataset._task_group]

        if is_valid:
            self.random_shuffle = False
            self.greedy_intokens = 0
            self.mix_datasets = create_dataset_instance(
                "concat",
                datasets_list,
                datasets_prob,
                ("upsampling" if self.mix_strategy == "interleave_under" else "oversampling"),
                self.random_seed,
                self.random_shuffle,
                self.num_samples_each_epoch,
            )
        else:
            if self.mix_strategy not in [
                "random",
                "concat",
                "interleave_under",
                "interleave_over",
            ]:
                raise ValueError(f"Unsupported mix strategy: {self.mix_strategy}")
            else:
                self.mix_datasets = create_dataset_instance(
                    self.mix_strategy,
                    datasets_list,
                    datasets_prob,
                    ("upsampling" if self.mix_strategy == "interleave_under" else "oversampling"),
                    self.random_seed,
                    self.random_shuffle,
                    self.num_samples_each_epoch,
                    self.reverse,
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

    def __iter_func(self):
        """Core iterator function for sequence generation.

        Returns:
            Sequence: A processed sequence containing token IDs and labels.
        """

        # prepare epoch data
        batch_sequence, cur_len = [], 0
        dataset_iterator = iter(self.mix_datasets)

        # pre-training:
        # 1. tokenize all the samples in the sampling pool,
        # 2. combine them into one large sample
        # 3. truncate it into multiple new samples based on the max_seq_len.
        if self.is_pretraining:
            all_tokenized_tokens = []
            for _ in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                actual_example_num = 1
                tokens = self._postprocess_pretraining_sequence(example, actual_example_num)
                if tokens is None:
                    if self.estimate:
                        self.unused_samples += actual_example_num
                    continue
                if self.estimate:
                    self.used_samples += actual_example_num

                all_tokenized_tokens.extend(tokens)

                while len(all_tokenized_tokens) >= self.max_seq_len:
                    cut_tokens = all_tokenized_tokens[: self.max_seq_len]
                    cut_tokens = cut_tokens + [self.tokenizer.eos_token_id]
                    all_tokenized_tokens = all_tokenized_tokens[self.max_seq_len :]

                    res_tokens = cut_tokens[:-1]
                    res_labels = cut_tokens[1:]
                    loss_mask = [1] * len(res_tokens)
                    pos_ids = list(range(len(res_tokens)))
                    sequence = Sequence(
                        token_ids=res_tokens,
                        position_ids=pos_ids,
                        labels=res_labels,
                        loss_mask=loss_mask,
                        num_examples=actual_example_num,
                    )
                    batch_sequence = [sequence]
                    yield batch_sequence

                    if self.estimate:
                        self.used_estimate_samples += actual_example_num
                        if self.used_estimate_samples >= self.max_estimate_samples:
                            self.used_estimate_samples = 0
                            # Set flag to False and yield empty list to signal the end of estimation
                            self.estimate = False
                            yield []

            # If the entire dataset has been fully traversed, return the remaining data.
            if len(all_tokenized_tokens) > 0:
                cut_tokens = all_tokenized_tokens
                cut_tokens = cut_tokens + [self.tokenizer.eos_token_id]
                res_tokens = cut_tokens[:-1]
                res_labels = cut_tokens[1:]
                loss_mask = [1] * len(res_tokens)
                pos_ids = list(range(len(res_tokens)))
                sequence = Sequence(
                    token_ids=res_tokens,
                    position_ids=pos_ids,
                    labels=res_labels,
                    loss_mask=loss_mask,
                    num_examples=actual_example_num,
                )
                batch_sequence = [sequence]
                yield batch_sequence
                if self.estimate:
                    self.used_estimate_samples += actual_example_num
                    if self.used_estimate_samples >= self.max_estimate_samples:
                        self.used_estimate_samples = 0
                        # Set flag to False and yield empty list to signal the end of estimation
                        self.estimate = False
                        yield []
        else:
            if not self.packing:
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    actual_example_num = 1
                    sequence = self._postprocess_sequence(example, actual_example_num)
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
                        actual_example_num = 1
                        sequence = self._postprocess_sequence(example, actual_example_num)
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
                        actual_example_num = 1
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
        """Iterator interface for the dataset.

        Yields:
            Sequence: The generated sequences.
        """
        if self.is_valid:
            yield from self.__iter_func()
        else:
            while True:
                yield from self.__iter_func()

    def _postprocess_pretraining_sequence(self, example, actual_example_num):
        # tokens
        content = example.request["messages"][0]["content"]
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))

        return tokens

    def _postprocess_sequence(self, example, actual_example_num):
        """Process code completion examples into token sequences.

        Args:
            example: The input example containing code components.
            actual_example_num (int): Number of examples used.

        Returns:
            Sequence: Processed sequence or None if invalid.
        """
        if self.use_template:
            if not self.tokenizer.chat_template:
                self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
            if example.is_function_call:
                encoded_messages = postprocess_fc_sequence(self.tokenizer, example.request)
            else:
                encoded_messages = self.tokenizer.encode_chat_inputs(
                    example.request, encode_one_turn=self.encode_one_turn
                )
        else:
            encoded_messages = self.tokenizer.encode_chat_inputs_with_no_template(
                example.request, encode_one_turn=self.encode_one_turn
            )

        num_reserved_tokens_for_each_dialog = 1  # only break_turn_token or end_token
        num_reserved_tokens_for_each_turn = 8

        cur_len = num_reserved_tokens_for_each_dialog

        turn_index = len(encoded_messages) - 1

        tokens = []
        loss_mask = []
        while turn_index >= 0:
            tokens_src, tokens_target = encoded_messages[turn_index]
            if len(tokens_src) + len(tokens_target) > (
                self.max_seq_len + 1 - cur_len - num_reserved_tokens_for_each_turn
            ):
                break

            tokens = tokens_src + tokens_target + tokens

            loss_mask = (
                [0] * (len(tokens_src) - 1) + [example.label[turn_index]] * (len(tokens_target) + 1) + loss_mask
            )
            assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"

            cur_len = len(tokens)

            turn_index -= 1

        # Not even one turn can be added, so need to do warning and skip this example
        if len(tokens) <= num_reserved_tokens_for_each_dialog + num_reserved_tokens_for_each_turn:
            try:
                # For print log
                sub_src = example.src[0].strip()[:5]
                sub_tgt = example.tgt[-1].strip()[-5:]
                global LOGGER_COUNT
                LOGGER_COUNT += 1
                if LOGGER_COUNT <= 5:
                    logger.warning(f"even one turn, example_output:'{{'src':[{sub_src}, ……],'tgt':[……{sub_tgt}]}}'")
            except Exception:
                logger.warning("[SKIP] wrong example")

            return None

        if self.use_template:
            if self.begin_token_id is not None and self.end_of_response_id is not None:
                # Maybe left truncated, so need to add begin_token
                if tokens[0] != self.begin_token_id:
                    tokens = [self.begin_token_id] + tokens
                    loss_mask = [0] + loss_mask

                if len(tokens) > self.max_seq_len:
                    raise RuntimeError(f"token_ids is too long: {len(tokens)}")

                # Add EOS token at the end
                del tokens[-1]
                del loss_mask[-1]
                labels = tokens[1:] + [self.tokenizer.eos_token_id]

                # end_of_response is a special token that indicates the end of the turn.
                # end_token is a special token that indicates the end of the answer.
                labels = [
                    label if label != self.end_of_response_id else self.tokenizer.eos_token_id for label in labels
                ]
            else:
                tokens = tokens[:-1] + [self.tokenizer.eos_token_id]
                labels = tokens[1:] + [-100]
                if len(tokens) > self.max_seq_len:
                    raise RuntimeError(f"token_ids is too long: {len(tokens)}")
        else:
            oral_tokens = tokens
            tokens = oral_tokens[:-1]
            labels = oral_tokens[1:]
            loss_mask = loss_mask[1:]
            if len(tokens) > self.max_seq_len:
                raise RuntimeError(f"token_ids is too long: {len(tokens)}")

        pos_ids = list(range(len(tokens)))

        if sum(loss_mask) == 0:
            logger.warning(f"[SKIP] all labels set to 0: {example}")
            return None

        assert len(tokens) == len(loss_mask), f"{len(tokens)}-{len(loss_mask)}"
        assert len(tokens) == len(labels), f"{len(tokens)}-{len(labels)}"

        def print_debug_info(tokenizer, data, label):
            """Helper function to print tokenized data debug info"""
            try:
                decoded = tokenizer.decode(data)
                logger.info(f"[dataset debug] {label}: {decoded}")
            except (TypeError, ValueError, OverflowError) as e:
                logger.info(f"[dataset debug] tokenizer decode {label} error: {str(e)}")

        enable_dataset_debug = os.getenv("FLAGS_enable_dataset_debug", "false").lower() in ("true", "1", "t")

        if enable_dataset_debug:
            logger.info("\n" + "=" * 50)
            logger.info("[dataset debug] Debug mode enabled")

            if hasattr(self, "tokenizer"):
                print_debug_info(self.tokenizer, tokens, "input")
                labels = [x for x in labels if x != -100]  # remove -100
                print_debug_info(self.tokenizer, labels, "labels")
                logger.info(f"[dataset debug] loss mask: {loss_mask}")
            else:
                logger.info("[dataset debug] Tokenizer not available")

            logger.info("=" * 50 + "\n")
        return Sequence(
            token_ids=tokens,
            position_ids=pos_ids,
            labels=labels,
            loss_mask=loss_mask,
            num_examples=actual_example_num,
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
            sequence = self._postprocess_sequence(examples[index], actual_example_num_list[index])
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


def gen_self_attn_mask(batch_token_ids: List[List[int]], max_seq_len: int):
    """Generate self-attention mask for multi-sequence batches.

    Args:
        batch_token_ids (List[List[int]]): List of token ID sequences.
        max_seq_len (int): Maximum sequence length.

    Returns:
        ndarray: 4D attention mask array.
    """
    input_mask_data = np.zeros((1, 1, max_seq_len, max_seq_len), dtype="float32")
    offset = 0
    for index, token_ids in enumerate(batch_token_ids):
        cur_len = len(token_ids)
        b = np.tril(np.ones([cur_len, cur_len]), 0)
        input_mask_data[0, 0, offset : offset + cur_len, offset : offset + cur_len] = b
        offset += cur_len
    return input_mask_data


def gen_attn_mask_startend_row_indices(batch_token_ids: List[List[int]], max_seq_len: int):
    """Generate row indices for flash attention masks.

    Args:
        batch_token_ids (List[List[int]]): List of token ID sequences.
        max_seq_len (int): Maximum sequence length.

    Returns:
        ndarray: Row indices array with dtype int32.
    """
    offset = 0
    attn_mask_startend_row_indices = []
    for token_ids in batch_token_ids:
        cur_len = len(token_ids)
        attn_mask_startend_row_indices.extend([offset + cur_len] * cur_len)
        offset += cur_len
    if offset < max_seq_len:
        attn_mask_startend_row_indices.extend(list(range(offset, max_seq_len)))
    # NOTE(hehuang): The dtype of attn_mask_startend_row_indices must be np.int32
    return np.array(attn_mask_startend_row_indices, dtype=np.int32)[None, None, ..., None]  # add dimension modify
