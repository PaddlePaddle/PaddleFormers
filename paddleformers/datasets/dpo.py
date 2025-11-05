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

"""DPO dataset."""

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from paddle.io import IterableDataset
from scipy.linalg import block_diag

from paddleformers.utils.env import NONE_CHAT_TEMPLATE

from ..utils.log import logger
from .base import MultiSourceDataset
from .data_utils import postprocess_fc_sequence
from .mix_datasets import create_dataset_instance

LOGGER_COUNT = 0


@dataclass
class Example:
    """Dataset example."""

    chosen: str
    rejected: str
    source: str
    session_start_index: int
    score_delta: float
    is_function_call: bool


@dataclass
class Sequence:
    """Sequence."""

    input_ids: Optional[List[int]]
    position_ids: Optional[List[int]]
    attention_mask: Optional[List[List[int]]]
    attn_mask_startend_row_indices: Optional[List[int]]
    chosen_labels: List[int]
    rejected_labels: List[int]
    response_index: List[int]
    score_delta: float


def create_dataset(**dataset_config):
    """Create DPO dataset.

    Args:
        **dataset_config: Configuration parameters including:
            - task_dataset_path (str): Path of each dataset
            - task_dataset_prob (str): Prob of each dataset
            - sub_dataset_type (str): type of each dataset
            - tokenizer: Text tokenization module
            - max_seq_len (int): Total sequence length limit
            - max_prompt_len (int): Total prompt length
            - num_samples_each_epoch (int): number of sample per training epoch
            - is_valid (bool, optional): Validation mode flag. Defaults to False
            - random_seed (int): Reproduction seed for shuffling
            - greedy_intokens (bool): Greedy intokens strategy
            - buffer_size (int): Preloading buffer capacity
            - use_attn_mask_startend_row_indices (bool): Sparse attention mode
            - mask_out_eos_token (bool): EOS loss masking

    Returns:
        SequenceDataset: Configured dataset pipeline with:
            - Multi-source data loading
            - Dynamic sequence generation
            - Session-aware processing (when enabled)
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
        process_fn=process_session_example,
        process_fn_fc=process_fc,
    )
    sequence_dataset = SequenceDataset(
        dataset=example_dataset,
        tokenizer=dataset_config["tokenizer"],
        max_seq_len=dataset_config["max_seq_len"],
        max_prompt_len=dataset_config["max_prompt_len"],
        num_samples_each_epoch=dataset_config["num_samples_each_epoch"],
        is_valid=dataset_config.get("is_valid", False),
        random_seed=dataset_config["random_seed"],
        random_shuffle=dataset_config["random_shuffle"],
        greedy_intokens=dataset_config["greedy_intokens"],
        buffer_size=dataset_config["buffer_size"],
        use_attn_mask_startend_row_indices=dataset_config.pop("use_attn_mask_startend_row_indices", True),
        mask_out_eos_token=dataset_config["mask_out_eos_token"],
        packing=dataset_config["packing"],
        mix_strategy=dataset_config["mix_strategy"],
        encode_one_turn=dataset_config["encode_one_turn"],
    )
    return sequence_dataset


def process_fc(data, input_file):
    multi_turns_messages = data["messages"]
    tools_list = data["tools"]
    label = data["label"] if "label" in data else None

    # be default, all assistant output should be learned, labels are all 1
    if label is None:
        label = []
        for index, turn in enumerate(multi_turns_messages):
            # if "assistant" in turn["role"]:
            if index % 2 == 1:
                label.append(1)

    def split_dpo_messages(full_messages):
        chosen_m, rejected_m = deepcopy(full_messages), deepcopy(full_messages)
        chosen_m[-1] = full_messages[-1]["preferred_output"]
        rejected_m[-1] = full_messages[-1]["non_preferred_output"]
        return chosen_m, rejected_m

    assistant_index = 0
    for index, turn in enumerate(multi_turns_messages):
        is_assistant_turn = turn["role"] == "assistant" if "role" in turn else "non_preferred_output" in turn
        if is_assistant_turn:
            if label[assistant_index]:
                chosen_m, rejected_m = split_dpo_messages(multi_turns_messages[: index + 1])
                ex = Example(
                    chosen={"messages": chosen_m, "tools": tools_list},
                    rejected={"messages": rejected_m, "tools": tools_list},
                    session_start_index=0,  # single turn
                    score_delta=1.0,
                    source=input_file,
                    is_function_call=True,
                )
                yield ex

            assistant_index += 1


def collate_fn(
    batch,
    tokenizer,
    max_seq_len=None,
    use_sparse_head_and_loss_fn=True,
    use_fused_head_and_loss_fn=True,
    use_response_score_delta=False,
):
    """Convert batch data into tensor for DPO.

    Args:
        batch (List[List[Sequence]]): Batch of input sequences containing multiple data samples.
            Each sample is a list of Sequence objects containing tokenized data components.
        tokenizer (Tokenizer): Text tokenizer for processing sequence components.
        max_seq_len (int, optional): Maximum sequence length for padding/truncation.
            If None, will raise ValueError. Defaults to None.
        use_sparse_head_and_loss_fn (bool, optional): Whether to use sparse indexing for loss calculation.
            Enables memory-efficient indexing for large sequences. Defaults to True.
        use_fused_head_and_loss_fn (bool, optional): Whether to use fused kernel to calculate lm head and loss.
            Optimizes for memory access patterns. Defaults to True.

    Returns:
        Dict[str, np.ndarray]: Processed tensor dictionary containing:
            - input_ids (int32): Padded token ids [batch_size, max_seq_len]
            - position_ids (int32): Position ids [batch_size, max_seq_len]
            - chosen_labels (int32): Preferred response labels [batch_size, max_seq_len]
            - rejected_labels (int32): Unpreferred response labels [batch_size, max_seq_len]
            - response_indexs (int32): Response span indices [batch_size, 4]
            - attention_mask (float32, optional): Attention mask matrix [batch_size, 1, max_seq_len, max_seq_len]
            - attn_mask_startend_row_indices (int32, optional): Sparse attention row indices [batch_size, max_seq_len]
    """
    if max_seq_len is None:
        max_seq_len = max(len(item.input_ids) for sequence in batch for item in sequence)

    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "chosen_labels": [],
        "rejected_labels": [],
        "response_indexs": [],
    }
    if use_response_score_delta:
        input_dict["score_deltas"] = []

    sequence = batch[0][0]
    if sequence.attn_mask_startend_row_indices is not None:
        input_dict["attn_mask_startend_row_indices"] = []
        use_attn_mask_startend_row_indices = True
    elif sequence.attention_mask is not None:
        input_dict["attention_mask"] = []
        use_attn_mask_startend_row_indices = False
    else:
        raise ValueError("attention_mask and attn_mask_startend_row_indices are both None.")
    sequence_sum_flatten = 0
    for i, sequences in enumerate(batch):
        difference = max_seq_len - sum([len(sequence.input_ids) for sequence in sequences])

        input_dict["input_ids"].append(sum([sequence.input_ids for sequence in sequences], []) + [0] * difference)
        input_dict["position_ids"].append(
            sum([sequence.position_ids for sequence in sequences], []) + [0] * difference
        )
        input_dict["chosen_labels"].append(
            sum([sequence.chosen_labels for sequence in sequences], []) + [0] * difference
        )
        input_dict["rejected_labels"].append(
            sum([sequence.rejected_labels for sequence in sequences], []) + [0] * difference
        )
        if use_attn_mask_startend_row_indices:
            start_row_indices = []
            sequence_sum = 0
            for sequence in sequences:
                start_row_indices += [indice + sequence_sum for indice in sequence.attn_mask_startend_row_indices]
                sequence_sum += len(sequence.input_ids)
            input_dict["attn_mask_startend_row_indices"].append(
                [start_row_indices + list(range(start_row_indices[-1], max_seq_len))]
            )
        else:
            input_dict["attention_mask"].append(
                # (s,s) -> (1,s,s)
                np.expand_dims(
                    # pad to max_loength
                    np.pad(
                        # block attention_mask
                        block_diag(*[sequence.attention_mask for sequence in sequences]),
                        pad_width=((0, difference), (0, difference)),
                        mode="constant",
                        constant_values=False,
                    ),
                    axis=0,
                )
            )
        sequence_sum = 0
        for sequence in sequences:
            # bs, chosen_response_start_index, rejeted_response_start_index, rejeted_response_end_index + 1
            if use_sparse_head_and_loss_fn:
                response_index = [
                    i,
                    sequence_sum_flatten,
                    sequence.response_index[1] - sequence.response_index[0] + sequence_sum_flatten,
                    sequence.response_index[2] - sequence.response_index[0] + sequence_sum_flatten,
                ]
                sequence_sum_flatten += sequence.response_index[2] - sequence.response_index[0]
            elif use_fused_head_and_loss_fn:
                response_index = [
                    i,
                    sequence.response_index[0] + sequence_sum_flatten,
                    sequence.response_index[1] + sequence_sum_flatten,
                    sequence.response_index[2] + sequence_sum_flatten,
                ]
                sequence_sum_flatten += len(sequence.input_ids)
            else:
                response_index = [
                    i,
                    sequence.response_index[0] + sequence_sum,
                    sequence.response_index[1] + sequence_sum,
                    sequence.response_index[2] + sequence_sum,
                ]
                sequence_sum += len(sequence.input_ids)
            input_dict["response_indexs"].append(response_index)
            if use_response_score_delta:
                input_dict["score_deltas"].append(sequence.score_delta)

    for key in input_dict:
        if key == "attention_mask":
            input_dict[key] = np.array(input_dict[key], dtype=np.float32)
        elif key == "attn_mask_startend_row_indices":
            input_dict[key] = np.array(input_dict[key], dtype=np.int32)[..., None]
        else:
            input_dict[key] = np.array(input_dict[key])
    return input_dict


def process_session_example(data, input_file):
    """Convert raw format example to Example.

    Args:
        data (dict): Raw session data dictionary containing:
            - src (str/list): Multi-turn dialogue context (user inputs sequence)
            - tgt (str/list): Assistant responses sequence (must be 1 shorter than src)
            - response (List[List[str]]): Pair of multi-turn response candidates [each is list of strings]
            - sort (List[int]): Ranking scores for response pairs [length must be 2]
            - system (str, optional): System-level instruction for dialogue
        input_file (str): Source file path for data provenance tracking

    Returns:
        Example: Standardized data container with fields:
            - src (list): Full context sequence (with system prompt if exists)
            - tgt (list): Expected response sequence
            - is_system (int): System prompt presence flag (0/1)
            - chosen/rejected (list): Selected best/worst multi-turn responses
            - source: Original data file path
            - data_format: Format identifier "sft"
    """
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        raise ValueError(
            f"Data format error. src length must be tgt length + 1. "
            f"But got src_length:{len(data['src'])} tgt_length:{len(data['tgt'])}"
        )
    if (len(data["response"]) != 2) or (len(data["response"]) != len(data["sort"])):
        raise ValueError(
            f"Response and sort length must be 2. "
            f"But got response_length:{len(data['response'])} sort_length:{len(data['sort'])}."
        )
    if data["sort"][0] == data["sort"][1]:
        raise ValueError(f"Sort field must be different." f" But got 'sort':{data['sort']}")
    if isinstance(data["response"][0], str) and isinstance(data["response"][1], str):
        data["response"] = [[data["response"][0]], [data["response"][1]]]
    for response in data["response"]:
        if not isinstance(response, list):
            raise ValueError(f"Session level response should be List[List[str]], but got List of {type(response)}")
        if len(response) % 2 != 1:
            raise ValueError("The number of responses should be even, but an odd number of responses were obtained.")
        for r in response:
            if len(r.strip()) < 1:
                raise ValueError(f"Response field must be longer than 1." f" But got 'response':{data['response']}.")

    if len(data["response"][0]) < 1 or len(data["response"][1]) < 1:
        raise ValueError(f"Ignore empty response." f" But got 'response':{data['response']}.")
    if data["sort"][0] > data["sort"][1]:
        chosen = data["response"][0]
        rejected = data["response"][1]
    else:
        chosen = data["response"][1]
        rejected = data["response"][0]

    if "is_system" not in data:
        # If is_system is 1, it indicates that the sample includes system settings
        # and no other sample should be concatenated before it.
        data["is_system"] = 0

    if data["is_system"] == 1:
        data["system"] = data["src"][0]
        data["src"] = data["src"][1:]
        data["tgt"] = data["tgt"][1:]

    if "system" in data:
        if not isinstance(data["system"], str):
            raise ValueError("System field must be a string.")

    # convert to OpenAI format
    data["messages"] = []
    if "system" in data:
        data["messages"].append({"role": "system", "content": data["system"]})
    for idx in range(len(data["src"])):
        data["messages"].append({"role": "user", "content": data["src"][idx]})
        if idx != len(data["src"]) - 1:
            data["messages"].append({"role": "assistant", "content": data["tgt"][idx]})

    chosen_m, rejected_m = data["messages"], deepcopy(data["messages"])
    session_start_index = (
        len(data["messages"]) if data["messages"][0]["role"] != "system" else len(data["messages"]) - 1
    )
    for idx in range(len(chosen)):
        if idx % 2 == 0:
            # assistant
            chosen_m.append({"role": "assistant", "content": chosen[idx]})
            rejected_m.append({"role": "assistant", "content": rejected[idx]})
        else:
            # user
            chosen_m.append({"role": "user", "content": chosen[idx]})
            rejected_m.append({"role": "user", "content": rejected[idx]})

    return Example(
        chosen={"messages": chosen_m},
        rejected={"messages": rejected_m},
        session_start_index=session_start_index,
        source=input_file,
        score_delta=1.0,
        is_function_call=False,
    )


class SequenceDataset(IterableDataset):
    """Stateful dataset for generating token sequences from multi-source examples.

    Args:
        dataset (MultiSourceDataset): Source dataset containing examples to process
        tokenizer (Tokenizer): Tokenizer for text processing and token conversion
        max_seq_len (int, optional): Maximum sequence length. Defaults to 4096
        max_prompt_len (int, optional): Maximum prompt context length. Defaults to 2048
        num_samples_each_epoch (int, optional): number of sample per epoch. Defaults to 1e5
        is_valid (bool, optional): Validation mode flag (disable randomization). Defaults to False
        random_seed (int, optional): Seed for reproducible shuffling. Defaults to 11
        random_shuffle (bool, optional): Enable random shuffling. Defaults to True
        greedy_intokens (bool, optional): Greedy intokens  strategy. Defaults to False
        buffer_size (int, optional): Preload buffer size for optimization. Defaults to 500
        use_attn_mask_startend_row_indices (bool, optional): Use sparse attention indexing. Defaults to True
        mask_out_eos_token (bool, optional): Exclude EOS from loss calculation. Defaults to True
    """

    def __init__(
        self,
        dataset: MultiSourceDataset,
        tokenizer,
        max_seq_len: int = 4096,
        max_prompt_len: int = 2048,
        num_samples_each_epoch: int = 100000,
        is_valid: bool = False,
        random_seed: int = 11,
        random_shuffle: bool = True,
        greedy_intokens: bool = False,
        buffer_size: int = 500,
        use_attn_mask_startend_row_indices: bool = True,
        mask_out_eos_token: bool = True,
        packing: bool = False,
        mix_strategy: str = "random",
        encode_one_turn: bool = True,
    ):
        self.example_dataset = dataset
        self.tokenizer = tokenizer
        self.start_token = tokenizer.bos_token
        self.end_token = tokenizer.eos_token
        self.break_token = tokenizer.sep_token
        self.break_turn_token = tokenizer.cls_token
        self.sys_start_token = getattr(tokenizer, "sys_start_token", None)
        self.sys_end_token = getattr(tokenizer, "sys_end_token", None)

        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        if self.max_prompt_len > self.max_seq_len:
            raise ValueError(
                f"max_prompt_len should be less than max_seq_len, but got {self.max_prompt_len} > {self.max_seq_len}"
            )
        self.is_valid = is_valid
        self.random_seed = random_seed
        self.random_shuffle = random_shuffle
        self.greedy_intokens = greedy_intokens
        self.buffer_size = buffer_size
        self.origin_dataset_num = 0
        self.use_attn_mask_startend_row_indices = use_attn_mask_startend_row_indices
        self.mask_out_eos_token = mask_out_eos_token
        self.packing = packing
        self.mix_strategy = mix_strategy
        self.encode_one_turn = encode_one_turn
        self.num_samples_each_epoch = num_samples_each_epoch

        # For new data concatenation mode
        self.begin_of_query = self.tokenizer.tokenize("User: ")
        self.begin_of_response = self.tokenizer.tokenize("\nAssistant: ")
        self.end_of_response = "<|end_of_sentence|>"
        self.begin_token = "<|begin_of_sentence|>"  # Same effect as sys_start_token
        self.newline_token = self.tokenizer.tokenize("\n")  # Same effect as sys_end_token

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
                )

    def __iter_func(self):
        """
            The __iter_func method implements iteration over the dataset.
            Each iteration returns a Sequence-type element.
            Within the current epoch, samples are randomly generated using epoch_rng and are only valid for that epoch.
            If multiple workers exist, data is partitioned according to worker ID.

        Args:
            None (no parameters)

        Returns:
            Sequence (class): A Sequence-type element containing input IDs, input masks, and labels.

        Raises:
            No exceptions raised.
        """

        # prepare epoch data
        batch_sequence, cur_len = [], 0
        dataset_iterator = iter(self.mix_datasets)

        if not self.packing:
            for _ in range(len(self.mix_datasets)):
                example = next(dataset_iterator)
                sequence = self._postprocess_sequence(example)
                if sequence is None:
                    continue

                batch_sequence, cur_len = [sequence], len(sequence.input_ids)
                yield batch_sequence

            if len(batch_sequence) > 0:
                yield batch_sequence
        else:
            if not self.greedy_intokens:
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    sequence = self._postprocess_sequence(example)
                    if sequence is None:
                        continue

                    if cur_len + len(sequence.input_ids) <= self.max_seq_len:
                        batch_sequence.append(sequence)
                        cur_len += len(sequence.input_ids)
                    else:
                        yield batch_sequence
                        batch_sequence, cur_len = [sequence], len(sequence.input_ids)

                if len(batch_sequence) > 0:
                    yield batch_sequence
            else:
                sequence_buffer = []
                buffer_size = self.buffer_size
                for _ in range(len(self.mix_datasets)):
                    example = next(dataset_iterator)
                    sequence = self._postprocess_sequence(example)
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
        """Generate sequence packs using greedy bin packing algorithm for efficient batching.

        Args:
            sequences (List[Sequence]): List of input sequences containing:
                - input_ids (List[int]): Tokenized sequence
                [Other sequence attributes...]

        Returns:
            List[List[Sequence]]: Packed sequences grouped into batches where:
                - Each sublist represents a batch
                - Sum of sequence lengths in batch <= self.max_seq_len
                - Batches ordered by descending remaining capacity
        """
        left_len_list = np.array([])
        sequence_pack = []
        for sequence in sequences:
            sequence_len = len(sequence.input_ids)
            if len(left_len_list) > 0:
                max_left_len_index = left_len_list.argmax()

            if len(left_len_list) == 0 or left_len_list[max_left_len_index] < sequence_len:
                sequence_pack.append([sequence])
                left_len_list = np.append(left_len_list, np.array([self.max_seq_len - sequence_len]))
            else:
                sequence_pack[max_left_len_index].append(sequence)
                left_len_list[max_left_len_index] -= sequence_len
        return sequence_pack

    def __postprocess_before_concat(self, example):
        """Process multi-turn conversation data into tokenized sequences with dynamic truncation.

        Args:
            example (Example): Input data object containing:
                - src (List[str]): Conversation history prompts
                - tgt (List[str]): Corresponding responses
                - chosen/rejected (List[str]): Preferred/unpreferred response paths
                - is_system (int): System prompt presence flag
                - system (str): System settings

        Returns:
            tuple: (prompt_ids, response_ids_list, label_ids_list, response_lens, total_len) containing:
                - prompt_token_ids (List[int]): Main conversation context token ids
                - response_token_ids_list (List[List[int]]): [chosen_path, rejected_path] response token ids
                - response_label_ids_list (List[List[int]]): Each response label ids（mask included）
                - response_len_list (List[int]): Valid response token length（special token excluded）
                - cur_len (int): Final input ids length
        """
        prompt_token_ids = []

        cur_len = 0

        # encoded_messages: List[List[int]]
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = NONE_CHAT_TEMPLATE
        if example.is_function_call:
            chosen_encoded_messages = postprocess_fc_sequence(self.tokenizer, example.chosen)
            rejected_encoded_messages = postprocess_fc_sequence(self.tokenizer, example.rejected)
        else:
            chosen_encoded_messages = self.tokenizer.encode_chat_inputs(example.chosen)
            rejected_encoded_messages = self.tokenizer.encode_chat_inputs(example.rejected)

        # chosen/rejected response
        response_token_ids_list = []
        response_label_ids_list = []
        response_len_list = []
        for responses in [
            chosen_encoded_messages[example.session_start_index // 2 :],
            rejected_encoded_messages[example.session_start_index // 2 :],
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
        turn_index = len(chosen_encoded_messages) - 1
        while turn_index >= 0:
            if turn_index == len(chosen_encoded_messages) - 1:
                cur_turn_token = chosen_encoded_messages[turn_index][0]
            else:
                cur_turn_token = chosen_encoded_messages[turn_index][0] + chosen_encoded_messages[turn_index][1]

            if cur_len + len(cur_turn_token) > self.max_seq_len:
                break

            prompt_token_ids = cur_turn_token + prompt_token_ids
            cur_len += len(cur_turn_token)
            turn_index -= 1

        # at least one turn
        if turn_index == len(chosen_encoded_messages) - 1:
            sub_src = example.chosen["messages"][0]["content"].strip()[:5]
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
        """Assemble processed components into final training sequence with attention controls.

        Args:
            example (Example): Input data object containing raw fields:
                - data_format (str): Specifies processing mode ("ec3_completion" or others)
                - [Other fields depending on data_format]

        Returns:
            Sequence: Processed training sequence containing:
                - input_ids (List[int]): Concatenated token IDs [prompt + chosen + rejected]
                - position_ids (List[int]): Position indices with special structure:
                    * prompt positions: 0~N
                    * chosen positions: N~N+M
                    * rejected positions: N~N+K (reuses prompt start index)
                - chosen_labels (List[int]): Masked labels for chosen response:
                    * 0 for prompt/rejected sections
                    * Shifted response tokens for chosen
                - rejected_labels (List[int]): Masked labels for rejected response
                - response_index (List[int]): Span indices [start, chosen_end, total_end]
                - attention controls (mask or indices):
                    * attention_mask (np.ndarray): Causal mask matrix if enabled
                    * attn_mask_startend_row_indices (List[int]): Sparse attention indices
                - score_delta (float): Score delta between chosen and rejected responses
        """
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
        # support use_sparse_head_and_loss_fn only
        response_index = [0, response_len_list[0], sum(response_len_list)]

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
        # 2. return sequence
        return Sequence(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            response_index=response_index,
            score_delta=example.score_delta,
        )
