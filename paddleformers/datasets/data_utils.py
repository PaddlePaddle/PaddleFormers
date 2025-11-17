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
"""Useful data utility."""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from paddleformers.utils.env import NONE_CHAT_TEMPLATE

from ..utils.log import logger

INF = 1000000
OPT_MULTI_OF = 256


@dataclass
class Example:
    """Data format for raw SFT (Supervised Fine-Tuning) examples."""

    request: Dict
    system: str
    label: List[int]
    is_system: int
    source: str
    is_function_call: bool = False


def round_up_to_multiple_of_8(n):
    """round up to multiple of 8"""
    return (n + 7) & ~7


def pad_batch_data(
    insts,
    pad_idx=0,
    return_pos=False,
    max_seq_len=None,
    return_input_mask=False,
    return_max_len=False,
    return_num_token=False,
    return_seq_lens=False,
):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max_seq_len if max_seq_len is not None else max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([inst + list([pad_idx] * (max_len - len(inst))) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst)) for inst in insts])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    if return_seq_lens:
        seq_lens = np.array([len(inst) for inst in insts])
        return_list += [seq_lens.astype("int64").reshape([-1, 1])]

    return return_list if len(return_list) > 1 else return_list[0]


def convert_to_tokens_for_pt(
    dial: List[dict],
    tokenizer,
    max_src_len,
):
    """Convert a dial to tokens for PT model."""
    # content_1+"\n"+content_2+"\n"+content_3
    sentence = "\n".join([x["content"] for x in dial])
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_src_len:
        logger.warning(
            f"The length of text ({len(tokens)}) cannot "
            f"be greater than max input length \
            ({max_src_len}). \
            We will truncate it."
        )
        # NOTE: LLM lost in middle
        tokens = tokens[: max_src_len // 2] + tokens[-max_src_len:]

    return tokens


def convert_to_tokens_for_sft(
    dial: List[dict],
    tokenizer,
    max_src_len,
):
    """
    Convert dialogue format into token sequences for supervised fine-tuning (SFT).

    Args:
        dial: Dialogue history as list of message dictionaries with:
              - role: "system", "knowledge", "user" or "assistant"
              - content: Text content
        tokenizer: Tokenizer instance for text processing
        max_src_len: Maximum allowed length for source tokens

    Returns:
        List of processed tokens ready for model input
    """
    if not tokenizer.chat_template:
        tokenizer.init_chat_template(NONE_CHAT_TEMPLATE)
    encoded_messages = tokenizer.encode_chat_inputs({"messages": dial})

    num_reserved_tokens_for_each_dialog = 1  # only break_turn_token or end_token
    num_reserved_tokens_for_each_turn = 8

    cur_len = num_reserved_tokens_for_each_dialog

    turn_index = len(encoded_messages) - 1

    tokens = []
    tokens = encoded_messages[turn_index][0]
    turn_index -= 1

    while turn_index >= 0:
        tokens_src, tokens_target = encoded_messages[turn_index]
        if len(tokens_src) + len(tokens_target) > (max_src_len + 1 - cur_len - num_reserved_tokens_for_each_turn):
            break

        tokens = tokens_src + tokens_target + tokens
        cur_len = len(tokens)
        turn_index -= 1

    return tokens


def convert_to_input_ids(
    dials: List[List[dict]],
    tokenizer,
    data_format,
    max_src_len,
) -> Tuple[List[List[int]], int]:
    """Convert batch dialogue into input_ids.

    The API support multiple data format: `pt`, `sft.

    Args:
        dials (List[List[dict]]): A batch of dialogue.
        tokenizer (Ernie4_5_Tokenizer): The used tokenizer.
        data_format (str): The data format for converting dialogue to input_ids,
            support `base`, `chat`.
        max_src_len (int): The maximum length of input_ids.

    Returns:
        input_ids (List[List[int]]): The raw input_ids with truncation, but without padding.
        num_input_tokens (int): The total input tokens in a batch.

    Raises:
        ValueError: Invalid data format.
    """
    input_ids = []
    num_input_tokens = 0
    for dial in dials:
        if data_format == "base":
            tokens = convert_to_tokens_for_pt(dial, tokenizer, max_src_len)
            input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        elif data_format == "chat":
            input_ids.append(convert_to_tokens_for_sft(dial, tokenizer, max_src_len))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        num_input_tokens += len(input_ids[-1])
    return input_ids, num_input_tokens


def function_call_chat_template(tokenizer, messages, tools):
    history = messages[:-1]
    input_dict = dict()
    input_dict["messages"] = history
    if tools is not None:
        input_dict["tools"] = tools
    history_str = tokenizer.apply_chat_template(
        input_dict,
        add_generation_prompt=True,
        tokenize=False,
    )
    history_len = len(history_str)
    input_dict["messages"] = messages
    all_str = tokenizer.apply_chat_template(
        input_dict,
        add_generation_prompt=False,
        tokenize=False,
    )
    # (21b think model) remove generation content
    s = "<|im_end|>\n\n<|im_start|>assistant\n<think>\n"
    if all_str.endswith(s):
        all_str = all_str[: -len(s)]
    response_str = all_str[history_len:]
    history_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(history_str))
    response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_str))
    return [history_id, response_id]


def postprocess_fc_sequence(tokenizer, example):
    messages = example["messages"]
    tools = example["tools"]
    encoded_messages = [function_call_chat_template(tokenizer, messages, tools)]
    return encoded_messages


def estimate_training(train_dataset, data_args, training_args, model_args):
    """
    Estimate required training steps based on dataset.

    Args:
        train_dataset: Dataset used for training estimation.
        data_args: Configuration object containing data parameters.
        training_args: Configuration object containing training parameters.
        model_args: Configuration object containing model parameters.

    Returns:
        dict: Contains estimated training steps and related parameters.
    """
    train_dataset.estimate = True
    logger.info("Start to estimate max training steps...")

    train_dataset_path_list = [path for path in str(data_args.train_dataset_path).replace(" ", "").split(",")]
    if len(train_dataset_path_list) > 1:
        logger.warning("Suggest to use max_steps instead of num_train_epochs for multi source dataset.")
        logger.info(
            "Multi source dataset detected, number of samples will be estimated by following rule. "
            "num_samples = (source1_num_samples * prob1 + source2_num_samples * prob2 + ...) * epochs"
        )

    max_samples = train_dataset.max_estimate_samples

    if training_args.max_estimate_samples != -1:
        # Set estimate samples to max_estimate_samples
        logger.warning("The results between sampling and non-sampling methods may differ.")
        train_dataset.max_estimate_samples = min(
            training_args.max_estimate_samples, train_dataset.max_estimate_samples
        )

    if train_dataset.max_estimate_samples > 0:
        train_batches = 0
        train_tokens = 0
        for sequences in train_dataset:
            if not train_dataset.estimate:
                break
            train_batches += 1
            for sequence in sequences:
                train_tokens += len(sequence.token_ids)

        train_tokens *= training_args.num_train_epochs
        train_batches *= training_args.num_train_epochs
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * max(training_args.data_parallel_degree, 1)
            * max(training_args.sharding_parallel_degree, 1)
        )
        max_steps = train_batches / global_batch_size

        if max_samples != train_dataset.max_estimate_samples:
            max_steps *= max_samples / train_dataset.max_estimate_samples
            train_tokens *= max_samples / train_dataset.max_estimate_samples
            train_dataset.used_samples *= max_samples / train_dataset.max_estimate_samples
            train_dataset.unused_samples *= max_samples / train_dataset.max_estimate_samples

        max_steps = int(np.ceil(max_steps))

        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": max_steps,
            "train_tokens": int(train_tokens),
            "global_batch_size": int(global_batch_size),
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "warmup_steps": int(np.ceil(0.1 * max_steps)),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree": int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(training_args.sharding_parallel_degree),
            "seed": training_args.seed,
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "max_seq_len": int(training_args.max_seq_len),
            "valid": True,
            "train_samples": int(max_samples * training_args.num_train_epochs),
            "estimate_samples": int(train_dataset.max_estimate_samples),
            "actual_train_samples": int(train_dataset.used_samples * training_args.num_train_epochs),
            "skip_samples": int(train_dataset.unused_samples * training_args.num_train_epochs),
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if train_batches / training_args.num_train_epochs / global_batch_size < 1:
            logger.warning("This dataset is too small, you'd better enlarge your dataset.")
            res["valid"] = False

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

        return max_steps
    else:
        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": 0,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "train_tokens": 0,
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "tensor_parallel_degree": int(training_args.tensor_parallel_degree),
            "pipeline_parallel_degree": int(training_args.pipeline_parallel_degree),
            "sharding_parallel_degree": int(training_args.sharding_parallel_degree),
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "max_seq_len": int(data_args.max_seq_len),
            "seed": data_args.seed,
            "valid": False,
            "train_samples": 0,
        }
        if hasattr(training_args, "num_of_gpus"):
            res["num_of_gpus"] = training_args.num_of_gpus

        if getattr(training_args, "estimation_output_file", None):
            with open(training_args.estimation_output_file, "w", encoding="utf-8") as f:
                json.dump(res, f)

        logger.error("No valid data found, please check your dataset format.")
        return 0
