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
