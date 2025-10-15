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

from typing import Dict, List

import numpy as np


def collate_fn(batch: List[List[Dict]], tokenizer, model_args, max_seq_len: int):
    """Convert batch of Dict into training tensors.

    Args:
        batch (List[List[Dict]]): Batch of input dicts
        tokenizer: Tokenizer for text conversion
        model_args: Model configuration parameters
        max_seq_len (int): Maximum sequence length for padding

    Returns:
        dict: Dictionary containing:
            - input_ids: Padded token IDs
            - labels: Shifted labels for prediction
    """
    input_keys = ["input_ids", "labels"]
    if model_args.num_nextn_predict_layers > 0:
        input_keys.append("nbatch_pack_offset")
    if model_args.use_attn_mask_startend_row_indices:
        input_keys.append("attn_mask_startend_row_indices")
    else:
        input_keys.append("attention_mask")
    return_list = []

    if max_seq_len is None:
        max_seq_len = max(len(item["input_ids"]) for sequence in batch for item in sequence)
    for batch_sequence in batch:
        original_token_ids = [seq["input_ids"] for seq in batch_sequence]
        token_ids = [sum(original_token_ids, [])]
        labels = [sum([seq["labels"] for seq in batch_sequence], [])]
        # padding
        padded_token_ids = pad_batch_data(token_ids, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        padded_labels = pad_batch_data(labels, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        return_list.append(
            [
                padded_token_ids,
                padded_labels,
            ]
        )

        if model_args.num_nextn_predict_layers > 0:
            # each sequence end index
            batch_sequence_len = [len(sequence) for sequence in original_token_ids]
            nbatch_pack_offset = [0] * sum(batch_sequence_len)
            prefix_sum = 0
            for sequence_len in batch_sequence_len[:-1]:
                prefix_sum += sequence_len
                nbatch_pack_offset[prefix_sum - 1] = 1
            padded_nbatch_pack_offset = pad_batch_data([nbatch_pack_offset], pad_idx=0, max_seq_len=max_seq_len)
            return_list[-1].append(padded_nbatch_pack_offset)

        if model_args.use_attn_mask_startend_row_indices:
            return_list[-1].append(gen_attn_mask_startend_row_indices(original_token_ids, max_seq_len))
        else:
            return_list[-1].append(gen_self_attn_mask(original_token_ids, max_seq_len))

    return_list = [np.concatenate(tensor_list) for tensor_list in zip(*return_list)]
    input_dict = dict(zip(input_keys, return_list))
    return input_dict


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
