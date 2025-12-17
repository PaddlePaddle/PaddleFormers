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

from typing import List

import numpy as np
from scipy.linalg import block_diag

from .SFTDataset import Sequence


def dpo_collate_fn(
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
        max_seq_len = max(len(item.token_ids) for sequence in batch for item in sequence)

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
        difference = max_seq_len - sum([len(sequence.token_ids) for sequence in sequences])

        input_dict["input_ids"].append(sum([sequence.token_ids for sequence in sequences], []) + [0] * difference)
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
                sequence_sum += len(sequence.token_ids)
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
                sequence_sum_flatten += len(sequence.token_ids)
            else:
                response_index = [
                    i,
                    sequence.response_index[0] + sequence_sum,
                    sequence.response_index[1] + sequence_sum,
                    sequence.response_index[2] + sequence_sum,
                ]
                sequence_sum += len(sequence.token_ids)
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
    """
    input_keys = ["input_ids", "labels", "position_ids"]
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
        labels = [sum([seq.labels for seq in batch_sequence], [])]
        position_ids = [sum([seq.position_ids for seq in batch_sequence], [])]
        # padding
        padded_token_ids = pad_batch_data(token_ids, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        padded_labels = pad_batch_data(labels, pad_idx=-100, max_seq_len=max_seq_len)
        padded_position_ids = pad_batch_data(position_ids, pad_idx=0, max_seq_len=max_seq_len)
        return_list.append(
            [
                padded_token_ids,
                padded_labels,
                padded_position_ids,
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

        if model_args.use_attn_mask_startend_row_indices:
            return_list[-1].append(
                gen_attn_mask_startend_row_indices(original_token_ids, max_seq_len, model_args.use_global_causal_attn)
            )
        else:
            return_list[-1].append(
                gen_self_attn_mask(original_token_ids, max_seq_len, model_args.use_global_causal_attn)
            )

    return_list = [np.concatenate(tensor_list) for tensor_list in zip(*return_list)]
    input_dict = dict(zip(input_keys, return_list))
    return input_dict


def mm_collate_fn(
    batch: List[List[Sequence]], template, processor, tokenizer, training_args, model_args, max_seq_len: int
):
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

    batch_images, batch_videos, batch_audios = [], [], []
    batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []
    for batch_sequence in batch:
        images = []
        for seq in batch_sequence:
            images.extend(seq.images)
        videos = []
        for seq in batch_sequence:
            videos.extend(seq.videos)
        audios = []
        for seq in batch_sequence:
            audios.extend(seq.audios)
        batch_images.extend(images)
        batch_videos.extend(videos)
        batch_audios.extend(audios)
        batch_imglens.append(len(images))
        batch_vidlens.append(len(videos))
        batch_audlens.append(len(audios))
        batch_input_ids.append([seq.token_ids for seq in batch_sequence])

    mm_inputs = template.mm_plugin.get_mm_inputs(
        batch_images,
        batch_videos,
        batch_audios,
        batch_imglens,
        batch_vidlens,
        batch_audlens,
        batch_input_ids,
        processor,
    )

    input_keys = ["input_ids", "labels"]
    if training_args.num_nextn_predict_layers > 0:
        input_keys.append("nbatch_pack_offset")
    if model_args.use_attn_mask_startend_row_indices:
        input_keys.append("attn_mask_startend_row_indices")
    else:
        input_keys.append("attention_mask")

    input_keys.append("pixel_values")
    input_keys.append("image_grid_thw")
    return_list = []
    if max_seq_len is None:
        max_seq_len = max(len(item.token_ids) for sequence in batch for item in sequence)
    for batch_sequence in batch:
        original_token_ids = [seq.token_ids for seq in batch_sequence]
        token_ids = [sum(original_token_ids, [])]
        labels = [sum([seq.labels for seq in batch_sequence], [])]
        # padding
        padded_token_ids = pad_batch_data(token_ids, pad_idx=tokenizer.pad_token_id, max_seq_len=max_seq_len)
        padded_labels = pad_batch_data(labels, pad_idx=-100, max_seq_len=max_seq_len)
        return_list.append(
            [
                padded_token_ids,
                padded_labels,
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
                return_list[-1].append(
                    gen_attn_mask_startend_row_indices(
                        original_token_ids, max_seq_len, model_args.use_global_causal_attn
                    )
                )
            else:
                return_list[-1].append(
                    gen_self_attn_mask(original_token_ids, max_seq_len, model_args.use_global_causal_attn)
                )

        if "pixel_values" in mm_inputs:
            return_list[-1].append(mm_inputs["pixel_values"])
        if "image_grid_thw" in mm_inputs:
            return_list[-1].append(mm_inputs["image_grid_thw"])

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


def gen_self_attn_mask(batch_token_ids: List[List[int]], max_seq_len: int, use_global_causal_attn: bool):
    """Generate self-attention mask for multi-sequence batches.

    Args:
        batch_token_ids (List[List[int]]): List of token ID sequences.
        max_seq_len (int): Maximum sequence length.

    Returns:
        ndarray: 4D attention mask array.
    """
    input_mask_data = np.zeros((1, 1, max_seq_len, max_seq_len), dtype="float32")
    offset = 0
    if use_global_causal_attn:
        total_len = 0
        for index, token_ids in enumerate(batch_token_ids):
            total_len += len(token_ids)
        b = np.tril(np.ones([total_len, total_len]), 0)
        input_mask_data[0, 0, offset : offset + total_len, offset : offset + total_len] = b
    else:
        for index, token_ids in enumerate(batch_token_ids):
            cur_len = len(token_ids)
            b = np.tril(np.ones([cur_len, cur_len]), 0)
            input_mask_data[0, 0, offset : offset + cur_len, offset : offset + cur_len] = b
            offset += cur_len
    return input_mask_data


def gen_attn_mask_startend_row_indices(
    batch_token_ids: List[List[int]], max_seq_len: int, use_global_causal_attn: bool
):
    """Generate row indices for flash attention masks.

    Args:
        batch_token_ids (List[List[int]]): List of token ID sequences.
        max_seq_len (int): Maximum sequence length.

    Returns:
        ndarray: Row indices array with dtype int32.
    """
    offset = 0
    attn_mask_startend_row_indices = []
    if use_global_causal_attn:
        total_len = 0
        for token_ids in batch_token_ids:
            total_len += len(token_ids)
        attn_mask_startend_row_indices.extend([offset + total_len] * total_len)
        offset += total_len
        if offset < max_seq_len:
            attn_mask_startend_row_indices.extend(list(range(offset, max_seq_len)))
    else:
        for token_ids in batch_token_ids:
            cur_len = len(token_ids)
            attn_mask_startend_row_indices.extend([offset + cur_len] * cur_len)
            offset += cur_len
        if offset < max_seq_len:
            attn_mask_startend_row_indices.extend(list(range(offset, max_seq_len)))
    # NOTE(hehuang): The dtype of attn_mask_startend_row_indices must be np.int32
    return np.array(attn_mask_startend_row_indices, dtype=np.int32)[None, None, ..., None]  # add dimension modify
