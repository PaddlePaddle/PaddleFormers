# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle


def check_preference_data(data):
    if isinstance(data["src"], str):
        data["src"] = [data["src"]]
    if isinstance(data["tgt"], str):
        data["tgt"] = [data["tgt"]]
    if len(data["src"]) != len(data["tgt"]) + 1:
        raise ValueError(
            "The number of src and tgt should differ by 1, but got {} and {}".format(
                len(data["src"]), len(data["tgt"])
            )
        )
    if (len(data["response"]) != 2) or (len(data["response"]) != len(data["sort"])):
        raise ValueError(
            "The number of response and sort should be 2, but got {} and {}".format(
                len(data["response"]), len(data["sort"])
            )
        )
    if len(data["response"][0]) == 0 or len(data["response"][1]) == 0:
        raise ValueError(f"The response should not be empty, but got {data}.")
    if data["sort"][0] == data["sort"][1]:
        raise ValueError("The two sort should be different, but got {data}.")
    # [chosen, rejected]
    if data["sort"][1] > data["sort"][0]:
        data["response"] = [data["response"][1], data["response"][0]]
        data["sort"] = [data["sort"][1], data["sort"][0]]

    return data


def preprocess_preference_data(data, tokenizer, data_args, model_args):
    """Convert raw format example to Example."""
    # 1. Check data format
    data = check_preference_data(data)

    response_0 = data["response"][0]
    response_1 = data["response"][1]
    response_0_encode_tokens = []
    for idx in range(len(data["src"])):
        if idx < len(data["tgt"]):
            if tokenizer.chat_template is not None:
                response_0_encode_tokens.append(
                    [
                        data["src"][idx].strip(),
                        data["tgt"][idx].strip(),
                    ]
                )
            else:
                response_0_encode_tokens.append(
                    [
                        tokenizer.encode(data["src"][idx].strip(), add_special_tokens=True)["input_ids"],
                        tokenizer.encode(data["tgt"][idx].strip(), add_special_tokens=False)["input_ids"]
                        + [tokenizer.eos_token_id],
                    ]
                )
        else:
            if tokenizer.chat_template is not None:
                response_0_encode_tokens.append(
                    [
                        data["src"][idx].strip(),
                        response_0.strip(),
                    ]
                )
            else:
                response_0_encode_tokens.append(
                    [
                        tokenizer.encode(data["src"][idx].strip(), add_special_tokens=True)["input_ids"],
                        tokenizer.encode(response_0.strip(), add_special_tokens=False)["input_ids"]
                        + [tokenizer.eos_token_id],
                    ]
                )
    if tokenizer.chat_template is not None:
        chat_input_list = response_0_encode_tokens
        response_0_encode_tokens = tokenizer.encode_chat_inputs(chat_input_list)["conversations"]
        # convert to response_1 response_0_encode_tokens
        chat_input_list[-1][-1] = response_1.strip()
        response_1_encode_tokens = tokenizer.encode_chat_inputs(chat_input_list)["conversations"]

        """Post process sequence: tokenization & truncation."""
        tokens_prompt = response_0_encode_tokens[-1][0][:-1]
        eos_token_id = response_0_encode_tokens[-1][-1][-1]
        tokens_response_0 = response_0_encode_tokens[-1][0][-1:] + response_0_encode_tokens[-1][-1][:-1]
        tokens_response_1 = response_0_encode_tokens[-1][0][-1:] + response_1_encode_tokens[-1][-1][:-1]
    else:
        eos_token_id = tokenizer.eos_token_id
        tokens_prompt = response_0_encode_tokens[-1][0][:-1]
        tokens_response_0 = (
            response_0_encode_tokens[-1][0][-1:]
            + tokenizer.encode(response_0.strip(), add_special_tokens=False)["input_ids"]
        )
        tokens_response_1 = (
            response_0_encode_tokens[-1][0][-1:]
            + tokenizer.encode(response_1.strip(), add_special_tokens=False)["input_ids"]
        )

    if len(tokens_prompt) + len(tokens_response_0) + len(tokens_response_1) > data_args.max_seq_len:
        # truncate prompt
        tokens_prompt = tokens_prompt[-data_args.max_prompt_len :]
        if (len(tokens_prompt) + len(tokens_response_0) + len(tokens_response_1)) > data_args.max_seq_len:
            max_response_len = data_args.max_seq_len - len(tokens_prompt)
            # 按比例截断
            max_response_0_len = int(
                len(tokens_response_0) / (len(tokens_response_0) + len(tokens_response_1)) * max_response_len
            )
            max_response_1_len = max_response_len - max_response_0_len
            tokens_response_0 = tokens_response_0[:max_response_0_len]
            tokens_response_1 = tokens_response_1[:max_response_1_len]

    cur_len = len(tokens_prompt) + len(tokens_response_0) + len(tokens_response_1)
    turn_index = len(response_0_encode_tokens) - 2

    # append former dialog contents
    while turn_index >= 0:
        tokens_src = response_0_encode_tokens[turn_index][0]
        tokens_target = response_0_encode_tokens[turn_index][1]
        turn_index -= 1

        if len(tokens_src) + len(tokens_target) > data_args.max_seq_len - cur_len:
            break
        tokens_prompt = tokens_src + tokens_target + tokens_prompt
        cur_len += len(tokens_src) + len(tokens_target)

    input_ids = tokens_prompt + tokens_response_0 + tokens_response_1
    prompt_len = len(tokens_prompt)
    response_0_len = len(tokens_response_0)
    response_1_len = len(tokens_response_1)
    seq_len = len(input_ids)
    # make position ids & labels

    position_ids = (
        list(range(prompt_len))  # prompt
        + list(range(prompt_len, prompt_len + response_0_len))  # response_0
        + list(range(prompt_len, prompt_len + response_1_len))  # response_1
    )
    response_0_labels = [0] * prompt_len + tokens_response_0[1:] + [eos_token_id] + [0] * response_1_len
    response_1_labels = [0] * prompt_len + [0] * response_0_len + tokens_response_1[1:] + [eos_token_id]

    # response index with sort
    response_indexs = [prompt_len, prompt_len + response_0_len, seq_len, data["sort"][0]]
    output_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "response_0_labels": response_0_labels,
        "response_1_labels": response_1_labels,
        "response_indexs": response_indexs,
    }

    # attention mask
    if model_args.flash_mask:
        output_dict["attn_mask_startend_row_indices"] = (
            [seq_len] * prompt_len + [prompt_len + response_0_len] * response_0_len + [seq_len] * response_1_len
        )
    else:
        attention_mask = np.tri(seq_len, seq_len, dtype=bool)
        attention_mask[(prompt_len + response_0_len) :, prompt_len : (prompt_len + response_0_len)] = False
        output_dict["attention_mask"] = attention_mask
    return output_dict


def preference_collate_fn(batch, max_seq_len=None, data_type="pairwise"):
    """Convert batch data into tensor."""
    if max_seq_len is None:
        raise ValueError("max_seq_len is None.")
    input_dict = {
        "input_ids": [],
        "position_ids": [],
        "response_0_labels": [],
        "response_1_labels": [],
        "response_indexs": [],
        "reference_chosen_logps": paddle.to_tensor([0], dtype="float32"),
        "reference_rejected_logps": paddle.to_tensor([0], dtype="float32"),
    }
    sequence = batch[0]
    if "attn_mask_startend_row_indices" in sequence:
        input_dict["attn_mask_startend_row_indices"] = []
        use_attn_mask_startend_row_indices = True
    elif "attention_mask" in sequence:
        input_dict["attention_mask"] = []
        use_attn_mask_startend_row_indices = False
    else:
        raise ValueError("attention_mask and attn_mask_startend_row_indices are both None.")

    for i, sequence in enumerate(batch):
        difference = max_seq_len - len(sequence["input_ids"])

        input_dict["input_ids"].append(sequence["input_ids"] + [0] * difference)
        input_dict["position_ids"].append(sequence["position_ids"] + [0] * difference)
        input_dict["response_0_labels"].append(sequence["response_0_labels"] + [0] * difference)
        input_dict["response_1_labels"].append(sequence["response_1_labels"] + [0] * difference)
        if use_attn_mask_startend_row_indices:
            input_dict["attn_mask_startend_row_indices"].append(
                [
                    sequence["attn_mask_startend_row_indices"]
                    + [sequence["attn_mask_startend_row_indices"][-1]] * difference
                ]
            )
        else:
            input_dict["attention_mask"].append(
                np.pad(
                    sequence["attention_mask"],
                    pad_width=((0, 0), (0, difference), (0, difference)),
                    mode="constant",
                    constant_values=False,
                )
            )
        for ri in sequence["response_indexs"]:
            if data_type == "pairwise":
                input_dict["response_indexs"].append([i] + ri[:-1])
            else:
                input_dict["response_indexs"].append([i] + ri)
    if data_type == "pairwise":
        input_dict["chosen_labels"] = input_dict.pop("response_0_labels")
        input_dict["rejected_labels"] = input_dict.pop("response_1_labels")
    else:
        input_dict["response_labels"] = input_dict.pop("response_0_labels")
        input_dict["response_kl_labels"] = input_dict.pop("response_1_labels")

    for key in input_dict:
        if key == "attention_mask":
            input_dict[key] = np.array(input_dict[key], dtype=bool)
        elif key == "attn_mask_startend_row_indices":
            input_dict[key] = np.array(input_dict[key], dtype=np.int32)
        else:
            input_dict[key] = np.array(input_dict[key])
    return input_dict


def preference_collate_fn_auto_parallel(batch, max_seq_len=None, data_type="pairwise", enable_auto_parallel=False):
    input_dict = preference_collate_fn(batch, max_seq_len, data_type)
    if "attn_mask_startend_row_indices" in input_dict:
        input_dict["attention_mask"] = np.array(paddle.to_tensor([0], dtype="float32"))
    result = {
        "input_ids": [
            input_dict["input_ids"],
            input_dict["position_ids"],
            input_dict["response_indexs"],
            input_dict["attention_mask"],
            input_dict["chosen_labels"],
            input_dict["rejected_labels"],
        ],
        "labels": [
            input_dict["chosen_labels"],
            input_dict["rejected_labels"],
            input_dict["response_indexs"],
            input_dict["reference_chosen_logps"],
            input_dict["reference_rejected_logps"],
        ],
    }
    if "attn_mask_startend_row_indices" in input_dict:
        result["input_ids"].append(input_dict["attn_mask_startend_row_indices"])
    return result
