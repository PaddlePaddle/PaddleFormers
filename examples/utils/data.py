# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import random

import numpy as np

from paddleformers.peft import LoRAModel, PrefixModelForCausalLM


def convert_multi_rounds_to_single_round(example, tokenizer):
    # 1. convert multi-rounds to single-round data format with chat_template
    example["src"] = example["src"] if isinstance(example["src"], list) else [example["src"]]
    example["tgt"] = example["tgt"] if isinstance(example["tgt"], list) else [example["tgt"]]

    src = tokenizer.chat_template.render_system()
    conversations = list(zip(example["src"], example["tgt"]))

    for index, conversation in enumerate(conversations[:-1]):
        src += "".join(tokenizer.chat_template.render_conversation(conversation, index=index))

    last_user, last_bot = tokenizer.chat_template.render_conversation(conversations[-1], index=len(conversations) - 1)

    example["src"] = [src + last_user]
    example["tgt"] = [last_bot]
    return example


def get_convert_example(model):
    if isinstance(model, LoRAModel) or isinstance(model, PrefixModelForCausalLM):
        base_model_prefix = model.model.base_model_prefix
    else:
        base_model_prefix = model.base_model_prefix

    if base_model_prefix == "chatglm":
        return convert_example_chatglm
    elif base_model_prefix in [
        "chatglm_v2",
        "llama",
        "bloom",
        "opt",
        "qwen",
        "mixtral",
        "mistral",
        "gemma",
        "qwen2",
        "qwen2_moe",
        "gpt",
        "yuan",
        "jamba",
        "deepseek_v2",
        "deepseek_v3",
        "model",
    ]:
        return convert_example_common
    else:
        raise ValueError(
            f"Unknown base_model_prefix: {model.base_model_prefix}. Supported base_model_prefix list: chatglm, bloom, llama, qwen, mixtral, gemma, qwen2, qwen2_moe, yuan, jamba,deepseek_v2, deepseek_v3",
        )


class DataFormatError(ValueError):
    pass


def tokenize_unsupervised_example(tokenizer, example, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if "src" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
    else:
        raise DataFormatError(
            f"Example format is wrong, please check: {example} or rewrite tokenize_example in data.py "
        )
    tokenized_source = tokenizer(
        source,
        truncation=False,
        padding=True,
        max_length=data_args.src_length,
        add_special_tokens=True,
    )

    if data_args.use_pose_convert:
        tokenized_source = get_example_pose(tokenized_source, tokenizer, data_args)

    return tokenized_source


def tokenize_example(tokenizer, example, data_args):
    if "src" in example and "tgt" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
        target = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    else:
        raise DataFormatError(
            f"Example format is wrong, please check: {example} or rewrite tokenize_example in data.py "
        )
    tokenized_source = tokenizer(
        source,
        max_length=data_args.src_length,
        truncation=True,
        add_special_tokens=True,
    )

    tgt_max_length = data_args.max_length - len(tokenized_source["input_ids"])
    tokenized_target = tokenizer(
        target,
        max_length=tgt_max_length,
        truncation=True,
        add_special_tokens=False,
    )

    tokenized_target_input_ids = tokenized_target["input_ids"]
    # Add eos_token_id at the end of sequence if the sentence is not truncated.
    # Attention! In some cases(ex. ChatGLMv2), tokenized eos_token is not equal to eos_token_id.
    if len(tokenized_target_input_ids) < tgt_max_length:
        tokenized_target_input_ids += [tokenizer.eos_token_id]

    return tokenized_source, tokenized_target_input_ids


def tokenize_rounds_example(tokenizer, example, data_args, **kwargs):
    """tokenize multi-rounds examples with chat_template.json

    Args:
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        example (dict[str, str | list[str]]):
                the example instance, which can be: {"src": "src-sentence", "tgt": "tgt-sentence"}
                or {"src": ["src-sentence-1", ..., "src-sentence-N"], "tgt": ["tgt-sentence-1", ..., "tgt-sentence-N"]}
        data_args (DataArgument): the data_argument instance of data processing

    Returns:
        dict[str, list[int]]: return input_ids and labels fields
    """

    # 0. prepare data
    context_data = example.get("context", {})
    context_data["is_training"] = True

    example["src"] = example["src"] if isinstance(example["src"], list) else [example["src"]]
    example["tgt"] = example["tgt"] if isinstance(example["tgt"], list) else [example["tgt"]]

    assert len(example["src"]) == len(example["tgt"]), "the length of `src` and `tgt` field must be same."

    conversations = [[src, tgt] for src, tgt in zip(example["src"], example["tgt"])]

    # 1. only tokenize input_ids
    conversation_result: list[tuple[list[int], list[int]]] = tokenizer.encode_chat_inputs(
        conversations, context_data=context_data, **kwargs
    )
    system_ids = conversation_result.pop("system", []) or []

    # 2. truncate conversations based on conversation unit
    input_ids, labels = [], []
    conversations_ids = conversation_result.pop("conversations")

    assert (
        len(system_ids) < data_args.max_length
    ), f"the length of system_ids<{len(system_ids)}> should be smaller than max_length<{data_args.max_length}>."
    max_length = data_args.max_length - len(system_ids)

    should_break = False
    for index in range(len(conversations_ids) - 1, -1, -1):
        user_input_ids, bot_input_ids = conversations_ids[index][0], conversations_ids[index][1]

        # break when the length of current conversations is greater than max_length
        if len(input_ids) + len(user_input_ids) + len(bot_input_ids) > max_length:

            # when the length of last conversation is lager than max_length, we should not break: at least one round
            if index < len(conversations_ids) - 1:
                break

            user_input_ids = user_input_ids[: data_args.src_length - len(system_ids)]
            bot_input_ids = bot_input_ids[: max_length - len(user_input_ids)]

            should_break = True

        input_ids = user_input_ids + bot_input_ids + input_ids
        labels = len(user_input_ids) * [-100] + bot_input_ids + labels

        if should_break:
            break

    input_ids = system_ids + input_ids
    labels = [-100] * len(system_ids) + labels
    tokenized_source = {"input_ids": input_ids}
    sequence_length = len(input_ids)

    if "position_ids" in tokenizer.model_input_names:
        tokenized_source["position_ids"] = list(range(sequence_length))

    return tokenized_source, labels


def convert_example_common(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if data_args.autoregressive:
        tokenized_source = tokenize_unsupervised_example(
            tokenizer, example, data_args, is_test=True, zero_padding=False, flash_mask=False
        )
        input_ids = tokenized_source["input_ids"]
        if "labels" in tokenized_source:
            labels = tokenized_source["labels"]
        else:
            labels = input_ids
            input_ids = input_ids[:-1] + [tokenizer.eos_token_id]
            labels = labels[1:] + [-100]
        features = {"input_ids": input_ids, "labels": labels}
        if "position_ids" in tokenized_source:
            features["position_ids"] = tokenized_source["position_ids"]
    else:
        if tokenizer.chat_template is not None:
            return convert_rounds_example_common(example, tokenizer, data_args, is_test, zero_padding, flash_mask)
        else:
            tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)

            if is_test:
                return {
                    **tokenized_source,
                    "labels": tokenized_target_input_ids,
                }
            else:
                input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
                source_length = len(tokenized_source["input_ids"])
                labels = [-100] * source_length + input_ids[source_length:]
                # shift input_ids and labels
                input_ids, labels = input_ids[:-1], labels[1:]
                seq_length = len(input_ids)
                features = {"input_ids": input_ids, "labels": labels}
                if "position_ids" in tokenized_source:
                    features["position_ids"] = list(range(seq_length))
    # maybe change here to suit flash_mask with longlora
    if zero_padding:
        if flash_mask:
            features["attn_mask_startend_row_indices"] = [seq_length] * seq_length
        else:
            features["attention_mask"] = np.tri(seq_length, seq_length, dtype=bool)
    return features


def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


# layers * intervention tokens
def get_intervention_locations(positions, last_position, num_interventions):
    """
    This function generates the intervention locations.
    """
    _first_n, _last_n = parse_positions(positions)

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1

    position_list = (
        [i for i in range(first_n)]
        + [i for i in range(last_position - last_n, last_position)]
        + [pad_position for _ in range(pad_amount)]
    )
    intervention_locations = [position_list] * num_interventions

    return intervention_locations


def get_src_last_position(labels):
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] == -100:
            return i + 2


# reft
def convert_example_for_reft(
    example,
    tokenizer,
    data_args,
    is_test=True,
    zero_padding=False,
    flash_mask=False,
    positions="f7+l7",
    num_interventions=32,
):
    features = convert_example_common(example, tokenizer, data_args, is_test, zero_padding, flash_mask)
    # src的最后一个位置
    if not is_test:
        last_position = get_src_last_position(features["labels"])
    else:
        last_position = len(features["input_ids"])
    # add positions
    intervention_locations = get_intervention_locations(positions, last_position, num_interventions)
    features["intervention_locations"] = intervention_locations
    return features


def convert_rounds_example_common(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    """convert multi-rounds conversation example

    Args:
        example (dict): the source of example
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        data_args (DataArgument): data argument for data preprocessing
        is_test (bool, optional): whether is testing stage. Defaults to True.
        zero_padding (bool, optional): whether use in_tokens. Defaults to False.

    Returns:
        dict[str, np.ndarray]: the features of example
    """
    rounds_inputs, labels = tokenize_rounds_example(tokenizer, example, data_args)

    if is_test:
        return {
            **rounds_inputs,
            "labels": labels,
        }

    input_ids = rounds_inputs.pop("input_ids")
    # shift input_ids and labels
    input_ids, labels = input_ids[:-1], labels[1:]

    seq_length = len(input_ids)
    features = {"input_ids": input_ids, "labels": labels}
    if zero_padding:
        if flash_mask:
            features["attn_mask_startend_row_indices"] = [seq_length] * seq_length
        else:
            features["attention_mask"] = np.tri(seq_length, seq_length, dtype=bool)

    if "position_ids" in rounds_inputs:
        rounds_inputs["position_ids"] = rounds_inputs["position_ids"][:-1]

    rounds_inputs.update(features)
    return rounds_inputs


def convert_example_chatglm(example, tokenizer, data_args, is_test=True, zero_padding=False, flash_mask=False):
    if flash_mask:
        raise ValueError("chatglm does not support flash mask for now!")
    if tokenizer.chat_template is not None:
        # chatglm only support single-round finetune
        example = convert_multi_rounds_to_single_round(example, tokenizer)

    tokenized_source, tokenized_target_input_ids = tokenize_example(tokenizer, example, data_args)

    if is_test:
        return {
            **tokenized_source,
            "labels": tokenized_target_input_ids,
        }
    else:
        input_ids = tokenized_source["input_ids"] + tokenized_target_input_ids
        bos_position = len(tokenized_source["input_ids"]) - 1
        labels = [-100] * bos_position + input_ids[bos_position:]
        # shift input_ids and labels
        input_ids, labels = input_ids[:-1], labels[1:]
        features = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if zero_padding:
            seq_length = len(input_ids)
            # attention_mask
            attention_mask = np.tri(seq_length, seq_length, dtype=bool)
            attention_mask[:, :bos_position] = 1
            features["attention_mask"] = attention_mask
            # 2d position_ids
            position_ids = np.arange(seq_length, dtype=np.int64)
            position_ids[:bos_position] = bos_position - 1
            block_position_ids = np.concatenate(
                [
                    np.zeros(bos_position, dtype=np.int64),
                    np.arange(1, seq_length - bos_position + 1, dtype=np.int64),
                ]
            )
            features["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)

        return features


def get_example_pose(tokenized_source, tokenizer, data_args):

    ids = tokenized_source["input_ids"]
    len_chunk = min(len(ids), data_args.max_length)
    if len(tokenized_source["input_ids"]) <= data_args.max_length:
        tokenized_source["input_ids"] += [tokenizer.eos_token_id]

    len_input = len(ids)

    lt1 = 0  # chunk1 start pos
    rt1 = random.randint(1, (len_chunk) // 2)  # chunk1 end pos

    rt2 = random.randint(lt1 + len_chunk, len_input - 1)  # chunk2 end pos
    lt2 = rt2 - (len_chunk - (rt1 - lt1))  # chunk2 start pos
    chunked_ids = ids[lt1:rt1] + ids[lt2:rt2]
    labels = ids[lt1 + 1 : rt1 + 1] + ids[lt2 + 1 : rt2 + 1]

    pos_ids = range(len(chunked_ids))
    pos_ids = [x + lt1 if i < rt1 - lt1 else x + (lt2 - (rt1 - lt1)) for i, x in enumerate(pos_ids)]

    features = {"input_ids": chunked_ids, "labels": labels, "position_ids": pos_ids}

    return features
