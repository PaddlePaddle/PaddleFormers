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
from __future__ import annotations

import glob
import math
import os
import shutil
import struct
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.incubate.multiprocessing as mp
from paddle.distributed import fleet
from sklearn.metrics import accuracy_score

from ..datasets import ZeroPaddingIterableDataset
from ..generation import GenerationConfig
from ..trainer import TrainerCallback
from ..trainer.trainer_utils import IterableDatasetShard
from ..transformers import (  # ChatGLMv2Tokenizer,
    AutoTokenizer,
    DeepseekV2ForCausalLMPipe,
    DeepseekV3ForCausalLMPipe,
    LlamaForCausalLMPipe,
    PretrainedConfig,
    Qwen2ForCausalLMPipe,
    Qwen2MoeForCausalLMPipe,
)
from ..transformers.tokenizer_utils import PretrainedTokenizer
from ..utils.log import logger


def compute_metrics(eval_preds):
    flattened_preds = np.array(eval_preds.predictions).flatten()
    flattened_labels = np.array(eval_preds.label_ids).flatten()
    filtered_preds = flattened_preds[flattened_labels != -100]
    filtered_labels = flattened_labels[flattened_labels != -100]
    accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
    return {
        "accuracy": accuracy,
    }


def get_prefix_tuning_params(model):
    if model.base_model_prefix == "chatglm":
        from ..peft.prefix import chatglm_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "chatglm_v2":
        from ..peft.prefix import chatglm_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = chatglm_postprocess_past_key_value
        multi_query_group_num = model.config.multi_query_group_num  # num_key_value_heads
    elif model.base_model_prefix == "bloom":
        from ..peft.prefix import bloom_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.n_embed
        postprocess_past_key_value = bloom_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "llama":
        from ..peft.prefix import llama_postprocess_past_key_value

        num_attention_heads = model.config.n_head
        num_hidden_layers = model.config.n_layer
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = llama_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "mistral":
        from ..peft.prefix import mistral_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = mistral_postprocess_past_key_value
        multi_query_group_num = model.config.num_key_value_heads
    elif model.base_model_prefix == "qwen":
        from ..peft.prefix import qwen_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = qwen_postprocess_past_key_value
        multi_query_group_num = None
    elif model.base_model_prefix == "qwen2":
        from ..peft.prefix import qwen_postprocess_past_key_value

        num_attention_heads = model.config.num_attention_heads
        num_hidden_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        postprocess_past_key_value = qwen_postprocess_past_key_value
        multi_query_group_num = model.config.num_key_value_heads  # num_key_value_heads
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}. ")
    return dict(
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        postprocess_past_key_value=postprocess_past_key_value,
        multi_query_group_num=multi_query_group_num,
    )


def get_lora_target_modules(model):
    # Not yet support RowParallelLinear
    if model.base_model_prefix == "chatglm":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix == "chatglm_v2":
        target_modules = [
            ".*query.*",
            ".*key.*",
            ".*value.*",
            ".*dense.*",
            ".*dense_h_to_4h.*",
            ".*dense_4h_to_h.*",
        ]
    elif model.base_model_prefix == "gpt":
        target_modules = [
            ".*qkv_proj.*",
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*linear1.*",
            ".*linear2.*",
            ".*out_proj.*",
        ]
    elif model.base_model_prefix == "bloom":
        target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
    elif model.base_model_prefix in ["llama", "jamba"] or isinstance(model, LlamaForCausalLMPipe):
        target_modules = [
            ".*q_proj.*",
            ".*v_proj.*",
            ".*k_proj.*",
            ".*o_proj.*",
            ".*qkv_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
            ".*gate_up_fused_proj.*",
        ]
    elif model.base_model_prefix == "opt":
        target_modules = [
            ".*project_in.*",
            ".*project_out.*",
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*qkv_proj.*",
            ".*out_proj.*",
            ".*linear1.*",
            ".*linear2.*",
        ]
    elif model.base_model_prefix == "qwen":
        target_modules = [
            ".*attn.c_attn.*",
            ".*attn.c_proj.*",
            ".*mlp.w1.*",
            ".*mlp.w2.*",
            ".*mlp.c_proj.*",
        ]
    elif model.base_model_prefix == "qwen2" or isinstance(model, Qwen2ForCausalLMPipe):
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*down_proj.*",
            ".*up_proj.*",
        ]
    elif model.base_model_prefix == "mixtral":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            # ".*gate.*", # TODO(DrownFish19): Does the gate weight require training?
            ".*w1.*",
            ".*w2.*",
            ".*w3.*",
        ]
    elif model.base_model_prefix == "mistral":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate.*",
            ".*w1.*",
            ".*w2.*",
            ".*w3.*",
        ]
    elif model.base_model_prefix == "qwen2_moe" or isinstance(model, Qwen2MoeForCausalLMPipe):
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            # ".*gate.*", # TODO(DrownFish19): Does the gate weight require training?
            ".*gate_proj.*",
            ".*up_proj.*",
            ".*down_proj.*",
        ]
    elif model.base_model_prefix in ["deepseek_v2", "deepseek_v3"] or isinstance(
        model, (DeepseekV2ForCausalLMPipe, DeepseekV3ForCausalLMPipe)
    ):
        target_modules = [
            ".*q_proj.*",
            ".*q_a_proj.*",
            ".*q_b_proj.*",
            ".*kv_a_proj_with_mqa.*",
            ".*kv_b_proj.*",
            ".*kv_b_proj.*",
            ".*o_proj.*",
            ".*mlp.gate_proj.*",
            ".*mlp.up_proj.*",
            ".*mlp.down_proj.*",
        ]
    elif model.base_model_prefix == "yuan":
        target_modules = [
            ".*q_proj.*",
            ".*k_proj.*",
            ".*v_proj.*",
            ".*o_proj.*",
            ".*gate_proj.*",
            ".*up_proj.*",
            ".*down_proj.*",
        ]
    else:
        raise ValueError(f"Unknown base_model_prefix: {model.base_model_prefix}.")
    return target_modules


class ZeroPaddingIterDatasetCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.

    """

    def on_step_end(self, args, state, control, **kwargs):
        train_dataloader = kwargs["train_dataloader"]
        if isinstance(train_dataloader.dataset, ZeroPaddingIterableDataset):
            dataset = train_dataloader.dataset
        elif isinstance(train_dataloader.dataset, IterableDatasetShard) and isinstance(
            train_dataloader.dataset.dataset, ZeroPaddingIterableDataset
        ):
            dataset = train_dataloader.dataset.dataset
        else:
            raise ValueError(
                "Unexpected dataset format: ZeroPaddingIterDatasetCallback expects `..datasets.ZeroPaddingIterableDataset`"
            )
        if state.trial_params is None:
            state.trial_params = {}
        state.trial_params["zero_padding_global_step"] = dataset.zero_padding_global_step


def get_infer_model_path(input_dir, model_prefix):
    if dist.get_world_size() > 1:
        local_rank = dist.get_rank()
        return os.path.join(input_dir, "rank_{}".format(local_rank), model_prefix)
    else:
        return os.path.join(input_dir, model_prefix)


def deserialize_from_file(fp):
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = np.arange(1, 1 + closest_power_of_2)
    slopes = np.power(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = np.arange(1, 1 + 2 * num_remaining_heads, 2)
        slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)], axis=0)

    return slopes.astype("float32")


def pad_batch_data(insts, masks=None, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad sequences to the max sequence length in batch."""
    max_len = max(map(len, insts))
    if pad_style == "left":
        inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts])
    else:
        inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])

    if masks is not None:
        if pad_style == "left":
            inst_mask = np.array([[0] * (max_len - len(inst)) + list(inst) for inst in masks])
        else:
            inst_mask = np.array([list(inst) + [0] * (max_len - len(inst)) for inst in masks])

    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        if masks is None:
            return inst_data.astype("int64").reshape([-1, max_len]), seq_len
        else:
            return (
                inst_data.astype("int64").reshape([-1, max_len]),
                inst_mask.astype("int64").reshape([-1, max_len]),
                seq_len,
            )
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


def dybatch_preprocess(
    tokenizer,
    texts: list[str],
    src_length: int,
    max_length: int,
    architectures: str,
    top_p: float,
    temperature: float,
    eos_token_id: int | list[list[int]],
    pre_caches_length: int = 0,
    benchmark: bool = False,
    pad_style: str = "None",
):
    """Pre-process generation inputs."""
    inputs = {}
    if "chatglmforcausallm" == architectures.lower():
        input_ids = []
        position_ids = []

        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding=True,
                max_length=src_length,
                # if use chat_template, it will not add special_tokens
                add_special_tokens=tokenizer.chat_template is None,
                # add_special_tokens=tokenizer.chat_template is None or isinstance(tokenizer, ChatGLMv2Tokenizer),
            )
            input_ids.append(tokens["input_ids"][0])
            position_ids.append(tokens["position_ids"][0])

        pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][0]
        inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
        bs = inputs["input_ids"].shape[0]
        max_len = max(map(len, input_ids))

        inst_data_pos = []
        for i in range(len(position_ids)):
            inst_data_pos.append(np.array([list(inst) + [0] * (max_len - len(inst)) for inst in position_ids[i]]))
        inputs["position_ids"] = paddle.to_tensor(np.array(inst_data_pos))
    elif "gpt" in architectures:
        input_ids = []
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            tokens = tokenizer(
                text,
                return_tensors="np",
                padding=False,
                max_length=src_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids.append(tokens["input_ids"][0])

        pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
        inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
        bs = inputs["input_ids"].shape[0]
        max_len = max(map(len, input_ids))

        position_ids = paddle.arange(sum(seq_len), dtype="int64")
        pre_len = seq_len[0]
        for length in seq_len[1:]:
            position_ids[pre_len : length + pre_len] = position_ids[pre_len : length + pre_len] - pre_len
            pre_len += length
        inputs["position_ids"] = position_ids
    else:
        input_ids = []
        attention_mask = []
        if isinstance(texts, str):
            texts = [texts]

        if pad_style == "left":
            return_attention_mask = True
            truncation = True
            for text in texts:
                tokens = tokenizer(
                    text,
                    return_tensors="np",
                    padding="max_length",
                    max_length=src_length,
                    truncation=truncation,
                    return_attention_mask=return_attention_mask,
                    return_token_type_ids=False,
                    add_special_tokens=tokenizer.chat_template is None,
                    # add_special_tokens=tokenizer.chat_template is None or isinstance(tokenizer, ChatGLMv2Tokenizer),
                )
                input_ids.append(tokens["input_ids"][0])
                attention_mask.append(tokens["attention_mask"][0])

            pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
            inputs["input_ids"], inputs["attention_mask"], seq_len = pad_batch_data(
                input_ids, attention_mask, pad_id=pad_token_id, return_seq_len=True, pad_style=pad_style
            )
            bs = inputs["input_ids"].shape[0]
            max_len = max(map(len, input_ids))

            position_ids = paddle.zeros(shape=[bs, max_length + src_length], dtype="int64")

            for i in range(bs):
                position_ids[
                    i, pre_caches_length + max_len - seq_len[i] : pre_caches_length + max_len
                ] = paddle.arange(seq_len[i]).unsqueeze(axis=0)
                seq_len[i] = max_len
            inputs["position_ids"] = position_ids
        else:
            for text in texts:
                tokens = tokenizer(
                    text,
                    return_tensors="np",
                    padding=False,
                    max_length=src_length,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=tokenizer.chat_template is None
                    # add_special_tokens=tokenizer.chat_template is None or isinstance(tokenizer, ChatGLMv2Tokenizer),
                )
                input_ids.append(tokens["input_ids"][0])

            pad_token_id = tokenizer([tokenizer.pad_token], return_tensors="np")["input_ids"][0][-1]
            inputs["input_ids"], seq_len = pad_batch_data(input_ids, pad_id=pad_token_id, return_seq_len=True)
            bs = inputs["input_ids"].shape[0]
            max_len = max(map(len, input_ids))

            position_ids = paddle.zeros(shape=[bs, max_length + src_length], dtype="int64")

            for i in range(bs):
                position_ids[i, pre_caches_length : pre_caches_length + seq_len[i]] = paddle.arange(seq_len[i])
            inputs["position_ids"] = position_ids

    tgt_ids = [input[-1:] for input in input_ids]
    tgt_pos = []
    for i, valid_len in enumerate(map(len, input_ids)):
        tgt_pos.append(valid_len - 1)

    step_idx = [
        0,
    ] * bs
    tgt_pos = np.array(tgt_pos).astype("int64")

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    inputs["eos_token_id"] = np.array(eos_token_id * bs).reshape(-1, 1).astype("int64")

    inputs["top_p"] = (
        np.array(
            [
                top_p,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["temperature"] = (
        np.array(
            [
                temperature,
            ]
            * bs
        )
        .reshape(-1, 1)
        .astype("float32")
    )
    inputs["seq_len_encoder"] = seq_len.astype("int32").reshape(-1, 1)
    inputs["seq_len_decoder"] = (seq_len + pre_caches_length).astype("int32").reshape(-1, 1)
    inputs["step_idx"] = np.array(step_idx).astype("int64").reshape(-1, 1)
    inputs["tgt_ids"] = np.array(tgt_ids).astype("int64").reshape(-1, 1)
    inputs["tgt_pos"] = tgt_pos.reshape(-1, 1)
    inputs["max_length"] = np.array(max_length - pre_caches_length).astype("int64").reshape((-1, 1))
    inputs["min_length"] = (
        np.array(
            [
                1
                if not benchmark
                else max_length
                - pre_caches_length,  # Note(Zhengzekang): When in benchmark mode, we need to set a fixed decode length.
            ]
            * bs
        )
        .astype("int64")
        .reshape((-1, 1))
    )
    inputs["penalty_score"] = (
        np.array(
            [
                1.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["frequency_score"] = (
        np.array(
            [
                0.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["presence_score"] = (
        np.array(
            [
                0.0,
            ]
            * bs
        )
        .astype("float32")
        .reshape((-1, 1))
    )
    inputs["stop_flags"] = (
        np.array(
            [
                0,
            ]
            * bs
        )
        .astype("bool")
        .reshape((-1, 1))
    )
    inputs["stop_nums"] = np.array([bs]).astype("int64")
    return inputs


def load_real_time_tokens():
    tokens = []
    files = glob.glob(os.path.join("./real_time_save.*"))
    for j in range(1, len(files) + 1):
        filename = "./real_time_save.temp_ids_rank_0_step_{}".format(j)
        if not os.path.exists(filename):
            break
        fp = open(filename, "rb+")
        fp.read(1)
        data_list = deserialize_from_file(fp)
        fp.close()
        tokens.append(np.array(data_list).reshape(-1, 1))
    os.system("rm -f ./real_time_save.temp_ids_rank_*")
    tokens = np.concatenate(tokens, axis=1)
    return tokens


def init_chat_template(
    tokenizer: PretrainedTokenizer, model_name_or_path: str, chat_template_file: Optional[str] = None
):
    """init chat template for the given tokenizer.

        If is None, it will not use `chat_template.json`;
        If is equal with `model_name_or_path`, it will use the default loading;
        If is directory, it will find the `chat_template.json` under the directory;
        If is file, it will load it.

    Args:
        tokenizer (PretrainedTokenizer): the instance of tokenizer
        model_name_or_path (str): _description_
        chat_template_file (Optional[str], optional): _description_. Defaults to None.
    """
    # 1. use the default chat_template file
    if chat_template_file is None:
        return

    if str(chat_template_file).lower() == "none":
        # delete the chat_template from tokenizer if not use chat_template.
        # why do this: it will load the `chat_template.json` file by default
        tokenizer.chat_template = None
        return

    # it will load the `chat_template.json` file by default, so do nothing
    if chat_template_file == model_name_or_path:
        if tokenizer.chat_template is None:
            logger.warning(f"there is not `chat_template.json` file in the `{model_name_or_path}`")
        return

    if os.path.isdir(chat_template_file):
        local_chat_template_file_path = os.path.join(chat_template_file, "chat_template.json")
        if os.path.exists(local_chat_template_file_path):
            chat_template_file = local_chat_template_file_path
        else:
            logger.warning(f"there is not `chat_template.json` file in the `{model_name_or_path}`")
            return

    if not os.path.exists(chat_template_file):
        logger.warning(f"there is not `chat_template.json` file from path<`{model_name_or_path}`>")
        return

    logger.info(f"loading `chat_template.json` from `{chat_template_file}`")
    tokenizer.init_chat_template(chat_template_file)


def get_model_max_position_embeddings(config: PretrainedConfig) -> Optional[int]:
    names = [
        "max_position_embeddings",  # most of models
        "max_sequence_length",  # GLM model
        "seq_length",  # llama model
    ]
    for name in names:
        max_length = config.get(name, None)
        if max_length is not None:
            return max_length
    return None


def read_res(
    model_name_or_path: str,
    tensor_queue: mp.Queue,
    result_queue: mp.Queue,
    done_event: mp.Event,
):
    from ..utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=USE_FAST_TOKENIZER)

    paddle.device.set_device("cpu")
    paddle.disable_static()
    outputs = []
    output_tensor = tensor_queue.get(timeout=1)
    done_event.set()
    logger.info("Start read result message")
    logger.info(f"Current path is {os.getcwd()}")

    from paddlenlp_ops import get_output

    while True:
        get_output(output_tensor, 0, True)
        if int(output_tensor[0, 0]) == -2:  # read none
            continue
        bsz = int(output_tensor[1, 0])
        output_numpy = output_tensor[2 : bsz + 2].numpy()
        output_numpy[output_numpy == -1] = tokenizer.eos_token_id
        outputs.append(output_numpy)
        if int(output_tensor[0, 0]) == -1:
            break
    output = np.concatenate(outputs, axis=1).tolist()
    seqs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, (out, seq) in enumerate(zip(output, seqs)):
        result_queue.put([i, out, seq])

    logger.info("Finish read result message")


def read_res_dynamic_insert(
    model_name_or_path: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    done_event: mp.Event,
    total_request_num: int,
    detokenize: bool,
):
    from ..utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=USE_FAST_TOKENIZER)

    paddle.device.set_device("cpu")
    paddle.disable_static()

    outputs = [[] for _ in range(total_request_num)]
    count = 0

    done_event.set()
    logger.info("Start read result dynamic insert")

    while count < total_request_num:
        try:
            task_id, token_ids = task_queue.get(block=True, timeout=None)

            if task_id < 0 or task_id >= total_request_num:
                logger.warning(f"Invalid task ID received: {task_id}")
                continue

            if len(outputs[task_id]) == 0:
                output_numpy = token_ids.reshape([1, -1])
                output_numpy[output_numpy == -1] = tokenizer.eos_token_id
                outputs[task_id] = output_numpy
                count += 1
                logger.info(f"Post-processing task {task_id} ({count}/{total_request_num})")

        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            continue
    output = np.concatenate(outputs, axis=0).tolist()
    if detokenize:
        seqs = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    else:
        seqs = [None] * len(output)
    for i, (out, seq) in enumerate(zip(output, seqs)):
        result_queue.put([i, out, seq])
    logger.info("Finish read result message")


def speculate_read_res(
    model_name_or_path: str,
    tensor_queue: mp.Queue,
    result_queue: mp.Queue,
    done_event: mp.Event,
):
    from ..utils.env import USE_FAST_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", use_fast=USE_FAST_TOKENIZER)
    paddle.device.set_device("cpu")
    paddle.disable_static()
    outputs = []
    from ..utils.env import MAX_DRAFT_TOKENS, SPECULATE_MAX_BSZ

    for _ in range(SPECULATE_MAX_BSZ):
        outputs.append([])
    output_tensor = tensor_queue.get(timeout=1)
    done_event.set()
    logger.info("Start speculate read result message")
    logger.info(f"Current path is {os.getcwd()}")

    from paddlenlp_ops import speculate_get_output

    while True:
        speculate_get_output(output_tensor, 0, True)
        if int(output_tensor[0, 0]) == -2:  # read none
            continue
        bsz = int(output_tensor[1])
        accept_num = output_tensor[2 : bsz + 2].numpy()
        for bi in range(bsz):
            output_numpy = output_tensor[
                2
                + SPECULATE_MAX_BSZ
                + bi * MAX_DRAFT_TOKENS : 2
                + SPECULATE_MAX_BSZ
                + bi * MAX_DRAFT_TOKENS
                + int(accept_num[bi]),
                0,
            ].numpy()
            output_numpy[output_numpy == -1] = tokenizer.eos_token_id
            outputs[bi].extend(output_numpy.tolist())
        if int(output_tensor[0, 0]) == -1:
            break

    seqs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, (out, seq) in enumerate(zip(outputs, seqs)):
        result_queue.put([i, out, seq])

    logger.info("Finish read result message")


def get_rotary_position_embedding(position_ids, head_dim, rope_theta=10000.0, rope_scaling: dict = None):
    """
    Pre-calculate rotary position embedding for position_ids.

    Args:
        position_ids: [1, S]
        head_dim: D

    Returns:
        rot_emb: [2, 1, S, 1, D], cos + sin
    """
    bsz, max_seq_len = position_ids.shape[:2]
    rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
    inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", None)
        if rope_type is not None and rope_type == "llama3":
            factor = rope_scaling.get("factor", 8.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 8192)

            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
            inv_freq = paddle.to_tensor(new_freqs, dtype=inv_freq.dtype)

    # shape: [B, S, D/2]
    freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
    # shape: [B, S, 1, D]
    emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

    rot_emb[0] = paddle.cos(emb)
    rot_emb[1] = paddle.sin(emb)
    return rot_emb


def init_dist_env():
    """
    Initialize the distributed environment and obtain tensor parallel degree and rank.

    Returns:
        tuple: A tuple containing tensor parallel rank and degree.
    """
    world_size = paddle.distributed.get_world_size()  # Get the total number of distributed nodes

    if world_size > 1:
        is_fleet_init = True
        try:
            # Try to get the hybrid communicate group to check if Fleet has been initialized
            hcg = fleet.get_hybrid_communicate_group()
        except AttributeError:
            is_fleet_init = False  # Fleet has not been initialized

        if is_fleet_init:
            # If Fleet is already initialized, get tensor parallel degree and rank
            tensor_parallel_degree = hcg.get_model_parallel_world_size()
            tensor_parallel_rank = hcg.get_model_parallel_rank()
        else:
            # If Fleet is not initialized, set up the distributed strategy and initialize Fleet
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,  # Data parallelism degree
                "mp_degree": world_size,  # Model parallelism degree (to be determined or set)
                "pp_degree": 1,  # Pipeline parallelism degree
                "sharding_degree": 1,  # Sharding parallelism degree
            }
            fleet.init(is_collective=True, strategy=strategy)  # Initialize Fleet
            hcg = fleet.get_hybrid_communicate_group()  # Get the hybrid communicate group after initialization

            # Get tensor parallel degree and rank after Fleet initialization
            tensor_parallel_degree = hcg.get_model_parallel_world_size()
            tensor_parallel_rank = hcg.get_model_parallel_rank()
    else:
        # If not in a distributed environment, set tensor parallel degree and rank to 1 and 0 respectively
        tensor_parallel_degree = 1
        tensor_parallel_rank = 0

    return tensor_parallel_rank, tensor_parallel_degree


def get_eos_token_id(
    tokenizer: PretrainedTokenizer, generation_config: Optional[GenerationConfig] = None
) -> List[List[int]]:
    """get eos_token_id from generation_config or tokenizer

    Returns:
        List[int]: eos_token_id to stop the generation
    """
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    if generation_config is not None and generation_config.eos_token_id is not None:
        if isinstance(generation_config.eos_token_id, int):
            eos_token_ids.append(generation_config.eos_token_id)
        else:
            eos_token_ids.extend(generation_config.eos_token_id)

    eos_token_ids_dict = {str(item): item for item in eos_token_ids}
    return list(eos_token_ids_dict.values())


def set_triton_cache(model_name_or_path, mode):
    """
    Set triton cache.
    """
    valid_modes = {"export", "static", "dynamic"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
    mp_id = paddle.distributed.get_rank()
    triton_dir = f"triton_ops_rank_{mp_id}"
    triton_kernel_cache_dir = f"{model_name_or_path}/{triton_dir}"
    if mode == "export":
        os.environ["TRITON_KERNEL_CACHE_DIR"] = triton_kernel_cache_dir
        if os.path.exists(triton_kernel_cache_dir):
            # del old triton_ops
            shutil.rmtree(triton_kernel_cache_dir)
    elif mode == "static":
        os.environ["TRITON_KERNEL_CACHE_DIR"] = triton_kernel_cache_dir
        for root, dirs, files in os.walk(triton_kernel_cache_dir):
            for file in files:
                if file.endswith("_package.so"):
                    so_full_path = os.path.join(root, file)
                    paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_full_path)
    else:
        os.environ["TRITON_KERNEL_CACHE_DIR"] = f"/root/.paddleformers/{triton_dir}"
