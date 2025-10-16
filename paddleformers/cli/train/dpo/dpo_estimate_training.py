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

""" Estimate DPO """

import json
import os

import numpy as np
import paddle

from paddleformers.utils.log import logger

# isort: off
# fmt: off
# isort: on
from paddleformers.datasets.dpo import create_dataset


def calculate_acc_steps(num_samples, train_batch, dataset_world_size, per_device_train_batch_size):
    """calculate_acc_steps

    Args:
        num_samples (int): Total training samples in dataset
        train_batch (int): Target global batch size
        dataset_world_size (int): Number of dataset parallel training devices
        per_device_train_batch_size (int): Batch size per GPU/device

    Returns:
        int: Number of gradient accumulation steps needed to achieve:
            - Global batch size target
            - Full dataset coverage
    """
    samples_per_batch = per_device_train_batch_size * dataset_world_size * num_samples / train_batch
    if num_samples < 100:
        recommend_bs = 8
    elif num_samples < 1000:
        recommend_bs = 16
    elif num_samples < 10000:
        recommend_bs = 32
    elif num_samples < 100000:
        recommend_bs = 64
    else:
        recommend_bs = 128
    return min(np.ceil(recommend_bs / samples_per_batch), 32)


def dpo_estimate_training(tokenizer, data_args, training_args, config, train_dataset=None):
    """ dpo_estimate_training

    Args:
        tokenizer (PreTrainedTokenizer): Text tokenization
        data_args (DataArguments): Datasets configuration
        training_args (TrainingArguments): Training configuration
        config (PretrainedConfig): Model configuration
        train_dataset (Dataset, optional): Preloaded dataset

    Returns:
        training_args (TrainingArguments): Training configuration with max_steps setting
        res (Dict): Training estimate results
    """

    if training_args.should_save or training_args.should_save_model_state:
        os.makedirs(training_args.output_dir, exist_ok=True)
    if train_dataset is None:
        dataset_config = {
            "tokenizer": tokenizer,
            "max_seq_len": data_args.max_seq_len,
            "max_prompt_len": data_args.max_prompt_len,
            "random_seed": training_args.seed,
            "num_replicas": 1,
            "rank": 0,
            "num_samples_each_epoch": data_args.num_samples_each_epoch,
            "random_shuffle": data_args.random_shuffle,
            "greedy_intokens": data_args.greedy_intokens,
            "buffer_size": data_args.buffer_size,
            "mask_out_eos_token": data_args.mask_out_eos_token,
            "packing": data_args.packing,
            "mix_strategy": data_args.mix_strategy,
            "encode_one_turn": data_args.encode_one_turn,
        }
        train_dataset = create_dataset(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config
        )
    max_samples = len(train_dataset.mix_datasets)
    if max_samples > 0 :
        if training_args.num_of_gpus > 0:
            dataset_world_size = (
                training_args.num_of_gpus
                // max(1, training_args.tensor_parallel_degree)
                // max(1, training_args.pipeline_parallel_degree))
            if dataset_world_size < 1:
                raise ValueError("dataset_world_size must be positive, please verify your config")
        else:
            dataset_world_size = training_args.dataset_world_size

        num_samples = 0
        train_tokens = 0
        train_batch = 0
        for sequences in train_dataset:
            if num_samples >= max_samples:
                break
            train_batch += 1
            for sequence in sequences:
                train_tokens += len(sequence.input_ids)
                num_samples += 1
        if training_args.gradient_accumulation_steps < 0:
            training_args.gradient_accumulation_steps = calculate_acc_steps(
                num_samples, train_batch, dataset_world_size, training_args.per_device_train_batch_size)
        max_samples *= training_args.num_train_epochs
        train_tokens *= training_args.num_train_epochs
        train_batch *= training_args.num_train_epochs
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * dataset_world_size
        )
        if training_args.num_of_gpus < 0:
            training_args.num_of_gpus = paddle.distributed.get_world_size()

        training_args.max_steps = np.ceil(train_batch / global_batch_size)
        total_tokens = training_args.max_steps * data_args.max_seq_len * global_batch_size
        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": int(training_args.max_steps),
            "train_samples": int(max_samples),
            "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
            "num_of_gpus": int(training_args.num_of_gpus),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "pipeline_parallel_degree": int(max(1, training_args.pipeline_parallel_degree)),
            "tensor_parallel_degree": int(max(1, training_args.tensor_parallel_degree)),
            "seed": int(training_args.seed),
            "num_samples_each_epoch": int(data_args.num_samples_each_epoch),
            "max_seq_len": int(data_args.max_seq_len),
            "max_prompt_len": int(data_args.max_prompt_len),
            "total_tokens": int(total_tokens),
            "train_tokens": int(train_tokens),
            "valid": True,
        }
        if train_batch / training_args.num_train_epochs / global_batch_size < 1:
            logger.warning("This dataset is too small, you'd better enlarge your dataset.")
            res["valid"] = False
    else:
        training_args.max_steps = 0
        logger.error("No valid data found, please check your dataset format.")
        res = {
            "num_train_epochs": int(training_args.num_train_epochs),
            "max_steps": int(training_args.max_steps),
            "train_samples": 0,
            "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
            "num_of_gpus": int(training_args.num_of_gpus),
            "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
            "pipeline_parallel_degree": int(max(1, training_args.pipeline_parallel_degree)),
            "tensor_parallel_degree": int(max(1, training_args.tensor_parallel_degree)),
            "seed": int(training_args.seed),
            "num_samples_each_epoch": 6000000,
            "max_seq_len": int(data_args.max_seq_len),
            "max_prompt_len": int(data_args.max_prompt_len),
            "valid": False,
        }

    logger.info(f"training argument: {res}")
    # NOTE(gongenlei): if not int, broadcast will overflow
    training_args.max_steps = int(training_args.max_steps)
    with open(os.path.join(training_args.output_dir, "dpo_train_args.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    return training_args, res
