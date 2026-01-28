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

import os
from dataclasses import fields
from typing import Any, Optional

import numpy as np

from paddleformers.data.indexed_dataset import SFTMMapIndexedDatasetBuilder
from paddleformers.datasets.data_utils import estimate_training
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.SFTDataset import TextSequence
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.trainer import RuntimeTimer
from paddleformers.transformers import (
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from paddleformers.utils.log import logger

from ..hparams import get_train_args, read_args
from .make_data_utils import DataGenerator


def run_make_sft_data(args: Optional[dict[str, Any]] = None) -> None:
    """
    Convert the dataset to the MapDataset format that can be used by the SFT training.
    """

    args = read_args(args)
    model_args, data_args, preprocess_args, generating_args, training_args = get_train_args(args)
    training_args.max_seq_len = data_args.max_seq_len

    runtime_timer = RuntimeTimer("Creating SFT MapDataset")

    # Load tokenizer & processor & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    dataset_config = {
        "tokenizer": tokenizer,
        "processor": processor,
        "max_seq_len": data_args.max_seq_len,
        "random_seed": training_args.seed,
        "num_replicas": training_args.dataset_world_size,
        "rank": training_args.dataset_rank,
        "num_samples_each_epoch": data_args.num_samples_each_epoch,
        "random_shuffle": data_args.random_shuffle,
        "greedy_intokens": data_args.greedy_intokens,
        "packing": data_args.packing,
        "mix_strategy": data_args.mix_strategy,
        "encode_one_turn": data_args.encode_one_turn,
        "use_template": data_args.use_template,
        "is_pretraining": True if model_args.stage.lower() == "pt" else False,
        "truncate_packing": data_args.truncate_packing,
        "stage": model_args.stage,
        "template_backend": data_args.template_backend,
        "split_multi_turn": data_args.split_multi_turn,
    }

    dataset_config.update(
        {
            "template": data_args.template,
            "tool_format": None,
            "default_system": None,
        }
    )

    if dataset_config["template_backend"] == "custom":
        template_instance = get_template_and_fix_tokenizer(dataset_config)
    else:
        template_instance = None
    dataset_config.update(
        {
            "template_instance": template_instance,
        }
    )

    dataclass = TextSequence

    global_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(training_args.data_parallel_size, 1)
        * max(training_args.sharding_parallel_size, 1)
    )

    if training_args.do_train and data_args.train_dataset_path:
        runtime_timer.start("Create SFT Train MapDataset")
        os.makedirs(os.path.join(data_args.dataset_output_dir, "train"), exist_ok=True)

        train_output_idx_files = os.path.join(data_args.dataset_output_dir, "train", "index.idx")
        train_dataset = create_dataset_sft(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config,
        )
        if training_args.max_steps == -1:
            training_args.estimation_output_file = (
                "estimate_training.json"
                if training_args.estimation_output_file is None
                else training_args.estimation_output_file
            )
            training_args.max_steps = estimate_training(train_dataset, data_args, training_args, model_args)

        train_samples = training_args.max_steps * global_batch_size
        logger.info(f"train_samples : {train_samples}")

        output_file_dict = {}
        train_dir = os.path.join(data_args.dataset_output_dir, "train")
        for field in fields(dataclass):
            output_path = os.path.join(train_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path
        train_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype)

        train_sample_generator = DataGenerator(train_dataset)
        used_samples = 0
        while used_samples < train_samples:
            train_sample = next(train_sample_generator)
            for sequence in train_sample:
                train_builder.add_item(sequence)
            train_builder.end_document()
            used_samples += 1
        train_builder.finalize(train_output_idx_files)
        logger.info(f"{runtime_timer.log()}")

    if training_args.do_eval and data_args.eval_dataset_path:
        runtime_timer.start("Create SFT Eval MapDataset")
        os.makedirs(os.path.join(data_args.dataset_output_dir, "eval"), exist_ok=True)

        eval_output_idx_files = os.path.join(data_args.dataset_output_dir, "eval", "index.idx")
        eval_dataset = create_dataset_sft(
            task_group=data_args.eval_dataset_path,
            task_group_prob=data_args.eval_dataset_prob,
            sub_dataset_type=data_args.eval_dataset_type,
            is_valid=True,
            **dataset_config,
        )
        output_file_dict = {}
        eval_dir = os.path.join(data_args.dataset_output_dir, "eval")
        for field in fields(dataclass):
            output_path = os.path.join(eval_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path
        eval_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype)

        for sequences in eval_dataset:
            for sequence in sequences:
                eval_builder.add_item(sequence)
            eval_builder.end_document()
        eval_builder.finalize(eval_output_idx_files)
        logger.info(f"{runtime_timer.log()}")
