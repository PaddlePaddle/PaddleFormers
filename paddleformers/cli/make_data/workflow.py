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

"""Training Ernie Model."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from typing import Any, Optional

import numpy as np
import paddle

from paddleformers.cli.utils.process import add_new_special_tokens
from paddleformers.data.indexed_dataset import SFTMMapIndexedDatasetBuilder
from paddleformers.datasets.loader import create_dataset as create_dataset_sft
from paddleformers.datasets.SFTDataset import TextSequence
from paddleformers.datasets.template.template import get_template_and_fix_tokenizer
from paddleformers.trainer import RuntimeTimer, set_random_seed, set_seed
from paddleformers.transformers import (
    AutoProcessor,
    AutoTokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
)
from paddleformers.utils.log import logger

from ..hparams import get_train_args, read_args
from .make_data_utils import DataGenerator

# Fine-tune Environment Variables to support sharding stage1 overlap optimization.
os.environ["USE_CASUAL_MASK"] = "False"


def run_make_data(args: Optional[dict[str, Any]] = None) -> None:
    """
    Run the data processing pipeline.
    """
    # parse args
    args = read_args(args)
    model_args, data_args, _, _, finetuning_args = get_train_args(args)

    # setup training
    training_args = finetuning_args
    training_args.max_seq_len = data_args.max_seq_len
    training_args.model_name_or_path = model_args.model_name_or_path
    training_args.download_hub = model_args.download_hub
    training_args.copy_custom_file_list = model_args.copy_custom_file_list

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Train")

    if training_args.pre_alloc_memory > 0:
        memory_size = int(training_args.pre_alloc_memory * 1024 * 1024 * 1024)
        x = paddle.empty([memory_size], dtype=paddle.uint8)
        logger.info(f"pre_alloc_memory size {x.shape}")
        del x

    # Setup GPU & distributed training
    paddle.set_device(training_args.device)
    set_random_seed(seed_=training_args.seed)
    set_seed(seed=training_args.seed)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    runtime_timer = RuntimeTimer("Creating SFT MapDataset")

    # Load tokenizer & processor & dataset
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    add_new_special_tokens(tokenizer, data_args.new_special_tokens_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # if using chat_template, data_args.eval_with_do_generation must be false
    if tokenizer.chat_template is not None:
        data_args.eval_with_do_generation = False

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "VL" in model_args.stage and training_args.dataloader_num_workers > 0:
        data_args.processor_use_fast = False
        logger.warning_once(
            f"Detected dataloader_num_workers={training_args.dataloader_num_workers} (>0). "
            "Since the CPU version of the 'interpolate' operator is currently unsupported, "
            "some models may use a fast image processor which can cause errors in dataloader workers. "
            "Temporarily fallback to the slow image processor (`use_fast=False`) by default to avoid potential issues. "
            "You can also explicitly set `processor_use_fast=False` or `dataloader_num_workers=0` to avoid this warning."
        )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=data_args.processor_use_fast)

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
        "is_pretraining": True if "pt" in model_args.stage.lower() else False,
        "truncate_packing": data_args.truncate_packing,
        "stage": model_args.stage,
        "template_backend": data_args.template_backend,
        "split_multi_turn": data_args.split_multi_turn,
        "dataset_num_proc": finetuning_args.dataset_num_proc,
        "binpacking": data_args.binpacking,
        "packing_interval": data_args.packing_interval,
        "dataloader_num_workers": training_args.dataloader_num_workers,
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
    # make offline dataset
    import time

    if tokenizer.vocab_size < 2**16 - 1:
        save_dtype = np.uint16
    else:
        save_dtype = np.int32
    dataclass = TextSequence

    global_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(training_args.data_parallel_size, 1)
        * max(training_args.sharding_parallel_size, 1)
    )

    logger.info(f"training_args.per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    logger.info(f"training_args.gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"training_args.data_parallel_size: {training_args.data_parallel_size}")
    logger.info(f"training_args.sharding_parallel_size: {training_args.sharding_parallel_size}")
    logger.info(f"global_batch_size: {global_batch_size}")

    def fetch_and_serialize(generator, dtype):
        sample = next(generator)
        result = []
        for sequence in sample:
            serialized = []
            for key in train_builder._data_file_dict.keys():
                tensor = np.array(getattr(sequence, key), dtype=dtype)
                serialized.append((key, tensor.tobytes(order="C"), tensor.size))
            result.append(serialized)
        return result

    if (
        training_args.do_train
        and data_args.train_dataset_path
        and training_args.should_load_dataset
        and paddle.distributed.get_rank() == 0
    ):
        runtime_timer.start("Create SFT Train MapDataset")
        os.makedirs(os.path.join(data_args.dataset_output_dir, "train"), exist_ok=True)

        train_output_idx_files = os.path.join(data_args.dataset_output_dir, "train", "index.idx")
        train_dataset = create_dataset_sft(
            task_group=data_args.train_dataset_path,
            task_group_prob=data_args.train_dataset_prob,
            sub_dataset_type=data_args.train_dataset_type,
            **dataset_config,
        )
        output_file_dict = {}
        train_dir = os.path.join(data_args.dataset_output_dir, "train")
        index_file = os.path.join(data_args.dataset_output_dir, "train", "index.idx")
        for field in fields(dataclass):
            output_path = os.path.join(train_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path
        train_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype, index_file=index_file)
        train_sample_generator = DataGenerator(train_dataset)
        count = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(fetch_and_serialize, train_sample_generator, save_dtype)
            while not train_dataset.iter_all_examples:
                serialized_sequences = future.result()
                future = executor.submit(fetch_and_serialize, train_sample_generator, save_dtype)
                for serialized in serialized_sequences:
                    train_builder.add_item_bytes(serialized)
                train_builder.end_document()
                count += 1
                if count % 1000 == 0:
                    logger.info(
                        f"Processed {count} samples in {time.time()-start_time:.2f} seconds, average speed: {count/(time.time()-start_time):.2f} samples/second"
                    )
        train_builder.finalize(train_output_idx_files)
        logger.info(f"{runtime_timer.log()}")

    if (
        training_args.do_eval
        and data_args.eval_dataset_path
        and training_args.should_load_dataset
        and paddle.distributed.get_rank() == 0
    ):
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
        index_file = os.path.join(data_args.dataset_output_dir, "eval", "index.idx")
        for field in fields(dataclass):
            output_path = os.path.join(eval_dir, f"{field.name}.bin")
            output_file_dict[field.name] = output_path
        eval_builder = SFTMMapIndexedDatasetBuilder(output_file_dict, save_dtype, index_file=index_file)
        for sequences in eval_dataset:
            for sequence in sequences:
                eval_builder.add_item(sequence)
            eval_builder.end_document()
        eval_builder.finalize(eval_output_idx_files)
        logger.info(f"{runtime_timer.log()}")
    logger.info("Make SFT Offline DataSet Done.")
    return
