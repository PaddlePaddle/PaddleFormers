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

import json
import numpy as np
from pprint import pprint
from paddleformers.datasets2.processor.vision_processor import Qwen2VLVisionProcessor
from paddleformers.datasets2.processor.auto_processor import Qwen2VLProcessor
from paddleformers.transformers import AutoTokenizer
from paddleformers.hparams.data_args import DataArguments
from paddleformers.datasets2.processor import SupervisedDatasetProcessor, ProcessDataset


from paddleformers.datasets2.reader.mix_datasets import MultiSourceDataset, create_dataset_instance


def create_dataset(**dataset_config):

    # data loader
    multi_source_dataset = MultiSourceDataset(**dataset_config)

    # data mix
    mix_datasets = create_dataset_instance(
        dataset_config["mix_strategy"],
        multi_source_dataset,
        **dataset_config,
    )

    # debug
    for item in mix_datasets:
        print(item)
        break

    # dataset_loader = _get_dataset_loader(dataset_config)  # 默认MultiSourceDataset
    # dataset = dataset_loader(dataset_config)

    # # data processor
    # dataset = _get_preprocessed_dataset(dataset, dataset_config)

    # """
    # dataset_processor = _get_dataset_processor(
    #     data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    # )

    data_args = DataArguments(
        max_seq_len=16384,
        min_pixels=3136,
        max_pixels=4816896,
        video_min_frames=4,
        video_max_frames=768,
        render_timestamp=True,
    )  
    auto_processor = Qwen2VLProcessor(data_args=data_args)
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/paddlejob/workspace/env/output/lrl/PaddleFormers/Qwen3-0.6B-base",
        trust_remote_code=True,    
    )
    vision_processor = Qwen2VLVisionProcessor(data_args=data_args)
    processor = SupervisedDatasetProcessor(
        auto_processor=auto_processor,
        tokenizer=dataset_config["tokenizer"],
        vision_processor=vision_processor,
        data_args=data_args,
    )
    processed_dataset = ProcessDataset(mix_datasets, processor)
    
    # debug
    for item in processed_dataset:
        print(item)
        break

    return processed_dataset

    # """

    # # packing
    # if dataset_config["packing"] == true:
    #     dataset = dataset.map(
    #         Packing_processor,
    #         **kwargs,
    #     )
    # else:
    #     dataset = dataset = dataset.map(
    #         NoPacking_processor,
    #         **kwargs,
    #     )

    # # sampler
    # Sampler = _get_sampler(dataset_config)
    # dataset = IterDataset(
    #     Sampler(
    #         dataset,
    #         **kwargs,
    #     )
    # )
    # return dataset

if __name__ == "__main__":

    from paddleformers.transformers import (
        AutoTokenizer,
    )

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained("/root/paddlejob/workspace/env/output/lrl/PaddleFormers/Qwen3-0.6B-base")
    # if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, Llama3Tokenizer):
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_config = {
        "tokenizer": tokenizer,
        "max_seq_len": 8192,
        "random_seed": 42,
        "num_replicas": 1,
        "rank": 0,
        "num_samples_each_epoch": 6000000,
        "random_shuffle": True,
        "greedy_intokens": True,
        "packing": True,
        "mix_strategy": "concat",
        "encode_one_turn": True,
        "use_template": True,
        "reverse": True,
        "task_group": "/root/paddlejob/workspace/env/output/lrl/PaddleFormers/data/sft/train.jsonl",
        "task_group_prob": "1.0",
        "sub_dataset_type": "erniekit",
    }

    create_dataset(
        **dataset_config,
    )
