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

from paddleformers.datasets2.reader.file_reader import MultiSourceDataset
from paddleformers.datasets2.reader.mix_datasets import create_dataset_instance


def create_dataset(**dataset_config):

    # data loader
    task_dataset_path = [path for path in str(dataset_config["task_group"]).replace(" ", "").split(",") if path != ""]
    task_dataset_prob = [
        float(prob) for prob in str(dataset_config["task_group_prob"]).replace(" ", "").split(",") if prob != ""
    ]
    sub_dataset_type = [
        type_ for type_ in str(dataset_config["sub_dataset_type"]).replace(" ", "").split(",") if type_ != ""
    ]

    if not (len(task_dataset_path) == len(task_dataset_prob) == len(sub_dataset_type)):
        raise ValueError("The len of dataset path, prob, type are inconsistent, please check the configuration.")

    if len(task_dataset_path) == 0:
        raise ValueError("The len of dataset path is zero, please check the configuration.")

    multi_source_dataset = MultiSourceDataset(
        task_dataset_path=task_dataset_path,
        task_dataset_prob=task_dataset_prob,
        sub_dataset_type=sub_dataset_type,
    )

    datasets_list = [task["dataset"] for task in multi_source_dataset._task_group]
    datasets_prob = [task["prob"] for task in multi_source_dataset._task_group]

    if dataset_config["mix_strategy"] not in [
        "random",
        "concat",
        "interleave_under",
        "interleave_over",
    ]:
        raise ValueError(f"Unsupported mix strategy: {dataset_config['mix_strategy']}")
    else:
        mix_datasets = create_dataset_instance(
            dataset_config["mix_strategy"],
            datasets_list,
            datasets_prob,
            ("upsampling" if dataset_config["mix_strategy"] == "interleave_under" else "oversampling"),
            dataset_config["random_seed"],
            dataset_config["random_shuffle"],
            dataset_config["num_samples_each_epoch"],
            dataset_config["reverse"],
        )

    print(mix_datasets)
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
    # dataset = dataset.map(
    #     dataset_processor.preprocess_dataset,
    #     batched=True,
    #     batch_size=data_args.preprocessing_batch_size,
    #     remove_columns=column_names,
    #     **kwargs,
    # )
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
    }

    create_dataset(
        task_group="/root/paddlejob/workspace/env/output/lrl/PaddleFormers/data/sft/train.jsonl",
        task_group_prob="1.0",
        sub_dataset_type="erniekit",
        **dataset_config,
    )

